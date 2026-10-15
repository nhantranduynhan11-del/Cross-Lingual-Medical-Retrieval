"""Stage 1: 9 nhánh độc lập = câu hỏi {vi, en, zh} × {Dense, LSR, BM25}; mỗi nhánh trả về danh sách ĐOẠN CHA.

Câu hỏi: qtrans/queries_{vi,en,zh}.parquet (id, query, lang) do `mirret.translate build` tạo (vi = câu gốc của BTC).
Mỗi nhánh (lang, method):
  - tìm ở từng mức trong retrieve.levels (BM25: bm25.levels); kết quả mức đoạn con được dồn về đoạn cha (đoạn con tốt
    nhất đại diện) → run mức: runs/<lang>_<method>@<mức>.parquet (giữ lại để đo đóng góp của đoạn con);
  - nhiều mức → gộp RRF trong nhánh → runs/<lang>_<method>.parquet (độ sâu retrieve.depth đoạn cha).
Dense: FAISS index/dense_<mức>.faiss; LSR: đọc thẳng sparse.npz của các khối emb; BM25: index/bm25/<mức>_<lang>.

  python -m mirret.retrieve qencode --device cuda:0 [--langs vi,en,zh]     # → qemb/<lang>.npz
  python -m mirret.retrieve run [--langs vi,en,zh] [--methods dense,lsr,bm25] [--emb a,b]
"""
import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from . import bm25, config, fuse, runs, sparse_index
from .encode import done_blocks, load_qemb


def queries_path(cfg, lang):
    return cfg.path("qtrans") / f"queries_{lang}.parquet"


def load_queries(path):
    t = pq.read_table(path).to_pydict()
    return [int(x) for x in t["id"]], list(t["query"])


def qencode(cfg, langs, device, log=print):
    from .encode import M3Encoder, encode_texts, resolve_device
    ec = cfg["encode"]
    enc = M3Encoder.load(ec["model"], resolve_device(device), ec["fp16"], ec["max_length"])
    for lg in langs:
        p = queries_path(cfg, lg)
        if not p.exists():
            log(f"  [{lg}] chưa có {p} → bỏ qua")
            continue
        qids, texts = load_queries(p)
        out = encode_texts(enc, qids, texts, cfg.path("qemb") / f"{lg}.npz", ec["batch_tokens"])
        log(f"  [{lg}] {len(qids)} câu → {out}")


def _depth_for(level, depth):
    return depth * 3 if level == "children" else depth   # nhiều đoạn con cùng một đoạn cha → lấy sâu hơn


def run_level(cfg, lang, method, level, depth, emb=None, allowed=None, log=print):
    """→ run mức đoạn cha (đã dồn đoạn con về đoạn cha), hoặc None nếu thiếu dữ liệu."""
    rc = cfg["retrieve"]
    d = _depth_for(level, depth)
    if method in ("dense", "lsr"):
        qp = cfg.path("qemb") / f"{lang}.npz"
        if not qp.exists():
            log(f"    thiếu {qp}")
            return None
        qids, dense, sp = load_qemb(qp)
        if method == "dense":
            from .index import dense_search
            if not (cfg.path("index") / f"dense_{level}.faiss").exists():
                log(f"    thiếu chỉ mục dense_{level}")
                return None
            res = dense_search(cfg, level, dense, d)
        else:
            src = done_blocks(emb or [cfg.path("emb")], level)
            if not src:
                log(f"    thiếu khối emb mức {level}")
                return None
            res = sparse_index.search(src, sp, d, rc["query_batch"], allowed=allowed)
    else:
        from .index import bm25_dir
        bd = bm25_dir(cfg, level, lang)
        qp = queries_path(cfg, lang)
        if not (bd / "vocab.parquet").exists() or not qp.exists():
            log(f"    thiếu chỉ mục {bd} hoặc {qp}")
            return None
        qids, texts = load_queries(qp)
        qe = bm25.QueryEncoder(bd, cfg["bm25"]["vi_tokenizer"])
        res = sparse_index.search(bd, [qe(t, lang) for t in texts], d, rc["query_batch"])
    df = runs.from_topk(qids, [r[0] for r in res], [r[1] for r in res])
    if level == "children":
        df = runs.to_parents(df)
    return df[df["rank"] <= depth].reset_index(drop=True)


def run_all(cfg, langs, methods, emb=None, log=print):
    rc, rdir = cfg["retrieve"], cfg.path("runs")
    depth = rc["depth"]
    allowed = {}                                           # LSR: chỉ vector khớp bản chunks hiện tại
    for method in methods:
        levels = cfg["bm25"]["levels"] if method == "bm25" else rc["levels"]
        for lang in langs:
            t0 = time.time()
            level_runs = {}
            for level in levels:
                al = None
                if method == "lsr":
                    if level not in allowed:
                        from .index import current_pairs
                        allowed[level] = {f"{u}:{h}" for u, h in current_pairs(cfg.path("chunks") / level).items()}
                    al = allowed[level]
                df = run_level(cfg, lang, method, level, depth, emb, al, log)
                if df is None:
                    continue
                level_runs[level] = df
                runs.write(df, rdir / f"{lang}_{method}@{level}.parquet",
                           {"lang": lang, "method": method, "level": level, "depth": depth})
            if not level_runs:
                log(f"  {lang}_{method}: KHÔNG có mức nào chạy được")
                continue
            if len(level_runs) == 1:
                br = next(iter(level_runs.values()))
            else:
                br = fuse.rrf(list(level_runs.values()), cfg["fuse"]["rrf_k"], depth)
            runs.write(br, rdir / f"{lang}_{method}.parquet",
                       {"lang": lang, "method": method, "levels": list(level_runs), "depth": depth,
                        "level_fusion": "rrf" if len(level_runs) > 1 else None})
            log(f"  {lang}_{method}: {br['qid'].nunique()} câu hỏi, mức {list(level_runs)}, {time.time() - t0:.0f}s")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qencode", "run"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--langs", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--emb", default=None, help="thư mục emb cho LSR (phẩy ngăn cách); mặc định <work>/emb")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = config.load(a.config)
    rc = cfg["retrieve"]
    langs = a.langs.split(",") if a.langs else rc["langs"]
    if a.cmd == "qencode":
        qencode(cfg, langs, a.device)
    else:
        methods = a.methods.split(",") if a.methods else rc["methods"]
        emb = [Path(x) for x in a.emb.split(",")] if a.emb else None
        run_all(cfg, langs, methods, emb)


if __name__ == "__main__":
    main()
