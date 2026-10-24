"""Stage 1: 9 nhánh độc lập = câu hỏi {vi, en, zh} × {Dense, LSR, BM25}; mỗi nhánh trả về danh sách ĐOẠN CHA.

Câu hỏi: qtrans/queries_{vi,en,zh}.parquet (id, query, lang) do `mirret.translate build` tạo (vi = câu gốc của BTC).
Mỗi nhánh (lang, method):
  - tìm ở từng mức trong retrieve.levels (BM25: bm25.levels); kết quả mức đoạn con được dồn về đoạn cha (đoạn con tốt
    nhất đại diện) → run mức: runs/<lang>_<method>@<mức>.parquet (giữ lại để đo đóng góp của đoạn con);
  - nhiều mức → gộp RRF trong nhánh → runs/<lang>_<method>.parquet (độ sâu retrieve.depth đoạn cha).
Dense: FAISS index/dense_<mức>.faiss; LSR: đọc thẳng sparse.npz của các khối emb; BM25: index/bm25/<mức>_<lang>.

  python -m mirret.retrieve qencode --device cuda:0 [--langs vi,en,zh]     # → qemb/<lang>.npz
  python -m mirret.retrieve run [--langs vi,en,zh] [--methods dense,lsr,bm25] [--emb a,b]
Biến thể câu hỏi (mirret/expand.py, vd Query2Doc / HyDE): --variant q2d → câu hỏi ở qexp/q2d/queries_<lang>.parquet
(BM25 dùng queries_<lang>_bm25.parquet nếu có), qemb/q2d_<lang>.npz, runs/q2d_<lang>_<method>[@mức].parquet.

Rewrite + thuật ngữ (mirret/query_expand.py, cờ trong qexpand; tắt hết → không chạy gì, `run` không đổi):
  python -m mirret.retrieve qencode --variant rw ; python -m mirret.retrieve qencode --variant rwt   (+ câu gốc: qencode)
  python -m mirret.retrieve qx [--langs …] [--methods …] [--emb …]
  → runs/<qexpand.prefix>_<lang>_<method>[@mức].parquet, chỉ cho nhánh có cờ bật:
    use_rewrite_dense : dense tìm bằng [câu gốc, rewrite], gộp fuse_variants(qexpand.fuse_how) ở từng mức
    use_rewrite_sparse: LSR thêm phiên bản rewrite; BM25 thêm rewrite × w_rewrite
    use_terms         : LSR thêm phiên bản "rewrite ; thuật ngữ" (qexp/rwt); BM25 thêm thuật ngữ × w_terms
    w_cand > 0        : BM25 thêm candidates (chẩn đoán phỏng đoán) × w_cand
  Gộp với các nhánh còn lại: python -m mirret.fuse --runs qx_vi_dense,vi_lsr,… --linear 0.7 --name lin07qx
"""
import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from . import bm25, config, fuse, runs, sparse_index
from .encode import done_blocks, load_qemb


def queries_path(cfg, lang, variant=None, bm25=False):
    if not variant:
        return cfg.path("qtrans") / f"queries_{lang}.parquet"
    d = cfg.path("qexp") / variant
    if bm25 and (d / f"queries_{lang}_bm25.parquet").exists():
        return d / f"queries_{lang}_bm25.parquet"
    return d / f"queries_{lang}.parquet"


def prefix(variant):
    return f"{variant}_" if variant else ""


def qemb_path(cfg, lang, variant=None):
    return cfg.path("qemb") / f"{prefix(variant)}{lang}.npz"


def load_queries(path):
    t = pq.read_table(path).to_pydict()
    return [int(x) for x in t["id"]], list(t["query"])


def qencode(cfg, langs, device, variant=None, log=print):
    from .encode import M3Encoder, encode_texts, resolve_device
    ec = cfg["encode"]
    enc = M3Encoder.load(ec["model"], resolve_device(device), ec["fp16"], ec["max_length"])
    for lg in langs:
        p = queries_path(cfg, lg, variant)
        if not p.exists():
            log(f"  [{lg}] chưa có {p} → bỏ qua")
            continue
        qids, texts = load_queries(p)
        out = encode_texts(enc, qids, texts, qemb_path(cfg, lg, variant), ec["batch_tokens"])
        log(f"  [{lg}] {len(qids)} câu → {out}")


def _depth_for(level, depth):
    return depth * 3 if level == "children" else depth   # nhiều đoạn con cùng một đoạn cha → lấy sâu hơn


def run_level(cfg, lang, method, level, depth, emb=None, allowed=None, variant=None, log=print, bm25_q=None):
    """→ run mức đoạn cha (đã dồn đoạn con về đoạn cha), hoặc None nếu thiếu dữ liệu.
    bm25_q: hàm (thư mục chỉ mục BM25) → (qids, [(id từ, trọng số)]) thay cho câu hỏi đọc từ file (BM25 có trọng số)."""
    rc = cfg["retrieve"]
    d = _depth_for(level, depth)
    if method in ("dense", "lsr"):
        qp = qemb_path(cfg, lang, variant)
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
        if bm25_q is not None:
            if not (bd / "vocab.parquet").exists():
                log(f"    thiếu chỉ mục {bd}")
                return None
            qids, encoded = bm25_q(bd)
            res = sparse_index.search(bd, encoded, d, rc["query_batch"])
        else:
            qp = queries_path(cfg, lang, variant, bm25=True)
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


def run_all(cfg, langs, methods, emb=None, variant=None, log=print):
    rc, rdir = cfg["retrieve"], cfg.path("runs")
    depth = rc["depth"]
    allowed = {}                                           # LSR: chỉ vector khớp bản chunks hiện tại
    for method in methods:
        levels = cfg["bm25"]["levels"] if method == "bm25" else rc["levels"]
        for lang in langs:
            t0 = time.time()
            level_runs = {}
            name = f"{prefix(variant)}{lang}_{method}"
            for level in levels:
                al = None
                if method == "lsr":
                    if level not in allowed:
                        from .index import current_pairs
                        allowed[level] = {f"{u}:{h}" for u, h in current_pairs(cfg.path("chunks") / level).items()}
                    al = allowed[level]
                df = run_level(cfg, lang, method, level, depth, emb, al, variant, log)
                if df is None:
                    continue
                level_runs[level] = df
                runs.write(df, rdir / f"{name}@{level}.parquet",
                           {"lang": lang, "method": method, "level": level, "depth": depth, "variant": variant})
            write_branch(cfg, rdir / f"{name}.parquet", level_runs, depth, t0,
                         {"lang": lang, "method": method, "variant": variant}, log)


def write_branch(cfg, path, level_runs, depth, t0, meta, log=print):
    """Các run mức → run nhánh (một mức: giữ nguyên; nhiều mức: RRF) → path."""
    if not level_runs:
        log(f"  {path.stem}: KHÔNG có mức nào chạy được")
        return
    if len(level_runs) == 1:
        br = next(iter(level_runs.values()))
    else:
        br = fuse.rrf(list(level_runs.values()), cfg["fuse"]["rrf_k"], depth)
    runs.write(br, path, {**meta, "levels": list(level_runs), "depth": depth,
                          "level_fusion": "rrf" if len(level_runs) > 1 else None})
    log(f"  {path.stem}: {br['qid'].nunique()} câu hỏi, mức {list(level_runs)}, {time.time() - t0:.0f}s")


def qx_flags(cfg):
    """Cờ của qexpand → (bật dense, bật LSR, bật BM25, cấu hình)."""
    xc = cfg.get("qexpand") or {}
    rd, rs, te = (bool(xc.get(k)) for k in ("use_rewrite_dense", "use_rewrite_sparse", "use_terms"))
    cand = float(xc.get("w_cand") or 0) > 0
    return rd, rs or te, rs or te or cand, xc


def run_qx(cfg, langs, methods, emb=None, log=print):
    """Nhánh có rewrite / thuật ngữ (mirret/query_expand.py) theo cờ qexpand → runs/<prefix>_<lang>_<method>[@mức].
    Dense / LSR: mỗi phiên bản câu hỏi (câu gốc qemb/<lang>.npz, rewrite qemb/rw_<lang>.npz, rewrite + thuật ngữ
    qemb/rwt_<lang>.npz) một lượt tìm, gộp bằng query_expand.fuse_variants ở từng mức (đoạn con đã dồn về đoạn cha);
    BM25: một câu hỏi có trọng số (WeightedQueryEncoder) từ qexp/<name>/terms_<lang>.parquet."""
    from . import query_expand as qx
    use_dense, use_lsr, use_bm25, xc = qx_flags(cfg)
    if not (use_dense or use_lsr or use_bm25):
        log("qexpand: mọi cờ đang tắt (use_rewrite_dense / use_rewrite_sparse / use_terms / w_cand) → không chạy gì")
        return
    rc, rdir = cfg["retrieve"], cfg.path("runs")
    depth, name, pre, how = rc["depth"], xc.get("name", "rw"), xc.get("prefix", "qx"), xc.get("fuse_how", "max")
    vsets = {"dense": [None, name],
             "lsr": [None] + [name] * bool(xc.get("use_rewrite_sparse")) + [f"{name}t"] * bool(xc.get("use_terms"))}
    meta = {k: xc.get(k) for k in ("use_rewrite_dense", "use_rewrite_sparse", "use_terms", "w_base", "w_rewrite",
                                   "w_terms", "w_cand", "max_df_ratio", "fuse_how", "name")}
    allowed = {}
    for method in methods:
        if not {"dense": use_dense, "lsr": use_lsr, "bm25": use_bm25}[method]:
            continue
        levels = cfg["bm25"]["levels"] if method == "bm25" else rc["levels"]
        for lang in langs:
            t0, level_runs, bname = time.time(), {}, f"{pre}_{lang}_{method}"
            terms = qx.read_terms(cfg, lang, name) if method == "bm25" else None

            def bm25_q(bd, terms=terms, lang=lang):
                enc = qx.WeightedQueryEncoder(
                    bd, cfg["bm25"]["vi_tokenizer"], xc.get("w_base", 1.0),
                    xc.get("w_rewrite", 1.0) if xc.get("use_rewrite_sparse") else 0.0,
                    xc.get("w_terms", 0.5) if xc.get("use_terms") else 0.0, xc.get("max_df_ratio"), xc.get("w_cand", 0.0))
                qids = sorted(terms)
                return qids, [enc(lang, *terms[q]) for q in qids]
            for level in levels:
                al = None
                if method == "lsr":
                    if level not in allowed:
                        from .index import current_pairs
                        allowed[level] = {f"{u}:{h}" for u, h in current_pairs(cfg.path("chunks") / level).items()}
                    al = allowed[level]
                if method == "bm25":
                    df = run_level(cfg, lang, method, level, depth, emb, al, None, log, bm25_q=bm25_q)
                else:
                    dfs = [run_level(cfg, lang, method, level, depth, emb, al, v, log) for v in vsets[method]]
                    if any(d is None for d in dfs):
                        log(f"    {bname}@{level}: thiếu phiên bản {[v for v, d in zip(vsets[method], dfs) if d is None]}"
                            " (chạy retrieve qencode [--variant …])")
                        continue
                    df = qx.fuse_variants(dfs, how, depth=depth)
                if df is None:
                    continue
                level_runs[level] = df
                runs.write(df, rdir / f"{bname}@{level}.parquet",
                           {"lang": lang, "method": method, "level": level, "depth": depth, "qexpand": meta,
                            "variants": vsets.get(method, ["bm25 có trọng số"])})
            write_branch(cfg, rdir / f"{bname}.parquet", level_runs, depth, t0,
                         {"lang": lang, "method": method, "qexpand": meta}, log)


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qencode", "run", "qx"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--langs", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--emb", default=None, help="thư mục emb cho LSR (phẩy ngăn cách); mặc định <work>/emb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", default=None, help="biến thể câu hỏi trong qexp/ (mirret.expand build), vd q2d, hyde")
    a = ap.parse_args()
    cfg = config.load(a.config)
    rc = cfg["retrieve"]
    langs = a.langs.split(",") if a.langs else rc["langs"]
    if a.cmd == "qencode":
        qencode(cfg, langs, a.device, a.variant)
    else:
        methods = a.methods.split(",") if a.methods else rc["methods"]
        emb = [Path(x) for x in a.emb.split(",")] if a.emb else None
        if a.cmd == "qx":
            run_qx(cfg, langs, methods, emb)
        else:
            run_all(cfg, langs, methods, emb, a.variant)


if __name__ == "__main__":
    main()
