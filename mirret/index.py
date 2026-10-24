"""Dựng chỉ mục.

  python -m mirret.index dense --level parents [--emb thư_mục_emb_1,thư_mục_2]   # FAISS IVF-SQ8 (hoặc Flat nếu nhỏ)
  python -m mirret.index bm25-tokenize --level parents --lang vi [--shard i/N]   # tách từ (chia việc được)
  python -m mirret.index bm25-build    --level parents --lang vi                 # thống kê toàn cục + ma trận BM25
  python -m mirret.index bm25 --level parents                                    # cả 3 ngôn ngữ, tách từ + dựng

Kết quả: index/dense_<mức>.faiss (+ _ids.parquet, .json), index/bm25/<mức>_<lang>/.
Chỉ lấy vector khớp bản chunks hiện tại (cặp unit_id + h); đơn vị chưa có vector → cảnh báo (cần mã hoá bổ sung).
LSR không cần dựng: mirret.retrieve đọc thẳng sparse.npz của các khối emb.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import bm25, config
from .encode import done_blocks


def current_pairs(chunk_dir):
    """{unit_id: h} của bản chunks hiện tại."""
    out = {}
    for p in sorted(Path(chunk_dir).glob("part-*.parquet")):
        t = pq.read_table(p, columns=["unit_id", "h"])
        out.update(zip(t.column("unit_id").to_pylist(), t.column("h").to_pylist()))
    return out


def emb_dirs(cfg, override=None):
    if override:
        return [Path(x) for x in str(override).split(",") if x]
    return [cfg.path("emb")]


def _selection(blocks, cur):
    """Mỗi khối → mảng chỉ số dòng dùng được (khớp h hiện tại, chưa lấy ở khối trước)."""
    taken, sel = set(), []
    for b in blocks:
        t = pq.read_table(b / "ids.parquet", columns=["unit_id", "h"])
        ids, hs = t.column("unit_id").to_pylist(), t.column("h").to_pylist()
        rows = []
        for i, (u, h) in enumerate(zip(ids, hs)):
            if cur.get(u) == h and u not in taken:
                taken.add(u)
                rows.append(i)
        sel.append((b, np.array(rows, dtype=np.int64), [ids[i] for i in rows]))
    return sel, taken


def build_dense(cfg, level, emb=None, force=False, log=print):
    import faiss
    dc = cfg["index"]["dense"]
    out = cfg.path("index")
    out.mkdir(parents=True, exist_ok=True)
    fidx = out / f"dense_{level}.faiss"
    if fidx.exists() and not force:
        log(f"{fidx} đã có → bỏ qua (--force để dựng lại)")
        return
    blocks = done_blocks(emb_dirs(cfg, emb), level)
    if not blocks:
        raise SystemExit(f"Không có khối emb mức {level} trong {emb_dirs(cfg, emb)}")
    cur = current_pairs(cfg.path("chunks") / level)
    sel, taken = _selection(blocks, cur)
    total = sum(len(r) for _, r, _ in sel)
    missing = len(cur) - len(taken)
    if missing:
        log(f"[CẢNH BÁO] {missing:,}/{len(cur):,} đơn vị {level} chưa có vector (chưa mã hoá hoặc đã đổi nội dung) → "
            f"chạy `mirret.encode run --level {level} --skip <emb hiện có>` rồi dựng lại")
    dim = np.load(blocks[0] / "dense.npy", mmap_mode="r").shape[1]
    if total < dc["flat_below"]:
        index, kind = faiss.IndexFlatIP(dim), "flat"
    else:
        nlist = min(dc["nlist"], max(1, total // 39))
        index = faiss.index_factory(dim, f"IVF{nlist},SQ8", faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(cfg["seed"])
        frac = min(1.0, dc["train_size"] / total)
        xs = []
        for b, rows, _ in sel:
            if len(rows):
                pick = rows[rng.random(len(rows)) < frac]
                if len(pick):
                    xs.append(np.asarray(np.load(b / "dense.npy", mmap_mode="r")[pick], dtype=np.float32))
        x = np.vstack(xs)
        t0 = time.time()
        index.train(x)
        kind = f"IVF{nlist},SQ8"
        log(f"huấn luyện {kind} trên {len(x):,} vector: {time.time() - t0:.0f}s")
        del x
    ids = []
    for b, rows, bid in sel:
        if len(rows):
            index.add(np.asarray(np.load(b / "dense.npy", mmap_mode="r")[rows], dtype=np.float32))
            ids += bid
    faiss.write_index(index, str(fidx))
    pq.write_table(pa.table({"unit_id": pa.array(ids, pa.string())}), out / f"dense_{level}_ids.parquet")
    (out / f"dense_{level}.json").write_text(json.dumps({"kind": kind, "n": index.ntotal, "dim": dim,
                                                         "missing": missing, "blocks": [str(b) for b, _, _ in sel]},
                                                        indent=1), encoding="utf-8")
    log(f"→ {fidx}: {index.ntotal:,} vector ({kind})")


def dense_search(cfg, level, qvecs, k):
    import faiss
    out = cfg.path("index")
    index = faiss.read_index(str(out / f"dense_{level}.faiss"))
    try:
        faiss.extract_index_ivf(index).nprobe = cfg["index"]["dense"]["nprobe"]
    except Exception:
        pass
    ids = pq.read_table(out / f"dense_{level}_ids.parquet").column("unit_id").to_pylist()
    D, I = index.search(np.asarray(qvecs, dtype=np.float32), min(k, index.ntotal))
    return [([ids[j] for j in row if j >= 0], [float(s) for s, j in zip(drow, row) if j >= 0])
            for drow, row in zip(D, I)]


def bm25_dir(cfg, level, lang):
    from pathlib import Path as _P
    base = _P(cfg["bm25"]["index_dir"]) if cfg["bm25"].get("index_dir") else cfg.path("index") / "bm25"
    return base / f"{level}_{lang}"


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dense", "bm25-tokenize", "bm25-build", "bm25"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--level", default="parents", choices=["parents", "children"])
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--emb", default=None, help="thư mục emb (phẩy ngăn cách); mặc định <work>/emb")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    bc = cfg["bm25"]
    if a.cmd == "dense":
        build_dense(cfg, a.level, a.emb, a.force)
        return
    langs = cfg["retrieve"]["langs"] if a.cmd == "bm25" else [a.lang]
    for lg in langs:
        d = bm25_dir(cfg, a.level, lg)
        if a.cmd in ("bm25-tokenize", "bm25"):
            ud = bc.get("zh_userdict")
            if ud and not Path(ud).is_absolute():
                ud = config.ROOT / ud                     # đi kèm mã, như translate.glossary
            bm25.tokenize_parts(cfg.path("chunks") / a.level, d, lg, a.shard, bc["workers"], bc["vi_tokenizer"],
                                a.force, en_stem=bc.get("en_stem", False), zh_t2s=bc.get("zh_t2s", False),
                                zh_userdict=ud)
        if a.cmd in ("bm25-build", "bm25"):
            bm25.build(d, bc["k1"], bc["b"], a.force)


if __name__ == "__main__":
    main()
