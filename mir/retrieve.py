"""T5 (phần 1): mã hoá câu hỏi và truy hồi từng nhánh, mỗi nhánh một run file độ sâu `retrieve.depth`.

  python -m mir.retrieve qencode --config ... --device cuda:0      # → <work>/qemb/<tên tập câu hỏi>.npz (dense + sparse)
  python -m mir.retrieve run --branch dense|lsr|bm25|all --config ...
       → <work>/runs/<strategy>_<nhánh>.parquet (+ .json cấu hình)
Gộp nhánh (RRF, có trọng số): mir.fuse.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import bm25, config, runs, sparse_index
from .index import dense_search


def load_queries(path):
    t = pq.read_table(path).to_pydict()
    return [int(x) for x in t["id"]], list(t["query"])


def encode_queries(enc, qids, texts, path, batch_tokens=16384, force=False):
    """Câu hỏi → npz: qids, dense float32 [n, H], sparse CSR (indptr, indices, data). Không thêm tiêu đề."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    dense, sparse, trunc = enc.encode(texts, batch_tokens=batch_tokens)
    indptr = np.concatenate([[0], np.cumsum([len(t) for t, _ in sparse])]).astype(np.int64)
    np.savez(path, qids=np.array(qids, dtype=np.int64), dense=dense.astype(np.float32), indptr=indptr,
             indices=np.concatenate([t for t, _ in sparse] + [np.zeros(0, np.int32)]).astype(np.int32),
             data=np.concatenate([w for _, w in sparse] + [np.zeros(0, np.float32)]).astype(np.float32),
             truncated=np.array(trunc))
    return path


def load_qemb(path):
    z = np.load(path)
    sp = [(z["indices"][a:b], z["data"][a:b]) for a, b in zip(z["indptr"][:-1], z["indptr"][1:])]
    return [int(q) for q in z["qids"]], z["dense"], sp


def run_branch(branch, index_dir, qemb_path, qids, qtexts, depth, batch=64, nprobe=64, log=print, lsr_source=None):
    """lsr_source: thư mục (hoặc danh sách) chỉ mục LSR; mặc định <index_dir>/lsr. Có thể là các thư mục emb (đọc thẳng)."""
    t0 = time.time()
    if branch == "dense":
        eq, dense, _ = load_qemb(qemb_path)
        assert eq == qids, "qemb không khớp tập câu hỏi"
        res = dense_search(index_dir, dense, depth, nprobe)
    elif branch == "lsr":
        eq, _, sp = load_qemb(qemb_path)
        assert eq == qids, "qemb không khớp tập câu hỏi"
        res = sparse_index.search(lsr_source or (Path(index_dir) / "lsr"), sp, depth, batch, log)
    elif branch == "bm25":
        qe = bm25.QueryEncoder(Path(index_dir) / "bm25")
        res = sparse_index.search(Path(index_dir) / "bm25", [qe(t) for t in qtexts], depth, batch, log)
    else:
        raise ValueError(branch)
    log(f"  {branch}: {len(qids)} câu hỏi, {time.time() - t0:.0f}s")
    return runs.from_topk(qids, [r[0] for r in res], [r[1] for r in res])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qencode", "run"])
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--queries", default=None, help="file câu hỏi parquet (id, query); mặc định retrieve.queries")
    ap.add_argument("--qemb", default=None, help="mặc định <work>/qemb/<tên file câu hỏi>.npz")
    ap.add_argument("--index", default=None, help="mặc định <work>/index/<strategy>")
    ap.add_argument("--out", default=None, help="thư mục runs (mặc định <work>/runs)")
    ap.add_argument("--branch", default="all")
    ap.add_argument("--emb", default=None, help="nhánh LSR đọc thẳng từ các thư mục emb (a,b,c) nếu chưa dựng index/lsr")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    rc = cfg["retrieve"]
    strategy = a.strategy or cfg["chunk"]["strategy"]
    qpath = cfg.path("queries", override=a.queries)
    qemb = Path(a.qemb) if a.qemb else cfg.path("qemb") / (qpath.stem + ".npz")
    qids, qtexts = load_queries(qpath)
    if a.cmd == "qencode":
        from .encode import M3Encoder
        ec = cfg["encode"]
        enc = M3Encoder.load(ec["model"], a.device, ec["fp16"], ec["max_length"])
        print("→", encode_queries(enc, qids, qtexts, qemb, ec["batch_tokens"], a.force))
        return
    index_dir = cfg.path("index", strategy, a.index)
    out = cfg.path("runs", override=a.out)
    nprobe = cfg["index"]["dense"]["nprobe"]
    lsr_src = None
    if not sparse_index.shards(index_dir / "lsr"):
        from .index import emb_dirs
        lsr_src = emb_dirs(a.emb) if a.emb else [cfg.path("emb", strategy)]
    for br in (["dense", "lsr", "bm25"] if a.branch == "all" else a.branch.split(",")):
        dst = out / f"{strategy}_{br}.parquet"
        if dst.exists() and not a.force:
            print(f"{dst} đã có → bỏ qua")
            continue
        df = run_branch(br, index_dir, qemb, qids, qtexts, rc["depth"], rc["query_batch"], nprobe, lsr_source=lsr_src)
        runs.write(df, dst, {"branch": br, "strategy": strategy, "queries": str(qpath), "index": str(index_dir),
                             "depth": rc["depth"], "nprobe": nprobe, "config": {k: cfg[k] for k in
                                                                                    ("index", "bm25", "encode")}},
                   force=a.force)
        print(f"→ {dst} ({len(df):,} dòng)")


if __name__ == "__main__":
    main()
