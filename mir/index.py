"""T4: dựng chỉ mục cho 3 nhánh truy hồi.

  python -m mir.index dense        --config ...   # FAISS: IndexFlatIP (< flat_below đoạn) hoặc IVF{nlist},SQ8
  python -m mir.index dense-tune   --config ...   # đo Recall@100 so với Flat trên mẫu ~100k đoạn → chọn nlist/nprobe
  python -m mir.index lsr          --config ...   # lexical weights (emb/*/sparse.npz) → chỉ mục đảo CSC theo shard
  python -m mir.index bm25-tokenize --config ... [--shard i/N]   # tách từ pyvi/jieba từng part (chia việc được)
  python -m mir.index bm25-build   --config ...   # thống kê toàn cục + ma trận BM25 theo shard

Kết quả: <work>/index/<strategy>/{dense.faiss, dense_ids.parquet, lsr/shard-*.npz, bm25/shard-*.npz, ...}
Bộ nhớ: mỗi nhánh nạp riêng khi truy hồi (T5); LSR/BM25 duyệt từng shard nên không cần nạp toàn bộ cùng lúc.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import bm25, config, sparse_index
from .encode import load_sparse


def emb_dirs(emb):
    """Một thư mục, chuỗi "a,b,c" hoặc danh sách (kết quả T3 của nhiều người nằm ở nhiều dataset)."""
    if isinstance(emb, (list, tuple)):
        return [Path(x) for x in emb]
    return [Path(x) for x in str(emb).split(",") if x]


def emb_shards(emb):
    """Mọi shard đã xong (_DONE.json) trong các thư mục emb, sắp theo số khối; báo lỗi nếu trùng hoặc thiếu khối."""
    out = sorted((d for e in emb_dirs(emb) for d in e.glob("shard-*") if (d / "_DONE.json").exists()),
                 key=lambda d: int(d.name.split("-")[1]))
    if not out:
        raise SystemExit(f"Không có shard nào đã xong trong {emb}")
    nums = [int(d.name.split("-")[1]) for d in out]
    if len(set(nums)) != len(nums):
        raise SystemExit("Trùng khối giữa các thư mục emb: " + str(sorted({n for n in nums if nums.count(n) > 1})))
    missing = sorted(set(range(max(nums) + 1)) - set(nums))
    if missing:
        print(f"[CẢNH BÁO] thiếu {len(missing)} khối emb: {missing[:20]}{'…' if len(missing) > 20 else ''}")
    return out


def _sample_rows(shards, n, seed):
    """Lấy ngẫu nhiên n vector (đều trên toàn bộ) → (ma trận float32, chunk_ids)."""
    sizes = [json.loads((d / "_DONE.json").read_text(encoding="utf-8"))["n"] for d in shards]
    total = sum(sizes)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total, size=min(n, total), replace=False))
    vecs, ids, off = [], [], 0
    for d, s in zip(shards, sizes):
        loc = pick[(pick >= off) & (pick < off + s)] - off
        if len(loc):
            x = np.load(d / "dense.npy", mmap_mode="r")
            vecs.append(np.asarray(x[loc], dtype=np.float32))
            cid = pq.read_table(d / "chunk_ids.parquet").column("chunk_id").to_pylist()
            ids += [cid[i] for i in loc]
        off += s
    return np.vstack(vecs), ids


def build_dense(emb_dir, out_dir, dc, seed=42, force=False, log=print):
    import faiss
    out = Path(out_dir)
    if (out / "dense.faiss").exists() and not force:
        log(f"{out / 'dense.faiss'} đã có → bỏ qua")
        return
    out.mkdir(parents=True, exist_ok=True)
    shards = emb_shards(emb_dir)
    total = sum(json.loads((d / "_DONE.json").read_text(encoding="utf-8"))["n"] for d in shards)
    dim = np.load(shards[0] / "dense.npy", mmap_mode="r").shape[1]
    if total < dc["flat_below"]:
        index, kind = faiss.IndexFlatIP(dim), "flat"
    else:
        nlist = min(dc["nlist"], max(1, total // 39))
        index = faiss.index_factory(dim, f"IVF{nlist},SQ8", faiss.METRIC_INNER_PRODUCT)
        x, _ = _sample_rows(shards, dc["train_size"], seed)
        t0 = time.time()
        index.train(x)
        kind = f"IVF{nlist},SQ8"
        log(f"huấn luyện {kind} trên {len(x):,} vector: {time.time() - t0:.0f}s")
    ids = []
    for d in shards:
        index.add(np.asarray(np.load(d / "dense.npy"), dtype=np.float32))
        ids += pq.read_table(d / "chunk_ids.parquet").column("chunk_id").to_pylist()
        log(f"  thêm {d.name}: tổng {index.ntotal:,}")
    faiss.write_index(index, str(out / "dense.faiss"))
    pq.write_table(pa.table({"chunk_id": ids}), out / "dense_ids.parquet")
    (out / "dense.json").write_text(json.dumps({"kind": kind, "n": index.ntotal, "dim": dim, "config": dc,
                                                "emb_dir": str(emb_dir)}, indent=1), encoding="utf-8")


def dense_search(index_dir, qvecs, k, nprobe):
    import faiss
    index = faiss.read_index(str(Path(index_dir) / "dense.faiss"))
    try:
        faiss.extract_index_ivf(index).nprobe = nprobe
    except Exception:                                    # IndexFlatIP: không có nprobe
        pass
    ids = pq.read_table(Path(index_dir) / "dense_ids.parquet").column("chunk_id").to_pylist()
    D, I = index.search(np.asarray(qvecs, dtype=np.float32), min(k, index.ntotal))
    return [([ids[j] for j in row if j >= 0], [float(s) for s, j in zip(drow, row) if j >= 0]) for drow, row in zip(D, I)]


def tune_dense(emb_dir, tc, qvecs=None, seed=42, log=print):
    """Recall@k của IVF-SQ8 (nhiều nlist × nprobe) so với Flat, trên mẫu tc.sample đoạn."""
    import faiss
    shards = emb_shards(emb_dir)
    x, _ = _sample_rows(shards, tc["sample"] + (0 if qvecs is not None else tc["n_queries"]), seed)
    if qvecs is None:                                     # không có câu hỏi: giữ lại một phần đoạn làm truy vấn
        q, x = x[:tc["n_queries"]], x[tc["n_queries"]:]
        qsrc = "đoạn giữ lại"
    else:
        q, qsrc = np.asarray(qvecs, dtype=np.float32)[:tc["n_queries"]], "câu hỏi thật"
    k = min(tc["k"], len(x))
    flat = faiss.IndexFlatIP(x.shape[1])
    flat.add(x)
    _, gt = flat.search(q, k)
    rows = []
    for nlist in tc["nlists"]:
        if nlist * 39 > len(x):
            log(f"  bỏ nlist={nlist}: cần >= {nlist * 39:,} vector để huấn luyện")
            continue
        idx = faiss.index_factory(x.shape[1], f"IVF{nlist},SQ8", faiss.METRIC_INNER_PRODUCT)
        idx.train(x)
        idx.add(x)
        ivf = faiss.extract_index_ivf(idx)
        for nprobe in tc["nprobes"]:
            if nprobe > nlist:
                continue
            ivf.nprobe = nprobe
            t0 = time.time()
            _, I = idx.search(q, k)
            ms = (time.time() - t0) / len(q) * 1000
            rec = np.mean([len(set(a) & set(b)) / k for a, b in zip(I.tolist(), gt.tolist())])
            rows.append({"nlist": nlist, "nprobe": nprobe, f"recall@{k}": round(float(rec), 4), "ms_per_query": round(ms, 2)})
            log(f"  nlist={nlist} nprobe={nprobe}: Recall@{k}={rec:.4f}, {ms:.2f} ms/câu")
    md = [f"# T4 — Chọn nlist/nprobe cho IVF-SQ8 (so với IndexFlatIP)", "",
          f"Mẫu {len(x):,} đoạn, {len(q):,} truy vấn ({qsrc}), k = {k}.", "",
          f"| nlist | nprobe | Recall@{k} | ms/câu |", "|---:|---:|---:|---:|"]
    md += [f"| {r['nlist']} | {r['nprobe']} | {r[f'recall@{k}']:.4f} | {r['ms_per_query']} |" for r in rows]
    md += ["", "Ghi chú: trên toàn kho (~8,6 triệu đoạn) nên giữ cùng tỷ lệ nprobe/nlist hoặc đo lại trên mẫu lớn hơn."]
    return rows, "\n".join(md) + "\n"


def build_lsr(emb_dir, out_dir, force=False, log=print):
    """Tuỳ chọn: lưu bản CSC của lexical weights. T5 cũng đọc thẳng được từ emb (retrieve --lsr-from-emb)."""
    out = Path(out_dir) / "lsr"
    for k, d in enumerate(emb_shards(emb_dir)):
        n = int(d.name.split("-")[1])
        if (out / f"shard-{n:04d}.npz").exists() and not force:
            continue
        X = load_sparse(d / "sparse.npz")
        ids = pq.read_table(d / "chunk_ids.parquet").column("chunk_id").to_pylist()
        sparse_index.write_shard(out, n, X, ids)
        log(f"  lsr {d.name}: {X.shape[0]:,} đoạn, nnz {X.nnz:,}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dense", "dense-tune", "lsr", "bm25-tokenize", "bm25-build"])
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--emb", default=None, help="mặc định <work>/emb/<strategy>; nhiều thư mục: a,b,c")
    ap.add_argument("--chunks", default=None, help="mặc định <work>/chunks/<strategy>")
    ap.add_argument("--out", default=None, help="mặc định <work>/index/<strategy>")
    ap.add_argument("--qemb", default=None, help="dense-tune: file vector câu hỏi (qemb .npz) làm truy vấn")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    strategy = a.strategy or cfg["chunk"]["strategy"]
    emb = a.emb if a.emb else cfg.work_dir / cfg["encode"]["out_dir"] / strategy
    chunks = Path(a.chunks) if a.chunks else cfg.work_dir / cfg["chunk"]["out_dir"] / strategy
    out = Path(a.out) if a.out else cfg.work_dir / cfg["index"]["out_dir"] / strategy
    if a.cmd == "dense":
        build_dense(emb, out, cfg["index"]["dense"], cfg["seed"], a.force)
    elif a.cmd == "dense-tune":
        qv = np.load(a.qemb)["dense"] if a.qemb else None
        rows, md = tune_dense(emb, cfg["index"]["tune"], qv, cfg["seed"])
        res = cfg.work_dir / "results"
        res.mkdir(parents=True, exist_ok=True)
        (res / "t4_dense_tune.md").write_text(md, encoding="utf-8")
        (res / "t4_dense_tune.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(md)
    elif a.cmd == "lsr":
        build_lsr(emb, out, a.force)
    elif a.cmd == "bm25-tokenize":
        bm25.tokenize_parts(chunks, out / "bm25", a.shard, cfg["bm25"]["tokenize_workers"], a.force)
    else:
        bm25.build(out / "bm25", cfg["bm25"]["k1"], cfg["bm25"]["b"], a.force)


if __name__ == "__main__":
    main()
