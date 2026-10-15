"""Chỉ mục đảo dạng ma trận CSC theo shard — dùng chung cho LSR (lexical weights BGE-M3) và BM25 (từ mir/ cũ).

Shard chỉ mục: <dir>/shard-XXXX.npz (CSC, hàng = đơn vị, cột = token/từ, data float16) + shard-XXXX_ids.parquet.
Shard emb (LSR đọc thẳng kết quả mã hoá, không lưu bản sao): <emb>/<khối>/sparse.npz (CSR) + ids.parquet.
Điểm = tích vô hướng vector đơn vị × vector câu hỏi trên các cột của câu hỏi; lấy top-k từng shard rồi gộp.
"""
import heapq
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def save_csc(path, m):
    m = m.tocsc()
    np.savez(path, format=np.array("csc"), shape=np.array(m.shape), indptr=m.indptr.astype(np.int64),
             indices=m.indices.astype(np.int32), data=m.data.astype(np.float16))


def load_csc(path, dtype=np.float32):
    import scipy.sparse as sp
    z = np.load(path)
    return sp.csc_matrix((z["data"].astype(dtype), z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def save_csr(path, indptr, indices, data, n_cols):
    """CSR với data float16 (scipy không lưu float16 → lưu từng mảng)."""
    np.savez(path, format=np.array("csr"), shape=np.array([len(indptr) - 1, n_cols]),
             indptr=np.asarray(indptr, dtype=np.int64), indices=np.asarray(indices, dtype=np.int32),
             data=np.asarray(data, dtype=np.float16))


def load_csr(path, dtype=np.float32):
    import scipy.sparse as sp
    z = np.load(path)
    return sp.csr_matrix((z["data"].astype(dtype), z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def write_shard(dirpath, k, matrix, unit_ids):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    save_csc(d / f"shard-{k:04d}.npz", matrix)
    pq.write_table(pa.table({"unit_id": list(unit_ids)}), d / f"shard-{k:04d}_ids.parquet")


def index_shards(dirpath):
    return sorted(Path(dirpath).glob("shard-*.npz"))


def providers(sources, allowed=None):
    """[(tên, hàm nạp → (CSC, ids))]. sources: thư mục chỉ mục CSC, hoặc danh sách thư mục khối emb
    (<khối>/sparse.npz + ids.parquet). allowed: tập "unit_id:h" hợp lệ (bỏ vector cũ của đoạn đã đổi nội dung)."""
    out = []
    for d in ([sources] if isinstance(sources, (str, Path)) else sources):
        d = Path(d)
        if (d / "sparse.npz").exists():                   # một khối emb
            blocks = [d]
        else:
            blocks = sorted(x for x in d.glob("*") if (x / "_DONE.json").exists() and (x / "sparse.npz").exists())
        for p in index_shards(d):
            def load_idx(p=p):
                return load_csc(p), pq.read_table(str(p).replace(".npz", "_ids.parquet")).column("unit_id").to_pylist()
            out.append((p.name, load_idx))
        for b in blocks:
            def load_blk(b=b):
                X = load_csr(b / "sparse.npz")
                t = pq.read_table(b / "ids.parquet", columns=["unit_id", "h"])
                ids, hs = t.column("unit_id").to_pylist(), t.column("h").to_pylist()
                if allowed is not None:
                    keep = np.array([f"{i}:{h}" in allowed for i, h in zip(ids, hs)], dtype=bool)
                    if not keep.all():
                        X, ids = X[keep], [i for i, k in zip(ids, keep) if k]
                return X.tocsc(), ids
            out.append((b.name, load_blk))
    return out


def search(sources, queries, k, batch=64, log=None, allowed=None):
    """queries: list[(term_ids, weights)] → list[(unit_ids, scores)] giảm dần, mỗi câu tối đa k."""
    nq = len(queries)
    heaps = [[] for _ in range(nq)]
    for name, load in providers(sources, allowed):
        X, ids = load()
        for b0 in range(0, nq, batch):
            qb = queries[b0:b0 + batch]
            cols = np.unique(np.concatenate([np.asarray(q[0], dtype=np.int64) for q in qb] +
                                            [np.zeros(0, np.int64)]))
            cols = cols[cols < X.shape[1]]
            if len(cols) == 0:
                continue
            pos = {c: i for i, c in enumerate(cols.tolist())}
            Q = np.zeros((len(cols), len(qb)), dtype=np.float32)
            for j, (t, w) in enumerate(qb):
                for tt, ww in zip(np.asarray(t).tolist(), np.asarray(w).tolist()):
                    if tt in pos:
                        Q[pos[tt], j] += ww
            S = np.asarray(X[:, cols] @ Q)
            kk = min(k, S.shape[0])
            for j in range(len(qb)):
                col = S[:, j]
                nz = np.flatnonzero(col > 0)
                if len(nz) == 0:
                    continue
                if len(nz) > kk:
                    nz = nz[np.argpartition(-col[nz], kk - 1)[:kk]]
                h = heaps[b0 + j]
                for r in nz.tolist():
                    item = (float(col[r]), ids[r])
                    if len(h) < k:
                        heapq.heappush(h, item)
                    elif item > h[0]:
                        heapq.heapreplace(h, item)
        if log:
            log(f"    {name}: {X.shape[0]:,} đơn vị")
    out = []
    for h in heaps:
        h = sorted(h, key=lambda x: (-x[0], x[1]))
        seen, ids, sc = set(), [], []
        for s_, c in h:                                   # cùng đơn vị ở hai khối (mã hoá lặp) → giữ một
            if c not in seen:
                seen.add(c)
                ids.append(c)
                sc.append(s_)
        out.append((ids, sc))
    return out
