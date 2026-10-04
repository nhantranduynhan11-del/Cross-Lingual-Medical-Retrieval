"""Chỉ mục đảo dạng ma trận CSC theo shard (dùng chung cho LSR và BM25).

Mỗi shard: <dir>/shard-XXXX.npz (CSC: hàng = đoạn, cột = token/từ; data float16) + shard-XXXX_ids.parquet.
Điểm = tích vô hướng giữa vector đoạn và vector câu hỏi trên các cột của câu hỏi:
  với mỗi lô câu hỏi, cắt X[:, các cột xuất hiện trong lô] (truy cập theo cột của CSC = danh sách đăng),
  nhân với ma trận câu hỏi thu gọn → điểm [số đoạn × số câu hỏi], rồi lấy top-k và gộp giữa các shard.
"""
import heapq
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def save_csc(path, m):
    """m: scipy.sparse (bất kỳ) → lưu CSC với data float16."""
    m = m.tocsc()
    np.savez(path, format=np.array("csc"), shape=np.array(m.shape), indptr=m.indptr.astype(np.int64),
             indices=m.indices.astype(np.int32), data=m.data.astype(np.float16))


def load_csc(path, dtype=np.float32):
    import scipy.sparse as sp
    z = np.load(path)
    return sp.csc_matrix((z["data"].astype(dtype), z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def write_shard(dirpath, k, matrix, chunk_ids):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    save_csc(d / f"shard-{k:04d}.npz", matrix)
    pq.write_table(pa.table({"chunk_id": list(chunk_ids)}), d / f"shard-{k:04d}_ids.parquet")


def shards(dirpath):
    return sorted(p for p in Path(dirpath).glob("shard-*.npz"))


def providers(dirs):
    """Nguồn shard: thư mục chỉ mục (shard-XXXX.npz, CSC) hoặc thư mục emb (shard-XXXX/sparse.npz, CSR — LSR đọc
    thẳng từ kết quả T3, chuyển CSC trong bộ nhớ, không cần lưu bản sao). dirs: một hoặc nhiều thư mục."""
    from .encode import load_sparse
    out = []
    for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs):
        d = Path(d)
        for p in shards(d):
            out.append((p.name, lambda p=p: load_csc(p), str(p).replace(".npz", "_ids.parquet")))
        for s in sorted(x for x in d.glob("shard-*") if (x / "_DONE.json").exists()):
            out.append((s.name, lambda s=s: load_sparse(s / "sparse.npz").tocsc(), str(s / "chunk_ids.parquet")))
    return out


def search(dirpath, queries, k, batch=64, log=None):
    """queries: list[(term_ids np.int, weights np.float)] → list[(chunk_ids, scores)] đã sắp giảm dần, mỗi câu tối đa k.
    dirpath: thư mục (hoặc danh sách thư mục) chỉ mục CSC hay emb."""
    nq = len(queries)
    heaps = [[] for _ in range(nq)]                      # heap nhỏ nhất: (score, chunk_id)
    for name, load, ids_path in providers(dirpath):
        X = load()
        ids = pq.read_table(ids_path).column("chunk_id").to_pylist()
        for b0 in range(0, nq, batch):
            qb = queries[b0:b0 + batch]
            cols = np.unique(np.concatenate([q[0] for q in qb] + [np.zeros(0, np.int64)])).astype(np.int64)
            cols = cols[cols < X.shape[1]]
            if len(cols) == 0:
                continue
            pos = {c: i for i, c in enumerate(cols.tolist())}
            Q = np.zeros((len(cols), len(qb)), dtype=np.float32)
            for j, (t, w) in enumerate(qb):
                for tt, ww in zip(np.asarray(t).tolist(), np.asarray(w).tolist()):
                    if tt in pos:
                        Q[pos[tt], j] += ww
            S = X[:, cols] @ Q                           # [số đoạn, số câu hỏi trong lô]
            S = np.asarray(S)
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
            log(f"  {name}: {X.shape[0]:,} đoạn")
    out = []
    for h in heaps:
        h = sorted(h, key=lambda x: (-x[0], x[1]))
        out.append(([c for _, c in h], [s for s, _ in h]))
    return out
