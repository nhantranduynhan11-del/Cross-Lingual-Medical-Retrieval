import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from mir import bm25, encode, index, sparse_index

DOCS = [("vi", "Bệnh tăng huyết áp là bệnh mạn tính, cần điều trị lâu dài và giảm muối."),
        ("vi", "Đái tháo đường type 2 cần kiểm soát đường huyết, ăn uống hợp lý."),
        ("vi", "Tăng huyết áp ở người cao tuổi dễ gây đột quỵ nếu không điều trị."),
        ("zh", "高血压患者应该少吃盐，多运动，按时服药控制血压。"),
        ("zh", "糖尿病患者要控制血糖，注意饮食。"),
        ("vi", "Cách phòng bệnh cúm mùa: tiêm vắc xin, rửa tay thường xuyên.")]


def _write_chunks(tmp_path, split=3):
    d = tmp_path / "chunks"
    d.mkdir()
    rows = [{"chunk_id": f"{i}_0", "doc_id": str(i), "lang": l, "text": t} for i, (l, t) in enumerate(DOCS)]
    pq.write_table(pa.Table.from_pylist(rows[:split]), d / "part-00000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[split:]), d / "part-00001.parquet")
    return d


def test_tokenize():
    assert "huyết_áp" in bm25.tokenize("Bệnh tăng huyết áp, cần điều trị.", "vi")
    assert "高血压" in bm25.tokenize("高血压患者应该少吃盐", "zh")
    assert all(t == t.lower() and t not in (",", "。") for t in bm25.tokenize("Bệnh A, B. 高血压。", "vi"))


def test_bm25_matches_bm25s_across_shards(tmp_path):
    chunks = _write_chunks(tmp_path)
    out = tmp_path / "bm25"
    bm25.tokenize_parts(chunks, out, workers=1)
    bm25.build(out, k1=1.5, b=0.75)
    q = "điều trị tăng huyết áp"
    qe = bm25.QueryEncoder(out)
    (ids, scores), = sparse_index.search(out, [qe(q)], k=10)
    toks = [bm25.tokenize(t, l) for l, t in DOCS]
    ref = bm25.bm25s_reference_scores(toks, bm25.tokenize(q, "vi"))
    want = {f"{i}_0": s for i, s in enumerate(ref) if s > 0}
    assert set(ids) == set(want)                                        # 2 shard, IDF toàn cục = bm25s trên cả kho
    for c, s in zip(ids, scores):
        assert abs(s - want[c]) < 1e-2 * max(1, want[c])
    assert ids[0] in ("0_0", "2_0")
    stats = json.loads((out / "stats.json").read_text())
    assert stats["n_docs"] == len(DOCS)


def test_sparse_search_matches_bruteforce(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.random(300, 50, density=0.1, random_state=1, format="csr", dtype=np.float32)
    for k, (a, b) in enumerate([(0, 120), (120, 300)]):
        sparse_index.write_shard(tmp_path, k, X[a:b], [f"d{i}" for i in range(a, b)])
    qs = [(np.array(sorted(rng.choice(50, 5, replace=False))), rng.random(5).astype(np.float32)) for _ in range(7)]
    res = sparse_index.search(tmp_path, qs, k=10, batch=3)
    Xd = X.toarray().astype(np.float16).astype(np.float32)             # chỉ mục lưu float16
    for (t, w), (ids, sc) in zip(qs, res):
        s = Xd[:, t] @ w
        top = [i for i in np.argsort(-s, kind="stable")[:10] if s[i] > 0]
        assert np.allclose(sc, s[top], atol=1e-4)
        assert ids == [f"d{i}" for i in top]


def _fake_emb(tmp_path, n=600, dim=16, shard=250, fp="abc"):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    emb = tmp_path / "emb"
    for k in range(0, n, shard):
        d = emb / f"shard-{k // shard:04d}"
        d.mkdir(parents=True)
        np.save(d / "dense.npy", x[k:k + shard].astype(np.float16))
        m = sp.random(min(shard, n - k), 100, density=0.05, random_state=k, format="csr", dtype=np.float32)
        encode.save_sparse(d / "sparse.npz", m.indptr, m.indices, m.data, 100)
        pq.write_table(pa.table({"chunk_id": [f"{i}_0" for i in range(k, min(k + shard, n))]}), d / "chunk_ids.parquet")
        (d / "_DONE.json").write_text(json.dumps({"n": min(shard, n - k), "chunks_total": n, "shard_size": shard,
                                                  "chunks_fp": fp}))
    return emb, x


def test_emb_shards_checks_version_and_missing(tmp_path, capsys):
    emb, _ = _fake_emb(tmp_path)
    assert len(index.emb_shards(emb)) == 3
    import shutil
    shutil.rmtree(emb / "shard-0002")                                   # thiếu khối cuối → cảnh báo
    assert len(index.emb_shards(emb)) == 2 and "thiếu 1/3 khối" in capsys.readouterr().out
    meta = json.loads((emb / "shard-0001" / "_DONE.json").read_text())
    (emb / "shard-0001" / "_DONE.json").write_text(json.dumps(dict(meta, chunks_fp="khac")))
    with pytest.raises(SystemExit, match="bản chunks khác nhau"):
        index.emb_shards(emb)


def test_dense_flat_and_ivf(tmp_path):
    emb, x = _fake_emb(tmp_path)
    dc = {"flat_below": 1000, "nlist": 8, "nprobe": 8, "train_size": 600}
    index.build_dense(emb, tmp_path / "flat", dc)
    res = index.dense_search(tmp_path / "flat", x[:3], 5, 8)
    assert [r[0][0] for r in res] == ["0_0", "1_0", "2_0"]               # chính nó gần nhất
    index.build_dense(emb, tmp_path / "ivf", dict(dc, flat_below=10))
    assert json.loads((tmp_path / "ivf" / "dense.json").read_text())["kind"].startswith("IVF")
    res = index.dense_search(tmp_path / "ivf", x[:3], 5, 8)              # nprobe = nlist → như vét cạn
    assert [r[0][0] for r in res] == ["0_0", "1_0", "2_0"]
    rows, md = index.tune_dense(emb, {"sample": 500, "nlists": [4, 8], "nprobes": [1, 4, 8], "n_queries": 50, "k": 10})
    full = [r for r in rows if r["nprobe"] == r["nlist"]]
    assert full and all(r["recall@10"] > 0.85 for r in full) and "Recall@10" in md


def test_lsr_index_from_emb(tmp_path):
    emb, _ = _fake_emb(tmp_path)
    index.build_lsr(emb, tmp_path / "idx")
    assert len(sparse_index.shards(tmp_path / "idx" / "lsr")) == 3
    X = sparse_index.load_csc(tmp_path / "idx" / "lsr" / "shard-0000.npz")
    Y = encode.load_sparse(emb / "shard-0000" / "sparse.npz")
    assert (abs(X - Y) > 1e-3).nnz == 0


def test_lsr_search_from_emb_equals_index(tmp_path):
    emb, _ = _fake_emb(tmp_path)
    index.build_lsr(emb, tmp_path / "idx")
    rng = np.random.default_rng(3)
    qs = [(np.array(sorted(rng.choice(100, 6, replace=False))), rng.random(6).astype(np.float32)) for _ in range(5)]
    a = sparse_index.search(tmp_path / "idx" / "lsr", qs, 20)
    half = tmp_path / "emb2"                                           # kết quả T3 chia ở 2 thư mục (2 người)
    half.mkdir()
    (emb / "shard-0002").rename(half / "shard-0002")
    b = sparse_index.search([emb, half], qs, 20)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert len(index.emb_shards(f"{emb},{half}")) == 3
