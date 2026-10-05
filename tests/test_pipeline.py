"""Kiểm thử đầu-cuối T3→T8 trên sample/ thật (corpus_clean + chunks/structure + 50 câu hỏi), model BGE-M3 tí hon
(trọng số ngẫu nhiên — chỉ kiểm đường ống, KHÔNG có ý nghĩa về chất lượng). Nhãn đánh giá là nhãn GIẢ."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mir import bm25, encode, evaluate, fuse, index, rerank, retrieve, runs, store, submit

ROOT = Path(__file__).resolve().parents[1]
CLEAN, CHUNKS, QUERIES = ROOT / "sample/corpus_clean", ROOT / "sample/chunks/structure", ROOT / "sample/query.parquet"
LINKS = ROOT / "Data/links_corpus.parquet"
pytestmark = pytest.mark.skipif(not (CHUNKS.exists() and QUERIES.exists() and LINKS.exists()),
                                reason="chưa có sample/chunks/structure hoặc Data/")


def test_end_to_end(tiny_encoder, tmp_path):
    enc = tiny_encoder
    ec = {"model": "tiny", "max_length": 512, "fp16": False, "batch_tokens": 32768, "max_batch": 256,
          "shard_size": 2000, "sparse_min_weight": 0.0}
    # T3
    encode.run(enc, CHUNKS, CLEAN, tmp_path / "emb", ec, log=lambda *_: None)
    n_chunks = pq.read_table(CHUNKS, columns=["chunk_id"]).num_rows
    shards = sorted((tmp_path / "emb").glob("shard-*"))
    assert len(shards) == -(-n_chunks // 2000) and all((d / "_DONE.json").exists() for d in shards)
    # T4
    idx = tmp_path / "index"
    index.build_dense(tmp_path / "emb", idx, {"flat_below": 10**6, "nlist": 64, "nprobe": 8, "train_size": 5000},
                      log=lambda *_: None)
    index.build_lsr(tmp_path / "emb", idx, log=lambda *_: None)
    bm25.tokenize_parts(CHUNKS, idx / "bm25", workers=2, log=lambda *_: None)
    bm25.build(idx / "bm25", log=lambda *_: None)
    # T5
    qids, qtexts = retrieve.load_queries(QUERIES)
    qemb = retrieve.encode_queries(enc, qids, qtexts, tmp_path / "qemb.npz")
    br = {b: retrieve.run_branch(b, idx, qemb, qids, qtexts, depth=100, nprobe=8, log=lambda *_: None)
          for b in ("dense", "lsr", "bm25")}
    for b, df in br.items():
        assert df.groupby("qid")["rank"].max().max() <= 100
        assert df["qid"].nunique() >= (len(qids) if b == "dense" else len(qids) // 2), b
    allids = set(pq.read_table(CHUNKS, columns=["chunk_id"]).column("chunk_id").to_pylist())
    assert set(br["bm25"]["chunk_id"]) <= allids
    G = fuse.rrf(list(br.values()), 60, 100)
    W = fuse.weighted(list(br.values()), [0.4, 0.3, 0.3], 100)
    assert G["qid"].nunique() == len(qids) and W["qid"].nunique() == len(qids)
    runs.write(G, tmp_path / "runs/structure_G.parquet", {})
    # T6 (5 câu hỏi, top 10)
    small = G[G["qid"].isin(qids[:5]) & (G["rank"] <= 10)]
    texts = rerank.candidate_texts(small["chunk_id"].unique().tolist(), CHUNKS, CLEAN)
    rr = rerank.rerank_run(enc, small, dict(zip(qids, qtexts)), texts,
                           {"query_max_length": 512, "doc_max_length": 512, "batch_tokens": 32768}, log=lambda *_: None)
    assert len(rr) == len(small) and set(rr["chunk_id"]) == set(small["chunk_id"])
    # T7 — nhãn GIẢ: tài liệu đứng đầu nhánh BM25 của mỗi câu hỏi, chunk_text = đoạn đó
    top = br["bm25"][br["bm25"]["rank"] == 1]
    ctext = store.lookup(CHUNKS, "chunk_id", top["chunk_id"].tolist(), ["text"])
    qrels = pd.DataFrame({"qid": top["qid"], "doc_id": top["doc_id"],
                          "chunk_text": [ctext[c]["text"] for c in top["chunk_id"]]})
    langs = {k: v["lang"] for k, v in store.lookup(CLEAN, "doc_id", qrels["doc_id"].tolist(), ["lang"]).items()}
    tG = {k: v["text"] for k, v in store.lookup(CHUNKS, "chunk_id", G[G["rank"] <= 100]["chunk_id"].unique().tolist(),
                                                ["text"]).items()}
    s, _ = evaluate.evaluate(G, qrels, tG, langs, {"ks": [10, 50, 100], "match": "overlap", "overlap_threshold": 0.8}, 10, 10)
    assert s["n_queries"] == len(qrels) and s["doc_R@100"] >= 0.99            # nhãn lấy từ BM25 → phải nằm trong G
    # T8
    sub = submit.build(G, qids, 10, 10, tG)
    link_ids = set(pq.read_table(LINKS, columns=["id"]).column("id").to_pylist())
    used = {c["doc_id"] for x in sub for c in x["relevant_chunks"]}
    clean = {k: v["text"] for k, v in store.lookup(CLEAN, "doc_id", list(used), ["text"]).items()}
    assert submit.validate(sub, qids, link_ids, clean) == []
    _, zp = submit.write(sub, tmp_path / "subs", "test")
    assert zp.exists() and len(sub) == len(qids)
