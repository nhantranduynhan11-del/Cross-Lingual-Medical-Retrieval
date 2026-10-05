import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mir import evaluate, fuse, rerank, runs, submit


def _run(rows):
    return runs.from_topk([q for q, _ in rows], [[c for c, _ in r] for _, r in rows], [[s for _, s in r] for _, r in rows])


# ---------------- T5: gộp ----------------
def test_rrf_values_and_order():
    b1 = _run([(1, [("a_0", 9), ("b_0", 8), ("c_0", 7)])])
    b2 = _run([(1, [("b_0", 5), ("d_0", 4)])])
    r = fuse.rrf([b1, b2], k=60)
    assert list(r["chunk_id"]) == ["b_0", "a_0", "d_0", "c_0"]
    assert r["score"].iloc[0] == pytest.approx(1 / 62 + 1 / 61)
    assert list(r["rank"]) == [1, 2, 3, 4] and list(r["doc_id"]) == ["b", "a", "d", "c"]
    single = fuse.rrf([b1], k=60)                                         # cấu hình 1 nhánh = giữ thứ tự nhánh
    assert list(single["chunk_id"]) == ["a_0", "b_0", "c_0"]


def test_weighted_minmax():
    b1 = _run([(1, [("a_0", 10), ("b_0", 0)])])
    b2 = _run([(1, [("b_0", 3), ("a_0", 1)])])
    r = fuse.weighted([b1, b2], [0.3, 0.7])
    assert list(r["chunk_id"]) == ["b_0", "a_0"]                         # b: 0.7, a: 0.3
    assert r["score"].tolist() == pytest.approx([0.7, 0.3])


def test_run_io_and_doc_ranking(tmp_path):
    df = _run([(1, [("5_0", 3), ("5_1", 2), ("7_0", 1)]), (2, [("9_p0", 1)])])
    runs.write(df, tmp_path / "x.parquet", {"a": 1})
    with pytest.raises(FileExistsError):
        runs.write(df, tmp_path / "x.parquet")
    back = runs.read(tmp_path / "x.parquet")
    assert back["doc_id"].tolist() == ["5", "5", "7", "9"]
    d = runs.doc_ranking(back)
    assert d[d.qid == 1]["doc_id"].tolist() == ["5", "7"]


# ---------------- T6: xếp hạng lại ----------------
def test_maxsim():
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    d = np.array([[1, 0], [0.6, 0.8]], dtype=np.float32)
    assert rerank.maxsim(q, d) == pytest.approx((1 + 0.8) / 2)


def test_rerank_run_with_tiny_model(tiny_encoder):
    df = _run([(1, [("1_0", 3), ("2_0", 2), ("3_0", 1)])])
    texts = {"1_0": "t\nmột hai ba", "2_0": "t\nbệnh tăng huyết áp", "3_0": "t\n高血压"}
    rc = {"query_max_length": 64, "doc_max_length": 64, "batch_tokens": 4096}
    out = rerank.rerank_run(tiny_encoder, df, {1: "tăng huyết áp"}, texts, rc)
    qv = tiny_encoder.colbert(["tăng huyết áp"])[0]
    dv = tiny_encoder.colbert([texts[c] for c in ["1_0", "2_0", "3_0"]])
    want = sorted(zip([rerank.maxsim(qv, d) for d in dv], ["1_0", "2_0", "3_0"]), reverse=True)
    assert out["chunk_id"].tolist() == [c for _, c in want]
    assert out["score"].tolist() == pytest.approx([s for s, _ in want], abs=1e-3)


def test_to_parent(tmp_path):
    pq.write_table(pa.table({"chunk_id": ["1_0", "1_1", "2_0"], "parent_id": ["1_p0", "1_p0", "2_p0"]}),
                   tmp_path / "part-00000.parquet")
    df = _run([(1, [("1_1", 3), ("2_0", 2), ("1_0", 1)])])
    p = rerank.to_parent(df, tmp_path)
    assert p["chunk_id"].tolist() == ["1_p0", "2_p0"] and p["rank"].tolist() == [1, 2]


# ---------------- T7: đánh giá với nhãn GIẢ ----------------
EC = {"ks": [1, 2, 3], "match": "overlap", "overlap_threshold": 0.8}


def test_match_rules():
    assert evaluate.match("a  b\nc", "a b c", "exact")
    assert evaluate.match("xx a b c yy", "a b c", "contains") and not evaluate.match("a b", "a b c", "exact")
    assert evaluate.match("bệnh tăng huyết áp cần điều trị", "bệnh tăng huyết áp cần điều trị sớm", "overlap", 0.8)
    assert not evaluate.match("hoàn toàn khác", "bệnh tăng huyết áp", "overlap", 0.8)


def test_metrics_on_fake_qrels():
    run = _run([(1, [("10_0", 5), ("11_0", 4), ("10_1", 3)]), (2, [("20_0", 1)])])
    qrels = pd.DataFrame({"qid": [1, 1, 2], "doc_id": ["10", "12", "21"],
                          "chunk_text": ["đoạn đúng của bài mười", "", ""]})
    texts = {"10_0": "khác hẳn", "11_0": "x", "10_1": "đoạn đúng của bài mười"}
    s, per_q = evaluate.evaluate(run, qrels, texts, {"10": "vi", "12": "zh", "21": "vi"}, EC, k_doc=2, k_chunk=3)
    q1 = per_q.set_index("qid").loc[1]
    assert q1["doc_R@1"] == 0.5 and q1["doc_R@3"] == 0.5 and q1["MRR@10"] == 1.0
    assert q1["doc_R@3_vi"] == 1.0 and q1["doc_R@3_zh"] == 0.0
    assert q1["doc_P"] == 0.5 and q1["doc_R"] == 0.5 and q1["doc_F2"] == pytest.approx(0.5)
    assert q1["chunk_R@2"] == 0.0 and q1["chunk_R@3"] == 1.0
    assert q1["chunk_P"] == pytest.approx(1 / 3) and q1["chunk_F2"] == pytest.approx(5 * (1 / 3) / (4 / 3 + 1))
    q2 = per_q.set_index("qid").loc[2]
    assert q2["doc_R@3"] == 0.0 and q2["nDCG@10"] == 0.0
    assert s["n_queries"] == 2 and s["n_queries_with_chunks"] == 1
    assert s["doc_F2"] == pytest.approx((0.5 + 0.0) / 2)


# ---------------- T8: bài nộp ----------------
def test_submission_build_validate_zip(tmp_path):
    run = _run([(1, [("10_0", 3), ("10_1", 2), ("11_0", 1)])])
    texts = {"10_0": "đoạn A", "10_1": "đoạn B", "11_0": "đoạn C"}
    sub = submit.build(run, [1, 2], k_doc=1, k_chunk=2, chunk_texts=texts)
    assert sub[0] == {"id": 1, "relevant_docs": ["10"], "relevant_chunks": [{"doc_id": "10", "chunk_text": "đoạn A"},
                                                                         {"doc_id": "10", "chunk_text": "đoạn B"}]}
    assert sub[1] == {"id": 2, "relevant_docs": [], "relevant_chunks": []}
    clean = {"10": "mở đầu đoạn A rồi đoạn B", "11": "đoạn C"}
    assert submit.validate(sub, [1, 2], {10, 11}, clean) == []
    bad = json.loads(json.dumps(sub))
    bad[0]["relevant_chunks"][0]["chunk_text"] = "viết lại"
    bad[0]["relevant_docs"] = ["999"]
    errs = submit.validate(bad, [1, 2, 3], {10, 11}, clean)
    assert len(errs) == 3
    js, zp = submit.write(sub, tmp_path, "sub")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["sub.json"] and json.loads(z.read("sub.json")) == sub
    with pytest.raises(FileExistsError):
        submit.write(sub, tmp_path, "sub")
