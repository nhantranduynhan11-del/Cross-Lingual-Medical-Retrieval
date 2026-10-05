import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mir import chunk

CC = {"tokenizer": "BAAI/bge-m3", "max_tokens": 120, "min_tokens": 20, "overlap_tokens": 0,
      "parent_child": {"parent_max_tokens": 150, "child_max_tokens": 40}}

try:
    TOK = chunk.load_tokenizer(CC["tokenizer"])
except Exception:                                          # không có mạng và chưa có tokenizer trong cache
    TOK = None
pytestmark = pytest.mark.skipif(TOK is None, reason="chưa có tokenizer BGE-M3")

VI = ("Nguyên nhân\n" + "Bệnh tăng huyết áp do nhiều yếu tố như di truyền, ăn mặn, ít vận động và căng thẳng kéo dài. " * 4
      + "\nTriệu chứng:\n" + "Người bệnh có thể đau đầu, chóng mặt, mệt mỏi, khó thở khi gắng sức, nhìn mờ. " * 5
      + "\n1. Điều trị\n" + "Dùng thuốc theo chỉ định của bác sĩ, giảm muối, tập thể dục đều đặn mỗi ngày. " * 6)
ZH = ("【处方】 黄连 防风 荆芥 山栀 黄芩 牛蒡子 滑石 玄参 知母 石膏各3克\n【功效与作用】 "
      + "清心解毒，治心经火旺，酷暑时生天疱，发及遍身者。" * 12 + "\n【用法用量】 水400毫升，灯心20根，煎至320毫升，空腹时服。")


def _doc(text):
    return chunk.Doc(text, TOK.encode(text, add_special_tokens=False).offsets)


@pytest.mark.parametrize("strategy", chunk.STRATEGIES)
@pytest.mark.parametrize("text", [VI, ZH, "Một câu ngắn.", "长" * 2000, "x " * 1500])
def test_invariants(strategy, text):
    doc = _doc(text)
    spans, parents, owner = chunk.chunk_doc(doc, strategy, CC)
    mx = CC["parent_child"]["child_max_tokens"] if strategy == "parent_child" else CC["max_tokens"]
    for a, b in spans:
        assert 0 <= a < b <= len(text) and text[a:b] == text[a:b].strip()
        assert doc.ntok(a, b) <= mx
    covered = "".join(text[a:b] for a, b in spans)                  # không chồng lấn + không bỏ sót chữ
    assert "".join(covered.split()) == "".join(text.split())
    assert all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))
    if strategy == "parent_child":
        for (a, b), j in zip(spans, owner):
            pa_, pb = parents[j]
            assert pa_ <= a and b <= pb
        assert all(doc.ntok(a, b) <= CC["parent_child"]["parent_max_tokens"] for a, b in parents)


def test_structure_cuts_at_section_starts():
    doc = _doc(VI)
    starts = {a for a, _ in chunk.sections(doc)}
    spans, _, _ = chunk.chunk_doc(doc, "structure", CC)
    assert len(spans) > 1
    # ranh giới đoạn rơi vào đầu một mục, trừ khi mục đó tự dài hơn max_tokens
    long_secs = [(a, b) for a, b in chunk.sections(doc) if doc.ntok(a, b) > CC["max_tokens"]]
    for a, _ in spans:
        assert a in starts or any(s < a < e for s, e in long_secs)


def test_is_heading():
    for h in ["Nguyên nhân", "Triệu chứng:", "1. Điều trị", "【处方】 黄连 防风", "一、病因", "（二）诊断", "病情分析："]:
        assert chunk.is_heading(h), h
    for l in ["Bệnh tăng huyết áp do nhiều yếu tố như di truyền, ăn mặn.", "清心解毒，治心经火旺。", "2023"]:
        assert not chunk.is_heading(l), l


def test_overlap():
    doc = _doc(VI)
    cc = dict(CC, overlap_tokens=15)
    spans, _, _ = chunk.chunk_doc(doc, "recursive", cc)
    assert any(b1 > a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))
    assert all(doc.ntok(a, b) <= cc["max_tokens"] + 15 for a, b in spans)


def test_run_writes_and_skips(tmp_path):
    src = tmp_path / "clean"
    src.mkdir()
    pq.write_table(pa.Table.from_pylist([{"doc_id": "1", "host": "a.vn", "lang": "vi", "title": "t", "text": VI},
                                         {"doc_id": "2", "host": "b.cn", "lang": "zh", "title": "t", "text": ZH}]),
                   src / "part-00000.parquet")
    for s in chunk.STRATEGIES:
        chunk.run(src, tmp_path / "chunks", s, CC)
    t = pq.read_table(tmp_path / "chunks" / "structure" / "part-00000.parquet").to_pydict()
    assert t["chunk_id"][0] == "1_0" and set(t["doc_id"]) == {"1", "2"}
    pcs = pq.read_table(tmp_path / "chunks" / "parent_child" / "part-00000.parquet").to_pydict()
    par = pq.read_table(tmp_path / "chunks" / "parent_child" / "parents-00000.parquet").to_pydict()
    assert set(pcs["parent_id"]) <= set(par["parent_id"])
    f = tmp_path / "chunks" / "structure" / "part-00000.parquet"
    m = f.stat().st_mtime_ns
    chunk.run(src, tmp_path / "chunks", "structure", CC)               # đã có → bỏ qua
    assert f.stat().st_mtime_ns == m
    t = f.stat().st_mtime + 10                                         # corpus_clean làm lại sau khi chia đoạn
    os.utime(src / "part-00000.parquet", (t, t))
    chunk.run(src, tmp_path / "chunks", "structure", CC)               # part chunks cũ hơn nguồn → chia lại
    assert f.stat().st_mtime_ns != m
    md, summ = chunk.report(src, tmp_path / "chunks", CC, est_docs=1000)
    assert all(v["invariant_errors"] == 0 for v in summ.values()) and summ["parent_child"]["parent_invariant_errors"] == 0
