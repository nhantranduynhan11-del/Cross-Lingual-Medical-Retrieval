from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mir import config, store

CFG = "configs/baseline.yaml"


def test_paths_relative_to_repo_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MIR_DATA_DIR", raising=False)
    monkeypatch.delenv("MIR_WORK_DIR", raising=False)
    c = config.load(config.ROOT / CFG)
    assert c.work_dir.resolve() == config.ROOT and c.data_dir.resolve() == config.ROOT
    assert c.path("chunks", "structure") == c.work_dir / "chunks" / "structure"
    assert c.path("chunks") == c.work_dir / "chunks"
    assert c.path("emb", "parent_child") == c.work_dir / "emb" / "parent_child"
    assert c.path("runs") == c.work_dir / "runs" and c.path("qemb") == c.work_dir / "qemb"
    assert c.path("queries") == c.data_dir / "Data" / "query.parquet"
    assert c.path("chunks", "structure", override="x/y") == Path("x/y")
    with pytest.raises(KeyError):
        c.path("chunk")                                                   # tên sai → lỗi, không đoán
    monkeypatch.setenv("MIR_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MIR_WORK_DIR", str(tmp_path / "w"))
    c = config.load(config.ROOT / CFG)
    assert c.links_path == tmp_path / "d" / "Data" / "links_corpus.parquet"
    assert c.path("clean") == tmp_path / "w" / "corpus_clean"
    assert c.results() == tmp_path / "w" / "results" and (tmp_path / "w" / "results").is_dir()


def test_unit_rows_chunk_and_parent(tmp_path):
    pq.write_table(pa.table({"chunk_id": ["1_0", "1_1"], "doc_id": ["1", "1"], "text": ["a", "b"]}),
                   tmp_path / "part-00000.parquet")
    pq.write_table(pa.table({"parent_id": ["1_p0"], "doc_id": ["1"], "text": ["a b"]}),
                   tmp_path / "parents-00000.parquet")
    assert store.unit_rows(tmp_path, ["1_1"]) == {"1_1": {"text": "b"}}
    assert store.unit_rows(tmp_path, ["1_p0"], ["doc_id", "text"]) == {"1_p0": {"doc_id": "1", "text": "a b"}}
    assert store.unit_rows(tmp_path, []) == {}
    assert store.is_parent_id("12_p3") and not store.is_parent_id("12_3")
