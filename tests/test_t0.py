import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from mir import config, make_sample, survey
from mir.crawlio import iter_docs


def _fake(tmp_path):
    crawl = tmp_path / "out"
    (crawl / "s0").mkdir(parents=True)
    docs, i = [], 0
    for host, lang, n in [("a.cn", "zh", 300), ("b.vn", "vi", 100), ("c.vn", "vi", 3)]:
        for _ in range(n):
            i += 1
            unit = ("中文内容" if lang == "zh" else "tiếng Việt có dấu ") if i % 50 else \
                ("重复重复" if lang == "zh" else "bài trùng lặp ")
            docs.append({"id": i, "url": f"https://{host}/{i}", "host": host, "lang": lang,
                         "title": f"t{i}", "text": unit * (30 + i)})
    with gzip.open(crawl / "s0" / "part-00000.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (crawl / "s0" / "progress.tsv").write_text("1\tok\n9999\thttp_404\n", encoding="utf-8")
    raw = gzip.compress("\n".join(json.dumps(d) for d in docs[:50]).encode())
    (crawl / "part-00001.jsonl.gz").write_bytes(raw[: len(raw) // 2])     # file bị cụt
    data = tmp_path / "Data"
    data.mkdir()
    pq.write_table(pa.table({"id": [1, 2], "url": ["https://a.cn/1", "https://b.vn/2"]}), data / "links_corpus.parquet")
    pq.write_table(pa.table({"id": list(range(100)), "query": [f"q{k}" for k in range(100)]}), data / "query.parquet")
    cfg = tmp_path / "c.yaml"
    p = tmp_path.as_posix()
    cfg.write_text(f'paths: {{data_dir: "{p}", work_dir: "{p}", queries: Data/query.parquet, '
                   f'links: Data/links_corpus.parquet, crawl_dir: out}}\n'
                   'seed: 1\nsample: {n_docs: 50, n_queries: 10, min_per_host: 5, dir: sample}\n', encoding="utf-8")
    return cfg, len(docs)


def test_truncated_gz_does_not_crash(tmp_path):
    cfg, n = _fake(tmp_path)
    c = config.load(cfg)
    assert len(list(iter_docs(c.crawl_path))) >= n


def test_survey(tmp_path):
    cfg, n = _fake(tmp_path)
    c = config.load(cfg)
    s = survey.survey(c.crawl_path, c.links_path)
    assert s["n_docs"] == n and s["lang"]["zh"] == 300 and s["progress_status"]["ok"] == 1
    assert "a.cn" in survey.to_markdown(s)


def test_sample_stratified_and_idempotent(tmp_path):
    cfg, _ = _fake(tmp_path)
    c = config.load(cfg)
    make_sample.build(c)
    f = tmp_path / "sample" / "corpus_raw.parquet"
    t = pq.read_table(f).to_pydict()
    assert len(t["id"]) == 50
    hosts = {h: t["host"].count(h) for h in set(t["host"])}
    assert hosts["c.vn"] == 3 and hosts["b.vn"] >= 5
    before = f.stat().st_mtime_ns
    make_sample.build(c)
    assert f.stat().st_mtime_ns == before
    assert len(pq.read_table(tmp_path / "sample" / "query.parquet")) == 10
