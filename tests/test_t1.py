import json

import pyarrow.parquet as pq

from mir import clean

CC = {"repeat_line_ratio": 0.3, "min_group_docs": 50, "min_docs_for_stats": 2000, "max_stats_docs": 5000,
      "min_chars": 100, "heading": {"max_len": 20, "min_followed_by_content": 0.6, "max_at_end": 0.3, "max_at_start": 0.5},
      "groups": [{"suffix": "39.net", "exclude": ["ask.39.net"], "name": "39.net (kênh tin)"}],
      "force_drop": {}, "force_keep": {}, "part_size": 7}
BODY = "Nội dung riêng của bài số {i}, đủ dài để không bị coi là dòng ngắn và không lặp giữa các bài viết."


def _docs():
    docs = []
    for i in range(60):                                   # nhóm lớn: a.vn
        lines = ["Trang chủ | Sức khỏe", "Nguyên nhân", BODY.format(i=i), "Điều trị", BODY.format(i=i + 1000),
                 "Bình luận của bạn", "Bản quyền thuộc báo A"]
        docs.append({"id": i, "host": "a.vn", "lang": "vi", "title": f"t{i}", "text": "\n".join(lines)})
    for i in range(5):                                    # nhóm nhỏ: không lọc
        docs.append({"id": 100 + i, "host": "b.vn", "lang": "vi", "title": "",
                     "text": "Bản quyền thuộc báo B\n" + BODY.format(i=i)})
    for k, h in enumerate(["baby.39.net", "care.39.net", "food.39.net"] * 20):   # 60 bài gộp thành 1 nhóm
        docs.append({"id": 200 + k, "host": h, "lang": "zh", "title": "",
                     "text": f"相关资讯\n{'中文正文内容' * 20}{k}\n39健康网专稿"})
    docs.append({"id": 999, "host": "a.vn", "lang": "vi", "title": "", "text": "Trang chủ | Sức khỏe\nNgắn quá"})
    return docs


def test_compute_lines_drop_and_keep_heading():
    lines, stats = clean.compute_lines(_docs(), CC, 1)
    a = lines["a.vn"]
    drop = {i["line"] for i in a["drop"]}
    keep = {i["line"] for i in a["keep_heading"]}
    assert drop == {"Trang chủ | Sức khỏe", "Bình luận của bạn", "Bản quyền thuộc báo A"}
    assert keep == {"Nguyên nhân", "Điều trị"}
    assert "b.vn" not in lines and stats["b.vn"]["filtered"] is False
    assert "39.net (kênh tin)" in lines and stats["39.net (kênh tin)"]["docs"] == 60
    assert {i["line"] for i in lines["39.net (kênh tin)"]["drop"]} == {"相关资讯", "39健康网专稿"}


def test_force_lists():
    cc = dict(CC, force_drop={"a.vn": ["Nguyên nhân"]}, force_keep={"a.vn": ["Bình luận của bạn"]})
    lines, _ = clean.compute_lines(_docs(), cc, 1)
    assert "Nguyên nhân" in {i["line"] for i in lines["a.vn"]["drop"]}
    assert "Bình luận của bạn" in {i["line"] for i in lines["a.vn"]["keep_heading"]}


def test_apply_writes_clean_corpus(tmp_path):
    docs = _docs()
    lines, _ = clean.compute_lines(docs, CC, 1)
    st = clean.apply(docs, lines, CC, tmp_path)
    t = pq.read_table(tmp_path).to_pydict()
    by = dict(zip(t["doc_id"], t["text"]))
    assert "999" not in by and st["dropped_short"] == 1              # còn < 100 ký tự → bỏ
    assert by["0"].split("\n") == ["Nguyên nhân", BODY.format(i=0), "Điều trị", BODY.format(i=1000)]
    assert by["100"].startswith("Bản quyền thuộc báo B")            # nhóm nhỏ giữ nguyên
    assert all(isinstance(x, str) for x in t["doc_id"]) and len(list(tmp_path.glob("part-*.parquet"))) > 1
    orig = {str(d["id"]): d["text"].split("\n") for d in docs}
    for i, x in by.items():                                          # chỉ bỏ dòng, không sửa/không đảo dòng
        it = iter(orig[i])
        assert all(l in it for l in x.split("\n"))


def test_review_md_lists_groups():
    lines, stats = clean.compute_lines(_docs(), CC, 1)
    md = clean.review_md(lines, stats, CC)
    assert "## a.vn" in md and "Bản quyền thuộc báo A" in md and "không (ít bài)" in md


def test_cli_does_not_overwrite(tmp_path, monkeypatch):
    import pyarrow as pa
    import sys
    pq.write_table(pa.Table.from_pylist(_docs()), tmp_path / "in.parquet")
    cfg = tmp_path / "c.yaml"
    p = tmp_path.as_posix()
    cfg.write_text(f'paths: {{data_dir: "{p}", work_dir: "{p}", queries: q, links: l, crawl_dir: out}}\nseed: 1\n'
                   "clean: " + json.dumps(dict(CC, lines_file="clean_lines.json", out_dir="corpus_clean")) + "\n",
                   encoding="utf-8")
    run = lambda *x: (monkeypatch.setattr(sys, "argv", ["clean", "--config", str(cfg), "--input",
                                                       str(tmp_path / "in.parquet"), *x]), clean.main())
    run("--stats")
    m1 = (tmp_path / "clean_lines.json").stat().st_mtime_ns
    run("--stats")                                                   # đã có → không ghi đè
    assert (tmp_path / "clean_lines.json").stat().st_mtime_ns == m1
    run("--apply")
    f = sorted((tmp_path / "corpus_clean").glob("part-*.parquet"))[0]
    m2 = f.stat().st_mtime_ns
    run("--apply")
    assert f.stat().st_mtime_ns == m2
    assert (tmp_path / "results" / "t1_review.md").exists()
