"""Kiểm thử crawl.py (không gọi mạng). Phần cuối dùng crawl_eval/html/ nếu có (bỏ qua nếu không)."""
import argparse
import gzip
import json
import os
import sys
import unicodedata

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import crawl  # noqa: E402
from mir.crawlio import iter_docs, read_progress  # noqa: E402

PAGE = """<html><head><meta charset="{cs}"><title>Tiêu đề - Báo</title></head><body>
<nav>Trang chủ | Sức khỏe | Thời sự</nav>
<article><h1>Tiêu đề</h1><p>{p1}</p><p>{p2}</p></article>
<footer>Bản quyền thuộc về báo</footer></body></html>"""
P1 = "Bệnh tăng huyết áp là tình trạng áp lực máu lên thành động mạch tăng cao kéo dài, cần được theo dõi thường xuyên."
P2 = "Người bệnh nên giảm muối, tập thể dục đều đặn và dùng thuốc theo chỉ định của bác sĩ để kiểm soát huyết áp."


# ---------- giải mã, phân tích ----------
def test_decode_gb_meta_and_heuristic():
    s = "<html><head><meta charset='gb2312'></head><body>中医治疗头痛的方法很多</body></html>"
    assert "中医治疗" in crawl.decode_html(s.encode("gb18030"))
    no_meta = "<html><body>中医治疗头痛的方法很多，针灸也有效果</body></html>".encode("gbk")
    assert "针灸" in crawl.decode_html(no_meta)          # không khai báo charset, không phải UTF-8 → GB18030
    assert crawl.decode_html("﻿abc".encode("utf-8")) == "abc"


def test_broken_html_root_closed_early():
    s = '<html lang="vi"></html><head><title>x</title></head><body><div class="c">' + P1 + "</div></body>"
    d = crawl._parse(s)
    assert P1[:20] in d.text_content()
    _, text, _ = crawl.extract(s.encode(), "example.vn")
    assert P1[:20] in text


def test_nfc_and_entities_and_junk():
    nfd = unicodedata.normalize("NFD", P1)
    html_ = PAGE.format(cs="utf-8", p1=nfd + " &amp;#8226;", p2=P2 + "<br>" + "QUJD" * 20 + "<br>{title} {publish}")
    _, text, lang = crawl.extract(html_.encode(), "example.vn")
    assert unicodedata.normalize("NFC", text) == text and P1 in text
    assert "•" in text and "&#8226;" not in text           # thực thể mã hoá 2 lần
    assert "QUJD" not in text and "{title}" not in text    # chuỗi mã hoá, mẫu trang trống
    assert lang == "vi"


def test_detect_lang():
    assert crawl.detect_lang("高血压患者应该少吃盐，多运动。1.5mg/kg ABC DEF GHI JKL MNO") == "zh"
    assert crawl.detect_lang(P1) == "vi"
    assert crawl.detect_lang("BMI 25 kg/m2", "example.vn") == "vi"


# ---------- bộ máy quy tắc ----------
@pytest.fixture
def rule(monkeypatch):
    def put(r):
        monkeypatch.setattr(crawl, "SITE_RULES", [dict(r, suffix="rule.test")] + crawl._RULES)
    return put


def test_rule_keep_drop_stop_cut(rule):
    rule({"keep": ['//div[@class="body"]'], "drop": './/div[@class="ad"]', "stop": {"Quảng cáo"},
          "stop_re": r"^Xem thêm", "cut_re": r"^Tin liên quan$", "title": "//h1", "since": 2})
    s = (f'<html><body><h1>Tiêu đề bài</h1><div class="body"><p>{P1}</p><div class="ad">Mua ngay giảm giá</div>'
         f'<p>Quảng cáo</p><p>Xem thêm: bài khác</p><p>{P2}</p><p>Tin liên quan</p><p>Bài A</p></div></body></html>')
    t, x, _ = crawl.extract(s.encode(), "www.rule.test")
    assert t == "Tiêu đề bài" and x == P1 + "\n" + P2


def test_rule_first_need_fallback(rule):
    rule({"keep": ['//div[@id="new"]', '//div[@id="old"]'], "first": True, "since": 2})
    s = f'<html><body><div id="old"><p>{P1}</p></div></body></html>'
    assert crawl.extract(s.encode(), "rule.test")[1] == P1          # bố cục mới không có → dùng bố cục cũ
    rule({"keep": ['//div[@id="sapo"]', '//div[@id="body"]'], "need": '//div[@id="body"]', "since": 2})
    s = (f'<html><body><nav>menu menu menu</nav><div id="sapo">Sapo ngắn của bài viết này dài hơn ba mươi ký tự</div>'
         f'<article><p>{P1}</p><p>{P2}</p></article></body></html>')
    x = crawl.extract(s.encode(), "rule.test")[1]
    assert P1 in x                                                   # thiếu thân bài → không chỉ trả về sapo


def test_rule_reorder_dedupe_join(rule):
    rule({"keep": ['//div[@id="sum"]', '//div[@id="t"]'], "reorder": r"jbk(\d+)", "join_labels": True,
          "min_chars": 1, "since": 2})
    s = ('<html><body><div id="sum">Đoạn một của bài viết……</div><div id="t">'
         '<div id="jbk2">Đoạn ba.</div><div id="jbk1">Đoạn hai.</div><div id="jbk0">Đoạn một của bài viết dài.</div>'
         '<p>发病部位：</p><p>眼</p></div></body></html>')
    x = crawl.extract(s.encode(), "rule.test")[1]
    assert x.split("\n") == ["Đoạn một của bài viết dài.", "Đoạn hai.", "Đoạn ba.", "发病部位： 眼"]


def test_cells_are_separated():
    d = crawl._parse("<table><tr><td>药方名称</td><td>阑尾1号</td></tr></table>")
    assert crawl._block_text(d.xpath("//table")[0]) == "药方名称 阑尾1号"


def test_rules_are_valid_xpath():
    import lxml.etree
    d = crawl._parse("<html><body></body></html>")
    for r in crawl.SITE_RULES:
        for xp in r["keep"] + [r.get("need")] + ([r["title"]] if isinstance(r.get("title"), str) else r.get("title") or []):
            if xp:
                d.xpath(xp)
        if r.get("drop"):
            d.xpath("//body")[0].xpath(r["drop"] if isinstance(r["drop"], str) else " | ".join(r["drop"]))
        assert r.get("since", 1) <= crawl.EXTRACT_VERSION, r["suffix"]


# ---------- ghi kết quả, chạy lại, trích lại ----------
def test_progress_xv_and_redo_stale_logic(tmp_path):
    w = crawl.Writer(str(tmp_path), 10)
    w.write(1, "ok", crawl.make_record(1, "u", "www.cnkang.com", "t", P1, "vi"))
    w.write(2, "http_404")
    w.close()
    with open(tmp_path / "progress.tsv", "a", encoding="utf-8") as f:
        f.write("3\tok\n")                                    # dòng kiểu cũ (2 cột) → xv = 1
    p = crawl.read_progress(str(tmp_path))
    assert p[1] == ("ok", crawl.EXTRACT_VERSION) and p[2] == ("http_404", 1) and p[3] == ("ok", 1)
    assert read_progress(str(tmp_path)) == {1: "ok", 2: "http_404", 3: "ok"}
    assert crawl.required_xv("www.cnkang.com") > 1            # bài cnkang trích bằng bản cũ → --redo-stale tải lại
    assert crawl.required_xv("vnexpress.net") == 1            # tên miền dùng bộ trích chung: bài cũ vẫn dùng được


def test_reextract_from_raw_and_reader_picks_newest(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    html_ = PAGE.format(cs="utf-8", p1=P1, p2=P2)
    t, x, lg, raw = crawl.process_page(html_.encode(), "example.vn", want_raw=True)
    assert crawl.process_page(raw, "example.vn")[:3] == (t, x, lg)    # trích từ bản gọn = trích từ bản gốc
    w = crawl.Writer(str(out), 10, save_raw=True)
    old = {"id": 7, "url": "u", "host": "example.vn", "lang": "vi", "title": "cũ", "text": "bản cũ"}   # bản ghi kiểu cũ
    w.write(7, "ok", old, {"id": 7, "url": "u", "host": "example.vn", "html": raw}, xv=1)
    w.close()
    a = argparse.Namespace(out=str(out), reextract="all", records_per_part=10, cpu=1, min_chars=50)
    crawl.reextract(a)
    docs = list(iter_docs(out))
    assert len(docs) == 1 and docs[0]["text"] == x and docs[0]["xv"] == crawl.EXTRACT_VERSION
    assert crawl.read_progress(str(out))[7] == ("ok", crawl.EXTRACT_VERSION)


def test_reader_upgrades_old_records(tmp_path):
    with gzip.open(tmp_path / "part-00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": 1, "url": "u", "host": "www.120ask.com", "lang": "vi", "title": "t",
                            "text": "高血压患者 &#8226; 应该少吃盐，多运动，按时服药控制血压"}, ensure_ascii=False) + "\n")
    d = next(iter_docs(tmp_path))
    assert d["lang"] == "zh" and "•" in d["text"] and d["xv"] == 1


# ---------- trên bộ HTML đánh giá (nếu có) ----------
EVAL = os.path.join(ROOT, "crawl_eval", "html")


@pytest.mark.skipif(not os.path.isdir(EVAL), reason="chưa có crawl_eval/html")
def test_eval_pages_raw_roundtrip_and_no_crash():
    files = sorted(os.listdir(EVAL))[::25]                   # ~40 trang rải đều các tên miền
    for f in files:
        host = f[:-8].split("__")[0]
        body = gzip.open(os.path.join(EVAL, f)).read()
        t, x, lg, raw = crawl.process_page(body, host, want_raw=True)
        assert crawl.process_page(raw, host)[:3] == (t, x, lg), f
        assert lg in ("vi", "zh")
