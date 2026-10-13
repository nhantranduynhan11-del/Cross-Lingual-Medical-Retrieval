"""Trích một trang web: HTML → bản ghi (tiêu đề, văn bản có cấu trúc, metadata, ngôn ngữ, loại tài liệu, trạng thái).

Chạy trong process con (ProcessPoolExecutor) nên chỉ nhận/trả kiểu dữ liệu đơn giản.
"""
import collections
import copy
import functools
import hashlib
import html as _html
import json
import re

import lxml.html

from . import XV
from .classify import detect_lang, doc_type, is_notice
from .htmlx import clean_lines, decode_html, parse, render, split_marker, strip_html, tei_lines, to_plain
from .meta import (clean_field, date_from_elements, date_from_header_lines, extract_meta, good_author,
                   strip_site_suffix, tidy_breadcrumb)
from .rules import find_rule

_LABEL = re.compile(r"^[^：:\s]{1,6}[：:]\s*")
# Dòng rác: chuỗi mã hoá (nội dung bị trang mã hoá, giải bằng JS — vd một nửa số bài baophutho.vn) và chỗ trống
# của mẫu trang chưa được điền ("{title} {publish}").
_JUNK_LINE = re.compile(r"^[A-Za-z0-9+/=]{60,}$|^(\{\w+\}\s*)+$")
_CRUMB_SEP = re.compile(r"\s*[>›»/]\s*|\s+-\s+")


@functools.lru_cache(maxsize=None)
def _rx(p):
    return re.compile(p) if p else None


def _drop_xpath(rule):
    d = rule.get("drop")
    return " | ".join(d) if isinstance(d, (list, tuple)) else d


def _apply_line_rules(lines, rule):
    stop = rule.get("stop") or set()
    stop_re, cut_re = _rx(rule.get("stop_re")), _rx(rule.get("cut_re"))
    out = []
    for line in lines:
        mk, c = split_marker(line)
        if cut_re and cut_re.search(c):                    # phần đuôi do mẫu trang chèn vào: bỏ từ đây trở đi
            break
        if c in stop or (stop_re and stop_re.search(c)):
            continue
        if out:
            pc = split_marker(out[-1])[1]
            if rule.get("dedupe_lines") and _LABEL.sub("", c) == _LABEL.sub("", pc):
                continue                                   # câu hỏi lặp: tiêu đề / mô tả / "补充说明：" giống hệt
            if rule.get("join_labels") and len(pc) <= 12 and pc.endswith(("：", ":")):
                out[-1] = pc + " " + c                     # "发病部位：" + "眼" → "发病部位： 眼"
                continue
        out.append(line)
    return out


def _region_lines(el, rule):
    return _apply_line_rules(render(el, _drop_xpath(rule), rule.get("reorder")), rule)


def _dedupe_parts(parts):
    """Sapo / tóm tắt lặp lại (toàn bộ hoặc gần hết, kể cả bản bị cắt "……") trong phần dài hơn → bỏ phần ngắn."""
    if len(parts) < 2:
        return parts
    keys = [re.sub(r"\s+", "", to_plain("\n".join(p))) for p in parts]
    grams = [{k[i:i + 6] for i in range(max(1, len(k) - 5))} for k in keys]
    keep = []
    for i in range(len(parts)):
        longer = set().union(*[grams[j] for j in range(len(parts))
                               if j != i and (len(keys[j]) > len(keys[i]) or (len(keys[j]) == len(keys[i]) and j < i))])
        if not grams[i] or len(grams[i] & longer) / len(grams[i]) < 0.8:
            keep.append(i)
    return [parts[i] for i in keep]


def _first_text(el, xps, max_len=200):
    for xp in ([xps] if isinstance(xps, str) else xps or []):
        for h in el.xpath(xp):
            t = clean_field(h if isinstance(h, str) else h.xpath("string()"), max_len)
            if t:
                return t
    return ""


def _n(lines):
    return sum(len(split_marker(x)[1]) for x in lines if x != "---")


def _site_extract(doc, rule):
    if rule.get("need") and not doc.xpath(rule["need"]):   # thiếu vùng bắt buộc (thân bài) → bố cục không khớp
        return []
    parts = []
    for xp in rule["keep"]:
        got = [ls for ls in (_region_lines(el, rule) for el in doc.xpath(xp)) if ls]
        if rule.get("first"):
            if sum(_n(g) for g in got) >= rule.get("min_chars", 30):
                parts = got
                break
        else:
            parts += got
    return [line for p in _dedupe_parts(parts) for line in p]


def _qa_extract(doc, rule):
    """(dòng văn bản, thông tin người trả lời) hoặc None nếu không thấy vùng câu hỏi."""
    q = rule["qa"]
    qels = [el for xp in q["question"] for el in doc.xpath(xp)]
    if not qels:
        return None
    lines = [line for el in qels for line in _region_lines(el, rule)]
    answers = []
    for a in doc.xpath(q["answer"]):
        src = a.xpath(q["a_text"]) if q.get("a_text") else []
        al = [line for el in src for line in _region_lines(el, rule)]
        if not al:                                         # HTML lồng sai (<p> trong <p>) làm vùng nội dung rỗng:
            al = _region_lines(a, rule)                    # lấy cả khối câu trả lời (đã bỏ phần giao diện)
        if not al:
            continue
        author = _first_text(a, q.get("a_author"), 60)
        title = _first_text(a, q.get("a_title"), 120)
        if author and title.startswith(author):
            title = title[len(author):].strip()
        info = {"author": author, "author_title": title,
                "time": date_from_elements(a.xpath(q["a_time"])) if q.get("a_time") else None}
        answers.append({k: v for k, v in info.items() if v})
        lines += ["---"] + al
    return lines, answers


def _generic(doc):
    import trafilatura
    xml = trafilatura.extract(copy.deepcopy(doc), output_format="xml", include_comments=False, include_tables=True,
                              favor_recall=True, deduplicate=False, include_images=False, include_links=False,
                              with_metadata=False)
    return tei_lines(xml)


def _largest_block(root):
    """Vùng có nhiều chữ trong thẻ <p> nhất (kiểu readability) — khi mọi cách khác trả về rỗng."""
    root = copy.deepcopy(root)
    for bad in root.xpath(".//nav|.//footer|.//aside"):    # không bỏ <form>: nhiều trang ASP.NET bọc cả bài trong form
        bad.drop_tree()
    score = collections.Counter()
    for p in root.iter("p"):
        n = len((p.text_content() or "").strip())
        if n < 25:
            continue
        par = p.getparent()
        if par is not None:
            score[par] += n
            g = par.getparent()
            if g is not None:
                score[g] += n // 2
    if not score:
        return []
    return render(max(score, key=score.get))


def _post(lines):
    out = []
    for line in lines:
        if "&" in line:                                    # trang mã hoá thực thể 2 lần (vd 120ask: "&#8226;")
            line = _html.unescape(line)
        if not _JUNK_LINE.match(split_marker(line)[1].strip()):
            out.append(line)
    return clean_lines("\n".join(out))


def extract_job(body, host, url, charset, want_raw, min_chars, min_chars_zh):
    """Chạy trong process con: không để một trang lỗi làm hỏng cả lô."""
    try:
        return process_page(body, host, url, charset, want_raw, min_chars, min_chars_zh)
    except Exception as e:
        return {"status": "err_extract", "extractor": f"{type(e).__name__}: {str(e)[:200]}", "xv": XV}, None


def text_hash(plain):
    return hashlib.md5(re.sub(r"\s+", "", plain.lower()).encode("utf-8")).hexdigest()[:16]


def process_page(body, host, url="", header_charset=None, want_raw=False, min_chars=30, min_chars_zh=10):
    """bytes/str HTML → (bản ghi, HTML gọn hoặc None). Bản ghi chưa có id/url/thời gian (nơi gọi bổ sung).
    Ngưỡng "empty" riêng cho tiếng Trung: mục từ điển Đông y ("石莲子：苦寒，清心开胃去湿热。") ngắn nhưng có nội dung."""
    src = strip_html(parse(decode_html(body, header_charset)))   # mọi bước dưới dùng bản gọn (= bản lưu vào raw/)
    doc = parse(src)
    meta = extract_meta(doc)
    rule = find_rule(host)
    lines, extra, extractor, title = [], {}, "generic", ""
    site_ok = False
    if rule:
        try:
            qa = _qa_extract(doc, rule) if rule.get("qa") else None
            if qa is not None:
                lines, answers = qa
                if answers:
                    extra["answers"] = answers
            else:
                lines = _site_extract(doc, rule)
        except Exception:
            lines = []
        site_ok = _n(lines) >= rule.get("min_chars", 30)
        if rule.get("strict") and not site_ok and any(doc.xpath(xp) for xp in rule["keep"]):
            site_ok = True                                 # có vùng nội dung nhưng rỗng (trang chỉ có ảnh) → không lấy menu
        if site_ok:
            extractor = "rule:" + rule["suffix"]
            title = _first_text(doc, rule.get("title"), 500)
            if rule.get("title_first_line") and lines:
                title = split_marker(lines[0])[1][:500]
        else:
            lines, extra = [], {}
    if not site_ok:
        lines = _generic(doc)
        if _n(lines) < 50:
            # HTML hỏng (vd thẻ <html> đóng ngay đầu trang) làm bộ trích chung không thấy nội dung:
            # lấy nhánh <body>/<html> nhiều chữ nhất rồi trích lần nữa, cuối cùng lấy vùng nhiều chữ nhất.
            root = max(doc.xpath("//body") + doc.xpath("//html") or [doc],
                       key=lambda e: len(e.text_content() or ""))
            try:
                lines2, how = _generic(parse(lxml.html.tostring(root, encoding="unicode"))), "generic"
                if _n(lines2) < 50:
                    lines2, how = _largest_block(root), "fallback"
                if _n(lines2) > _n(lines):
                    lines, extractor = lines2, how
            except Exception:
                pass
    lines = _post(lines)
    text = "\n".join(lines)
    plain = to_plain(text)

    pub = (date_from_elements([e for xp in rule.get("date", []) for e in doc.xpath(xp)]) if rule and site_ok
           else None) or meta["published"] or date_from_header_lines([split_marker(x)[1] for x in lines])
    author = (good_author(_first_text(doc, rule.get("author"), 100), meta["site_name"]) if rule and site_ok
              else "") or meta["author"]
    section = meta["section"]
    keywords = meta["keywords"]
    title = strip_site_suffix(clean_field(title or meta["title"], 500), meta["site_name"], host)
    if rule and site_ok and rule.get("section"):
        parts = []
        for el in doc.xpath(rule["section"]):
            parts += [p for p in _CRUMB_SEP.split(clean_field(el.xpath("string()"), 300)) if p]
        section = tidy_breadcrumb(parts, title) or section
    if rule and site_ok and rule.get("keywords"):
        kws = [clean_field(e.xpath("string()"), 50) for e in doc.xpath(rule["keywords"])]
        keywords = "; ".join(dict.fromkeys(k for k in kws if k)) or keywords
    if meta["canonical"]:
        extra["canonical"] = meta["canonical"]

    lang = detect_lang(plain or title, host, meta["lang_hint"])
    if is_notice(title, plain):
        status = "notice"
    elif len(plain) < (min_chars_zh if lang == "zh" else min_chars):
        status = "empty"                                   # vẫn giữ tiêu đề/metadata/chữ trong bản ghi
    else:
        status = "ok"
    rec = {
        "status": status, "lang": lang, "doc_type": doc_type(host, url),
        "title": title, "text": text, "description": meta["description"], "author": author, "published": pub or "",
        "modified": meta["modified"], "section": section, "keywords": keywords, "site_name": meta["site_name"],
        "extra": json.dumps(extra, ensure_ascii=False) if extra else "", "extractor": extractor, "xv": XV,
        "n_chars": len(plain), "text_hash": text_hash(plain) if plain else "",
    }
    return rec, (src if want_raw else None)
