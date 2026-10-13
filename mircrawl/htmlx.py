"""HTML → văn bản có cấu trúc.

Định dạng cột `text` (mỗi dòng một khối, theo thứ tự đọc):
  "# " … "###### "   tiêu đề mục (h1–h6; đoạn ngắn chỉ gồm chữ đậm cũng coi là tiêu đề "### ")
  "- "               mục danh sách
  "a | b | c"        một hàng bảng, các ô cách nhau " | "
  "---"              ranh giới giữa câu hỏi và từng câu trả lời (trang hỏi đáp)
  dòng còn lại       đoạn văn
Các dấu này chỉ đánh dấu cấu trúc, không thêm chữ. `to_plain()` bỏ chúng để lấy chữ thuần (vd khi tạo chunk_text).
"""
import copy
import re
import unicodedata

import lxml.etree
import lxml.html

# ---------------------------------------------------------------- giải mã bảng mã
_CHARSET = re.compile(rb'''<meta[^>]+charset\s*=\s*["']?\s*([\w-]+)|<\?xml[^>]+encoding\s*=\s*["']([\w-]+)''', re.I)
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def decode_html(html_bytes, header_charset=None):
    """bytes → str. Thứ tự: BOM, charset ở thẻ meta, charset ở header HTTP, UTF-8 chặt, GB18030.
    Họ GB (gb2312/gbk) luôn giải mã bằng gb18030 (tập cha) để không vỡ chữ hiếm."""
    if isinstance(html_bytes, str):
        return html_bytes
    if html_bytes.startswith(b"\xef\xbb\xbf"):
        return html_bytes[3:].decode("utf-8", "replace")
    cands = []
    m = _CHARSET.search(html_bytes[:6000])
    if m:
        cands.append((m.group(1) or m.group(2)).decode("ascii", "ignore").lower())
    if header_charset:
        cands.append(header_charset.lower())
    for enc in cands:
        if enc.startswith("gb") or enc in ("x-gbk", "cp936"):
            return html_bytes.decode("gb18030", "replace")
        if enc in ("utf8", "utf-8"):
            break                                          # khai báo UTF-8: kiểm tra chặt bên dưới (có trang khai sai)
        try:
            return html_bytes.decode(enc, "strict")
        except (LookupError, UnicodeDecodeError):
            continue
    try:
        return html_bytes.decode("utf-8", "strict")
    except UnicodeDecodeError:
        pass
    gb = html_bytes.decode("gb18030", "replace")
    u8 = html_bytes.decode("utf-8", "replace")
    return gb if gb.count("�") < u8.count("�") else u8


# ---------------------------------------------------------------- phân tích & làm gọn
def parse(src):
    """str → cây lxml của CẢ văn bản.
    - document_fromstring luôn đọc cả trang (fromstring đôi khi coi trang là "đoạn HTML" và chỉ trả phần tử đầu).
    - HTML hỏng kiểu `<html></html><head>…<body>…` (vd baotayninh.vn) tạo nhiều gốc <html> liền nhau:
      lấy gốc nhiều chữ nhất."""
    src = _XML_DECL.sub("", src, count=1)                  # lxml không nhận str có khai báo encoding
    src = _CTRL.sub("", src)                               # ký tự NULL / điều khiển (vd benhviennhitrunguong.gov.vn) làm lxml lỗi
    try:
        d = lxml.html.document_fromstring(src)
    except Exception:                                      # trang rỗng / không phải HTML
        return lxml.html.document_fromstring("<html><body></body></html>")
    sibs = [d] + [s for s in d.itersiblings() if isinstance(s.tag, str)]
    if len(sibs) > 1:
        best = max(sibs, key=lambda e: len(e.text_content() or ""))
        if best is not d:
            d = lxml.html.document_fromstring(lxml.html.tostring(best, encoding="unicode"))
    return d


_NOISE_DOC = ("//script[not(contains(translate(@type,'LDJSON','ldjson'),'ld+json'))]"
              "|//style|//noscript|//svg|//iframe|//template|//link[not(@rel='canonical')]|//comment()")


def drop(e):
    """Bỏ phần tử, giữ lại phần chữ đứng sau nó (tail). Dùng được cho cả cây HTML và XML."""
    p = e.getparent()
    if p is None:
        return
    if e.tail:
        prev = e.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + e.tail
        else:
            p.text = (p.text or "") + e.tail
    p.remove(e)


def strip_html(doc):
    """HTML gọn (sửa trực tiếp trên doc): bỏ script (trừ JSON-LD), style, comment, iframe, svg, link (trừ canonical),
    thuộc tính sự kiện và thuộc tính data-*. Giữ class/id/meta.
    Bộ trích LUÔN chạy trên bản gọn này nên trích lúc crawl và trích lại từ HTML đã lưu cho kết quả giống hệt."""
    for bad in doc.xpath(_NOISE_DOC):
        drop(bad)
    for e in doc.iter():
        if isinstance(e.tag, str):
            for k in [k for k in e.attrib if k.startswith(("on", "data-")) or k in ("srcset", "sizes")]:
                del e.attrib[k]
    return lxml.html.tostring(doc, encoding="unicode")


# ---------------------------------------------------------------- vùng nội dung → các dòng
HEADINGS = {f"h{i}": i for i in range(1, 7)}
BLOCK = {"p", "div", "dd", "dt", "section", "article", "table", "ul", "ol", "dl", "blockquote", "pre", "figure",
         "figcaption", "header", "footer", "main", "aside", "nav", "form", "hr", "center", "address", "details",
         "summary", "caption", "tbody", "thead", "tfoot", "tr"}
CELL = {"td", "th"}
_NOISE_EL = (".//script|.//style|.//noscript|.//template|.//button|.//select|.//textarea|.//input|.//svg"
             "|.//iframe|.//audio|.//video|.//object|.//embed")
_HAS_BLOCK = ".//p|.//div|.//li|.//table|.//br|.//h1|.//h2|.//h3|.//h4|.//h5|.//h6|.//ul|.//ol"
_NOT_BOLD = ".//text()[not(ancestor::strong) and not(ancestor::b)]"
_NO_HEAD_INSIDE = set(HEADINGS) | {"li", "td", "th", "a", "dt", "dd"}
_SENT_END = ".。,，;；…"


def _norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _reorder(el, pattern):
    """Sắp lại các khối con có id khớp pattern (vd jbk11, jbk10 … jbk0) theo số tăng dần, giữ nguyên vị trí cụm."""
    rx = re.compile(pattern)
    kids = [c for c in el if isinstance(c.tag, str) and rx.fullmatch(c.get("id") or "")]
    if len(kids) < 2:
        return
    anchor = kids[0].getprevious()
    for c in kids:
        el.remove(c)
    kids.sort(key=lambda c: int(rx.fullmatch(c.get("id")).group(1)))
    pos = 0 if anchor is None else list(el).index(anchor) + 1
    for i, c in enumerate(kids):
        el.insert(pos + i, c)


def _mark_bold_headings(el):
    """Đoạn ngắn (≤120 ký tự), không kết thúc bằng dấu câu, toàn bộ chữ in đậm → tiêu đề mục.
    Rất phổ biến ở báo Việt: <p><strong>1. Triệu chứng</strong></p>."""
    for p in el.iter("p", "div"):
        if p.get("data-mh") or p.xpath(_HAS_BLOCK):
            continue
        txt = _norm_ws(p.xpath("string()"))
        if not (2 <= len(txt) <= 120) or txt[-1] in _SENT_END:
            continue
        if not p.xpath(".//strong|.//b") or _norm_ws("".join(p.xpath(_NOT_BOLD))).strip(" :：-–—·•|"):
            continue
        if any(isinstance(a.tag, str) and a.tag.lower() in _NO_HEAD_INSIDE for a in p.iterancestors()):
            continue
        p.set("data-mh", "3")


def render(el, drop_xpath=None, reorder=None, bold_headings=True):
    """Một vùng nội dung → danh sách dòng theo định dạng ở đầu file. Không sửa cây gốc."""
    el = copy.deepcopy(el)
    for bad in el.xpath(_NOISE_EL):
        drop(bad)
    if drop_xpath:
        for bad in el.xpath(drop_xpath):
            drop(bad)
    if reorder:
        _reorder(el, reorder)
    if bold_headings:
        _mark_bold_headings(el)
    for e in el.iter():
        if not isinstance(e.tag, str):
            continue
        tag = e.tag.lower()
        lvl = HEADINGS.get(tag) or (3 if e.get("data-mh") else 0)
        if lvl:
            e.text = "\n" + "#" * lvl + " " + (e.text or "")
            e.tail = "\n" + (e.tail or "")
        elif tag == "li":
            e.text = "\n- " + (e.text or "")
            e.tail = "\n" + (e.tail or "")
        elif tag in CELL:                                   # ô cùng hàng: "a | b" (không dính "药方名称阑尾1号")
            e.tail = " | " + (e.tail or "")
        elif tag == "br":
            e.tail = "\n" + (e.tail or "")
        elif tag in BLOCK:
            e.text = "\n" + (e.text or "")
            e.tail = "\n" + (e.tail or "")
    return clean_lines(el.xpath("string()"))


_WS = re.compile(r"[ \t　   ]+")
_ZW = re.compile(r"[​‌‍⁠﻿­]")
_PIPE_EDGE = re.compile(r"^(?:\|\s*)+|(?:\s*\|)+$")
_MARK_ONLY = re.compile(r"^(#{1,6}|-)$")
_MARK = re.compile(r"^(#{1,6} |- )")


def norm_text(t):
    """Chuẩn hoá chung: NFC (một số báo Việt viết dấu dạng tổ hợp NFD), bỏ ký tự độ rộng 0, đổi khoảng trắng lạ."""
    t = unicodedata.normalize("NFC", t or "")
    return _ZW.sub("", t.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n"))


def clean_lines(t):
    out, pending = [], ""
    for line in norm_text(t).split("\n"):
        line = _PIPE_EDGE.sub("", _WS.sub(" ", line).strip()).strip()
        if not line:
            continue
        if _MARK_ONLY.match(line):                          # dấu cấu trúc đứng riêng (vd <li><p>…</p></li>): gắn vào dòng sau
            pending = line + " "
            continue
        if pending:
            if not _MARK.match(line):
                line = pending + line
            pending = ""
        out.append(line)
    return out


def split_marker(line):
    """'## Triệu chứng' → ('## ', 'Triệu chứng'); dòng thường → ('', dòng)."""
    m = _MARK.match(line)
    return (m.group(1), line[m.end():]) if m else ("", line)


_MARK_ML = re.compile(r"^(?:#{1,6} |- )", re.M)
_SEP_ML = re.compile(r"^---$\n?", re.M)


def to_plain(text):
    """Bỏ dấu cấu trúc (tiêu đề, danh sách, ranh giới hỏi đáp) → chữ thuần, vẫn mỗi khối một dòng."""
    return _SEP_ML.sub("", _MARK_ML.sub("", text or "")).strip()


# ---------------------------------------------------------------- XML của trafilatura → các dòng
_TEI = {"p": "p", "list": "ul", "item": "li", "table": "table", "row": "tr", "cell": "td", "lb": "br",
        "quote": "blockquote", "code": "pre", "main": "div", "doc": "div"}


def tei_lines(xml):
    """Kết quả trafilatura.extract(output_format="xml") → các dòng cùng định dạng với render()."""
    if not xml:
        return []
    root = lxml.etree.fromstring(xml.encode("utf-8"), lxml.etree.XMLParser(recover=True, huge_tree=True))
    if root is None:
        return []
    for bad in root.xpath("//comments|//graphic"):
        drop(bad)
    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        if e.tag == "head":
            r = (e.get("rend") or "h2").lower()
            e.tag = r if r in HEADINGS else "h2"
        else:
            e.tag = _TEI.get(e.tag, "span")
    return render(root, bold_headings=False)
