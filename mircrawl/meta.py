"""Metadata của trang: tiêu đề, mô tả, tác giả, ngày đăng/cập nhật, chuyên mục, từ khoá, tên trang.

Nguồn theo thứ tự tin cậy: quy tắc riêng của tên miền (rules.py) > JSON-LD > thẻ meta (OpenGraph, article:*, …)
> phần tử có itemprop. KHÔNG suy đoán từ thân bài: thiếu thì để trống.
Ngày luôn chuẩn hoá về "YYYY-MM-DD" (15/09/2026 | September 15, 2026 | 2026年9月15日 → 2026-09-15).
"""
import datetime
import json
import re
import unicodedata

from .htmlx import norm_text

# ---------------------------------------------------------------- ngày
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_D_YMD = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})(?!\d)")
_D_VN = re.compile(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*((?:19|20)\d{2})", re.I)
_D_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*((?:19|20)\d{2})(?!\d)")
_D_MDY_EN = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b")
_D_DMY_EN = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d{2})\b")
_D_EPOCH = re.compile(r"^\d{10}(?:\d{3})?$")


def _mk(y, m, d):
    try:
        dt = datetime.date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None
    if dt.year < 1900 or dt > datetime.date.today() + datetime.timedelta(days=1):
        return None
    return dt.isoformat()


def norm_date(s):
    """Chuỗi ngày bất kỳ → 'YYYY-MM-DD', hoặc None nếu không đọc được / không hợp lệ (vd ngày ở tương lai)."""
    if s is None:
        return None
    s = norm_text(str(s)).strip()
    if not s:
        return None
    if _D_EPOCH.match(s):
        try:
            ts = int(s[:10])
            return _mk(*datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).timetuple()[:3])
        except (OverflowError, OSError, ValueError):
            return None
    m = _D_YMD.search(s)
    if m and _mk(*m.groups()):
        return _mk(*m.groups())
    for rx in (_D_VN, _D_DMY):                               # tiếng Việt: ngày trước tháng
        m = rx.search(s)
        if m and _mk(m.group(3), m.group(2), m.group(1)):
            return _mk(m.group(3), m.group(2), m.group(1))
    m = _D_MDY_EN.search(s)
    if m and m.group(1)[:3].lower() in _MONTHS:
        r = _mk(m.group(3), _MONTHS[m.group(1)[:3].lower()], m.group(2))
        if r:
            return r
    m = _D_DMY_EN.search(s)
    if m and m.group(2)[:3].lower() in _MONTHS:
        return _mk(m.group(3), _MONTHS[m.group(2)[:3].lower()], m.group(1))
    return None


_DATE_LABEL = re.compile(r"^(ngày đăng|đăng ngày|ngày xuất bản|ngày cập nhật|cập nhật|thời gian|发布时间|发表时间|"
                         r"发布日期|更新时间|时间|日期|published|posted on|date)\s*[:：]?\s*", re.I)
_DT_TOKEN = re.compile(r"\d{1,4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,4}日?"
                       r"(?:\s*[-–,|]?\s*\d{1,2}[:h]\d{2}(?::\d{2})?)?(?:\s*\(?GMT\s*[+-]\s*\d+\)?)?", re.I)


def date_from_header_lines(lines, n=5):
    """Ngày ghi RÕ ở vài dòng đầu bài: dòng có nhãn ("Ngày đăng: 27/08/2009", "发布时间：2016-03-12") hoặc dòng gần
    như chỉ gồm ngày giờ ("19/02/2014 08:24", "Hà Vũ • 14/04/2025 - 19:25"). Không lấy ngày nằm trong câu văn."""
    for line in lines[:n]:
        line = line.lstrip("#- ").strip()
        if len(line) > 60:
            continue
        if _DATE_LABEL.match(line):
            d = norm_date(line)
        else:
            m = _DT_TOKEN.search(line)
            rest = (line[:m.start()] + line[m.end():]).strip(" •·|-–,") if m else "x" * 99
            d = norm_date(m.group(0)) if len(rest) <= 25 and not re.search(r"\d{3,}", rest) else None
        if d:
            return d
    return None


def date_from_elements(els):
    """Ngày đầu tiên đọc được từ các phần tử (thuộc tính datetime/content, rồi đến chữ)."""
    for e in els:
        if isinstance(e, str):
            d = norm_date(e)
        else:
            d = norm_date(e.get("datetime")) or norm_date(e.get("content")) or norm_date(e.xpath("string()"))
        if d:
            return d
    return None


# ---------------------------------------------------------------- JSON-LD
_ARTICLE_TYPES = {"article", "newsarticle", "blogposting", "medicalwebpage", "medicalscholarlyarticle",
                  "scholarlyarticle", "report", "reportagenewsarticle", "analysisnewsarticle", "techarticle",
                  "healthtopiccontent", "qapage", "question", "medicalcondition", "drug", "webpage", "posting"}
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _types(o):
    t = o.get("@type")
    t = t if isinstance(t, list) else [t]
    return {str(x).lower() for x in t if x}


def ld_objects(doc):
    out = []
    for s in doc.xpath('//script[contains(translate(@type,"LDJSON","ldjson"),"ld+json")]'):
        raw = (s.text or "").strip()
        if not raw:
            continue
        data = None
        for txt in (raw, _TRAILING_COMMA.sub(r"\1", _CTRL.sub(" ", raw))):
            try:
                data = json.loads(txt, strict=False)
                break
            except ValueError:
                continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, dict):
                out.append(x)
                for k in ("@graph", "mainEntity"):
                    if isinstance(x.get(k), (dict, list)):
                        stack.append(x[k])
    return out


def _names(v, limit=10):
    """author/publisher của JSON-LD (chuỗi, dict, list) → danh sách tên (bỏ URL)."""
    out = []
    for x in (v if isinstance(v, list) else [v]):
        if isinstance(x, dict):
            x = x.get("name") or ""
        if isinstance(x, list):
            x = " ".join(str(y) for y in x)
        x = clean_field(x, 100)
        if x and not _URLISH.match(x) and x not in out:
            out.append(x)
    return out[:limit]


# ---------------------------------------------------------------- tiện ích
_URLISH = re.compile(r"^(https?:)?//|^www\.|\.(com|vn|cn|net|org)(/|$)", re.I)


def clean_field(s, max_len=1000):
    s = re.sub(r"\s+", " ", norm_text(str(s or ""))).strip()
    if "&" in s:
        import html
        s = html.unescape(s)
    return s[:max_len]


_BAD_AUTHOR = {"tác giả", "author", "admin", "administrator", "biên tập viên", "ban biên tập", "btv", "phóng viên",
               "pv", "nguồn", "source", "editor", "staff", "guest", "unknown", "作者", "佚名", "未知", "管理员", "小编",
               "编辑", "责任编辑", "来源"}


_BAD_AUTHOR_RX = re.compile(r"^(admin\w*|system|root|user\d*|p\.?\s?v\.?|\d+|nguồn\s*[:：].*|source\s*:.*|"
                            r"(báo|tạp chí|cổng thông tin|trang tin|báo điện tử)\s.*)$", re.I)


def good_author(a, site=""):
    """Tên tác giả hợp lệ hoặc "" (bỏ chữ giữ chỗ "Tác giả", tài khoản hệ thống, mã số, "Nguồn: …", tên tờ báo,
    URL, tên trang, ngày tháng). Tác giả mặc định của cả trang (vd "VnExpress") do merge bỏ theo tần suất."""
    a = clean_field(a, 100).strip(" :：-–|")
    parts = re.split(r"\s+-\s+", a, maxsplit=1)
    if len(parts) == 2 and re.search(r"\d", parts[1]):     # "Tên - Số 8, Lý Nam Đế, …" (địa chỉ dính vào)
        a = parts[0]
    if len(a) < 2 or a.lower() in _BAD_AUTHOR or _BAD_AUTHOR_RX.match(a) or _URLISH.match(a) or norm_date(a):
        return ""
    if site and a.lower() == site.lower():
        return ""
    return a


def _fold(s):
    """'Sức Khỏe Cộng Đồng' → 'suckhoecongdong' (bỏ dấu, chỉ giữ a-z0-9) để so với tên miền."""
    s = unicodedata.normalize("NFD", (s or "").lower().replace("đ", "d"))
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if not unicodedata.combining(c)))


_TITLE_SEPS = (" | ", " • ", " - ", " – ", " — ", " _ ", "_", " :: ", " » ", " · ", "--", "-", "|")


def strip_site_suffix(title, site="", host=""):
    """Bỏ hậu tố tên trang ở tiêu đề: 'Sỏi thận … | Sức Khỏe Cộng Đồng' → 'Sỏi thận …'.
    Chỉ bỏ khi phần đuôi trùng tên trang (og:site_name) hoặc trùng tên miền sau khi bỏ dấu; phần còn lại do merge
    xử lý theo tần suất (hậu tố lặp lại ở nhiều bài cùng tên miền)."""
    title = (title or "").strip().rstrip(" •|-–—_·")
    hfold = _fold(host.split(":")[0])
    for sep in _TITLE_SEPS:
        if sep in title:
            head, _, tail = title.rpartition(sep)
            t = tail.strip()
            tf = _fold(t)
            if head.strip() and ((site and t.lower() == site.lower()) or (len(tf) >= 4 and tf in hfold)):
                return head.strip()
    return title


def _split_kw(s):
    return [k.strip() for k in re.split(r"[,，;；、|]", s or "") if k.strip()]


def meta_tags(doc):
    m = {}
    for e in doc.xpath("//meta[@content]"):
        k = (e.get("property") or e.get("name") or e.get("itemprop") or e.get("http-equiv") or "").strip().lower()
        v = clean_field(e.get("content"), 2000)
        if k and v and k not in m:
            m[k] = v
    return m


_PUB_KEYS = ("article:published_time", "datepublished", "pubdate", "publishdate", "publish-date", "date",
             "dc.date.issued", "dc.date", "dcterms.created", "citation_publication_date", "og:published_time",
             "bytedance:published_time", "content_date_published", "sailthru.date", "article.published")
_MOD_KEYS = ("article:modified_time", "og:updated_time", "datemodified", "last-modified", "dcterms.modified",
             "bytedance:updated_time", "article.updated")
_AUTHOR_KEYS = ("author", "article:author", "dable:author", "content_author", "byl", "dc.creator", "citation_author")
_HOME = {"trang chủ", "trang chu", "home", "首页", "主页", "网站首页", "trang chính"}


def breadcrumb_from_ld(objs):
    for o in objs:
        if "breadcrumblist" in _types(o):
            items = o.get("itemListElement") or []
            items = items if isinstance(items, list) else [items]
            try:
                items = sorted(items, key=lambda x: int(x.get("position", 0)) if isinstance(x, dict) else 0)
            except (TypeError, ValueError):
                pass
            names = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                item = it.get("item")
                n = clean_field(it.get("name") or (item.get("name") if isinstance(item, dict) else ""), 80)
                if n:
                    names.append(n)
            if names:
                return names
    return []


def breadcrumb_from_html(doc):
    for el in doc.xpath('//*[contains(translate(@class,"BREADCUM","breadcum"),"breadcrumb") or '
                        'contains(translate(@id,"BREADCUM","breadcum"),"breadcrumb")][1]'):
        parts = [clean_field(x, 80) for x in el.xpath(".//a//text() | .//li//text() | .//span//text()")]
        parts = [p for p in parts if p and p not in (">", "›", "»", "/", "|", "-")]
        if 1 <= len(parts) <= 10:
            return parts
    return []


def tidy_breadcrumb(parts, title=""):
    out = []
    for p in parts:
        p = p.strip(" >›»/|")
        if p and p.lower() not in _HOME and p not in out:
            out.append(p)
    if out and title and out[-1] == title:
        out.pop()
    return " > ".join(out[:6])


def extract_meta(doc):
    """Metadata chung (không phụ thuộc tên miền). Trả về dict các chuỗi (rỗng nếu không có)."""
    ld = ld_objects(doc)
    mt = meta_tags(doc)
    art = [o for o in ld if _types(o) & _ARTICLE_TYPES]
    # bài viết thật (Article/NewsArticle…) ưu tiên hơn WebPage chung chung
    art.sort(key=lambda o: 1 if _types(o) <= {"webpage"} else 0)

    def ld_get(*keys):
        for o in art:
            for k in keys:
                v = o.get(k)
                if v:
                    return v
        return None

    title = clean_field(ld_get("headline") or mt.get("og:title") or mt.get("twitter:title") or "", 500)
    if not title:
        h1 = [clean_field(h.xpath("string()"), 500) for h in doc.xpath("//h1")]
        h1 = [h for h in h1 if h]
        if len(h1) == 1:
            title = h1[0]
    if not title:
        t = doc.xpath("//title")
        title = clean_field(t[0].xpath("string()"), 500) if t else ""

    desc = clean_field(ld_get("description") or mt.get("og:description") or mt.get("description")
                       or mt.get("twitter:description") or "", 1000)
    published = norm_date(ld_get("datePublished", "dateCreated", "uploadDate"))
    if not published:
        published = next((d for d in (norm_date(mt.get(k)) for k in _PUB_KEYS) if d), None)
    if not published:
        published = date_from_elements(doc.xpath('//*[@itemprop="datePublished"]'))
    modified = norm_date(ld_get("dateModified"))
    if not modified:
        modified = next((d for d in (norm_date(mt.get(k)) for k in _MOD_KEYS) if d), None)

    site = clean_field(mt.get("og:site_name") or "", 200)
    if not site:
        pub = ld_get("publisher")
        site = (_names(pub) or [""])[0]
    authors = [a for a in (good_author(x, site) for x in _names(ld_get("author", "creator"))) if a]
    if not authors:
        for k in _AUTHOR_KEYS:
            a = good_author(mt.get(k), site)
            if a:
                authors = [a]
                break

    crumbs = breadcrumb_from_ld(ld) or breadcrumb_from_html(doc)
    section = tidy_breadcrumb(crumbs, title)
    if not section:
        sec = ld_get("articleSection")
        sec = sec[0] if isinstance(sec, list) and sec else sec
        section = clean_field(sec or mt.get("article:section") or "", 200)

    kws = []
    ldkw = ld_get("keywords")
    for src in (mt.get("news_keywords"), mt.get("keywords"), ", ".join(ldkw) if isinstance(ldkw, list) else ldkw):
        for k in _split_kw(src):
            if k not in kws:
                kws.append(k)
    lang = ld_get("inLanguage")
    lang = clean_field(lang if isinstance(lang, str) else "", 20) or mt.get("language") or \
        mt.get("content-language") or mt.get("page_language") or (doc.get("lang") or "")
    canon = doc.xpath("//link[@rel='canonical']/@href")
    return {
        "title": title, "description": desc, "author": "; ".join(authors), "published": published or "",
        "modified": modified or "", "section": section, "keywords": "; ".join(kws[:30]), "site_name": site,
        "lang_hint": clean_field(lang, 20).lower(), "canonical": clean_field(canon[0], 1000) if canon else "",
    }
