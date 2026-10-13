"""Phân loại trang: ngôn ngữ, loại tài liệu, trang thông báo (không phải bài viết)."""
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------- ngôn ngữ
_CJK = re.compile(r"[一-鿿]")
_VI = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)
_LATIN = re.compile(r"[A-Za-z]")
VI_HOSTS = {"hellobacsi.com", "benhvienvietduc.org", "pmc-ecm-healthblog.beta.pharmacity.io", "www.pharmacity.vn"}


def detect_lang(text, host="", hint=""):
    """'zh' | 'vi' | 'en'. So số chữ Hán, số chữ cái có dấu tiếng Việt và số chữ Latinh (không dùng thẻ <html lang>:
    nhiều trang tiếng Trung khai báo sai là 'en')."""
    s = (text or "")[:4000]
    cjk, vi, lat = len(_CJK.findall(s)), len(_VI.findall(s)), len(_LATIN.findall(s))
    if cjk >= 10 and cjk > 2 * vi:
        return "zh"
    if vi >= 5 and vi >= 0.02 * lat:
        return "vi"
    if lat >= 100 and vi < 0.005 * lat and cjk < 10:
        return "en"
    if host.endswith(".vn") or host in VI_HOSTS:
        return "vi"
    if cjk or host.endswith(".cn"):
        return "zh"
    return (hint or "")[:2] if (hint or "")[:2] in ("vi", "zh", "en") else ""


# ---------------------------------------------------------------- loại tài liệu
# qa: hỏi đáp | knowledge: bài kiến thức (bệnh, triệu chứng, chẩn đoán, điều trị) | book: sách, cẩm nang, bách khoa
# drug: thuốc, dược liệu, bài thuốc | news: tin tức | paper: bài báo khoa học (PubMed) | other
# (hậu tố tên miền, regex đường dẫn hoặc None, loại) — dòng đầu tiên khớp được dùng.
DOC_TYPES = [
    ("pubmed.ncbi.nlm.nih.gov", None, "paper"),
    ("120ask.com", None, "qa"), ("ask.39.net", None, "qa"), ("ask.familydoctor.com.cn", None, "qa"),
    ("thaythuocvietnam.vn", r"^/cau-hoi/", "qa"), ("thaythuocvietnam.vn", r"^/thuvien/", "paper"),
    ("jbk.39.net", None, "knowledge"), ("jb39.com", None, "knowledge"), ("baidu.com", r"^/bh/", "knowledge"),
    ("baidianfeng.familydoctor.com.cn", None, "knowledge"),
    ("msdmanuals.cn", None, "book"), ("a-hospital.com", None, "book"), ("zhongyibaodian.net", None, "book"),
    ("qihuangzhishu.com", None, "book"), ("pmphai.com", None, "book"), ("zysjonline.com", r"^/books/", "book"),
    ("zydcd.com", None, "drug"), ("zysjonline.com", r"^/(formulas|herbs?|medicines?)/", "drug"),
    ("wujue.com", r"^/(gjwx/fangji|zyzy)/", "drug"),
    ("nhathuoclongchau.com.vn", r"^/(thuoc|duoc-lieu|thuc-pham-chuc-nang|thanh-phan)/", "drug"),
    ("nhathuoclongchau.com.vn", None, "knowledge"), ("youmed.vn", r"^/tin-tuc/(thuoc|duoc-lieu)", "drug"),
    ("youmed.vn", None, "knowledge"), ("hellobacsi.com", None, "knowledge"), ("vinmec.com", None, "knowledge"),
    ("medlatec.vn", None, "knowledge"), ("tamanhhospital.vn", None, "knowledge"),
    ("tiemchunglongchau.com.vn", None, "knowledge"), ("pharmacity.io", None, "knowledge"),
    ("pharmacity.vn", None, "knowledge"), ("youlai.cn", None, "knowledge"), ("cnkang.com", None, "knowledge"),
    ("familydoctor.com.cn", None, "knowledge"), ("39.net", None, "knowledge"), ("wujue.com", None, "knowledge"),
    ("zysjonline.com", None, "knowledge"), ("iiyi.com", None, "knowledge"),
    ("familydoctor.cn", None, "knowledge"),
    ("people.com.cn", None, "news"), ("benhvienvietduc.org", None, "news"), ("vnexpress.net", None, "news"),
]
_DT = [(s, re.compile(p) if p else None, t) for s, p, t in DOC_TYPES]


def doc_type(host, url):
    path = urlparse(url).path or "/"
    for suf, rx, t in _DT:
        if (host == suf or host.endswith("." + suf)) and (rx is None or rx.search(path)):
            return t
    if host.endswith(".vn") or host.endswith(".org.vn"):
        return "news"                                      # báo, cổng thông tin bệnh viện Việt Nam
    return "other"


# ---------------------------------------------------------------- trang thông báo
# Trang tải được nhưng không phải bài viết: 404 mềm, bài đã gỡ, màn hình chọn ngôn ngữ, yêu cầu bật JavaScript,
# trang xác minh. Chỉ xét khi chữ ngắn để không bắt nhầm bài viết có nhắc tới các cụm từ này.
NOTICE = re.compile("|".join([
    # tiếng Việt
    r"(trang|bài viết|nội dung|đường dẫn)[^.\n]{0,40}(không tồn tại|không tìm thấy|đã bị (xóa|xoá|gỡ)|chưa có nội dung)",
    r"không tìm thấy (trang|bài viết|nội dung)", r"chọn ngôn ngữ", r"tiếp tục (để )?đọc (bằng|bản) tiếng",
    r"chưa có (nội dung|phiên bản) tiếng", r"vui lòng bật javascript", r"đang chuyển hướng",
    # tiếng Anh
    r"page (not found|does not exist|cannot be found)", r"404 (error|not found)", r"\berror 404\b",
    r"(select|choose) (your|a) (language|region)", r"not available in (english|your language)",
    r"(please )?enable javascript", r"access denied", r"just a moment\.\.\.", r"checking your browser",
    r"verify you are (a )?human", r"this (page|article|content) (has been|was) (removed|deleted)",
    # tiếng Trung
    r"(页面|网页|文章|内容|问题|帖子)(已|已经)?(不存在|被删除|删除|下架|失效|找不到)", r"(您|你)(访问|要找|浏览|请求)的(页面|网页|内容)",
    r"找不到(该|此|您要的)?(页面|网页)", r"抱歉[，,！!]?\s*(页面|您访问|该|此)", r"请选择(语言|地区)", r"访问出错",
    r"(请|需要)?(开启|启用)\s*javascript", r"安全验证", r"人机验证", r"滑动(验证|拼图)",
]), re.I)


def is_notice(title, text, max_chars=400):
    """True nếu trang là thông báo (lỗi, bài đã gỡ, chọn ngôn ngữ, xác minh…) chứ không phải nội dung."""
    if len(text) > max_chars:
        return False
    return bool(NOTICE.search(text) or NOTICE.search(title or ""))
