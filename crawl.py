"""
Crawl nội dung bài viết từ links_corpus.parquet (ViBioMIR) — có thể dừng/chạy tiếp.

Cài đặt:   pip install curl_cffi trafilatura lxml pyarrow      (aiohttp chỉ cần khi --engine aiohttp)
Thăm dò:   python crawl.py --probe 3                    (vài trang mỗi tên miền, in bảng lỗi theo từng trang)
Chạy thử:  python crawl.py --limit 300 --out out_test
Chạy thật: python crawl.py --out out --shard 0/3 --save-raw --redo-stale --skip-hosts familydoctor.com.cn,zysjonline.com
           (mỗi thành viên một shard khác nhau; mỗi máy một thư mục --out riêng)
Trích lại không cần mạng (sau khi sửa quy tắc, cần đã chạy với --save-raw):
           python crawl.py --out out --reextract stale

Kết quả (thư mục --out):
  part-00000.jsonl.gz ...  mỗi dòng: {"id","url","host","lang","title","text","xv"}
                           (một id có thể có nhiều dòng sau khi tải lại/trích lại: dùng dòng có xv lớn nhất,
                            cùng xv thì dòng sau — mir/crawlio.py đã làm việc này)
  raw/raw-00000.jsonl.gz   (--save-raw) {"id","url","host","html"}: HTML gọn (~12 KB/trang sau nén) để --reextract
  progress.tsv             id <tab> trạng thái (ok / empty / http_404 / err_timeout ...) [<tab> xv], để chạy tiếp
Chạy lại cùng lệnh = tiếp tục từ chỗ dừng. Thêm --retry-failed để thử lại các URL bị lỗi tạm thời,
--redo-stale để tải lại các bài đã trích bằng quy tắc cũ (xv nhỏ hơn quy tắc hiện tại của tên miền).

Bộ trích chữ: quy tắc riêng theo tên miền (SITE_RULES — vùng nội dung chính, bỏ giao diện) cho các tên miền
chiếm ~96% số URL; còn lại dùng trafilatura. Kiểm chứng bằng bộ công cụ trong crawl_eval/ (xem crawl_eval/README.md).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # tránh lỗi "OpenBLAS memory allocation failed" khi chạy nhiều process
import argparse, asyncio, collections, gzip, json, random, re, sys, time, unicodedata
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlparse

import pyarrow.parquet as pq

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CJK = re.compile(r"[\u4e00-\u9fff]")
FINAL_HTTP = {400, 401, 403, 404, 410, 451}      # không thử lại
RETRY_HTTP = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


# ---------- trích nội dung (chạy ở process con) ----------
# Phiên bản bộ trích chữ, ghi vào mỗi bản ghi ("xv") và cột 3 của progress.tsv.
# Đổi quy tắc riêng của một tên miền → tăng EXTRACT_VERSION và đặt "since" của quy tắc đó bằng số mới;
# bản ghi cũ hơn sẽ được làm lại bằng --redo-stale (tải lại) hoặc --reextract (từ HTML đã lưu, không cần mạng).
# Bản ghi không có "xv" (crawl.py trước 3/10/2026) coi như xv = 1.
EXTRACT_VERSION = 2
GENERIC_SINCE = 1      # bộ trích chung (trafilatura) không đổi kết quả từ phiên bản 1
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "section", "article", "table",
          "ul", "ol", "dl", "blockquote", "pre", "figure", "figcaption", "header", "footer", "main", "aside", "nav",
          "form", "hr", "center", "address", "details", "summary", "caption", "tbody", "thead", "tfoot"}
_CELL = {"td", "th"}          # ô bảng cùng hàng: cách nhau dấu cách (không dính "药方名称阑尾1号")

# ---------- QUY TẮC RIÊNG THEO TÊN MIỀN ----------
# Mỗi quy tắc (khớp tên miền và mọi tên miền con của "suffix"):
#   keep   : danh sách XPath vùng nội dung chính. Mặc định lấy MỌI vùng khớp, theo thứ tự trong danh sách.
#   first  : True → thử lần lượt từng XPath, dùng XPath đầu tiên cho ra chữ (trang có nhiều bố cục).
#   strict : True → tìm thấy vùng keep nhưng rỗng thì trả về rỗng (không rơi về bộ trích chung).
#   need   : XPath vùng bắt buộc (thường là thân bài). Không có → quy tắc không khớp (tránh chỉ lấy được sapo).
#   drop   : XPath tương đối (.//...) các phần giao diện nằm TRONG vùng keep, bỏ trước khi lấy chữ.
#   stop   : các dòng bỏ đi (so khớp nguyên dòng);  stop_re: biểu thức chính quy, dòng khớp (search) bị bỏ.
#   cut_re : dòng đầu tiên khớp → bỏ dòng đó và mọi dòng sau nó trong vùng (phần đuôi do mẫu trang chèn vào).
#   reorder: regex id của các khối con cần sắp lại theo số (cnkang: jbk11 ... jbk0 → jbk0 ... jbk11).
#   join_labels: ghép dòng nhãn ngắn kết thúc bằng "：" với dòng giá trị ngay sau (trang dạng bảng thông tin).
#   dedupe_lines: bỏ dòng giống hệt dòng ngay trước (bỏ qua nhãn ngắn đầu dòng như "补充说明：").
#   title  : XPath tiêu đề (mặc định lấy từ metadata trang); title_first_line: lấy dòng đầu nội dung.
#   min_chars: quy tắc cho ra ít hơn số ký tự này coi như không khớp bố cục → dùng bộ trích chung (mặc định 30).
#   since  : EXTRACT_VERSION lúc quy tắc được thêm/sửa lần cuối (bài cũ hơn sẽ bị --redo-stale/--reextract làm lại).
# Khi không có quy tắc hoặc quy tắc không khớp → trafilatura (+ phương án dự phòng cho HTML hỏng).
SITE_RULES = []      # được gán ở cuối phần quy tắc (xem _RULES bên dưới)


def _reorder(el, pattern):
    """Sắp lại các khối con có id khớp pattern (vd jbk11, jbk10 ... jbk0) theo số tăng dần, giữ nguyên vị trí cụm."""
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


_WS = re.compile(r"[ \t　]{2,}")
_ZW = re.compile(r"[​‌‍﻿]")


def _lines(t):
    # NFC: một số báo Việt viết dấu dạng tổ hợp (NFD); pyvi và so khớp chuỗi cần dạng dựng sẵn.
    t = unicodedata.normalize("NFC", t or "")
    t = _ZW.sub("", t.replace("\xa0", " ").replace("　", " ").replace("\r", "\n"))
    return "\n".join(_WS.sub(" ", l.strip()) for l in t.split("\n") if l.strip())


def _block_text(el):
    import copy
    el = copy.deepcopy(el)
    for bad in el.xpath(".//script|.//style|.//noscript|.//template"):
        if bad.getparent() is not None:
            bad.drop_tree()
    for e in el.iter():
        if not isinstance(e.tag, str):
            continue
        tag = e.tag.lower()
        if tag in _BLOCK:
            e.tail = "\n" + (e.tail or "")
            if tag != "br":
                e.text = "\n" + (e.text or "")
        elif tag in _CELL:
            e.tail = " " + (e.tail or "")
    return _lines(el.text_content())


_LABEL = re.compile(r"^[^：:\s]{1,6}[：:]\s*")


def _site_extract(doc, rule):
    import copy
    drop = rule.get("drop")
    if isinstance(drop, (list, tuple)):
        drop = " | ".join(drop)
    stop = rule.get("stop") or set()
    stop_re = re.compile(rule["stop_re"]) if rule.get("stop_re") else None
    cut_re = re.compile(rule["cut_re"]) if rule.get("cut_re") else None

    def region_text(el):
        if drop or rule.get("reorder"):
            el = copy.deepcopy(el)
        if drop:
            for bad in el.xpath(drop):
                if bad.getparent() is not None:
                    bad.drop_tree()
        if rule.get("reorder"):
            _reorder(el, rule["reorder"])
        out = []
        for l in _block_text(el).split("\n"):
            if cut_re and cut_re.search(l):                # phần đuôi do mẫu trang chèn vào: bỏ từ đây trở đi
                break
            if l not in stop and not (stop_re and stop_re.search(l)):
                if rule.get("dedupe_lines") and out and _LABEL.sub("", l) == _LABEL.sub("", out[-1]):
                    continue                               # câu hỏi lặp: tiêu đề / mô tả / "补充说明：" giống hệt nhau
                if rule.get("join_labels") and out and len(out[-1]) <= 12 and out[-1].endswith(("：", ":")):
                    out[-1] += " " + l                     # "发病部位：" + "眼" → "发病部位： 眼"
                else:
                    out.append(l)
        return "\n".join(out)

    parts = []
    if rule.get("need") and not doc.xpath(rule["need"]):   # thiếu vùng bắt buộc (thân bài) → coi như không khớp bố cục
        return "", ""
    for xp in rule["keep"]:
        got = [t for t in (region_text(el) for el in doc.xpath(xp)) if t]
        if rule.get("first"):
            if sum(map(len, got)) >= rule.get("min_chars", 30):
                parts = got
                break
        else:
            parts += got
    if len(parts) > 1:
        # sapo / tóm tắt lặp lại (toàn bộ hoặc gần hết, kể cả bản bị cắt "……") trong phần dài hơn → bỏ phần ngắn
        keys = [re.sub(r"\s+", "", p) for p in parts]
        grams = [{k[i:i + 6] for i in range(max(1, len(k) - 5))} for k in keys]
        keep_idx = []
        for i in range(len(parts)):
            longer = set().union(*[grams[j] for j in range(len(parts))
                                   if j != i and (len(keys[j]) > len(keys[i]) or (len(keys[j]) == len(keys[i]) and j < i))])
            if not grams[i] or len(grams[i] & longer) / len(grams[i]) < 0.8:
                keep_idx.append(i)
        parts = [parts[i] for i in keep_idx]
    title = ""
    for xp in ([rule["title"]] if isinstance(rule.get("title"), str) else rule.get("title") or []):
        h = doc.xpath(xp)
        title = _lines(h[0].text_content() if hasattr(h[0], "text_content") else str(h[0])) if h else ""
        if title:
            break
    if rule.get("title_first_line") and parts:
        title = parts[0].split("\n", 1)[0]
    return title, "\n".join(parts)


def _cls(*names):
    """XPath điều kiện: thẻ có một trong các class (so khớp nguyên token, không khớp nhầm "shortdescription")."""
    return " or ".join(f'contains(concat(" ",normalize-space(@class)," ")," {n} ")' for n in names)


def _news(suffix, sapo, body, drop=None, **kw):
    """Quy tắc trang tin: sapo (tuỳ chọn) + thân bài; thân bài là vùng bắt buộc."""
    r = {"suffix": suffix, "since": 2, "keep": ([sapo] if sapo else []) + [body], "need": body}
    if drop:
        r["drop"] = drop
    r.update(kw)
    return r


# MediaWiki (a-hospital.com, zhongyibaodian.net): bỏ hộp điều hướng, mục lục, danh mục, liên kết nguồn;
# cắt các mục đuôi do mẫu trang chèn vào ("中药方专题", "参看"...).
_MW = {
    "keep": ['//div[@id="bodyContent"]'],
    "drop": './/*[' + _cls("navbox", "nav", "hierarchy-breadcrumb", "hierarchy-list", "hierarchy-nav", "msg-table",
                           "catlinks", "toc", "printfooter", "editsection", "thumbcaption", "noprint") +
            ' or @id="fromlink" or @id="toc" or @id="jump-to-nav" or @id="contentSub" or @id="siteSub"'
            ' or @id="belownav" or @id="catlinks"]',
    "cut_re": r"^(中药方专题|参看|相关页面|相关条目|更多医学百科条目|附：中药材大全|外部链接)$",
    "stop_re": r"^(中医宝典|A\+医学百科|医学电子书) >",
    "title": '//h1[@id="firstHeading"]',
    "since": 2,
}

_RULES = [
    # ---- tiếng Trung ----
    # 120ask (918k URL): câu hỏi + các câu trả lời; bỏ hồ sơ bác sĩ, thời gian, nút bấm, thẻ audio.
    # min_chars 1: trang không có mô tả/trả lời bằng chữ (vd chỉ trả lời bằng audio) → giữ phần ít ỏi đó,
    # không rơi về trafilatura (sẽ lấy toàn menu). Bài < --min-chars sẽ bị ghi "empty".
    {"suffix": "120ask.com", "since": 2, "min_chars": 1, "dedupe_lines": True,
     "keep": ['//div[contains(@class,"b_askbox")]',
              '//div[contains(@class,"b_answerli") and not(contains(@class,"zhenshiAD"))]'],
     "drop": './/*[' + _cls("b_askab1", "b_answerarea", "b_askbot_ad", "tousu", "codeposition", "b_askbox_btn",
                            "b_answertop", "b_anscont_time", "b_ansaddti", "b_docaddti", "b_bico", "img") +
             ' or @id="area" or self::audio]',
     "title": '//h1',
     "stop": {"健康咨询描述：", "追问", "附件：点击查看大图", "图片涉及用户隐私，只有用户本人才能看到。"},
     "stop_re": r"^以上是对“.*”这个问题的建议"},
    # cnkang (963k URL): nhiều bố cục qua các năm → thử lần lượt. Bố cục cũ (#endText): đoạn văn nằm trong
    # div#jbk0..jbkN nhưng HTML xếp NGƯỢC → sắp lại; bỏ "核心提示" (tóm tắt bị cắt, lặp nội dung).
    # strict: vùng nội dung có nhưng rỗng (bài chỉ có ảnh) → bài rỗng, không rơi về trafilatura (sẽ lấy toàn menu).
    {"suffix": "cnkang.com", "since": 2, "first": True, "min_chars": 1, "strict": True,
     "keep": ['//div[' + _cls("detailc") + ']', '//div[@id="endText"]',
              '//td[' + _cls("style3") + ']', '//div[' + _cls("conmain") + ']'],
     "drop": './/div[' + _cls("article_intro") + '] | .//div[@id="jkpl"]',
     "reorder": r"jbk(\d+)",
     "title": ['//h1[@id="h1title"]', '//h1[' + _cls("detaila") + ']'],
     "stop_re": r"^(\[\d+\]\s*)+(下一页)?$|^(上一页|下一页)$|^温馨提示：以上资料仅供参考"},
    # ask.39.net (337k URL): câu hỏi + câu trả lời; bỏ thông tin bác sĩ, thẻ từ khoá, "其他人还在看", hộp bách khoa bệnh.
    {"suffix": "ask.39.net", "since": 2, "dedupe_lines": True,
     "keep": ['//div[' + _cls("cont_l") + ']'],
     "drop": './/*[@id="sub" or ' + _cls("mation", "txt_nametime", "txt_label", "txt_answer", "sys-box", "doctor_all",
                                         "doc_t_strip", "relevant_ask", "ask_doc_box", "sele_img", "doc_zwen") + ']',
     "title": '//h1',
     "stop": {"追问：", "回复:", "回复：", "回复追问：", "取消 提交", "取消", "提交", "我要咨询", "举报", "向医生提问"},
     "stop_re": r"^(精选回答|医生回答|网友回答)\(\d+\)$"},
    dict(_MW, suffix="a-hospital.com"),
    dict(_MW, suffix="zhongyibaodian.net"),
    # zydcd: trafilatura lấy cả menu và tách mỗi liên kết thành một dòng ("白术\n（二两）").
    {"suffix": "zydcd.com", "since": 2, "min_chars": 1,
     "keep": ['//div[' + _cls("info_txt") + ']'], "title": '//h1'},
    # wujue: trang ngắn (mục từ điển) → trafilatura lấy nhầm cột phải.
    {"suffix": "wujue.com", "since": 2, "min_chars": 1,
     "keep": ['//div[' + _cls("art-text") + ']'], "title": '//div[' + _cls("main_articletitle") + ']'},
    # qihuangzhishu: div.spider chỉ chứa nội dung, không có dòng "下载《...》 电子书打不开？".
    {"suffix": "qihuangzhishu.com", "since": 2, "min_chars": 1, "keep": ['//div[@class="spider"]']},
    # pmphai: <title> chung mọi trang → lấy dòng đầu nội dung làm tiêu đề.
    {"suffix": "pmphai.com", "since": 2,
     "keep": ['//div[' + _cls("gu_det_left_info") + ']'], "stop": {"收藏", "已收藏"}, "title_first_line": True},
    # jbk.39.net: trang tổng quan bệnh / triệu chứng / xét nghiệm. Chỉ lấy khối giới thiệu + thông tin cơ bản,
    # bỏ bác sĩ, thuốc, bình luận dùng thuốc, bài liên quan.
    {"suffix": "jbk.39.net", "since": 2, "first": True, "join_labels": True,
     "keep": ['(//div[' + _cls("list_left") + ']//div[' + _cls("disease_box") + '])[1]',
              '//article[' + _cls("w740") + ']'],
     "drop": './/*[' + _cls("catalog", "ad", "bdsharebuttonbox") + ']',
     "stop": {"基本信息", "在线购药：", "详细"}, "stop_re": r"^您现在的位置", "title": '//h1'},
    # youlai: class có mã băm (textContent--1jgN9) có thể đổi khi trang build lại → so khớp theo tiền tố.
    {"suffix": "youlai.cn", "since": 2, "keep": ['//div[contains(@class,"textContent--")]'], "title": '//h1'},
    # Các kênh tin 39.net (baby, care, food...): "核心提示" (tóm tắt) + thân bài; bỏ thanh bên, bác sĩ, tin liên quan.
    # Tóm tắt trùng thân bài (~85% số trang) tự bị bỏ; ~14% trang có tóm tắt mang nội dung riêng thì được giữ.
    _news("39.net", '//div[' + _cls("art_summary") + ']', '//div[' + _cls("art_content") + ']',
          title='//div[' + _cls("art_left") + ']/h1',
          stop_re=r"^（以上内容仅授权39健康网独家使用|^39健康网\(www\.39\.net\)专稿|^延伸阅读："),
    # ---- tiếng Việt (báo/trang tin): sapo (nếu có) + thân bài; "need" = thân bài, thiếu thì dùng bộ trích chung ----
    _news("suckhoecongdongonline.vn", '//div[' + _cls("article__sapo") + ']', '//div[' + _cls("article__body") + ']'),
    _news("suckhoedoisong.vn", '//*[' + _cls("detail-sapo") + ']', '//div[' + _cls("detail-content") + ']'),
    _news("thanhnien.vn", '//div[' + _cls("detail-sapo") + ']', '//div[' + _cls("detail-cmain") + ']',
          drop='.//*[' + _cls("detail__related", "box-related") + ']'),
    _news("laodong.vn", '//div[' + _cls("chappeau") + ']', '//div[@id="gallery-ctt"]',
          drop='.//*[' + _cls("tin-lien-quan") + ']'),
    _news("phunusuckhoe.giadinhonline.vn", None, '//div[@id="article-body"]'),
    _news("giadinhonline.vn", '//h2[' + _cls("intro") + ']', '//div[@id="explus-editor"]'),
    _news("vinmec.com", None, '//div[@id="main-article"]', drop='.//*[' + _cls("toc", "ez-toc-container") + ']',
          stop={"☰", "Mục lục"}, stop_re=r"^Để đặt lịch khám tại viện"),
    # nhathuoclongchau (80k URL): class Tailwind → bám theo cấu trúc: tóm tắt (khung xám) + khối thân bài
    # (div không class chứa div id=""); bỏ cỡ chữ, lưu ý cuối bài, dược sĩ duyệt, bài/sản phẩm liên quan.
    _news("nhathuoclongchau.com.vn",
          '//div[contains(@class,"max-w-[597px]")]/div[' + _cls("mt-5") + ' and ' + _cls("mb-6") + ']',
          '//div[contains(@class,"max-w-[597px]")]/div[not(@class) and div[@id=""]]'),
    _news("suckhoeviet.org.vn", '//div[' + _cls("article-detail-desc") + ']', '//div[' + _cls("__MASTERCMS_CONTENT") + ']',
          drop='.//*[' + _cls("article-detail-source") + ']'),
    {"suffix": "tamanhhospital.vn", "since": 2, "first": True,
     "keep": ['//div[@id="ftwp-postcontent"]', '//div[' + _cls("post_info") + ']'],
     "drop": './/*[' + _cls("content_insert", "ftwp-in-post", "div_related_bycat") + ']'},
    # baophutho / baothanhhoa (cùng hệ quản trị nội dung): một phần bài bị mã hoá, giải bằng JS → script KHÔNG giải mã;
    # bài đó thành rỗng (thay vì trafilatura lấy nhầm bài khác ở thanh bên). Bài không mã hoá đọc bình thường.
    _news("baophutho.vn", None, '//div[@id="dcontent"]'),
    _news("baothanhhoa.vn", None, '//div[@id="dcontent"]'),
    _news("medlatec.vn", '//div[' + _cls("block-posts-single") + ']/div[' + _cls("shortdescription") + ']',
          '//div[' + _cls("block-posts-single") + ']/div[' + _cls("description") + ']'),
    _news("baohaiphong.vn", None, '(//div[' + _cls("b-maincontent") + '])[1]'),
    _news("baonghean.vn", None, '(//div[' + _cls("b-maincontent") + '])[1]'),
    _news("baodanang.vn", None, '(//div[' + _cls("b-maincontent") + '])[1]'),
    _news("hanoimoi.vn", None, '(//div[' + _cls("b-maincontent") + '])[1]'),
    # vietnamnet: "maincontent" là id ở bố cục thường, là class ở bố cục ảnh tràn màn hình.
    _news("vietnamnet.vn", '//div[' + _cls("content-detail-sapo") + ']',
          '(//div[@id="maincontent" or ' + _cls("maincontent") + '])[1]'),
    _news("tiemchunglongchau.com.vn", None, '//div[@id="content-article"]'),
    _news("hellobacsi.com", None, '//div[' + _cls("unique-content-wrapper") + ']'),
    _news("baocantho.com.vn", None, '//div[@id="newscontents"]'),
    _news("baoangiang.com.vn", None, '//div[@id="newscontents"]'),
]
SITE_RULES = _RULES


def _fallback_extract(doc):
    """Khi bộ trích chung trả về rỗng: lấy vùng có nhiều chữ trong thẻ <p> nhất (kiểu readability)."""
    # không bỏ <form>: nhiều trang (ASP.NET) bọc cả bài viết trong một thẻ form
    for bad in doc.xpath("//script|//style|//noscript|//nav|//footer|//aside"):
        bad.drop_tree()
    score = collections.Counter()
    for p in doc.iter("p"):
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
        return ""
    best = max(score, key=score.get)
    return _block_text(best)


_CHARSET = re.compile(rb'''<meta[^>]+charset\s*=\s*["']?\s*([\w-]+)|<\?xml[^>]+encoding\s*=\s*["']([\w-]+)''', re.I)


_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>")
_JUNK_LINE = re.compile(r"^[A-Za-z0-9+/=]{60,}$|^(\{\w+\}\s*)+$")


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


_VI_CHARS = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđ]", re.I)


def detect_lang(text, host=""):
    """'zh' hoặc 'vi' — so số chữ Hán với số chữ cái có dấu tiếng Việt (không dùng ngưỡng tỷ lệ cố định:
    bài tiếng Trung nhiều số/chữ Latinh từng bị gán nhầm 'vi')."""
    s = text[:3000]
    cjk, vi = len(CJK.findall(s)), len(_VI_CHARS.findall(s))
    if cjk >= 10 and cjk > 2 * vi:
        return "zh"
    if vi >= 5:
        return "vi"
    return "vi" if host.endswith(".vn") or host in VI_HOSTS else "zh" if cjk else "vi"


VI_HOSTS = {"hellobacsi.com", "benhvienvietduc.org", "pmc-ecm-healthblog.beta.pharmacity.io"}


def _parse(src):
    """str → cây lxml của CẢ văn bản. (lxml.html.fromstring đôi khi coi trang là "đoạn HTML" và chỉ trả về
    phần tử đầu tiên — vd baotayninh.vn — làm mất toàn bộ nội dung; document_fromstring luôn đọc cả trang.)"""
    import lxml.html
    try:
        d = lxml.html.document_fromstring(src)
    except Exception:                                      # trang rỗng / không phải HTML
        return lxml.html.document_fromstring("<html><body></body></html>")
    # HTML hỏng kiểu `<html ...></html><head>...<body>...` (vd baotayninh.vn): lxml tạo nhiều gốc <html> liền nhau,
    # nội dung nằm ở gốc sau → lấy gốc nhiều chữ nhất làm văn bản.
    sibs = [d] + [s for s in d.itersiblings() if isinstance(s.tag, str)]
    if len(sibs) > 1:
        best = max(sibs, key=lambda e: len(e.text_content() or ""))
        if best is not d:
            d = lxml.html.document_fromstring(lxml.html.tostring(best, encoding="unicode"))
    return d


def strip_html(doc):
    """HTML gọn (sửa trực tiếp trên doc): bỏ script/style/comment/iframe/svg/link và thuộc tính sự kiện.
    Giữ class/id và các thuộc tính khác. Bộ trích chữ LUÔN chạy trên bản gọn này, nên trích lúc crawl và
    trích lại từ out/raw/ (--reextract) cho kết quả giống hệt nhau."""
    import lxml.html
    d = doc
    for bad in d.xpath("//script|//style|//noscript|//svg|//iframe|//link|//comment()"):
        p = bad.getparent()
        if p is None:
            continue
        if isinstance(bad.tag, str):
            bad.drop_tree()
        else:                                              # comment: giữ lại phần đuôi chữ
            if bad.tail:
                prev = bad.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + bad.tail
                else:
                    p.text = (p.text or "") + bad.tail
            p.remove(bad)
    for e in d.iter():
        if isinstance(e.tag, str):
            for k in [k for k in e.attrib if k.startswith("on") or k in ("srcset", "data-src", "data-original")]:
                del e.attrib[k]
    return lxml.html.tostring(d, encoding="unicode")


def find_rule(host):
    return next((r for r in SITE_RULES if host == r["suffix"] or host.endswith("." + r["suffix"])), None)


def required_xv(host):
    """Bản ghi trích bằng phiên bản nhỏ hơn số này được coi là cũ (--redo-stale / --reextract sẽ làm lại)."""
    r = find_rule(host)
    return max(GENERIC_SINCE, r.get("since", 1) if r else 1)


def extract(html_bytes, host: str = "", header_charset=None):
    """Trả về (title, text, lang). Nhận bytes (HTML gốc) hoặc str (HTML đã giải mã, vd từ --save-raw)."""
    return process_page(html_bytes, host, header_charset)[:3]


def process_page(html_bytes, host="", header_charset=None, want_raw=False):
    """(title, text, lang, raw_html | None). Chạy ở process con."""
    import copy
    import trafilatura
    import lxml.html
    src = _XML_DECL.sub("", decode_html(html_bytes, header_charset), count=1)   # lxml không nhận str có khai báo encoding
    src = strip_html(_parse(src))                          # mọi bước dưới đây dùng bản gọn (= bản lưu ở --save-raw)
    doc = _parse(src)
    title, text = "", ""

    def get_doc():
        return doc

    raw = src if want_raw else None
    rule = find_rule(host)
    if rule:
        try:
            title, text = _site_extract(doc, rule)
        except Exception:
            title, text = "", ""
    site_ok = len(text) >= (rule.get("min_chars", 30) if rule else 30)
    if rule and rule.get("strict") and not site_ok and any(doc.xpath(xp) for xp in rule["keep"]):
        site_ok = True                                     # có vùng nội dung nhưng rỗng (trang chỉ có ảnh) → bài rỗng, không lấy menu
    if not site_ok:                                        # không có quy tắc riêng, hoặc quy tắc không khớp bố cục
        title = ""                                         # tiêu đề theo quy tắc cũng không đáng tin khi bố cục không khớp
        text = _lines(trafilatura.extract(src, include_comments=False, include_tables=True,
                                          favor_recall=True, deduplicate=False))
    if not title:
        try:
            meta = trafilatura.extract_metadata(src)
            title = (meta.title or "") if meta else ""
        except Exception:
            pass
    if len(text) < 50 and not site_ok:                     # bộ trích chung trả về rỗng (bài ngắn theo quy tắc riêng thì giữ nguyên)
        try:
            # HTML hỏng (vd thẻ <html> đóng ngay đầu trang) làm bộ trích chung không thấy nội dung:
            # lấy lại nhánh <body>/<html> chứa nhiều chữ nhất rồi trích lần nữa.
            d = get_doc()
            root = max(d.xpath("//body") + d.xpath("//html") or [d], key=lambda e: len(e.text_content() or ""))
            text = _lines(trafilatura.extract(lxml.html.tostring(root, encoding="unicode"), include_comments=False,
                                              include_tables=True, favor_recall=True, deduplicate=False))
            if len(text) < 50:                             # vẫn rỗng → lấy vùng nhiều chữ nhất
                text = _fallback_extract(copy.deepcopy(root))
            h = d.xpath("//h1")
            if len(h) == 1 and _lines(h[0].text_content()):
                title = _lines(h[0].text_content())
        except Exception:
            pass
    import html as _html
    if "&" in text:                                        # trang mã hoá thực thể 2 lần (vd 120ask: "&#8226;")
        text = _html.unescape(text)
    # Dòng rác chung: chuỗi mã hoá (nội dung bị trang mã hoá, giải bằng JS — vd một nửa số bài baophutho.vn;
    # script KHÔNG giải mã) và chỗ trống của mẫu trang chưa được điền ("{title} {publish}").
    text = "\n".join(l for l in (text or "").split("\n") if not _JUNK_LINE.match(l.strip()))
    title = " ".join(_lines(_html.unescape(title or "")).split("\n"))
    text = _lines(text)
    return title.strip(), text.strip(), detect_lang(text, host), raw


# ---------- tải một URL ----------
BLOCKED = {"http_401", "http_403", "http_451", "blocked_cf", "blocked_tiny"}

# Quy tắc riêng theo tên miền: gộp mọi tên miền con vào MỘT hàng đợi, với số kết nối / nhịp nghỉ / thời gian chờ riêng.
SLOW_RULES = [
    # 39.net (các kênh tin): gọi dày là bật CAPTCHA kéo ghép hình → 1 kết nối, nghỉ lâu.
    # Đã thử 120 lần gọi ở nhịp này mà không bị chặn.
    {"suffix": "39.net", "exclude": {"ask.39.net", "jbk.39.net"}, "group": "39.net (các kênh tin)",
     "workers": 1, "delay": 2.0, "timeout": None, "retries": None},
    # youlai.cn: khoảng một nửa số trang treo tới hết thời gian chờ (không phụ thuộc tốc độ gọi), nửa còn lại trả về tức thì
    # → chờ ngắn, không thử lại ngay; các trang treo để --retry-failed xử lý sau.
    {"suffix": "youlai.cn", "exclude": set(), "group": "youlai.cn",
     "workers": 4, "delay": 0.3, "timeout": 8, "retries": 0},
    {"suffix": "qihuangzhishu.com", "exclude": set(), "group": "qihuangzhishu.com",
     "workers": 1, "delay": 1.0, "timeout": None, "retries": 1},
]
# Trang xác minh người dùng (CAPTCHA). Script KHÔNG giải CAPTCHA: gặp là dừng nhóm tên miền đó.
CAPTCHA = re.compile(rb"AJ-Captcha|slideVerify|\xe6\xbb\x91\xe5\x8a\xa8\xe6\x8b\xbc\xe5\x9b\xbe\xe9\xaa\x8c\xe8\xaf\x81|cf-turnstile|g-recaptcha|h-captcha", re.I)


def rate_group(host):
    """Trả về (tên nhóm, quy tắc hoặc None)."""
    for r in SLOW_RULES:
        suf = r["suffix"]
        if host not in r["exclude"] and (host == suf or host.endswith("." + suf)):
            return r["group"], r
    return host, None


LOGIN_PATH = re.compile(r"login|signin|sign-in|dang-nhap|passport", re.I)
BENIGN = {"http_400", "http_404", "http_410", "not_html", "gone_redirect"}   # trang không còn, không phải bị chặn


HTTP1_HOSTS = set()      # tên miền từng báo lỗi khung HTTP/2 (curl 16/92) → từ đó gọi bằng HTTP/1.1


async def _get(session, engine, url, timeout):
    if engine == "curl":
        kw = {}
        if urlparse(url).netloc.lower() in HTTP1_HOSTS:
            try:
                from curl_cffi.const import CurlHttpVersion
                kw["http_version"] = CurlHttpVersion.V1_1
            except Exception:
                kw["http_version"] = 2                       # giá trị số của HTTP/1.1 trong libcurl
        r = await session.get(url, timeout=timeout, allow_redirects=True, max_redirects=5, **kw)
        return r.status_code, r.headers, r.content, str(r.url)
    import aiohttp
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                           allow_redirects=True, max_redirects=5) as r:
        return r.status, r.headers, await r.read(), str(r.url)


JS_COOKIE = re.compile(rb'''document\.cookie\s*=\s*["']([A-Za-z0-9_\-]+)=([^"';]+)["']''')


def set_cookie(session, engine, url, name, value):
    host = urlparse(url).hostname
    if engine == "curl":
        session.cookies.set(name, value, domain=host, path="/")
    else:
        from yarl import URL
        session.cookie_jar.update_cookies({name: value}, response_url=URL(url))


async def fetch(session, engine, url, timeout, retries):
    last = "err_unknown"
    cookie_tried = False
    h2_tried = False
    attempt = -1
    while attempt < retries:
        attempt += 1
        try:
            status, hdr, body, final = await _get(session, engine, url, timeout)
            if hdr.get("cf-mitigated"):                     # Cloudflare đang đòi thử thách (challenge)
                return None, "blocked_cf"
            if status == 200:
                if len(body) > 8_000_000:
                    return None, "err_too_big"
                if len(body) < 600:
                    # Một số trang (vd laodong.vn) trả về trang nhỏ chỉ để đặt cookie rồi tải lại:
                    #   <script>document.cookie="D1N=...";window.location.reload(true);</script>
                    # Trình duyệt nào cũng tự làm việc này; ta đặt cookie đó rồi yêu cầu lại một lần.
                    m = JS_COOKIE.search(body)
                    if m and not cookie_tried:
                        cookie_tried = True
                        set_cookie(session, engine, url, m.group(1).decode(), m.group(2).decode())
                        attempt -= 1                         # lần gọi lại này không tính là một lần thử lỗi
                        continue
                    return None, "blocked_tiny"
                if len(body) < 8000 and CAPTCHA.search(body):
                    return None, "blocked_captcha"
                ctype = str(hdr.get("content-type", "")).lower()
                if (ctype and not any(x in ctype for x in ("html", "xml", "text/plain"))) \
                        or body[:3] == b"\xff\xd8\xff" or body[:4] in (b"%PDF", b"\x89PNG", b"GIF8"):
                    return None, "not_html"                  # URL trỏ tới ảnh / PDF, không phải bài viết
                o, f = urlparse(url), urlparse(final)
                if o.path.strip("/") and f.path != o.path and (not f.path.strip("/") or LOGIN_PATH.search(f.path)):
                    return None, "gone_redirect"             # bài đã gỡ: trang chuyển về trang chủ / trang đăng nhập
                return body, "ok"
            if status in FINAL_HTTP:
                return None, f"http_{status}"
            last = f"http_{status}"
            if status == 429:
                ra = hdr.get("Retry-After", "")
                await asyncio.sleep(min(60, int(ra)) if str(ra).isdigit() else 10 * (attempt + 1))
        except asyncio.TimeoutError:
            last = "err_timeout"
        except Exception as e:
            code = getattr(e, "code", None)                  # curl_cffi: mã lỗi libcurl (28 = quá thời gian, 56 = đứt kết nối...)
            last = "err_" + type(e).__name__ + (f"_curl{code}" if isinstance(code, int) and code else "")
            if code == 28:
                last = "err_timeout"
            host = urlparse(url).netloc.lower()
            if code in (16, 92) and host not in HTTP1_HOSTS and not h2_tried:
                # máy chủ trả khung HTTP/2 hỏng: chuyển tên miền này sang HTTP/1.1 và gọi lại ngay, không tính là lần thử
                HTTP1_HOSTS.add(host)
                h2_tried = True
                attempt -= 1
                continue
            if code in (16, 92) and not h2_tried:            # tên miền đã ở HTTP/1.1 từ trước mà vẫn lỗi → thử lại một lần
                h2_tried = True
                attempt -= 1
                continue
        if attempt < retries:
            await asyncio.sleep(2 ** attempt + random.random())
    return None, last


def open_session(a):
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}
    engine = a.engine
    if engine == "auto":
        try:
            import curl_cffi  # noqa: F401
            engine = "curl"
        except ImportError:
            engine = "aiohttp"
            print("[CẢNH BÁO] chưa cài curl_cffi → dùng aiohttp, nhiều trang sẽ trả 403. Chạy: pip install curl_cffi")
    print("Engine tải trang:", engine)
    if engine == "curl":
        from curl_cffi.requests import AsyncSession
        sess_cm = AsyncSession(impersonate="chrome", max_clients=a.concurrency,
                               headers={"Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"})
    else:
        import aiohttp
        sess_cm = aiohttp.ClientSession(connector=aiohttp.TCPConnector(
            limit=a.concurrency, limit_per_host=max(a.per_host_big, a.per_host), ttl_dns_cache=600, ssl=False),
            headers=headers, cookie_jar=aiohttp.CookieJar(unsafe=True))
    return engine, sess_cm


class _Parts:
    """Chuỗi file <prefix>-00000.jsonl.gz, mỗi file tối đa n bản ghi; chạy lại luôn mở file mới."""

    def __init__(self, folder, prefix, n_per):
        os.makedirs(folder, exist_ok=True)
        self.folder, self.prefix, self.n_per, self.cnt, self.fh = folder, prefix, n_per, 0, None
        L = len(prefix)
        existing = [f for f in os.listdir(folder) if f.startswith(prefix + "-") and f.endswith(".jsonl.gz")]
        self.k = max([int(f[L + 1:L + 6]) for f in existing], default=-1) + 1

    def write(self, rec):
        if self.fh is None or self.cnt >= self.n_per:
            if self.fh:
                self.fh.close()
            self.fh = gzip.open(os.path.join(self.folder, f"{self.prefix}-{self.k:05d}.jsonl.gz"), "at", encoding="utf-8")
            self.k += 1
            self.cnt = 0
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.cnt += 1

    def flush(self):
        if self.fh:
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


class Writer:
    """part-*.jsonl.gz: bài có nội dung; raw/raw-*.jsonl.gz: HTML gọn (--save-raw);
    progress.tsv: id <tab> trạng thái <tab> phiên bản bộ trích chữ (một dòng mỗi URL đã thử, dòng sau ghi đè dòng trước)."""

    def __init__(self, out, rec_per_part, save_raw=False):
        self.parts = _Parts(out, "part", rec_per_part)
        self.raw = _Parts(os.path.join(out, "raw"), "raw", rec_per_part) if save_raw else None
        self.prog = open(os.path.join(out, "progress.tsv"), "a", encoding="utf-8")

    def write(self, i, status, rec=None, raw=None, xv=EXTRACT_VERSION):
        if rec is not None:
            self.parts.write(rec)
        if raw is not None and self.raw is not None:
            self.raw.write(raw)
        self.prog.write(f"{i}\t{status}\t{xv}\n" if status in ("ok", "empty") else f"{i}\t{status}\n")

    def flush(self):
        self.parts.flush()
        if self.raw:
            self.raw.flush()
        self.prog.flush()

    def close(self):
        self.flush()
        self.parts.close()
        if self.raw:
            self.raw.close()
        self.prog.close()


def read_progress(out):
    """id -> (trạng thái, xv). Dòng sau ghi đè dòng trước; dòng cũ 2 cột có xv = 1."""
    done = {}
    pf = os.path.join(out, "progress.tsv")
    if os.path.exists(pf):
        for line in open(pf, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0].lstrip("-").isdigit():
                done[int(p[0])] = (p[1], int(p[2]) if len(p) > 2 and p[2].isdigit() else 1)
    return done


def make_record(i, u, host, title, text, lang):
    return {"id": i, "url": u, "host": host, "lang": lang, "title": title, "text": text, "xv": EXTRACT_VERSION}


async def main_async(a):
    os.makedirs(a.out, exist_ok=True)
    tbl = pq.read_table(a.input).to_pydict()
    ids, urls = tbl["id"], tbl["url"]

    # shard i/N
    si, sn = (int(x) for x in a.shard.split("/"))
    items = [(i, u) for i, u in zip(ids, urls) if i % sn == si]

    # bỏ qua những id đã xong
    done = read_progress(a.out)
    retry_status = {f"http_{c}" for c in RETRY_HTTP}
    n_stale = 0

    def finished(i, u):
        nonlocal n_stale
        if i not in done:
            return False
        s, xv = done[i]
        if a.retry_failed and (s.startswith(("err_", "blocked_")) or s in retry_status or s == "empty"):
            return False
        if a.redo_stale and s in ("ok", "empty") and xv < required_xv(urlparse(u).netloc.lower()):
            n_stale += 1
            return False
        return True
    items = [(i, u) for i, u in items if not finished(i, u)]
    if a.redo_stale:
        print(f"--redo-stale: {n_stale:,} bài trích bằng quy tắc cũ sẽ được tải lại")
    skip = {h.strip().lower() for h in a.skip_hosts.split(",") if h.strip()}
    if skip:
        n0 = len(items)
        def skipped(u):
            h = urlparse(u).netloc.lower()
            return any(h == x or h.endswith("." + x) or h == "www." + x for x in skip)
        items = [(i, u) for i, u in items if not skipped(u)]
        print(f"Bỏ qua {n0 - len(items):,} URL thuộc: {sorted(skip)}")
    if a.limit:
        random.Random(0).shuffle(items)
        items = items[:a.limit]

    by_host = collections.defaultdict(list)          # nhóm tốc độ -> [(id, url, host)]
    rule = {}
    hosts = set()
    for i, u in items:
        h = urlparse(u).netloc.lower()
        g, r = rate_group(h)
        by_host[g].append((i, u, h)); rule[g] = r; hosts.add(h)
    for g in by_host:
        if rule[g] is not None:                      # nhóm có quy tắc riêng, gồm nhiều tên miền con: trộn đều
            random.Random(0).shuffle(by_host[g])
        by_host[g] = collections.deque(by_host[g])
    total = len(items)
    print(f"Shard {a.shard}: {total:,} URL cần crawl, {len(hosts)} tên miền, đã xong trước đó: {len(done):,}")
    for g, q in by_host.items():
        r = rule[g]
        if r is not None and q:
            print(f"  Quy tắc riêng: {g}: {len(q):,} URL, {r['workers']} kết nối, nghỉ {r['delay']}s "
                  f"→ ít nhất ~{len(q) * r['delay'] / r['workers'] / 3600:.1f} giờ")
    if not total:
        return

    writer = Writer(a.out, a.records_per_part, a.save_raw)
    pool = ProcessPoolExecutor(max_workers=a.cpu)
    loop = asyncio.get_running_loop()
    glob = asyncio.Semaphore(a.concurrency)
    stats = collections.Counter()
    hstats = collections.defaultdict(collections.Counter)
    seen_err = set()
    html_saved = collections.Counter()
    t0 = time.time()
    stop = asyncio.Event()

    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}

    state = collections.defaultdict(lambda: {"consec": 0, "dead": False})

    async def host_worker(grp, q, delay, timeout, retries):
        while q and not stop.is_set():
            if state[grp]["dead"]:
                return
            i, u, host = q.popleft()
            async with glob:
                body, st = await fetch(session, engine, u, timeout, retries)
            if body is None:
                writer.write(i, st); stats[st] += 1; hstats[host][st] += 1
                if st not in BENIGN:
                    # gặp CAPTCHA thì dừng gần như ngay: gọi tiếp chỉ làm trang giữ lệnh chặn lâu hơn
                    state[grp]["consec"] += a.block_threshold // 3 + 1 if st == "blocked_captcha" else 1
                    if state[grp]["consec"] >= a.block_threshold and not state[grp]["dead"]:
                        state[grp]["dead"] = True
                        print(f"[DỪNG TÊN MIỀN] {grp}: lỗi liên tiếp ({st}) → dừng nhóm này trong lần chạy này. "
                              f"Còn {len(q):,} URL chưa thử (không ghi vào progress, lần sau chạy tiếp).", flush=True)
                else:
                    state[grp]["consec"] = 0
            else:
                state[grp]["consec"] = 0
                if html_saved[host] < a.save_html:
                    html_saved[host] += 1
                    os.makedirs(os.path.join(a.out, "html_samples"), exist_ok=True)
                    safe = re.sub(r"[^A-Za-z0-9.\-]", "_", host)
                    with gzip.open(os.path.join(a.out, "html_samples", f"{safe}__{i}.html.gz"), "wb") as hf:
                        hf.write(body)
                try:
                    title, text, lang, raw = await loop.run_in_executor(pool, process_page, body, host, None, a.save_raw)
                except Exception as e:
                    key = type(e).__name__
                    if key not in seen_err:
                        seen_err.add(key)
                        print(f"[CẢNH BÁO] lỗi trích nội dung ({key}): {str(e)[:300]} | url={u}", flush=True)
                    writer.write(i, "err_extract"); stats["err_extract"] += 1; hstats[host]["err_extract"] += 1
                    continue
                else:
                    raw_rec = {"id": i, "url": u, "host": host, "html": raw} if raw is not None else None
                    if len(text) < a.min_chars:
                        writer.write(i, "empty", raw=raw_rec); stats["empty"] += 1; hstats[host]["empty"] += 1
                    else:
                        writer.write(i, "ok", make_record(i, u, host, title, text, lang), raw_rec)
                        stats["ok"] += 1; hstats[host]["ok"] += 1
            await asyncio.sleep(delay * (0.5 + random.random()))

    async def reporter():
        while True:
            await asyncio.sleep(30)
            writer.flush()
            n = sum(stats.values()); el = time.time() - t0
            eta = (total - n) / max(n / el, 1e-9) / 3600
            print(f"[{el/60:6.1f} phút] {n:,}/{total:,} ({100*n/total:.1f}%) "
                  f"{n/el:.1f} trang/s | ok={stats['ok']:,} lỗi={n-stats['ok']:,} | còn ~{eta:.1f} giờ", flush=True)

    engine, sess_cm = open_session(a)
    async with sess_cm as session:
        tasks = []
        for grp, q in by_host.items():
            r = rule[grp]
            if r is None:
                k, d, to, re_ = (a.per_host_big if len(q) > 50_000 else a.per_host), a.delay, a.timeout, a.retries
            else:
                k, d = r["workers"], r["delay"]
                to = r["timeout"] or a.timeout
                re_ = a.retries if r["retries"] is None else r["retries"]
            tasks += [asyncio.create_task(host_worker(grp, q, d, to, re_)) for _ in range(k)]
        rep = asyncio.create_task(reporter())
        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            stop.set()
        finally:
            rep.cancel()
            writer.close()
            pool.shutdown(wait=False, cancel_futures=True)
    print("Xong. Thống kê:", dict(stats))
    with open(os.path.join(a.out, "host_stats.tsv"), "w", encoding="utf-8") as f:
        f.write("host\ttotal\tok\tlỗi_chính\n")
        for h, c in sorted(hstats.items(), key=lambda x: -sum(x[1].values())):
            bad = [(k, v) for k, v in c.most_common() if k != "ok"]
            f.write(f"{h}\t{sum(c.values())}\t{c['ok']}\t{bad[0][0] + ':' + str(bad[0][1]) if bad else ''}\n")
    print("Tên miền có nhiều lỗi nhất (xem đủ ở host_stats.tsv):")
    for h, c in sorted(hstats.items(), key=lambda x: -(sum(x[1].values()) - x[1]["ok"]))[:12]:
        tot = sum(c.values())
        if tot - c["ok"]:
            print(f"  {h:40s} ok {c['ok']}/{tot}  lỗi: {dict((k, v) for k, v in c.items() if k != 'ok')}")



# ====================== TRÍCH LẠI TỪ HTML ĐÃ LƯU (--reextract) ======================
def _process_raw(rec):
    t, x, lg, _ = process_page(rec["html"], rec["host"])
    return rec["id"], rec["url"], rec["host"], t, x, lg


def reextract(a):
    """Đọc out/raw/raw-*.jsonl.gz, chạy lại bộ trích chữ hiện tại, ghi bản ghi mới (xv mới) — không cần mạng.
    --reextract stale  : chỉ bài có xv cũ hơn quy tắc hiện tại của tên miền đó (mặc định)
    --reextract all    : mọi bài
    --reextract a.com,b.vn : mọi bài của các tên miền này"""
    raw_dir = os.path.join(a.out, "raw")
    files = sorted(f for f in os.listdir(raw_dir) if f.startswith("raw-") and f.endswith(".jsonl.gz")) \
        if os.path.isdir(raw_dir) else []
    if not files:
        print(f"Không có HTML đã lưu trong {raw_dir} (cần chạy crawl với --save-raw).")
        return
    mode = a.reextract.strip().lower()
    hosts = set() if mode in ("stale", "all") else {h.strip() for h in mode.split(",") if h.strip()}
    done = read_progress(a.out)

    def want(r):
        if mode == "all":
            return True
        if hosts:
            return any(r["host"] == h or r["host"].endswith("." + h) for h in hosts)
        return done.get(r["id"], ("", 1))[1] < required_xv(r["host"])

    def batches():
        b = []
        for f in files:
            try:
                with gzip.open(os.path.join(raw_dir, f), "rt", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if want(r):
                            b.append(r)
                            if len(b) >= 2000:
                                yield b
                                b = []
            except (EOFError, OSError):                    # file cuối bị cụt khi crawl bị ngắt
                continue
        if b:
            yield b

    writer = Writer(a.out, a.records_per_part)
    stats = collections.Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.cpu) as pool:
        for b in batches():
            for i, u, h, t, x, lg in pool.map(_process_raw, b, chunksize=50):
                if len(x) < a.min_chars:
                    writer.write(i, "empty"); stats["empty"] += 1
                else:
                    writer.write(i, "ok", make_record(i, u, h, t, x, lg)); stats["ok"] += 1
            writer.flush()
            print(f"[{(time.time() - t0) / 60:6.1f} phút] đã trích lại {sum(stats.values()):,} bài", flush=True)
    writer.close()
    print("Xong --reextract:", dict(stats))


# ====================== CHẾ ĐỘ THĂM DÒ (--probe) ======================
# Bước 1: mỗi tên miền gọi NHẸ vài trang (rải đều trong danh sách URL) → bắt lỗi luôn xảy ra:
#         bị chặn, trang xác minh, sai bảng mã, trích chữ rỗng.
# Bước 2: mỗi tên miền gọi một loạt ngắn ở ĐÚNG nhịp sẽ dùng khi chạy thật → bắt lỗi chỉ lộ ra khi gọi dày
#         (CAPTCHA theo tốc độ, quá thời gian) và đo tốc độ thật để ước lượng thời gian crawl.
def _spread(lst, n):
    """n phần tử rải đều trong lst (để chạm nhiều kiểu URL khác nhau của cùng một trang)."""
    if len(lst) <= n:
        return list(lst)
    return [lst[int((j + 0.5) * len(lst) / n)] for j in range(n)]


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


async def probe_async(a):
    os.makedirs(os.path.join(a.out, "html_samples"), exist_ok=True)
    tbl = pq.read_table(a.input).to_pydict()
    by = collections.defaultdict(list)
    for i, u in zip(tbl["id"], tbl["url"]):
        by[urlparse(u).netloc.lower()].append((i, u))
    skip = {h.strip().lower() for h in a.skip_hosts.split(",") if h.strip()}
    only = {h.strip().lower() for h in a.probe_hosts.split(",") if h.strip()}
    def match(h, names):
        return any(h == x or h.endswith("." + x) or h == "www." + x for x in names)
    hosts = [h for h in by if not match(h, skip) and (not only or match(h, only))]
    print(f"Thăm dò {len(hosts)} tên miền: bước 1 = {a.probe} trang/tên miền (gọi nhẹ), "
          f"bước 2 = {a.probe_load} trang/tên miền (nhịp chạy thật)")

    pool = ProcessPoolExecutor(max_workers=a.cpu)
    loop = asyncio.get_running_loop()
    glob = asyncio.Semaphore(a.concurrency)
    pages = collections.defaultdict(list)                 # host -> [dict]
    rng = random.Random(1)

    async def one(session, engine, host, i, u, phase, save, timeout, retries):
        async with glob:
            t = time.time()
            body, st = await fetch(session, engine, u, timeout, retries)
            fetch_sec = time.time() - t                    # chỉ tính thời gian tải, không tính lúc chờ trích chữ
        row = {"host": host, "id": i, "url": u, "phase": phase, "status": st, "fetch_sec": round(fetch_sec, 2),
               "html_bytes": len(body) if body else 0, "text_chars": 0, "title": "", "lang": "", "text": ""}
        if body is not None:
            if save:
                safe = re.sub(r"[^A-Za-z0-9.\-]", "_", host)
                with gzip.open(os.path.join(a.out, "html_samples", f"{safe}__{i}.html.gz"), "wb") as hf:
                    hf.write(body)
            try:
                title, text, lang = await loop.run_in_executor(pool, extract, body, host)
                row.update(title=title, lang=lang, text=text, text_chars=len(text))
                if len(text) < a.min_chars:
                    row["status"] = "empty"
            except Exception as e:
                row["status"] = "err_extract:" + type(e).__name__
        pages[host].append(row)
        return row["status"]

    groups = collections.defaultdict(list)
    rule = {}
    for h in hosts:
        g, r = rate_group(h)
        groups[g].append(h); rule[g] = r

    def settings(g, n_urls):
        r = rule[g]
        if r is None:
            return (a.per_host_big if n_urls > 50_000 else a.per_host), a.delay, a.timeout, a.retries
        return r["workers"], r["delay"], r["timeout"] or a.timeout, a.retries if r["retries"] is None else r["retries"]

    engine, sess_cm = open_session(a)
    t_all = time.time()
    plan = {}                                               # nhóm -> (số kết nối, nghỉ)
    async with sess_cm as session:
        # ---------- bước 1: gọi nhẹ ----------
        async def phase1(g):
            k, d, to, re_ = settings(g, sum(len(by[h]) for h in groups[g]))
            for h in groups[g]:
                for i, u in _spread(by[h], a.probe):
                    await one(session, engine, h, i, u, 1, True, to, min(re_, 1))
                    await asyncio.sleep(max(1.0, d))
        await asyncio.gather(*[phase1(g) for g in groups])
        ok1 = {h: sum(1 for r in pages[h] if r["status"] == "ok") for h in hosts}
        print(f"Bước 1 xong sau {time.time() - t_all:.0f}s: {sum(1 for h in hosts if ok1[h])}/{len(hosts)} tên miền lấy được nội dung")

        # ---------- bước 2: gọi ở nhịp chạy thật ----------
        async def phase2(g):
            live = [h for h in groups[g] if ok1[h] or all(r["status"] in BENIGN for r in pages[h])]
            if not live or a.probe_load <= 0:
                return
            k, d, to, re_ = settings(g, sum(len(by[h]) for h in groups[g]))
            plan[g] = (k, d)
            budget = a.probe_load if rule[g] is None else min(a.probe_load * 3, a.probe_load * len(live))
            per = max(1, budget // len(live))
            q = collections.deque()
            for h in live:
                used = {r["id"] for r in pages[h]}
                rest = [x for x in by[h] if x[0] not in used]
                for i, u in rng.sample(rest, min(per, len(rest))):
                    q.append((h, i, u))
            rng.shuffle(q)
            st = {"consec": 0, "dead": False}

            async def w():
                while q and not st["dead"]:
                    h, i, u = q.popleft()
                    s_ = await one(session, engine, h, i, u, 2, False, to, min(re_, 1))
                    if s_ in ("ok", "empty") or s_ in BENIGN:
                        st["consec"] = 0
                    else:
                        st["consec"] += 3 if s_ == "blocked_captcha" else 1
                        if st["consec"] >= 6:               # đừng gọi tiếp khi trang đã từ chối
                            st["dead"] = True
                    await asyncio.sleep(d * (0.5 + random.random()))
            await asyncio.gather(*[w() for _ in range(k)])
        await asyncio.gather(*[phase2(g) for g in groups])
    pool.shutdown(wait=False, cancel_futures=True)

    # ---------- báo cáo ----------
    import hashlib
    rows = []
    for h in hosts:
        g = rate_group(h)[0]
        p1 = [r for r in pages[h] if r["phase"] == 1]
        p2 = [r for r in pages[h] if r["phase"] == 2]
        allp = pages[h]
        okp = [r for r in allp if r["status"] == "ok"]
        s1 = collections.Counter(r["status"] for r in p1)
        s2 = collections.Counter(r["status"] for r in p2)
        gone = sum(1 for r in allp if r["status"] in BENIGN)
        med_txt = _median([r["text_chars"] for r in okp])
        med_html = _median([r["html_bytes"] for r in okp])
        bad1 = {k: v for k, v in s1.items() if k != "ok" and k not in BENIGN}
        bad2 = {k: v for k, v in s2.items() if k != "ok" and k not in BENIGN}
        dup = max(collections.Counter(hashlib.md5(r["text"].encode()).hexdigest() for r in okp).values(), default=0)
        note = ""
        if not okp:
            if all(k in BLOCKED or k.startswith("blocked_") for k in s1):
                verdict = "BỊ CHẶN"
            elif all(r["status"] in BENIGN for r in allp):
                verdict = "KHÔNG CÒN TRANG"
            elif all(k == "empty" for k in s1):
                verdict = "TRÍCH CHỮ RỖNG"
            else:
                verdict = "LỖI MÁY CHỦ / KẾT NỐI"
            note = json.dumps(dict(s1), ensure_ascii=False)
        elif p2 and sum(bad2.values()) > 0.2 * len(p2):
            verdict, note = "LỖI KHI GỌI DÀY", f"{sum(bad2.values())}/{len(p2)} trang lỗi ở bước 2 {json.dumps(bad2, ensure_ascii=False)}"
        elif bad1 or sum(bad2.values()) > 0.05 * max(1, len(p2)):
            verdict, note = "CHẬP CHỜN", json.dumps({**bad1, **bad2}, ensure_ascii=False)
        elif len(okp) >= 4 and dup >= 0.3 * len(okp):
            verdict, note = "CẦN XEM", f"{dup}/{len(okp)} trang có nội dung giống hệt nhau"
        elif med_txt < 150:
            verdict, note = "CẦN XEM", "chữ trích ra ít"
        else:
            verdict = "OK"
        if gone >= 0.3 * len(allp) and verdict in ("OK", "CHẬP CHỜN", "CẦN XEM"):
            note = (note + " | " if note else "") + f"{gone}/{len(allp)} trang đã gỡ (404 / chuyển về trang chủ)"
        hours = ""
        fs = [r["fetch_sec"] for r in p2 if r["status"] == "ok"]
        if g in plan and fs and verdict in ("OK", "CHẬP CHỜN", "CẦN XEM"):
            k, d = plan[g]
            n_grp = sum(len(by[x]) for x in groups[g])
            hours = f"{n_grp * (sum(fs) / len(fs) + d) / k / 3600:.1f}"   # giờ để crawl hết nhóm trên MỘT máy
        rows.append({"host": h, "url_trong_kho": len(by[h]), "ket_luan": verdict, "ghi_chu": note,
                     "buoc1": f"{s1.get('ok', 0)}/{len(p1)}", "buoc2": f"{s2.get('ok', 0)}/{len(p2)}" if p2 else "",
                     "da_go": gone, "html_trung_vi": med_html, "chu_trung_vi": med_txt, "gio_1_may": hours,
                     "nhom": g if g != h else "", "tieu_de_mau": (okp[0]["title"][:60] if okp else "")})
    rows.sort(key=lambda r: -r["url_trong_kho"])
    cols = list(rows[0].keys())
    with open(os.path.join(a.out, "probe_report.tsv"), "w", encoding="utf-8-sig") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    with gzip.open(os.path.join(a.out, "probe_pages.jsonl.gz"), "wt", encoding="utf-8") as f:
        for h in hosts:
            for r in pages[h]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(r["url_trong_kho"] for r in rows)
    print(f"\nXong sau {(time.time() - t_all) / 60:.1f} phút. Kết quả: {a.out}/probe_report.tsv (mở bằng Excel), "
          f"probe_pages.jsonl.gz (văn bản từng trang), html_samples/ (HTML gốc)\n")
    print(f"{'tên miền':38s} {'URL':>8s}  {'bước 1':>6s} {'bước 2':>7s} {'chữ':>6s}  kết luận")
    for r in rows:
        if r["ket_luan"] != "OK" or r["ghi_chu"]:
            print(f"{r['host'][:38]:38s} {r['url_trong_kho']:8,d}  {r['buoc1']:>6s} {r['buoc2']:>7s} {r['chu_trung_vi']:6d}  "
                  f"{r['ket_luan']}  {r['ghi_chu']}")
    agg = collections.Counter()
    for r in rows:
        agg[r["ket_luan"]] += r["url_trong_kho"]
    print("\nTỷ lệ kho theo kết luận:")
    for k, v in agg.most_common():
        print(f"  {k:28s} {v:10,d} URL  ({100 * v / total:.1f}%)")
    seen, slow = set(), []
    for r in sorted([r for r in rows if r["gio_1_may"]], key=lambda r: -float(r["gio_1_may"])):
        g = rate_group(r["host"])[0]
        if g not in seen:
            seen.add(g); slow.append((g, r["gio_1_may"], plan[g]))
    if slow:
        print("\nƯớc lượng thời gian crawl cho MỘT máy (chia 3 máy thì khoảng 1/3), 5 nhóm lâu nhất:")
        for g, hrs, (k, d) in slow[:5]:
            print(f"  {g:38s} ~{hrs:>6s} giờ   ({k} kết nối, nghỉ {d}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=r"E:\Medical Retrieval\Data\links_corpus.parquet")
    p.add_argument("--out", default="out")
    p.add_argument("--shard", default="0/1", help="i/N: chỉ lấy id % N == i (chia việc cho nhiều máy)")
    p.add_argument("--limit", type=int, default=0, help="chỉ crawl ngẫu nhiên N URL để chạy thử")
    p.add_argument("--concurrency", type=int, default=64, help="tổng số kết nối đồng thời")
    p.add_argument("--per-host", type=int, default=2, help="kết nối đồng thời / tên miền nhỏ")
    p.add_argument("--per-host-big", type=int, default=6, help="kết nối đồng thời / tên miền >50k URL")
    p.add_argument("--delay", type=float, default=0.3, help="nghỉ (giây) giữa 2 request của mỗi worker")
    p.add_argument("--timeout", type=int, default=25)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--records-per-part", type=int, default=20000)
    p.add_argument("--cpu", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)), help="số process trích nội dung")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--probe", type=int, default=0,
                   help="chế độ thăm dò: gọi nhẹ N trang mỗi tên miền rồi in bảng lỗi (không ghi vào progress)")
    p.add_argument("--probe-hosts", default="", help="chỉ thăm dò các tên miền này (cách nhau dấu phẩy)")
    p.add_argument("--slow-scale", type=float, default=1.0,
                   help="nhân nhịp nghỉ của các quy tắc riêng (0.5 = nhanh gấp đôi; chỉ giảm sau khi đã thử bằng --probe)")
    p.add_argument("--probe-load", type=int, default=40,
                   help="thăm dò bước 2: số trang mỗi tên miền gọi ở nhịp chạy thật (0 = bỏ bước này)")
    p.add_argument("--engine", choices=["auto", "curl", "aiohttp"], default="auto",
                   help="curl = curl_cffi (giống trình duyệt, ít bị 403 hơn)")
    p.add_argument("--save-html", type=int, default=3, help="lưu HTML gốc của N trang đầu mỗi tên miền (để kiểm tra bộ trích chữ)")
    p.add_argument("--block-threshold", type=int, default=25,
                   help="dừng một tên miền nếu lỗi từng này URL liên tiếp (bị chặn, máy chủ sập, quá thời gian)")
    p.add_argument("--skip-hosts", default="", help="các tên miền bỏ qua, cách nhau dấu phẩy, vd: zysjonline.com,laodong.vn")
    p.add_argument("--save-raw", action="store_true",
                   help="lưu HTML gọn của mọi trang vào out/raw/ (~11 KB/trang) để sau này --reextract không cần tải lại")
    p.add_argument("--redo-stale", action="store_true",
                   help="tải lại các bài đã trích bằng quy tắc cũ (xv nhỏ hơn quy tắc hiện tại của tên miền đó)")
    p.add_argument("--reextract", default="", metavar="stale|all|host1,host2",
                   help="không tải mạng: trích lại từ out/raw/ (cần đã chạy với --save-raw)")
    a = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for r in SLOW_RULES:
        r["delay"] *= a.slow_scale
    if a.probe and a.out == "out":
        a.out = "out_probe"
    if a.reextract:
        reextract(a)
        return
    try:
        asyncio.run(probe_async(a) if a.probe else main_async(a))
    except KeyboardInterrupt:
        print("\nĐã dừng. Chạy lại cùng lệnh để tiếp tục.")


if __name__ == "__main__":
    main()
