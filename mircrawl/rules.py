"""Quy tắc trích xuất riêng theo tên miền.

Mỗi quy tắc khớp tên miền "suffix" và mọi tên miền con. Các khoá:
  keep      : danh sách XPath vùng nội dung chính. Mặc định lấy MỌI vùng khớp, theo thứ tự trong danh sách.
  first     : True → thử lần lượt từng XPath, dùng XPath đầu tiên cho ra chữ (trang có nhiều bố cục).
  strict    : True → tìm thấy vùng keep nhưng rỗng thì trả về rỗng (không rơi về bộ trích chung, vốn sẽ lấy menu).
  need      : XPath vùng bắt buộc (thường là thân bài). Không có → coi như bố cục không khớp (tránh chỉ lấy sapo).
  drop      : XPath tương đối (.//…) các phần giao diện nằm TRONG vùng keep, bỏ trước khi lấy chữ.
  stop      : các dòng bỏ đi (so khớp nguyên dòng);  stop_re: dòng khớp regex (search) bị bỏ.
  cut_re    : dòng đầu tiên khớp → bỏ dòng đó và mọi dòng sau nó trong vùng (phần đuôi do mẫu trang chèn vào).
  reorder   : regex id của các khối con cần sắp lại theo số (cnkang: jbk11 … jbk0 → jbk0 … jbk11).
  join_labels : ghép dòng nhãn ngắn kết thúc bằng "：" với dòng giá trị ngay sau (trang dạng bảng thông tin).
  dedupe_lines: bỏ dòng giống hệt dòng ngay trước (bỏ qua nhãn ngắn đầu dòng như "补充说明：").
  min_chars : quy tắc cho ra ít hơn số ký tự này → coi như bố cục không khớp, dùng bộ trích chung (mặc định 30).
  title     : XPath tiêu đề (mặc định lấy từ metadata trang);  title_first_line: lấy dòng đầu nội dung.
  date      : danh sách XPath phần tử chứa ngày đăng (đọc @datetime, @content rồi đến chữ; dòng đầu đọc được).
  author    : danh sách XPath phần tử chứa tên tác giả.
  section   : XPath thanh "vị trí trang" (breadcrumb) → chuyên mục.
  keywords  : XPath các thẻ từ khoá (mỗi phần tử một từ khoá).
  qa        : trang hỏi đáp — tách câu hỏi và từng câu trả lời:
              question: [XPath vùng câu hỏi], answer: XPath mỗi câu trả lời,
              a_text / a_author / a_title / a_time: XPath tương đối trong câu trả lời (nội dung, người trả lời,
              chức danh, thời gian). "drop"/"stop"… của quy tắc cũng áp dụng cho câu hỏi và câu trả lời.
              Văn bản = câu hỏi + "---" + câu trả lời 1 + "---" + …; thông tin người trả lời ghi vào extra["answers"].
              Không tìm thấy vùng câu hỏi → dùng keep như trang thường.
Khi không có quy tắc, hoặc quy tắc không khớp → trafilatura (+ phương án dự phòng cho HTML hỏng).
"""


def _cls(*names):
    """Điều kiện XPath: thẻ có một trong các class (so khớp nguyên token, không khớp nhầm "shortdescription")."""
    return " or ".join(f'contains(concat(" ",normalize-space(@class)," ")," {n} ")' for n in names)


def _news(suffix, sapo, body, drop=None, **kw):
    """Quy tắc trang tin: sapo (tuỳ chọn) + thân bài; thân bài là vùng bắt buộc."""
    r = {"suffix": suffix, "keep": ([sapo] if sapo else []) + [body], "need": body}
    if drop:
        r["drop"] = drop
    r.update(kw)
    return r


# MediaWiki (a-hospital.com, zhongyibaodian.net): bỏ hộp điều hướng, mục lục, danh mục, liên kết nguồn;
# cắt các mục đuôi do mẫu trang chèn vào ("中药方专题", "参看"…).
_MW = {
    "keep": ['//div[@id="bodyContent"]'],
    "drop": './/*[' + _cls("navbox", "nav", "hierarchy-breadcrumb", "hierarchy-list", "hierarchy-nav", "msg-table",
                           "catlinks", "toc", "printfooter", "editsection", "mw-editsection", "thumbcaption",
                           "noprint") +
            ' or @id="fromlink" or @id="toc" or @id="jump-to-nav" or @id="contentSub" or @id="siteSub"'
            ' or @id="belownav" or @id="catlinks"]',
    "cut_re": r"^(中药方专题|参看|相关页面|相关条目|更多医学百科条目|附：中药材大全|外部链接)$",
    "stop_re": r"^(中医宝典|A\+医学百科|医学电子书) >",
    "title": '//h1[@id="firstHeading"]',
}

RULES = [
    # ================================================================ tiếng Trung
    # 120ask: câu hỏi + các câu trả lời (gồm cả phần hỏi tiếp / trả lời tiếp nằm ngoài khối nội dung chính của câu
    # trả lời → lấy cả khối b_answerli, bỏ phần giao diện). min_chars 1: trang chỉ trả lời bằng audio → giữ phần
    # ít ỏi đó (không rơi về trafilatura, vốn lấy cả menu); bài quá ngắn sẽ thành "empty".
    {"suffix": "120ask.com", "min_chars": 1, "dedupe_lines": True,
     "qa": {"question": ['//div[' + _cls("b_askbox") + ']'],
            "answer": '//div[contains(@class,"b_answerli") and not(contains(@class,"zhenshiAD"))]',
            "a_author": './/span[' + _cls("b_sp1") + ']/a',
            "a_title": './/span[' + _cls("b_sp1") + ']',
            "a_time": './/span[' + _cls("b_anscont_time") + ']'},
     "keep": ['//div[contains(@class,"b_askbox")]',
              '//div[contains(@class,"b_answerli") and not(contains(@class,"zhenshiAD"))]'],
     "drop": './/*[' + _cls("b_askab1", "b_answerarea", "b_askbot_ad", "tousu", "codeposition", "b_askbox_btn",
                            "b_answertop", "b_anscont_time", "b_ansaddti", "b_docaddti", "b_bico", "img") +
             ' or @id="area" or self::audio]',
     "title": '//h1',
     "date": ['//div[' + _cls("b_askab1") + ']/span'],
     "stop": {"健康咨询描述：", "追问", "附件：点击查看大图", "图片涉及用户隐私，只有用户本人才能看到。"},
     "stop_re": r"^以上是对“.*”这个问题的建议|^查看更多关于.*的医生回答>*$"},
    # ask.39.net: câu hỏi (tiêu đề, mô tả, "补充说明") + các câu trả lời; bỏ thẻ từ khoá, "其他人还在看",
    # hộp bách khoa bệnh (lặp ở mọi câu hỏi cùng thẻ bệnh).
    {"suffix": "ask.39.net", "dedupe_lines": True,
     "qa": {"question": ['//div[' + _cls("ask_cont") + ']'],
            "answer": '//div[' + _cls("sele_all") + ']',
            "a_text": './/p[' + _cls("sele_txt") + ']',
            "a_author": './/span[' + _cls("doc_name") + ']',
            "a_title": './/span[' + _cls("doc_yshi") + ']',
            "a_time": './/p[' + _cls("doc_time") + ']'},
     "keep": ['//div[' + _cls("cont_l") + ']'],
     "drop": './/*[@id="sub" or ' + _cls("mation", "txt_nametime", "txt_label", "txt_answer", "sys-box", "doctor_all",
                                         "doc_t_strip", "relevant_ask", "ask_doc_box", "sele_img", "doc_zwen") + ']',
     "title": '//h1',
     "date": ['//p[' + _cls("txt_nametime") + ']/span'],
     "section": '//div[@id="sub"]',
     "keywords": '//p[' + _cls("txt_label") + ']//a',
     "stop": {"追问：", "回复:", "回复：", "回复追问：", "取消 提交", "取消", "提交", "我要咨询", "举报", "向医生提问"},
     "stop_re": r"^(精选回答|医生回答|网友回答)\(\d+\)$"},
    # cnkang: nhiều bố cục qua các năm → thử lần lượt. Bố cục cũ (#endText): đoạn văn nằm trong div#jbk0..jbkN
    # nhưng HTML xếp NGƯỢC → sắp lại; bỏ "核心提示" (tóm tắt bị cắt, lặp nội dung).
    # strict: vùng nội dung có nhưng rỗng (bài chỉ có ảnh) → bài rỗng, không lấy menu.
    {"suffix": "cnkang.com", "first": True, "min_chars": 1, "strict": True,
     "keep": ['//div[' + _cls("detailc") + ']', '//div[@id="endText"]',
              '//td[' + _cls("style3") + ']', '//div[' + _cls("conmain") + ']'],
     "drop": './/div[' + _cls("article_intro") + '] | .//div[@id="jkpl"]',
     "reorder": r"jbk(\d+)",
     "title": ['//h1[@id="h1title"]', '//h1[' + _cls("detaila") + ']'],
     "date": ['//div[' + _cls("detailb1") + ']', '//div[' + _cls("detailb") + ']', '//div[@id="titleDate"]',
              '//td[' + _cls("info") + ']'],
     "stop_re": r"^(\[\d+\]\s*)+(下一页)?$|^(上一页|下一页)$|^温馨提示：以上资料仅供参考"},
    dict(_MW, suffix="a-hospital.com"),
    dict(_MW, suffix="zhongyibaodian.net"),
    # zydcd: trafilatura lấy cả menu và tách mỗi liên kết thành một dòng ("白术\n（二两）").
    {"suffix": "zydcd.com", "min_chars": 1, "keep": ['//div[' + _cls("info_txt") + ']'], "title": '//h1'},
    # wujue: trang ngắn (mục từ điển) → trafilatura lấy nhầm cột phải.
    {"suffix": "wujue.com", "min_chars": 1,
     "keep": ['//div[' + _cls("art-text") + ']'], "title": '//div[' + _cls("main_articletitle") + ']',
     "date": ['//div[' + _cls("main_articlelaiyuan") + ']']},
    # qihuangzhishu: div.spider chỉ chứa nội dung, không có dòng "下载《…》 电子书打不开？".
    {"suffix": "qihuangzhishu.com", "min_chars": 1, "keep": ['//div[@class="spider"]']},
    # pmphai: <title> chung mọi trang → lấy dòng đầu nội dung làm tiêu đề.
    {"suffix": "pmphai.com",
     "keep": ['//div[' + _cls("gu_det_left_info") + ']'], "stop": {"收藏", "已收藏"}, "title_first_line": True},
    # jbk.39.net: trang tổng quan bệnh / triệu chứng / xét nghiệm. Chỉ lấy khối giới thiệu + thông tin cơ bản,
    # bỏ bác sĩ, thuốc, bình luận dùng thuốc, bài liên quan.
    {"suffix": "jbk.39.net", "first": True, "join_labels": True,
     "keep": ['(//div[' + _cls("list_left") + ']//div[' + _cls("disease_box") + '])[1]',
              '//article[' + _cls("w740") + ']'],
     "drop": './/*[' + _cls("catalog", "ad", "bdsharebuttonbox") + ']',
     "stop": {"基本信息", "在线购药：", "详细"}, "stop_re": r"^您现在的位置|^详细>>", "title": '//h1'},
    # youlai: class có mã băm (textContent--1jgN9) có thể đổi khi trang build lại → so khớp theo tiền tố.
    {"suffix": "youlai.cn", "keep": ['//div[contains(@class,"textContent--")]'], "title": '//h1',
     "date": ['//span[contains(@class,"time--")]']},
    # Các kênh tin 39.net (baby, care, food…): "核心提示" (tóm tắt) + thân bài; bỏ thanh bên, bác sĩ, tin liên quan.
    # Tóm tắt trùng thân bài (~85% số trang) tự bị bỏ; ~14% trang có tóm tắt mang nội dung riêng thì được giữ.
    _news("39.net", '//div[' + _cls("art_summary") + ']', '//div[' + _cls("art_content") + ']',
          title='//div[' + _cls("art_left") + ']/h1',
          date=['//div[' + _cls("art_info") + ']//em', '//div[' + _cls("art_info") + ']'],
          stop_re=r"^（以上内容仅授权39健康网独家使用|^39健康网\(www\.39\.net\)专稿|^延伸阅读："),
    # ================================================================ tiếng Việt: sapo (nếu có) + thân bài
    _news("suckhoecongdongonline.vn", '//div[' + _cls("article__sapo") + ']', '//div[' + _cls("article__body") + ']',
          date=['//div[' + _cls("article__meta") + ']//time', '//div[' + _cls("article__meta") + ']'],
          stop_re=r"^Bài liên quan$"),
    _news("suckhoedoisong.vn", '//*[' + _cls("detail-sapo") + ']', '//div[' + _cls("detail-content") + ']',
          stop_re=r"^(Xem thêm video đang được quan tâm:|Mời quý vị và các bạn đón xem video dưới đây:)$"),
    _news("thanhnien.vn", '//div[' + _cls("detail-sapo") + ']', '//div[' + _cls("detail-cmain") + ']',
          drop='.//*[' + _cls("detail__related", "box-related") + ']', stop_re=r"^tin liên quan$"),
    _news("laodong.vn", '//div[' + _cls("chappeau") + ']', '//div[@id="gallery-ctt"]',
          drop='.//*[' + _cls("tin-lien-quan") + ']',
          stop_re=r"^(Bình luận:|Bạn nghĩ gì về nội dung này\?|Bạn cần đăng nhập để gửi bình luận\.|Gửi bình luận)$"),
    _news("phunusuckhoe.giadinhonline.vn", None, '//div[@id="article-body"]'),
    _news("giadinhonline.vn", '//h2[' + _cls("intro") + ']', '//div[@id="explus-editor"]'),
    _news("vinmec.com", None, '//div[@id="main-article"]', drop='.//*[' + _cls("toc", "ez-toc-container") + ']',
          stop={"☰", "Mục lục"}, stop_re=r"^Để đặt lịch khám tại viện"),
    # nhathuoclongchau (80k URL): class Tailwind → bám theo cấu trúc: tóm tắt (khung xám) + khối thân bài
    # (div không class chứa div id=""); ngày + tác giả ở hàng ngay dưới tiêu đề.
    _news("nhathuoclongchau.com.vn",
          '//div[contains(@class,"max-w-[597px]")]/div[' + _cls("mt-5") + ' and ' + _cls("mb-6") + ']',
          '//div[contains(@class,"max-w-[597px]")]/div[not(@class) and div[@id=""]]',
          date=['(//div[contains(@class,"max-w-[597px]")]/div[contains(@class,"justify-between")])[1]//p'],
          author=['(//div[contains(@class,"max-w-[597px]")]/div[contains(@class,"justify-between")])[1]/div[1]/p[1]'],
          stop_re=r"^Xem thêm:$"),
    _news("suckhoeviet.org.vn", '//div[' + _cls("article-detail-desc") + ']', '//div[' + _cls("__MASTERCMS_CONTENT") + ']',
          drop='.//*[' + _cls("article-detail-source") + ']'),
    {"suffix": "tamanhhospital.vn", "first": True,
     "keep": ['//div[@id="ftwp-postcontent"]', '//div[' + _cls("post_info") + ']'],
     "drop": './/*[' + _cls("content_insert", "ftwp-in-post", "div_related_bycat") + ']'},
    # baophutho / baothanhhoa (cùng hệ quản trị): một phần bài bị mã hoá, giải bằng JS → script KHÔNG giải mã;
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
    # hellobacsi: bỏ nhãn loại bài ("Góc nhìn" / "Hướng dẫn" + câu giải thích) chèn ở đầu bài.
    _news("hellobacsi.com", None, '//div[' + _cls("unique-content-wrapper") + ']',
          stop={"Góc nhìn", "Hướng dẫn"}, stop_re=r"^Đây là bài viết (có thể hiện|hướng dẫn|cung cấp|tổng hợp)"),
    _news("baocantho.com.vn", None, '//div[@id="newscontents"]'),
    _news("baoangiang.com.vn", None, '//div[@id="newscontents"]'),
]


def find_rule(host):
    return next((r for r in RULES if host == r["suffix"] or host.endswith("." + r["suffix"])), None)
