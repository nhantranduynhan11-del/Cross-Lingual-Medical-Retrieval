"""Dịch 1.200 câu hỏi vi → en → zh bằng LLM mở (<= 14B, phát hành trước 1/6/2026) và so sánh model / phiên bản prompt.

Prompt "t1" (15/10) = prompt v2 của dự án cũ (bảng khái niệm vi→en→zh trước, dịch en rồi zh, 2 ví dụ tự soạn).
Prompt "t2" (16/10, mặc định translate.prompt) sửa theo lỗi thật của t1 khi so Qwen2.5-14B / Qwen3-8B: quy tắc trong
system (thuật ngữ, triệu chứng không nâng thành chẩn đoán, đủ ý / không thêm / giữ phủ định, đúng người, chữ số, chữ viết
tắt, biệt dược), gợi ý thuật ngữ đã kiểm chứng theo từng câu (configs/med_glossary_vi.tsv), 3 ví dụ mới, khái niệm dạng
chuỗi "vi = en = zh". Kết quả: qtrans/<model> (t1) hoặc qtrans/<model>-<prompt> (vd qwen3_14b-t2).
Sinh tất định (greedy); câu JSON hỏng được sinh lại có lấy mẫu.

  python -m mirret.translate generate --model qwen3_14b [--prompt t2] [--shard 0/2] [--limit 20] [--batch 8]
  python -m mirret.translate import-old --qexp <thư mục qexp cũ> [--name old_qwen25_14b]   # dùng bản dịch cũ (v2)
  python -m mirret.translate compare --a qwen25_14b-t2 --b qwen3_14b-t2 [--n 100]   # chỉ số tự động + phiếu chấm mù
        (phiếu mới bỏ các câu đã có trong phiếu các lần trước; mỗi lần so một thư mục compare_* mới)
  python -m mirret.translate score --sheet <phiếu đã chấm.csv> [--sheet …]     # kết quả chấm của người
  python -m mirret.translate build [--use qwen3_14b-t2]                       # → qtrans/queries_{vi,en,zh}.parquet

Model: tên ngắn trong translate.models (vd qwen25_14b, qwen3_14b) hoặc id Hugging Face; compare/build nhận tên thư mục
trong qtrans/ (vd qwen25_14b = t1, qwen25_14b-t2).
"""
import argparse
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

PROMPT_VERSION = "t1"                                   # bản cũ, giữ để đọc lại/so sánh; bản mặc định: translate.prompt
SYSTEM = ("You are a medical terminology expert fluent in Vietnamese, Chinese and English. You translate search "
          "queries for a multilingual medical document retrieval system. You never answer the question.")
PROMPT = """Câu hỏi y khoa (tiếng Việt): "{query}"

Dịch câu hỏi để tìm tài liệu y khoa, làm lần lượt:
1. "concepts": liệt kê các khái niệm y khoa CÓ TRONG câu hỏi (bệnh, triệu chứng, thuốc, xét nghiệm, chỉ số, thủ thuật),
   mỗi khái niệm ghi thuật ngữ chuẩn bằng 3 ngôn ngữ {{"vi", "en", "zh"}} (zh: tiếng Trung giản thể, thuật ngữ y khoa
   chuẩn). Hiểu đúng nghĩa thông tục tiếng Việt trước khi dịch. Không ghi tên người, địa danh, bệnh viện. Tối đa 6.
2. "en": dịch TRỌN câu hỏi sang tiếng Anh, dùng đúng thuật ngữ ở bước 1.
3. "zh": dịch TRỌN câu hỏi sang tiếng Trung giản thể, tự nhiên như người Trung Quốc hỏi bác sĩ, dùng đúng thuật ngữ ở bước 1.
Quy tắc bắt buộc: dịch đủ mọi ý (câu hỏi dài nhiều ý thì dịch đủ, không tóm tắt); giữ đúng AI mắc bệnh (người hỏi, con,
vợ, bạn gái…); giữ nguyên mọi con số, đơn vị, tuổi, thời gian, tần suất; không thêm, bớt hay đổi triệu chứng / bệnh /
thuốc; tên thương mại của thuốc giữ nguyên chữ gốc; không trả lời câu hỏi, không đưa chẩn đoán.
Chỉ trả về MỘT đối tượng JSON, không giải thích."""

# Ví dụ tự soạn (từ mir/expand.py; đã kiểm không trùng câu nào trong data/query.parquet)
EXAMPLES = [
    ("Bị đau thắt ngực khi leo cầu thang, có phải bệnh mạch vành không?",
     {"concepts": [{"vi": "đau thắt ngực", "en": "angina pectoris", "zh": "心绞痛"},
                   {"vi": "bệnh mạch vành", "en": "coronary artery disease", "zh": "冠心病"}],
      "en": "Is angina when climbing stairs a sign of coronary artery disease?",
      "zh": "爬楼梯时出现心绞痛，是冠心病吗？"}),
    ("Con tôi 3 tuổi hay cáu gắt, ăn ít, sụt 1kg trong 2 tháng thì có sao không?",
     {"concepts": [{"vi": "cáu gắt", "en": "irritability", "zh": "易怒"},
                   {"vi": "chán ăn", "en": "poor appetite", "zh": "食欲不振"},
                   {"vi": "sụt cân", "en": "weight loss", "zh": "体重下降"}],
      "en": "My 3-year-old child is often irritable, eats little and has lost 1 kg in 2 months. Is something wrong?",
      "zh": "我孩子3岁，经常烦躁易怒，吃得少，2个月体重下降了1公斤，有问题吗？"}),
]

# ---------------------------------------------------------------- prompt "t2" (16/10/2026)
# Sửa theo lỗi thật của t1 (so Qwen2.5-14B / Qwen3-8B trên 1.200 câu: 367 khái niệm hai model hiểu khác nhau + chấm mù
# 100 câu): sai thuật ngữ thông tục (xương mác → xương đòn, viêm màng bồ đào → viêm thượng củng mạc, hiếm muộn → "rare"),
# nâng triệu chứng thành chẩn đoán (đau dạ dày → gastritis), bịa / bỏ ý, đảo nghĩa phủ định, nhầm người mắc bệnh, bỏ chữ
# viết tắt, đổi biệt dược sang thuốc khác, JSON hỏng ở bảng khái niệm.
# Khác t1: quy tắc nằm MỘT lần trong system (lượt ví dụ chỉ có câu hỏi → prompt ngắn, ít tốn bộ nhớ GPU); gợi ý thuật ngữ
# đã kiểm chứng (configs/med_glossary_vi.tsv) chèn theo từng câu; "concepts" là chuỗi "vi = en = zh" (ít dấu ngoặc).
SYSTEM_T2 = """You are a medical terminology expert fluent in Vietnamese, Chinese and English. You translate Vietnamese \
patient questions for a multilingual medical document search engine. You never answer the question.

Trả về MỘT đối tượng JSON với đúng 3 khoá theo thứ tự:
"concepts": danh sách chuỗi "tiếng Việt = English = 中文" cho các khái niệm y khoa CÓ TRONG câu hỏi (bệnh, triệu chứng,
  thuốc, xét nghiệm, chỉ số, thủ thuật, bộ phận cơ thể), tối đa 6. Hiểu đúng nghĩa thông tục tiếng Việt trước khi dịch.
"en": bản dịch TRỌN câu hỏi sang tiếng Anh.
"zh": bản dịch TRỌN câu hỏi sang tiếng Trung giản thể, tự nhiên như bệnh nhân Trung Quốc hỏi bác sĩ, dùng thuật ngữ y khoa
  chuẩn của Trung Quốc đại lục.

Quy tắc (vi phạm = bản dịch sai):
1. Đúng thuật ngữ: không bao giờ thay một bệnh / bộ phận / thuốc / xét nghiệm bằng một cái khác (xương mác là fibula, không
   phải clavicle; viêm màng bồ đào là uveitis, không phải episcleritis). Không chắc một từ thông tục thì dịch sát nghĩa
   mô tả, KHÔNG đoán ra một bệnh cụ thể. Có "Gợi ý thuật ngữ" thì dùng đúng thuật ngữ đó.
2. Triệu chứng giữ là triệu chứng, đúng mức độ: không tự nâng thành chẩn đoán (đau dạ dày = stomach pain, không phải
   gastritis; chậm kinh = late period, không phải amenorrhea; tim đập nhanh = rapid heartbeat).
3. Đủ ý, không thêm: dịch mọi câu hỏi và mọi chi tiết (tuổi, thời gian, liều, tần suất, tiền sử, thuốc đang dùng, kết quả
   xét nghiệm); không thêm triệu chứng, bệnh hay chi tiết không có; giữ đúng chiều nghĩa và phủ định ("giúp trẻ không
   còn tè dầm" = help the child stop bedwetting).
4. Đúng người: "em, mình, cháu, tôi, tớ" = người hỏi; "bé, con, cu, bé nhà em" = con của người hỏi; "người tôi" = cơ thể
   tôi; "mẹ em, chồng em…" = người thân đó. Giữ đúng ai mắc bệnh.
5. Con số và đơn vị giữ nguyên bằng CHỮ SỐ (2 tháng = 2 months = 2个月); dấu thập phân phẩy của tiếng Việt đổi thành
   dấu chấm (6,5 = 6.5).
6. Chữ viết tắt (HPV, HBsAg, AMH, IVF, TSH, CRP…): giữ nguyên; en có thể thêm tên đầy đủ trong ngoặc; zh dùng tên tiếng
   Trung chuẩn kèm chữ viết tắt trong ngoặc, vd 甲胎蛋白（AFP）.
7. Tên thuốc biệt dược giữ nguyên chữ; chỉ thêm hoạt chất trong ngoặc khi chắc chắn; không bao giờ đổi sang thuốc khác.
   Tên người, bệnh viện, địa danh giữ nguyên.
8. Không trả lời câu hỏi, không chẩn đoán, không khuyên. Chỉ trả về JSON, không giải thích."""

USER_T2 = 'Câu hỏi: "{query}"{hints}'

# Ví dụ tự soạn (đã kiểm không trùng câu nào trong data/query.parquet). Gợi ý thuật ngữ của ví dụ cũng sinh từ bảng
# thuật ngữ như câu thật, nên bản dịch mẫu phải khớp gợi ý.
EXAMPLES_T2 = [
    ("Mẹ em 62 tuổi hay bị hồi hộp, hoa mắt mỗi khi đứng dậy, đang uống Concor 2,5mg mỗi sáng thì có sao không ạ?",
     {"concepts": ["hồi hộp = palpitations = 心悸",
                   "hoa mắt khi đứng dậy = blurred vision on standing up = 起身时眼花",
                   "Concor = Concor (bisoprolol) = Concor（比索洛尔）"],
      "en": "My 62-year-old mother often has palpitations and blurred vision whenever she stands up. She takes Concor "
            "(bisoprolol) 2.5 mg every morning. Is this serious?",
      "zh": "我妈妈62岁，经常心悸，每次站起来都会眼花，每天早上服用Concor（比索洛尔）2.5mg，这样有问题吗？"}),
    ("Kết quả xét nghiệm của em: AFP 12 ng/ml, CEA 6,2 ng/ml, ALT 65 U/L. Các chỉ số này có cao không, có phải bị ung "
     "thư gan không?",
     {"concepts": ["AFP = alpha-fetoprotein (AFP) = 甲胎蛋白（AFP）",
                   "CEA = carcinoembryonic antigen (CEA) = 癌胚抗原（CEA）",
                   "ALT = alanine aminotransferase (ALT) = 谷丙转氨酶（ALT）",
                   "ung thư gan = liver cancer = 肝癌"],
      "en": "My test results: AFP (alpha-fetoprotein) 12 ng/mL, CEA (carcinoembryonic antigen) 6.2 ng/mL, ALT 65 U/L. "
            "Are these levels high? Do I have liver cancer?",
      "zh": "我的检查结果：甲胎蛋白（AFP）12 ng/mL，癌胚抗原（CEA）6.2 ng/mL，谷丙转氨酶（ALT）65 U/L。这些指标高吗？"
            "是不是得了肝癌？"}),
    ("Bé nhà em 4 tuổi vẫn chưa hết tè dầm, đêm nào cũng tè 1-2 lần. Có cần cho bé đi khám không và làm sao để bé "
     "không tè dầm nữa?",
     {"concepts": ["tè dầm = bedwetting (nocturnal enuresis) = 尿床（遗尿）"],
      "en": "My 4-year-old child still wets the bed 1-2 times every night. Should my child see a doctor, and how can I "
            "help my child stop bedwetting?",
      "zh": "我家孩子4岁了还在尿床，每晚尿1-2次。需要带孩子去看医生吗？怎样才能让孩子不再尿床？"}),
]

# ---------------------------------------------------------------- bảng thuật ngữ (gợi ý cho t2)
_GLOSSARY = {}


def load_glossary(path):
    """configs/med_glossary_vi.tsv → [(các alias đã biên dịch regex, en, zh, ghi chú, alias hiển thị)]."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path      # đi kèm mã (mir2-code)
    if path in _GLOSSARY:
        return _GLOSSARY[path]
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cols = (line.split("\t") + ["", "", "", ""])[:4]
            pats = []
            for al in (x.strip() for x in cols[0].split("|")):
                if not al:
                    continue
                cs = al.startswith("=")
                al = unicodedata.normalize("NFC", al.lstrip("="))
                pats.append(re.compile(r"(?<!\w)" + re.escape(al) + r"(?!\w)", 0 if cs else re.I))
            rows.append((pats, cols[1].strip(), cols[2].strip(), cols[3].strip(), cols[0].split("|")[0].strip("= ")))
    _GLOSSARY[path] = rows
    return rows


def glossary_hits(query, glossary, k=10):
    """Các mục thuật ngữ khớp câu hỏi: cụm dài trước, không chồng lấn; tối đa k mục, theo thứ tự xuất hiện."""
    q = unicodedata.normalize("NFC", query or "")
    found = []
    for i, (pats, *_rest) in enumerate(glossary):
        for p in pats:
            for m in p.finditer(q):
                found.append((m.start(), m.end(), i, m.group(0)))
    found.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    taken, used, out = [], set(), []
    for s, e, i, txt in found:
        if i in used or any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        used.add(i)
        out.append((s, i, txt))
    return [(txt, glossary[i]) for s, i, txt in sorted(out)[:k]]


def hints_text(hits):
    if not hits:
        return ""
    lines = [f'- "{txt}" → en: {g[1]}; zh: {g[2]}' + (f" ({g[3]})" if g[3] else "") for txt, g in hits]
    return "\nGợi ý thuật ngữ đã kiểm chứng (dùng khi từ đó mang nghĩa y khoa trong câu):\n" + "\n".join(lines)


def prompt_parts(cfg, version):
    """→ (system, hàm tạo lượt user từ câu hỏi, các lượt ví dụ [(user, assistant)])."""
    tc = cfg["translate"]
    if version == "t1":
        user = lambda q: PROMPT.format(query=q)
        return SYSTEM, user, [(user(q), json.dumps(a, ensure_ascii=False)) for q, a in EXAMPLES]
    if version != "t2":
        raise ValueError(f"không có prompt {version}")
    gl = load_glossary(tc.get("glossary") or "configs/med_glossary_vi.tsv")
    k = int(tc.get("glossary_max", 10))
    user = lambda q: USER_T2.format(query=q, hints=hints_text(glossary_hits(q, gl, k)))
    return SYSTEM_T2, user, [(user(q), json.dumps(a, ensure_ascii=False)) for q, a in EXAMPLES_T2]


def run_key(cfg, model, prompt=None):
    """Thư mục kết quả qtrans/<khoá>: t1 giữ tên cũ (vd qwen25_14b); bản khác thêm hậu tố (qwen25_14b-t2)."""
    prompt = prompt or cfg["translate"].get("prompt", "t1")
    k = model_key(cfg, model)
    return k if prompt == "t1" else f"{k}-{prompt}"


def resolve_key(cfg, name):
    """Tên dùng cho compare/build: thư mục có sẵn trong qtrans/ (vd 'qwen25_14b', 'qwen3_14b-t2') hoặc tên model
    (→ khoá theo translate.prompt hiện tại)."""
    if (cfg.path("qtrans") / name).is_dir():
        return name
    return run_key(cfg, name)

_HAN = re.compile(r"[一-鿿]")
_VI = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def model_id(cfg, key):
    return cfg["translate"]["models"].get(key, key)


def model_key(cfg, key_or_id):
    for k, v in cfg["translate"]["models"].items():
        if key_or_id in (k, v):
            return k
    name = re.split(r"[\/]", key_or_id.rstrip("/\\"))[-1]       # "Qwen/Qwen3-8B" → "qwen3_8b"
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


# ---------------------------------------------------------------- phân tích câu trả lời
def parse(text):
    s = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict) or not (d.get("en") and d.get("zh")):
        # JSON hỏng (thường do dấu ngoặc của "concepts", vd `]}, "en": …`): lấy chuỗi "en"/"zh" CUỐI CÙNG trong văn bản
        # (các "en"/"zh" trước đó nằm trong bảng khái niệm)
        d = {"concepts": (d or {}).get("concepts") if isinstance(d, dict) else []}
        for k in ("en", "zh"):
            hits = re.findall(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % k, s)
            if hits:
                try:
                    d[k] = json.loads(f'"{hits[-1]}"')
                except json.JSONDecodeError:
                    pass
        if not (d.get("en") and d.get("zh")):
            return None

    def one(x):
        if isinstance(x, list):
            x = " ".join(str(v) for v in x if isinstance(v, (str, int, float)))
        return str(x).strip() if isinstance(x, (str, int, float)) else ""
    concepts = []
    for c in d.get("concepts") or []:
        if isinstance(c, str) and "=" in c:                # t2: "vi = en = zh"
            parts = [x.strip() for x in c.split("=")] + ["", ""]
            c = {"vi": parts[0], "en": parts[1], "zh": parts[2]}
        if isinstance(c, dict):
            concepts.append(c)
    concepts = concepts[:6]
    return {"concepts": concepts, "en": one(d.get("en")), "zh": one(d.get("zh"))}


def script_ok(text, lang):
    """zh phải chủ yếu là chữ Hán; en không có chữ Hán, không còn chữ tiếng Việt có dấu."""
    if not text:
        return False
    han = len(_HAN.findall(text))
    if lang == "zh":
        # so với số TỪ Latin (không phải số ký tự): "HBV-DNA 231 IU/mL" chỉ là vài từ; câu tiếng Anh vẫn bị loại
        return han >= 2 and han >= 0.3 * (han + len(re.findall(r"[A-Za-z0-9]+", text)))
    return han == 0 and not _VI.search(text)


def numbers_kept(src, dst):
    """Mọi con số trong câu gốc có mặt trong câu dịch (so theo chữ số; "1,5" ~ "1.5")."""
    norm = lambda x: x.replace(",", ".")
    have = {norm(x) for x in _NUM.findall(dst or "")}
    need = {norm(x) for x in _NUM.findall(src or "")}
    return need <= have


def copies_example(rec):
    """Bản dịch chép lại câu trả lời mẫu thay vì dịch câu hỏi (gặp ở Qwen3-14B với câu mở đầu giống ví dụ, vd "Bé nhà
    em…"): en hoặc zh gần trùng (> 0,8) bản dịch của một ví dụ trong khi câu hỏi khác ví dụ đó."""
    import difflib
    q, en, zh = rec.get("query", ""), rec.get("en", ""), rec.get("zh", "")
    for eq, ea in EXAMPLES + EXAMPLES_T2:
        if difflib.SequenceMatcher(None, q, eq).ratio() > 0.8:
            continue
        if (en and difflib.SequenceMatcher(None, en, ea["en"]).ratio() > 0.8) or                 (zh and difflib.SequenceMatcher(None, zh, ea["zh"]).ratio() > 0.8):
            return True
    return False


# ---------------------------------------------------------------- bản ghi
def records_dir(cfg, key):
    return cfg.path("qtrans") / key


def load_records(cfg, key, ok_only=True):
    """{qid: bản ghi} của một model (mọi file records*.jsonl); bản hợp lệ được ưu tiên, rồi bản sau cùng."""
    done = {}
    for f in sorted(records_dir(cfg, key).glob("records*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ok") and r.get("raw"):              # đọc lại bằng parse() hiện tại (có thể cứu được JSON hỏng)
                d = parse(r["raw"])
                if d and d["en"] and d["zh"]:
                    r.update(d, ok=True, reparsed=True)
            if r.get("ok") and copies_example(r):            # chép ví dụ mẫu → coi như hỏng (generate sẽ dịch lại)
                r.update(ok=False, copied_example=True)
            if ok_only and not r.get("ok"):
                continue
            q = int(r["id"])
            if not (q in done and done[q].get("ok") and not r.get("ok")):
                done[q] = r
    return done


def vi_queries(cfg):
    t = pq.read_table(cfg.path("queries")).to_pydict()
    return [int(x) for x in t["id"]], list(t["query"])


# ---------------------------------------------------------------- sinh (GPU)
def load_llm(mid, quant=None, log=print):
    """Tokenizer (đệm trái) + model sinh; quant '4bit' | '8bit' (bitsandbytes, chỉ khi có GPU) | None (fp16)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    kw = {}
    if quant in ("4bit", "8bit") and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                        bnb_4bit_quant_type="nf4")
                                     if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
        kw["max_memory"] = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.8)
                            for i in range(torch.cuda.device_count())}
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()                    # bỏ thanh "Loading weights" (làm rác log)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, **kw).eval()
    log(f"  nạp xong {mid}: lượng tử {quant if 'quantization_config' in kw else 'không (fp16/CPU)'}")
    return tok, model


def generate(cfg, key, shard="0/1", limit=0, batch=None, quant=None, prompt=None, log=print):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    tc = dict(cfg["translate"])
    if quant:
        tc["quant"] = quant
    mid = model_id(cfg, key)
    prompt = prompt or tc.get("prompt", "t1")
    key = run_key(cfg, key, prompt)
    system, user_turn, shots = prompt_parts(cfg, prompt)
    thinking = bool(tc.get("thinking", False))
    qids, qtexts = vi_queries(cfg)
    if limit:
        qids, qtexts = qids[:limit], qtexts[:limit]
    si, sn = (int(x) for x in shard.split("/"))
    qids, qtexts = qids[si::sn], qtexts[si::sn]
    out = records_dir(cfg, key) / ("records.jsonl" if sn == 1 else f"records-{si}of{sn}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_records(cfg, key)
    tried = set(load_records(cfg, key, ok_only=False))
    todo = sorted([(q, t) for q, t in zip(qids, qtexts) if q not in done], key=lambda x: -len(x[1]))
    log(f"[{key}] {mid}: phần {shard}: {len(qids)} câu, đã có {len(qids) - len(todo)}, còn {len(todo)}")
    if not todo:
        return
    tok, model = load_llm(mid, tc.get("quant"), log)
    log(f"  lô {batch or tc['batch']}")

    def chat(q):
        msgs = [{"role": "system", "content": system}]
        for u, a_ in shots:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a_}]
        msgs.append({"role": "user", "content": user_turn(q)})
        # enable_thinking=False tắt chế độ suy nghĩ của Qwen3 (translate.thinking); model khác bỏ qua tham số này
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    def gen(batch, sample):
        enc = tok([chat(t) for _, t in batch], return_tensors="pt", padding=True).to(model.device)
        qlen = max(len(tok(t)["input_ids"]) for _, t in batch)
        max_new = int(min(1536, max(tc["max_new_tokens"], 160 + 3 * qlen)))   # câu dài nhiều ý → bản dịch dài
        if thinking:
            max_new += 1024                               # chỗ cho phần <think>…</think>
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                               **({"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample
                                  else {"do_sample": False}))
        return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    log(f"  prompt {prompt}, {len(shots)} ví dụ, prompt mẫu ~{len(tok(chat(todo[0][1]))['input_ids'])} token"
        + (", CÓ suy nghĩ (thinking)" if thinking else ""))
    t0, n_ok, n_done, bs = time.time(), 0, 0, int(batch or tc["batch"])
    # lượt 1: tất định (greedy) cho câu chưa thử; lượt 2: lấy mẫu cho câu đã hỏng ở phiên trước VÀ câu vừa hỏng ở lượt 1
    # (vd lặp vô hạn trong bảng khái niệm → JSON không đóng) — để không phải chạy lại notebook mới sửa được
    failed_now = []
    passes = [([x for x in todo if x[0] not in tried], False),
              (lambda: [x for x in todo if x[0] in tried] + failed_now, True)]
    for group, sample in passes:
        group = group() if callable(group) else group
        if sample and group:
            log(f"  dịch lại có lấy mẫu {len(group)} câu hỏng: {[q for q, _ in group][:10]}")
        k = 0
        while k < len(group):
            batch = group[k:k + bs]
            t1 = time.time()
            try:
                texts = gen(batch, sample)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log(f"  [CUDA hết bộ nhớ] giảm lô xuống {bs}")
                continue
            sec = (time.time() - t1) / len(batch)
            with out.open("a", encoding="utf-8") as f:
                for (q, t), raw in zip(batch, texts):
                    d = parse(raw)
                    ok = bool(d and d["en"] and d["zh"]) and not copies_example({"query": t, **d})
                    n_ok += ok
                    if not ok and not sample:
                        failed_now.append((q, t))
                    f.write(json.dumps({"id": q, "query": t, "ok": ok, **(d or {}), "raw": raw, "model": mid,
                                        "prompt": prompt, "thinking": thinking, "sampled": sample,
                                        "sec": round(sec, 2)},
                                       ensure_ascii=False) + "\n")
            k += len(batch)
            n_done += len(batch)
            mem = (f", GPU đỉnh {torch.cuda.max_memory_allocated() / 2**30:.1f}/"
                   f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB") if torch.cuda.is_available() else ""
            log(f"  {n_done}/{len(todo) + len(failed_now)} lượt dịch, hợp lệ {n_ok}, lô {bs}, {time.time() - t0:.0f}s ({sec:.1f}s/câu ở lô này){mem}")


def import_old(cfg, qexp_dir, name="old_qwen25_14b", log=print):
    """Bản mở rộng cũ (mir/expand.py, Qwen2.5-14B 4bit, prompt v2) → bản ghi dịch (chỉ lấy concepts, en, zh)."""
    files = sorted(Path(qexp_dir).glob("*expansions*.jsonl"))
    if not files:
        raise SystemExit(f"Không thấy expansions*.jsonl trong {qexp_dir}")
    out = records_dir(cfg, name) / "records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in files:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                ok = bool(r.get("ok") and r.get("en") and r.get("zh"))
                f.write(json.dumps({"id": r["id"], "query": r["query"], "ok": ok, "concepts": r.get("concepts") or [],
                                    "en": r.get("en", ""), "zh": r.get("zh", ""), "model": r.get("model"),
                                    "prompt": "v2-old", "sampled": r.get("sampled", False)}, ensure_ascii=False) + "\n")
                n += 1
    log(f"→ {out}: {n} bản ghi, hợp lệ {len(load_records(cfg, name))}")


# ---------------------------------------------------------------- dựng file câu hỏi
def build(cfg, use=None, log=print):
    key = resolve_key(cfg, use or cfg["translate"]["use"])
    qids, qtexts = vi_queries(cfg)
    done = load_records(cfg, key)
    if not done:
        raise SystemExit(f"Không có bản dịch nào trong {records_dir(cfg, key)} (có: "
                         f"{sorted(p.name for p in cfg.path('qtrans').iterdir() if p.is_dir())})")
    out = cfg.path("qtrans")
    out.mkdir(parents=True, exist_ok=True)
    rows = {"vi": [(q, t) for q, t in zip(qids, qtexts)]}
    for lg in ("en", "zh"):
        rows[lg] = [(q, done[q][lg]) for q in qids if q in done and script_ok(done[q].get(lg, ""), lg)]
    for lg, rs in rows.items():
        pq.write_table(pa.table({"id": pa.array([r[0] for r in rs], pa.int64()), "query": [r[1] for r in rs],
                                 "lang": [lg] * len(rs)}), out / f"queries_{lg}.parquet")
        log(f"  queries_{lg}: {len(rs)}/{len(qids)} câu" + ("" if lg == "vi" else f" (bản dịch {key})"))
    miss = [q for q in qids if q not in done]
    if miss:
        log(f"  [LƯU Ý] {len(miss)} câu chưa có bản dịch hợp lệ (vd {miss[:8]}): nhánh en/zh bỏ trống các câu này "
            "(không thay bằng câu gốc để khỏi nhân đôi trọng số câu tiếng Việt trong RRF).")
    (out / "queries_meta.json").write_text(json.dumps({"translation": key, "missing": miss}, indent=1),
                                           encoding="utf-8")


# ---------------------------------------------------------------- so sánh hai model
_ACR = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9-]*")   # HPV, HBsAg, IVF, beta-HCG…


def _head(term, lang):
    """Phần chính của thuật ngữ trong bảng (bỏ phần trong ngoặc / sau ';' / '/'): "tonsils (palatine…)" → "tonsils"."""
    t = re.split(r"[;/]", term)[0]
    t = re.sub(r"[（(].*?[）)]", "", t).strip()
    return t.lower() if lang == "en" else t


def auto_metrics(qids, qtexts, recs, glossary=None):
    """Chỉ số đo trực tiếp được (không cần người): hợp lệ, chữ viết, giữ số, giữ số câu hỏi, giữ chữ viết tắt, dùng đúng
    thuật ngữ trong bảng (khi câu có thuật ngữ khớp), độ dài, thời gian."""
    n = len(qids)
    m = {"n": n, "ok": 0, "en_script": 0, "zh_script": 0, "en_numbers": 0, "zh_numbers": 0, "en_len_ratio": [],
         "zh_len_ratio": [], "en_too_short": 0, "zh_too_short": 0, "sec": [], "multi_q": 0, "en_questions": 0,
         "zh_questions": 0, "acr": 0, "en_acr": 0, "zh_acr": 0, "gl": 0, "en_gl": 0, "zh_gl": 0}
    for q, t in zip(qids, qtexts):
        r = recs.get(q)
        if not r:
            continue
        m["ok"] += 1
        en, zh = r.get("en", ""), r.get("zh", "")
        m["en_script"] += script_ok(en, "en")
        m["zh_script"] += script_ok(zh, "zh")
        m["en_numbers"] += numbers_kept(t, en)
        m["zh_numbers"] += numbers_kept(t, zh)
        nq = t.count("?")
        if nq >= 2:                                       # câu nhiều câu hỏi: bản dịch còn đủ số dấu hỏi?
            m["multi_q"] += 1
            m["en_questions"] += en.count("?") >= nq
            m["zh_questions"] += (zh.count("？") + zh.count("?")) >= nq
        for a in set(_ACR.findall(t)):
            m["acr"] += 1
            m["en_acr"] += a.lower() in en.lower()
            m["zh_acr"] += a.lower() in zh.lower()
        for _txt, g in glossary_hits(t, glossary or []):
            m["gl"] += 1
            m["en_gl"] += _head(g[1], "en") in en.lower()
            m["zh_gl"] += _head(g[2], "zh") in zh
        re_ = len(en) / max(1, len(t))
        rz = len(zh) / max(1, len(t))
        m["en_len_ratio"].append(re_)
        m["zh_len_ratio"].append(rz)
        m["en_too_short"] += re_ < 0.5                   # nghi tóm tắt / bỏ ý (câu Anh thường dài ~0,9–1,2 câu Việt)
        m["zh_too_short"] += rz < 0.15                    # (câu Trung thường ~0,3–0,45 số ký tự câu Việt)
        if r.get("sec"):
            m["sec"].append(r["sec"])
    import statistics as st
    for k in ("en_len_ratio", "zh_len_ratio", "sec"):
        v = m[k]
        m[k] = round(st.median(v), 3) if v else None
    return m


def judged_before(cfg):
    """qid đã nằm trong phiếu chấm mù của các lần so trước (qtrans/compare_*/key.json)."""
    out = set()
    for f in cfg.path("qtrans").glob("compare_*/key.json"):
        out |= {int(r["qid"]) for r in json.loads(f.read_text(encoding="utf-8"))}
    return out


def compare(cfg, a, b, n=100, fresh=True, log=print):
    ka, kb = resolve_key(cfg, a), resolve_key(cfg, b)
    qids, qtexts = vi_queries(cfg)
    ra, rb = load_records(cfg, ka), load_records(cfg, kb)
    gl = load_glossary(cfg["translate"].get("glossary") or "configs/med_glossary_vi.tsv")
    ma, mb = auto_metrics(qids, qtexts, ra, gl), auto_metrics(qids, qtexts, rb, gl)
    pct = lambda m, k: f"{100 * m[k] / max(1, m['n']):.1f}%"
    frac = lambda m, k, d: f"{100 * m[k] / max(1, m[d]):.1f}% ({m[k]}/{m[d]})"
    L = [f"# So sánh bản dịch: {ka} vs {kb}", "",
         "Chỉ số tự động (đo trực tiếp, KHÔNG phải độ chính xác bản dịch — chỉ bắt lỗi hình thức):", "",
         f"| chỉ số | {ka} | {kb} |", "|---|---:|---:|",
         f"| JSON hợp lệ, đủ en + zh | {pct(ma, 'ok')} | {pct(mb, 'ok')} |",
         f"| en đúng chữ viết (không Hán, không dấu Việt) | {pct(ma, 'en_script')} | {pct(mb, 'en_script')} |",
         f"| zh đúng chữ viết | {pct(ma, 'zh_script')} | {pct(mb, 'zh_script')} |",
         f"| en giữ đủ con số | {pct(ma, 'en_numbers')} | {pct(mb, 'en_numbers')} |",
         f"| zh giữ đủ con số | {pct(ma, 'zh_numbers')} | {pct(mb, 'zh_numbers')} |",
         f"| en nghi bỏ ý (độ dài < 0,5 câu gốc) | {pct(ma, 'en_too_short')} | {pct(mb, 'en_too_short')} |",
         f"| zh nghi bỏ ý (< 0,15 câu gốc) | {pct(ma, 'zh_too_short')} | {pct(mb, 'zh_too_short')} |",
         f"| en giữ đủ số câu hỏi (câu có >= 2 dấu ?) | {frac(ma, 'en_questions', 'multi_q')} | "
         f"{frac(mb, 'en_questions', 'multi_q')} |",
         f"| zh giữ đủ số câu hỏi | {frac(ma, 'zh_questions', 'multi_q')} | {frac(mb, 'zh_questions', 'multi_q')} |",
         f"| en giữ chữ viết tắt (HPV, AMH…) | {frac(ma, 'en_acr', 'acr')} | {frac(mb, 'en_acr', 'acr')} |",
         f"| zh giữ chữ viết tắt | {frac(ma, 'zh_acr', 'acr')} | {frac(mb, 'zh_acr', 'acr')} |",
         f"| en dùng thuật ngữ của bảng (khi khớp) | {frac(ma, 'en_gl', 'gl')} | {frac(mb, 'en_gl', 'gl')} |",
         f"| zh dùng thuật ngữ của bảng (khi khớp) | {frac(ma, 'zh_gl', 'gl')} | {frac(mb, 'zh_gl', 'gl')} |",
         f"| tỷ lệ độ dài en/vi (trung vị) | {ma['en_len_ratio']} | {mb['en_len_ratio']} |",
         f"| tỷ lệ độ dài zh/vi (trung vị) | {ma['zh_len_ratio']} | {mb['zh_len_ratio']} |",
         f"| giây/câu (trung vị) | {ma['sec']} | {mb['sec']} |", ""]
    # khác biệt giữa hai model: câu mà hai bản dịch khác nhau nhiều (theo ký tự) → ưu tiên cho người đọc
    import difflib
    both = [q for q in qids if q in ra and q in rb]
    old = judged_before(cfg) if fresh else set()
    if old:                                               # phiếu mới: không lặp câu đã chấm (bảng thuật ngữ t2 lập
        both = [q for q in both if q not in old]          # một phần từ lỗi ở các câu đó → chấm lại sẽ thiên lệch)
        L.append(f"Phiếu mới bỏ {len(old)} câu đã có trong phiếu chấm các lần trước.")
        L.append("")
    diff = {q: 1 - difflib.SequenceMatcher(None, ra[q]["zh"], rb[q]["zh"]).ratio() for q in both}
    # phiếu chấm mù: n câu = một nửa chọn ngẫu nhiên phân tầng theo độ dài, một nửa là câu khác biệt nhiều nhất
    rng = random.Random(cfg["seed"])
    qtext = dict(zip(qids, qtexts))
    by_len = sorted(both, key=lambda q: len(qtext[q]))
    tier = [by_len[i::3] for i in range(3)]
    pick = []
    for t in tier:
        pick += rng.sample(t, min(len(t), n // 6))
    for q in sorted(both, key=lambda q: -diff[q]):
        if len(pick) >= n:
            break
        if q not in pick:
            pick.append(q)
    rng.shuffle(pick)
    out = cfg.path("qtrans") / f"compare_{ka}_vs_{kb}"
    k = 1
    while (out / "key.json").exists():                    # không ghi đè phiếu/khoá của lần so trước (có thể đã chấm)
        k += 1
        out = cfg.path("qtrans") / f"compare_{ka}_vs_{kb}_{k}"
    out.mkdir(parents=True, exist_ok=True)
    import csv
    key_rows = []
    with (out / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "qid", "query_vi", "en_X", "en_Y", "zh_X", "zh_Y",
                    "better_en (X/Y/=)", "better_zh (X/Y/=)", "severe_X (0/1)", "severe_Y (0/1)", "note"])
        for i, q in enumerate(pick):
            x, y = (ka, kb) if rng.random() < 0.5 else (kb, ka)
            rx, ry = (ra, rb) if x == ka else (rb, ra)
            w.writerow([i, q, qtext[q], rx[q]["en"], ry[q]["en"], rx[q]["zh"], ry[q]["zh"], "", "", "", "", ""])
            key_rows.append({"row": i, "qid": q, "X": x, "Y": y})
    (out / "key.json").write_text(json.dumps(key_rows, indent=1), encoding="utf-8")
    L += [f"Phiếu chấm mù: `{out / 'sheet.csv'}` ({len(pick)} câu: phân tầng theo độ dài + các câu hai model dịch khác "
          f"nhau nhiều nhất). X/Y được xáo ngẫu nhiên mỗi dòng; khoá ở `key.json` — người chấm KHÔNG mở khoá.", "",
          "Hướng dẫn chấm: với từng ngôn ngữ chọn bản tốt hơn (X / Y / = nếu tương đương). severe = 1 nếu bản đó có lỗi "
          "NẶNG làm sai thông tin cần tìm: sai thuật ngữ bệnh/thuốc/xét nghiệm, đổi người mắc bệnh, sai/mất con số, bỏ "
          "mất một ý hỏi, thêm triệu chứng/bệnh không có, trả lời câu hỏi thay vì dịch.",
          "Chấm xong: `python -m mirret.translate score --sheet <file đã chấm>`."]
    md = "\n".join(L) + "\n"
    (out / "auto_metrics.md").write_text(md, encoding="utf-8")
    log(md)


def score(cfg, sheets, log=print):
    """Đọc phiếu đã chấm (một hoặc nhiều người) + key.json → tỷ lệ thắng, lỗi nặng, khoảng tin cậy bootstrap."""
    import csv
    import math
    res = {}
    for sp in sheets:
        sp = Path(sp)
        key = {r["row"]: r for r in json.loads((sp.parent / "key.json").read_text(encoding="utf-8"))}
        with sp.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                k = key[int(row["row"])]
                for lg, col in (("en", "better_en (X/Y/=)"), ("zh", "better_zh (X/Y/=)")):
                    v = (row.get(col) or "").strip().upper()
                    if v in ("X", "Y", "="):
                        winner = "=" if v == "=" else k[v]
                        res.setdefault(lg, []).append(winner)
                for side in ("X", "Y"):
                    v = (row.get(f"severe_{side} (0/1)") or "").strip()
                    if v in ("0", "1"):
                        res.setdefault("severe", []).append((k[side], int(v)))
    models = sorted({m for lg in ("en", "zh") for m in res.get(lg, []) if m != "="} |
                    {m for m, _ in res.get("severe", [])})
    L = ["# Kết quả chấm mù bản dịch", ""]
    rng = random.Random(0)
    for lg in ("en", "zh"):
        v = res.get(lg, [])
        if not v or len(models) != 2:
            continue
        a, b = models
        wa, wb, tie = v.count(a), v.count(b), v.count("=")
        boot = []
        for _ in range(2000):
            s = [rng.choice(v) for _ in v]
            boot.append((s.count(a) - s.count(b)) / len(s))
        boot.sort()
        n = wa + wb
        p = sum(math.comb(n, i) for i in range(0, min(wa, wb) + 1)) * 2 / 2 ** n if n else 1.0   # dấu hai phía
        L.append(f"- **{lg}**: {a} thắng {wa}, {b} thắng {wb}, hoà {tie} (n={len(v)}). Chênh lệch (thắng−thua)/n = "
                 f"{(wa - wb) / len(v):+.3f}, KTC 95% bootstrap [{boot[50]:+.3f}, {boot[1949]:+.3f}], "
                 f"kiểm định dấu p = {min(1.0, p):.3f}")
    for m in models:
        s = [x for mm, x in res.get("severe", []) if mm == m]
        if s:
            L.append(f"- Lỗi nặng của {m}: {sum(s)}/{len(s)} câu ({100 * sum(s) / len(s):.1f}%)")
    L += ["", "Quy ước quyết định: chọn model có tỷ lệ lỗi nặng thấp hơn; nếu tương đương, chọn model thắng nhiều hơn ở "
          "tiếng Trung (nhánh zh là đường chính tới tài liệu tiếng Trung). Chênh lệch nằm trong KTC chứa 0 → coi như "
          "hoà, khi đó chọn theo kết quả truy hồi trên tập có nhãn (mirret.evaluate)."]
    md = "\n".join(L) + "\n"
    log(md)
    return md


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "import-old", "compare", "score", "build"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None, help="generate: số câu mỗi lô (mặc định translate.batch)")
    ap.add_argument("--quant", default=None, choices=[None, "4bit", "8bit", "none"], help="generate: ghi đè translate.quant")
    ap.add_argument("--qexp", default=None)
    ap.add_argument("--name", default="old_qwen25_14b")
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sheet", action="append", default=[])
    ap.add_argument("--use", default=None)
    ap.add_argument("--prompt", default=None, help="generate: phiên bản prompt (t1 | t2; mặc định translate.prompt)")
    ap.add_argument("--all-queries", action="store_true", help="compare: cho phép lặp lại câu đã chấm ở lần trước")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "generate":
        generate(cfg, a.model or cfg["translate"]["use"], a.shard, a.limit, a.batch, a.quant, a.prompt)
    elif a.cmd == "import-old":
        import_old(cfg, a.qexp, a.name)
    elif a.cmd == "compare":
        compare(cfg, a.a, a.b, a.n, not a.all_queries)
    elif a.cmd == "score":
        score(cfg, a.sheet)
    else:
        build(cfg, a.use)


if __name__ == "__main__":
    main()
