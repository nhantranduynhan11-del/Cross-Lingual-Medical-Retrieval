"""Viết lại câu hỏi + thuật ngữ mở rộng do LLM sinh (Qwen2.5-14B) — rewrite cho dense/LSR, thuật ngữ có trọng số cho BM25.

Mỗi câu hỏi 3 lượt (vi / en / zh), mỗi lượt từ câu hỏi của nhánh đó: vi = câu gốc, en/zh = bản dịch đang dùng
(qtrans/queries_{en,zh}.parquet, t2). Sinh tất định (greedy, KHÔNG phạt lặp — phạt lặp của transformers phạt cả token
của prompt → tiếng Việt né âm tiết của câu hỏi, sinh "hở …", "hẹn họng", trôi sang tiếng Trung; xem bản rw2). Đầu ra JSON:
  {"rewrite": "câu hỏi y khoa gọn", "terms": ["tên gọi khác", …(≤ 8)], "candidates": ["bệnh có khả năng", …(≤ 3)]}
candidates chỉ có khi câu hỏi hỏi nguyên nhân / bệnh gì — tách khỏi terms để bật / tắt riêng khi truy hồi (w_cand).
Gợi ý thuật ngữ trong prompt = các mục của configs/med_glossary_vi.tsv khớp câu gốc (translate.glossary_hits), viết bằng
ngôn ngữ của lượt; build còn tự thêm các tên đó vào terms (qexpand.glossary_terms) vì mô hình hay bỏ qua gợi ý.
JSON hỏng / rewrite sai ngôn ngữ → rewrite = câu hỏi của nhánh, terms = [], ok = False (greedy: sinh lại ra y hệt).
Term sai chữ viết (vi / en chứa chữ Hán…) bị bỏ.

  python -m mirret.query_expand generate [--qids 80queries.csv] [--limit 5] [--langs vi,en,zh] [--shard 0/2] [--batch 8]
         → qexp/rw3-<model>/records*.jsonl (cache theo phiên bản prompt + (qid, lang), có đầu ra thô; chạy lại = bỏ qua
           lượt đã có)
  python -m mirret.query_expand build   → qexp/rw/queries_<lang>.parquet (rewrite), qexp/rwt/queries_<lang>.parquet
                                          (rewrite + thuật ngữ, cho LSR), qexp/rw/terms_<lang>.parquet (cho BM25)
  python -m mirret.query_expand show [--limit 5]           đầu ra 3 ngôn ngữ + số lượt lỗi JSON
  python -m mirret.query_expand explain [--qid 36]         token + trọng số BM25 trước / sau mở rộng (cần chỉ mục BM25)

Dùng trong truy hồi: WeightedQueryEncoder (BM25 có trọng số), query_variants + fuse_variants (dense/LSR nhiều phiên bản).
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import bm25, config, expand, runs, translate

LANGS = ("vi", "en", "zh")
MODE = "rw3"                 # phiên bản prompt; cache qexp/<MODE>-<model>/ (rw: bịa chi tiết, terms chép câu hỏi;
                             # rw2: phạt lặp làm hỏng terms tiếng Việt, chẩn đoán lẫn trong terms)
MAX_TERMS, MAX_TERM_LEN, MAX_CANDS, MAX_ALL_TERMS = 8, 60, 3, 14

SYSTEM = {
    "vi": "Bạn là chuyên gia y khoa, hỗ trợ tìm kiếm tài liệu y khoa. Chỉ trả lời bằng JSON.",
    "en": "You are a medical expert helping with medical literature search. Answer only with JSON.",
    "zh": "你是协助检索医学资料的医学专家。只用JSON回答。",
}
INSTRUCT = {
    "vi": "Viết lại câu hỏi y khoa dưới đây và liệt kê thuật ngữ, hoàn toàn bằng tiếng Việt.\n"
          '- "rewrite": MỘT câu hỏi y khoa ngắn gọn (tối đa 40 từ; câu hỏi có nhiều ý thì giữ đủ các ý, tối đa 60 từ), '
          "giữ: triệu chứng, bệnh, thuốc, xét nghiệm, đối tượng và điều người hỏi muốn biết; bỏ lời chào, xưng hô và phần "
          "kể chuyện. CHỈ dùng thông tin có trong câu hỏi: không thêm tuổi, giới, chẩn đoán, thời gian hay chi tiết nào "
          "khác; giữ nguyên các con số, ngày tháng như trong câu hỏi, không quy đổi.\n"
          '- "terms": tối đa 8 cụm, ưu tiên TÊN GỌI KHÁC chưa có trong câu hỏi của các khái niệm chính (tên thông thường '
          "↔ tên y khoa, biệt dược ↔ hoạt chất, viết tắt ↔ đầy đủ); mỗi khái niệm chính có ít nhất một tên khác; thuật "
          "ngữ đã kiểm chứng (nếu có) phải được đưa vào. Không đưa từ chung chung (thuốc, bệnh, triệu chứng, nguyên "
          "nhân, điều trị) và không đưa chẩn đoán. Mỗi phần tử là một cụm, không dùng ký hiệu ↔.\n"
          '- "candidates": CHỈ khi câu hỏi hỏi nguyên nhân hoặc đang bị bệnh gì: tối đa 3 bệnh / tình trạng có khả năng '
          "nhất; các trường hợp khác để [].\n"
          'Chỉ trả về JSON dạng {{"rewrite": "...", "terms": ["..."], "candidates": ["..."]}}.{hints}\n\nCâu hỏi: {q}',
    "en": "Rewrite the medical question below and list its terms, entirely in English.\n"
          '- "rewrite": ONE concise medical question (at most 40 words; if the question has several parts keep all of '
          "them, at most 60 words) keeping: symptoms, diseases, drugs, tests, the patient and what is being asked; drop "
          "greetings and storytelling. Use ONLY information in the question: do not add age, sex, diagnoses, durations "
          "or any other detail; keep numbers and dates exactly as written, do not convert them.\n"
          '- "terms": up to 8 phrases, preferring OTHER NAMES not already in the question for its main concepts (lay ↔ '
          "medical name, brand ↔ active ingredient, abbreviation ↔ full form); give at least one other name for each "
          "main concept; verified terms (if any) must be included. No generic words (drug, disease, symptoms, causes, "
          "treatment) and no diagnoses. One phrase per item, do not use the ↔ symbol.\n"
          '- "candidates": ONLY if the question asks for a cause or what condition it is: up to 3 most likely '
          "conditions; otherwise [].\n"
          'Return only JSON: {{"rewrite": "...", "terms": ["..."], "candidates": ["..."]}}.{hints}\n\nQuestion: {q}',
    "zh": "请完全用简体中文改写下面的医学问题并列出术语。\n"
          '- "rewrite"：一个简洁的医学问题（不超过60个字；问题包含多个方面时保留全部，不超过90个字），保留症状、疾病、药物、检查、'
          "患者情况以及想问的内容；去掉问候语和叙述。只能使用问题中已有的信息：不要添加年龄、性别、诊断、时间或其他任何细节；"
          "数字和日期照原样保留，不要换算。\n"
          '- "terms"：最多8个词语，优先给出问题中主要概念的、问题里没有出现的其他名称（俗称 ↔ 医学名称、商品名 ↔ 有效成分、缩写 ↔ '
          "全称）；每个主要概念至少给出一个其他名称；如有已核实的术语必须列入。不要列出笼统的词（药物、疾病、症状、原因、治疗），"
          "也不要列出诊断。每项一个词语，不要使用 ↔ 符号。\n"
          '- "candidates"：仅当问题询问病因或是什么病时，列出最多3个最可能的疾病或情况；否则为[]。\n'
          '只返回JSON：{{"rewrite": "...", "terms": ["..."], "candidates": ["..."]}}。{hints}\n\n问题：{q}',
}
HINTS_HEAD = {"vi": "Thuật ngữ đã kiểm chứng (đưa vào terms):", "en": "Verified terms (include in terms):",
              "zh": "已核实的术语（列入terms）："}
# Ví dụ mẫu (câu tự soạn, không lấy từ bộ câu hỏi của BTC): giữ đúng chi tiết có trong câu, terms là tên gọi khác,
# chẩn đoán có khả năng để ở candidates (câu hỏi hỏi "có phải biến chứng không")
EXAMPLE = {
    "vi": ("Chào bác sĩ, mẹ em năm nay 60 tuổi bị tiểu đường mấy năm nay, đang uống Glucophage. Dạo này mẹ hay tê bì "
           "chân tay, đi lại khó khăn. Cho em hỏi có phải biến chứng không và nên làm gì ạ?",
           {"rewrite": "Người 60 tuổi bị tiểu đường đang uống Glucophage, tê bì chân tay, đi lại khó: có phải biến chứng "
                       "của tiểu đường không và nên làm gì?",
            "terms": ["đái tháo đường", "metformin", "dị cảm", "tê tay chân", "rối loạn dáng đi",
                      "biến chứng đái tháo đường"],
            "candidates": ["bệnh thần kinh ngoại biên do đái tháo đường", "thiếu vitamin B12 do metformin"]}),
    "en": ("Hello doctor, my mother is 60 and has had diabetes for several years; she takes Glucophage. Lately she often "
           "has numbness and tingling in her hands and feet and finds it hard to walk. Is this a complication, and what "
           "should we do?",
           {"rewrite": "60-year-old with diabetes taking Glucophage has numbness and tingling in hands and feet and "
                       "difficulty walking: is it a diabetes complication and what should be done?",
            "terms": ["diabetes mellitus", "metformin", "paresthesia", "pins and needles",
                      "gait disturbance", "diabetic complications"],
            "candidates": ["diabetic peripheral neuropathy", "metformin-induced vitamin B12 deficiency"]}),
    "zh": ("医生您好，我妈妈今年60岁，得糖尿病好几年了，一直在吃格华止。最近她经常手脚发麻，走路困难。请问这是并发症吗？应该怎么办？",
           {"rewrite": "60岁糖尿病患者服用格华止，手脚发麻、行走困难，是否为糖尿病并发症？应如何处理？",
            "terms": ["二甲双胍", "肢体麻木", "感觉异常", "步态异常", "糖尿病并发症"],
            "candidates": ["糖尿病周围神经病变", "二甲双胍相关维生素B12缺乏"]}),
}


# ---------------------------------------------------------------- bảng thuật ngữ
_ALIASES = {}


def glossary_aliases(path):
    """Các alias tiếng Việt của từng mục bảng thuật ngữ, cùng thứ tự với translate.load_glossary."""
    path = Path(path)
    if not path.is_absolute():
        path = config.ROOT / path
    if path not in _ALIASES:
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    rows.append([unicodedata.normalize("NFC", a.strip().lstrip("=")) for a in line.split("\t")[0].split("|")
                                 if a.strip()])
        _ALIASES[path] = rows
    return _ALIASES[path]


def _split_names(s):
    return [x.strip() for x in re.split(r"[；;（）()、/,，]|\s=\s", s or "") if len(x.strip()) >= 2]


def glossary_terms(vi_query, lang, glossary, aliases):
    """Tên đã kiểm chứng (bảng thuật ngữ) cho các mục khớp câu gốc, bằng ngôn ngữ lang; vi: các alias chưa có trong câu."""
    pos = {id(g): i for i, g in enumerate(glossary)}
    q, out = vi_query.casefold(), []
    for _txt, g in translate.glossary_hits(vi_query, glossary, 50):
        if lang == "vi":
            out += [a for a in aliases[pos[id(g)]] if a.casefold() not in q]
        else:
            out += _split_names(g[1] if lang == "en" else g[2])
    return filter_terms(clean_terms(out, max_n=50), lang)


# ---------------------------------------------------------------- prompt
def hints(vi_query, lang, glossary, k=10):
    """Các mục bảng thuật ngữ khớp câu gốc, viết bằng ngôn ngữ của lượt (vi: cụm khớp = tên chuẩn (en))."""
    lines = []
    for txt, g in translate.glossary_hits(vi_query, glossary, k):
        if lang == "vi":
            same = txt.casefold() == g[4].casefold()
            lines.append(f"- {txt}" + ("" if same else f" = {g[4]}") + (f" (en: {g[1]})" if g[1] else ""))
        else:
            v = g[1] if lang == "en" else g[2]
            if v:
                lines.append(f"- {v}")
    return ("\n" + HINTS_HEAD[lang] + "\n" + "\n".join(lines)) if lines else ""


def chat(tok, lang, query, hint_text="", example_hints=""):
    """Prompt: hệ thống + một ví dụ mẫu (EXAMPLE, kèm gợi ý thuật ngữ của chính ví dụ) + câu hỏi."""
    q, a = EXAMPLE[lang]
    msgs = [{"role": "system", "content": SYSTEM[lang]},
            {"role": "user", "content": INSTRUCT[lang].format(q=q, hints=example_hints)},
            {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)},
            {"role": "user", "content": INSTRUCT[lang].format(q=query, hints=hint_text)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ---------------------------------------------------------------- đọc đầu ra
_STR = r'"((?:[^"\\]|\\.)*)"'
_PAIR = re.compile(r"\s*(?:↔|<->|⇔|⟷|\s=\s)\s*")
FIELDS = ("rewrite", "terms", "candidates")


def parse_json(text):
    """Đầu ra thô → {"rewrite", "terms", "candidates"} hoặc None. Lấy khối {...} đầu tiên đọc được (bỏ <think>, ```json,
    chữ thừa trước/sau); JSON hỏng → lấy chuỗi "rewrite" và các danh sách bằng biểu thức chính quy."""
    s = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    s = re.sub(r"```(?:json)?", "", s)
    dec = json.JSONDecoder()
    a = s.find("{")
    while a >= 0:
        try:
            d, _ = dec.raw_decode(s, a)
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict) and ("rewrite" in d or "terms" in d):
            return {k: d.get(k) for k in FIELDS}
        a = s.find("{", a + 1)
    m = re.search(r'"rewrite"\s*:\s*' + _STR, s)
    if not m:
        return None

    def unq(x):
        try:
            return json.loads(f'"{x}"')
        except json.JSONDecodeError:
            return x

    def lst(key):
        t = re.search(r'"%s"\s*:\s*\[(.*?)(?:\]|$)' % key, s, re.S)
        return [unq(x) for x in re.findall(_STR, t.group(1))] if t else []
    return {"rewrite": unq(m.group(1)), "terms": lst("terms"), "candidates": lst("candidates")}


def clean_terms(terms, max_n=MAX_TERMS, max_len=MAX_TERM_LEN):
    """Bỏ rỗng, dài > max_len ký tự, trùng (không phân biệt hoa thường); tối đa max_n. Chuỗi → tách theo , ; xuống dòng.
    Phần tử gộp hai tên ("反酸 ↔ 胃食管反流", "a <-> b", "a = b") → tách thành hai thuật ngữ."""
    if isinstance(terms, str):
        terms = re.split(r"[,;\n，；、]", terms)
    flat = []
    for t in terms or []:
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            t = str(t)
        flat += _PAIR.split(t) if isinstance(t, str) else []
    out, seen = [], set()
    for t in flat:
        t = re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip().strip("-•*·\"'“”").strip()
        k = t.casefold()
        if not t or len(t) > max_len or k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_n:
            break
    return out


def filter_terms(terms, lang):
    """Bỏ term sai chữ viết: vi / en chứa chữ Hán, en chứa chữ tiếng Việt có dấu, zh không có chữ Hán lẫn chữ Latin
    (giữ viết tắt Latin trong zh như ALT, GERD — có ích cho BM25)."""
    out = []
    for t in terms:
        han = bool(translate._HAN.search(t))
        if lang in ("vi", "en") and han or lang == "en" and translate._VI.search(t):
            continue
        if lang == "zh" and not han and not re.search(r"[A-Za-z]", t):
            continue
        out.append(t)
    return out


def drop_artifacts(terms, lang, base):
    """Lỗi quen của Qwen2.5 với tiếng Việt (rw2: 148, rw3 không phạt lặp vẫn 93 / 1.200 lượt): cụm "hở …" vô nghĩa hoặc
    lạc đề ("hở van tim" cho câu hỏi về ho, "hở tăng cholesterol", "hở bím") → bỏ term / candidate tiếng Việt bắt đầu
    bằng "hở" khi câu hỏi không có chữ "hở" (câu hỏi về hở van tim, hở eo tử cung… vẫn giữ)."""
    if lang != "vi" or "hở" in (base or "").casefold():
        return list(terms)
    return [t for t in terms if not re.match(r"hở(\s|$)", t.casefold())]


def clean_rewrite(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().strip("\"'“”").strip()


def n_words(s, lang):
    return len(re.findall(r"[一-鿿]", s)) if lang == "zh" else len(s.split())


def to_record(qid, lang, base, raw, model="", sec=0.0):
    """Đầu ra thô → bản ghi cache. Hỏng (JSON / rewrite rỗng / sai ngôn ngữ) → rewrite = base, terms = candidates = [],
    ok = False. Term / candidate sai chữ viết bị bỏ (rewrite vẫn hợp lệ)."""
    d = parse_json(raw)
    why = ""
    rw, terms, cands = "", [], []
    if d:
        rw = clean_rewrite(d["rewrite"])
        terms = drop_artifacts(filter_terms(clean_terms(d["terms"]), lang), lang, base)
        cands = drop_artifacts(filter_terms(clean_terms(d.get("candidates"), max_n=MAX_CANDS), lang), lang, base)
    if d is None:
        why = "JSON hỏng"
    elif not rw:
        why = "rewrite rỗng"
    elif (lang == "vi" and translate._HAN.search(rw)) or (lang != "vi" and not translate.script_ok(rw, lang)):
        why = "sai ngôn ngữ"
    ok = not why
    if not ok:
        rw, terms, cands = base, [], []
    return {"id": int(qid), "lang": lang, "mode": MODE, "base": base, "rewrite": rw, "terms": terms,
            "candidates": cands, "ok": ok, "why": why, "n_words": n_words(rw, lang), "raw": raw, "model": model,
            "sec": round(sec, 2)}


# ---------------------------------------------------------------- sinh (GPU)
def generate(cfg, model=None, langs=LANGS, qids=None, shard="0/1", limit=0, batch=None, quant=None, log=print):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    xc = cfg["qexpand"]
    model = model or xc["model"]
    mid, mkey = translate.model_id(cfg, model), translate.model_key(cfg, model)
    vi = expand.queries(cfg, "vi")
    items = []
    for lg in langs:
        qs = expand.queries(cfg, lg)
        if not qs:
            log(f"  [{lg}] thiếu qtrans/queries_{lg}.parquet → bỏ qua")
        items += [(q, lg, t) for q, t in sorted(qs.items()) if qids is None or q in qids]
    if limit:
        keep = set(sorted({q for q, _, _ in items})[:limit])
        items = [x for x in items if x[0] in keep]
    si, sn = (int(x) for x in shard.split("/"))
    mine = set(sorted({q for q, _, _ in items})[si::sn])     # 3 ngôn ngữ của một câu cùng phần
    items = [x for x in items if x[0] in mine]
    out = expand.records_dir(cfg, MODE, mkey) / ("records.jsonl" if sn == 1 else f"records-{si}of{sn}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = expand.load_records(cfg, MODE, mkey, ok_only=False)
    todo = sorted([x for x in items if (x[0], x[1]) not in done], key=lambda x: -len(x[2]))
    rp = float(xc.get("repetition_penalty", 1.0))
    log(f"[{MODE}-{mkey}] phần {shard}: {len(items)} lượt ({len(mine)} câu × {len(langs)} ngôn ngữ), "
        f"đã có {len(items) - len(todo)}, còn {len(todo)}; repetition_penalty={rp}")
    if not todo:
        return
    gl = translate.load_glossary(cfg["translate"].get("glossary") or "configs/med_glossary_vi.tsv")
    k = int(xc.get("glossary_max", 10))
    ex_hints = {lg: hints(EXAMPLE["vi"][0], lg, gl, k) for lg in LANGS}
    tok, llm = translate.load_llm(mid, quant or cfg["translate"].get("quant"), log)
    expand.generate_loop(
        tok, llm, todo, out, lambda x: chat(tok, x[1], x[2], hints(vi.get(x[0], ""), x[1], gl, k), ex_hints[x[1]]),
        lambda x, raw, sample, sec: to_record(x[0], x[1], x[2], raw, mid, sec),
        int(batch or xc["batch"]), int(xc["max_new_tokens"]), retry_sample=False, repetition_penalty=rp, log=log)


def load(cfg, model=None):
    """{(qid, lang): bản ghi} (bản hợp lệ được ưu tiên)."""
    return expand.load_records(cfg, MODE, translate.model_key(cfg, model or cfg["qexpand"]["model"]), ok_only=False)


# ---------------------------------------------------------------- dựng file câu hỏi
def build(cfg, model=None, name=None, log=print):
    xc = cfg["qexpand"]
    name = name or xc.get("name", MODE)
    recs = load(cfg, model)
    if not recs:
        raise SystemExit(f"Chưa có bản ghi trong {expand.records_dir(cfg, MODE, translate.model_key(cfg, model or xc['model']))}")
    gpath = cfg["translate"].get("glossary") or "configs/med_glossary_vi.tsv"
    gl, al = translate.load_glossary(gpath), glossary_aliases(gpath)
    use_gl = bool(xc.get("glossary_terms", True))
    vi = expand.queries(cfg, "vi")
    out, out_t = cfg.path("qexp") / name, cfg.path("qexp") / f"{name}t"
    out.mkdir(parents=True, exist_ok=True)
    out_t.mkdir(parents=True, exist_ok=True)
    for lg in LANGS:
        qs = expand.queries(cfg, lg)
        rows, n_gl = [], 0
        for q in sorted({q for q, l in recs if l == lg}):
            r = recs[(q, lg)]
            ok = bool(r.get("ok"))
            base = qs.get(q, r["base"])
            rw = r["rewrite"] if ok else base
            terms = drop_artifacts(filter_terms(list(r.get("terms") or []), lg), lg, base) if ok else []
            cands = drop_artifacts(filter_terms(list(r.get("candidates") or []), lg), lg, base) if ok else []
            if use_gl:                                        # tên đã kiểm chứng: không phụ thuộc lượt LLM có hỏng không
                extra = [t for t in glossary_terms(vi.get(q, ""), lg, gl, al)
                         if t.casefold() not in {x.casefold() for x in terms}]
                n_gl += len(extra)
                terms = (terms + extra)[:MAX_ALL_TERMS]
            rows.append((q, base, rw, terms, cands, ok))
        if not rows:
            continue
        expand.write_queries(out / f"queries_{lg}.parquet", [(q, rw) for q, _, rw, _, _, _ in rows], lg)
        expand.write_queries(out_t / f"queries_{lg}.parquet",
                             [(q, " ; ".join([rw] + t)) for q, _, rw, t, _, _ in rows], lg)
        strs = pa.list_(pa.string())
        pq.write_table(pa.table({"id": pa.array([r[0] for r in rows], pa.int64()), "base": [r[1] for r in rows],
                                 "rewrite": [r[2] for r in rows], "terms": pa.array([r[3] for r in rows], strs),
                                 "candidates": pa.array([r[4] for r in rows], strs), "ok": [r[5] for r in rows]}),
                       out / f"terms_{lg}.parquet")
        log(f"  {name}/{lg}: {len(rows)} câu, lỗi {sum(not r[5] for r in rows)}, thêm {n_gl} tên từ bảng thuật ngữ, "
            f"{sum(bool(r[4]) for r in rows)} câu có candidates")
    (out / "meta.json").write_text(json.dumps({"mode": MODE, "model": model or xc["model"], "glossary_terms": use_gl,
                                               "variants": {name: "rewrite", f"{name}t": "rewrite + terms"}},
                                              indent=1), encoding="utf-8")
    log(f"→ {out}, {out_t}")


def read_terms(cfg, lang, name=None):
    """{qid: (base, rewrite, [terms], [candidates])} từ qexp/<name>/terms_<lang>.parquet."""
    p = cfg.path("qexp") / (name or cfg["qexpand"].get("name", MODE)) / f"terms_{lang}.parquet"
    t = pq.read_table(p).to_pydict()
    cands = t.get("candidates") or [[]] * len(t["id"])
    return {int(q): (b, r, list(x or []), list(c or []))
            for q, b, r, x, c in zip(t["id"], t["base"], t["rewrite"], t["terms"], cands)}


# ---------------------------------------------------------------- BM25 có trọng số
class WeightedQueryEncoder:
    """Câu hỏi BM25 = base·w_base + rewrite·w_rewrite + từng thuật ngữ·w_terms + từng candidate·w_cand → (id từ, trọng
    số) như QueryEncoder. Mỗi lần một token xuất hiện trong một phần, token được cộng trọng số của phần đó (chỉ base,
    w_base = 1 → đúng QueryEncoder). Token của THUẬT NGỮ / CANDIDATE có df/n_docs > max_df_ratio bị bỏ (base / rewrite
    giữ nguyên). Tách từ theo cấu hình ghi trong stats.json của chỉ mục (bm25.QueryEncoder)."""

    def __init__(self, index_dir, vi_mode="pyvi", w_base=1.0, w_rewrite=1.0, w_terms=0.5, max_df_ratio=0.05,
                 w_cand=0.0):
        index_dir = Path(index_dir)
        self.qe = bm25.QueryEncoder(index_dir, vi_mode)
        df = pq.read_table(index_dir / "vocab.parquet", columns=["df"]).column("df").to_numpy()
        n = json.loads((index_dir / "stats.json").read_text(encoding="utf-8"))["n_docs"]
        self.df_ratio = df.astype(np.float64) / max(1, n)
        self.w_base, self.w_rewrite, self.w_terms = float(w_base), float(w_rewrite), float(w_terms)
        self.w_cand = float(w_cand)
        self.max_df_ratio = max_df_ratio

    def tokens(self, text, lang):
        """Tách từ đúng như chỉ mục (QueryEncoder.tokens theo stats.json; bm25.py cũ chưa có hàm này → vi_mode)."""
        if hasattr(self.qe, "tokens"):
            return self.qe.tokens(text, lang)
        return bm25.tokenize(text, lang, self.qe.vi_mode)

    def parts(self, base, rewrite=None, terms=(), candidates=()):
        ps = [("base", base, self.w_base)]
        if rewrite and self.w_rewrite:
            ps.append(("rewrite", rewrite, self.w_rewrite))
        if self.w_terms:
            ps += [("terms", t, self.w_terms) for t in terms or ()]
        if self.w_cand:
            ps += [("candidates", t, self.w_cand) for t in candidates or ()]
        return ps

    def weights(self, lang, base, rewrite=None, terms=(), candidates=()):
        """→ ({id từ: trọng số}, {token: [nguồn]}, [(token bị lọc, df/n)])."""
        c, src, dropped = {}, {}, []
        for kind, text, w in self.parts(base, rewrite, terms, candidates):
            for t in self.tokens(text, lang):
                j = self.qe.vocab.get(t)
                if j is None:
                    continue
                if kind in ("terms", "candidates") and self.max_df_ratio is not None \
                        and self.df_ratio[j] > self.max_df_ratio:
                    dropped.append((t, float(self.df_ratio[j])))
                    continue
                c[j] = c.get(j, 0.0) + w
                src.setdefault(t, []).append(kind)
        return c, src, dropped

    def __call__(self, lang, base, rewrite=None, terms=(), candidates=()):
        c = self.weights(lang, base, rewrite, terms, candidates)[0]
        ks = sorted(c)
        return np.array(ks, dtype=np.int64), np.array([c[k] for k in ks], dtype=np.float32)

    def explain(self, lang, base, rewrite=None, terms=(), candidates=()):
        """Bảng token: trọng số chỉ base (trước) và sau mở rộng, nguồn, df/n; + token thuật ngữ bị lọc."""
        before = self.weights(lang, base)[0]
        after, src, dropped = self.weights(lang, base, rewrite, terms, candidates)
        words = {j: w for w, j in self.qe.vocab.items() if j in after or j in before}
        rows = [{"token": words[j], "trước": before.get(j, 0.0), "sau": after.get(j, 0.0),
                 "nguồn": ",".join(sorted(set(src.get(words[j], [])))), "df/n": round(float(self.df_ratio[j]), 4)}
                for j in sorted(set(after) | set(before), key=lambda j: (-after.get(j, 0), words[j]))]
        return pd.DataFrame(rows), dropped


# ---------------------------------------------------------------- dense / LSR nhiều phiên bản
def _norm_text(s):
    return re.sub(r"\W+", " ", (s or "").casefold()).strip()


def query_variants(base, rewrite=None, terms=(), with_terms=False):
    """[base, rewrite] (bỏ rewrite rỗng hoặc trùng base); with_terms: thêm "rewrite ; thuật ngữ…" (LSR)."""
    vs = [base]
    if rewrite and _norm_text(rewrite) != _norm_text(base):
        vs.append(rewrite)
    if with_terms and terms:
        vs.append(" ; ".join([rewrite or base] + list(terms)))
    return vs


def fuse_variants(dfs, how="max", weights=None, depth=None):
    """Các run của cùng một nhánh (mỗi phiên bản câu hỏi một run: qid, unit_id, doc_id, rank, score) → một run.
    Điểm mỗi phiên bản chuẩn hoá min-max theo câu hỏi (như fuse.branch_scores), rồi lấy max (how="max") hoặc trung bình
    có trọng số (how="mean"; đoạn phiên bản không trả về = 0). Bằng điểm → thứ hạng tốt nhất trước."""
    dfs = [d for d in dfs if d is not None and len(d)]
    if not dfs:
        raise ValueError("fuse_variants: không có run")
    w = list(weights) if weights is not None else [1.0] * len(dfs)
    parts = []
    for d, wi in zip(dfs, w):
        g = d.groupby("qid")["score"]
        lo, hi = g.transform("min"), g.transform("max")
        s = (d["score"] - lo) / (hi - lo).where(hi > lo, 1.0)
        parts.append(d.assign(score=s * (wi / sum(w)) if how == "mean" else s)[["qid", "unit_id", "doc_id", "rank",
                                                                                   "score"]])
    allr = pd.concat(parts, ignore_index=True)
    if how not in ("max", "mean"):
        raise ValueError("how phải là max hoặc mean")
    g = allr.groupby(["qid", "unit_id", "doc_id"], as_index=False).agg(score=("score", "max" if how == "max" else "sum"),
                                                                       rank=("rank", "min"))
    return runs.rerank_by_score(g, depth)[["qid", "unit_id", "doc_id", "rank", "score"]]


# ---------------------------------------------------------------- xem kết quả
def show(cfg, model=None, limit=5, log=print):
    recs = load(cfg, model)
    if not recs:
        raise SystemExit("Chưa có bản ghi")
    for lg in LANGS:
        rs = [r for (q, l), r in recs.items() if l == lg]
        if rs:
            bad = [r for r in rs if not r.get("ok")]
            long_ = sum(r["n_words"] > (90 if lg == "zh" else 60) for r in rs)
            log(f"[{lg}] {len(rs)} lượt, lỗi {len(bad)} ({', '.join(sorted({r['why'] for r in bad})) or '—'}), "
                f"rewrite quá dài: {long_}, TB {np.mean([len(r['terms']) for r in rs]):.1f} thuật ngữ, "
                f"{sum(bool(r.get('candidates')) for r in rs)} lượt có candidates, "
                f"{np.mean([r['sec'] for r in rs]):.1f}s/lượt")
    for q in sorted({q for q, _ in recs})[:limit]:
        log(f"\n=== câu {q}")
        for lg in LANGS:
            r = recs.get((q, lg))
            if r:
                log(f"[{lg}] base: {r['base']}")
                log(json.dumps({k: r.get(k) for k in FIELDS + ("ok", "why")}, ensure_ascii=False, indent=1))
                if not r["ok"]:
                    log(f"    thô: {r['raw'][:300]!r}")


def explain(cfg, qid=None, name=None, log=print):
    from .index import bm25_dir
    xc = cfg["qexpand"]
    for lg in LANGS:
        try:
            t = read_terms(cfg, lg, name)
        except FileNotFoundError:
            log(f"[{lg}] chưa build")
            continue
        q = qid if qid is not None else min(t)
        if q not in t:
            log(f"[{lg}] không có câu {q}")
            continue
        d = bm25_dir(cfg, cfg["bm25"]["levels"][0], lg)
        if not (d / "vocab.parquet").exists():
            log(f"[{lg}] thiếu chỉ mục BM25 {d}")
            continue
        enc = WeightedQueryEncoder(d, cfg["bm25"]["vi_tokenizer"], xc["w_base"], xc["w_rewrite"], xc["w_terms"],
                                   xc["max_df_ratio"], xc.get("w_cand", 0.0))
        base, rw, terms, cands = t[q]
        tab, dropped = enc.explain(lg, base, rw, terms, cands)
        tok = getattr(enc.qe, "tok", {"vi_mode": enc.qe.vi_mode})
        log(f"\n=== câu {q} [{lg}] chỉ mục {d.name}, tách từ {tok}\nbase: {base}\nrewrite: {rw}\nterms: {terms}"
            f"\ncandidates (w_cand={enc.w_cand}): {cands}")
        with pd.option_context("display.max_rows", 200, "display.width", 160):
            log(tab.to_string(index=False))
        log(f"tổng trọng số trước {tab['trước'].sum():.1f} → sau {tab['sau'].sum():.1f}; "
            f"token thuật ngữ bị lọc (df/n > {xc['max_df_ratio']}): {sorted(set(dropped))}")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "build", "show", "explain"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="tên trong translate.models hoặc id Hugging Face (mặc định qexpand.model)")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--qids", default=None, help="chỉ các câu này: file qid (.txt/.csv/.parquet) hoặc '1,2,3'")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="generate: chỉ N câu đầu; show: số câu in ra (mặc định 5)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--name", default=None, help="tên biến thể trong qexp/ (mặc định qexpand.name)")
    ap.add_argument("--qid", type=int, default=None, help="explain: câu cần in (mặc định câu nhỏ nhất)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "generate":
        generate(cfg, a.model, a.langs.split(","), runs.read_qids(a.qids), a.shard, a.limit, a.batch, a.quant)
    elif a.cmd == "build":
        build(cfg, a.model, a.name)
    elif a.cmd == "show":
        show(cfg, a.model, a.limit or 5)
    else:
        explain(cfg, a.qid, a.name)


if __name__ == "__main__":
    main()
