"""Dịch 1.200 câu hỏi vi → en → zh bằng LLM mở (<= 14B, phát hành trước 1/6/2026) và so sánh hai model.

Prompt "t1" = prompt v2 của dự án cũ (mir/expand.py, đã sửa sau khi đọc tay 20 câu: lập bảng khái niệm vi→en→zh
trước, dịch tiếng Anh trước rồi tiếng Trung, quy tắc giữ chủ thể / số liệu / triệu chứng / tên thuốc, 2 ví dụ tự soạn)
nhưng chỉ còn phần dịch. Sinh tất định (greedy); câu JSON hỏng được sinh lại có lấy mẫu.

  python -m mirret.translate generate --model qwen25_14b [--shard 0/2] [--limit 20]   # GPU, 4bit, mỗi GPU một phần
  python -m mirret.translate import-old --qexp <thư mục qexp cũ> [--name old_qwen25_14b]   # dùng bản dịch cũ (v2)
  python -m mirret.translate compare --a qwen25_14b --b qwen3_8b [--n 100]     # chỉ số tự động + phiếu chấm mù
  python -m mirret.translate score --sheet <phiếu đã chấm.csv> [--sheet …]     # kết quả chấm của người
  python -m mirret.translate build [--use qwen25_14b]                          # → qtrans/queries_{vi,en,zh}.parquet

Model: tên ngắn trong translate.models (vd qwen25_14b, qwen3_8b) hoặc id Hugging Face.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

PROMPT_VERSION = "t1"
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
        return None
    if not isinstance(d, dict):
        return None

    def one(x):
        if isinstance(x, list):
            x = " ".join(str(v) for v in x if isinstance(v, (str, int, float)))
        return str(x).strip() if isinstance(x, (str, int, float)) else ""
    concepts = [c for c in d.get("concepts") or [] if isinstance(c, dict)][:6]
    return {"concepts": concepts, "en": one(d.get("en")), "zh": one(d.get("zh"))}


def script_ok(text, lang):
    """zh phải chủ yếu là chữ Hán; en không có chữ Hán, không còn chữ tiếng Việt có dấu."""
    if not text:
        return False
    han = len(_HAN.findall(text))
    if lang == "zh":
        return han >= max(2, 0.3 * len(text.replace(" ", "")))
    return han == 0 and not _VI.search(text)


def numbers_kept(src, dst):
    """Mọi con số trong câu gốc có mặt trong câu dịch (so theo chữ số; "1,5" ~ "1.5")."""
    norm = lambda x: x.replace(",", ".")
    have = {norm(x) for x in _NUM.findall(dst or "")}
    need = {norm(x) for x in _NUM.findall(src or "")}
    return need <= have


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
def generate(cfg, key, shard="0/1", limit=0, log=print):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tc = cfg["translate"]
    mid = model_id(cfg, key)
    key = model_key(cfg, key)
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
    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    kw = {}
    if tc.get("quant") in ("4bit", "8bit") and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                        bnb_4bit_quant_type="nf4")
                                     if tc["quant"] == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
        kw["max_memory"] = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.8)
                            for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, **kw).eval()

    def chat(q):
        msgs = [{"role": "system", "content": SYSTEM}]
        for eq, ea in EXAMPLES:
            msgs += [{"role": "user", "content": PROMPT.format(query=eq)},
                     {"role": "assistant", "content": json.dumps(ea, ensure_ascii=False)}]
        msgs.append({"role": "user", "content": PROMPT.format(query=q)})
        # enable_thinking=False tắt chế độ suy nghĩ của Qwen3; model khác bỏ qua tham số này
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def gen(batch, sample):
        enc = tok([chat(t) for _, t in batch], return_tensors="pt", padding=True).to(model.device)
        qlen = max(len(tok(t)["input_ids"]) for _, t in batch)
        max_new = int(min(1536, max(tc["max_new_tokens"], 160 + 3 * qlen)))   # câu dài nhiều ý → bản dịch dài
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                               **({"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample
                                  else {"do_sample": False}))
        return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    t0, n_ok, n_done, bs = time.time(), 0, 0, tc["batch"]
    for group, sample in (([x for x in todo if x[0] not in tried], False), ([x for x in todo if x[0] in tried], True)):
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
                    ok = bool(d and d["en"] and d["zh"])
                    n_ok += ok
                    f.write(json.dumps({"id": q, "query": t, "ok": ok, **(d or {}), "raw": raw, "model": mid,
                                        "prompt": PROMPT_VERSION, "sampled": sample, "sec": round(sec, 2)},
                                       ensure_ascii=False) + "\n")
            k += len(batch)
            n_done += len(batch)
            log(f"  {n_done}/{len(todo)} câu, hợp lệ {n_ok}, lô {bs}, {time.time() - t0:.0f}s")


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
    key = model_key(cfg, use or cfg["translate"]["use"])
    qids, qtexts = vi_queries(cfg)
    done = load_records(cfg, key)
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
def auto_metrics(qids, qtexts, recs):
    """Chỉ số đo trực tiếp được (không cần người): tỷ lệ hợp lệ, đúng chữ viết, giữ số, độ dài."""
    n = len(qids)
    m = {"n": n, "ok": 0, "en_script": 0, "zh_script": 0, "en_numbers": 0, "zh_numbers": 0, "en_len_ratio": [],
         "zh_len_ratio": [], "en_too_short": 0, "zh_too_short": 0, "sec": []}
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


def compare(cfg, a, b, n=100, log=print):
    ka, kb = model_key(cfg, a), model_key(cfg, b)
    qids, qtexts = vi_queries(cfg)
    ra, rb = load_records(cfg, ka), load_records(cfg, kb)
    ma, mb = auto_metrics(qids, qtexts, ra), auto_metrics(qids, qtexts, rb)
    pct = lambda m, k: f"{100 * m[k] / max(1, m['n']):.1f}%"
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
         f"| tỷ lệ độ dài en/vi (trung vị) | {ma['en_len_ratio']} | {mb['en_len_ratio']} |",
         f"| tỷ lệ độ dài zh/vi (trung vị) | {ma['zh_len_ratio']} | {mb['zh_len_ratio']} |",
         f"| giây/câu (trung vị) | {ma['sec']} | {mb['sec']} |", ""]
    # khác biệt giữa hai model: câu mà hai bản dịch khác nhau nhiều (theo ký tự) → ưu tiên cho người đọc
    import difflib
    both = [q for q in qids if q in ra and q in rb]
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
    ap.add_argument("--qexp", default=None)
    ap.add_argument("--name", default="old_qwen25_14b")
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sheet", action="append", default=[])
    ap.add_argument("--use", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "generate":
        generate(cfg, a.model or cfg["translate"]["use"], a.shard, a.limit)
    elif a.cmd == "import-old":
        import_old(cfg, a.qexp, a.name)
    elif a.cmd == "compare":
        compare(cfg, a.a, a.b, a.n)
    elif a.cmd == "score":
        score(cfg, a.sheet)
    else:
        build(cfg, a.use)


if __name__ == "__main__":
    main()
