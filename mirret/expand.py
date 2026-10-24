"""Mở rộng câu hỏi bằng tài liệu giả do LLM sinh (generation-augmented retrieval) — task B1/B2 của lộ trình 20–31/10.

Hai chế độ, cài như R2MED (2025, bảng 3 + phụ lục E): mỗi câu hỏi một đoạn giả, sinh tất định (greedy):
  q2d  : Query2Doc — prompt few-shot (2 ví dụ), truy hồi bằng CÂU HỎI GHÉP ĐOẠN GIẢ
         (dense/LSR: "câu hỏi\\nđoạn giả"; BM25: câu hỏi lặp expand.bm25_repeat lần + đoạn giả, như Query2doc).
  hyde : HyDE — prompt zero-shot, truy hồi bằng RIÊNG ĐOẠN GIẢ (dense/LSR/BM25 đều dùng đoạn giả).
So sánh q2d với hyde = so cách dùng đoạn giả (ghép câu hỏi hay thay câu hỏi). Generative QE for CLIR (2025): câu hỏi dài
(như bộ này) thì ghép câu hỏi tốt hơn hẳn chỉ dùng đoạn giả.
Thêm biến thể không cần LLM:
  bm25c: BM25 với câu hỏi + các khái niệm y khoa (concepts) trong bản dịch t2 — chỉ nhánh BM25.

Đoạn giả viết THẲNG bằng ngôn ngữ của nhánh, từ câu hỏi của nhánh đó (vi gốc, en/zh = bản dịch đang dùng trong
qtrans/queries_{en,zh}.parquet). Reranker vẫn dùng câu hỏi gốc / bản dịch, không dùng đoạn giả.

  python -m mirret.expand generate --mode q2d [--model qwen25_14b] [--qids queries.csv] [--langs vi,en,zh]
                                   [--shard 0/2] [--limit 5] [--batch 8] [--quant 4bit]
         → qexp/<mode>-<model>/records*.jsonl  (chạy lại = chỉ sinh phần còn thiếu)
  python -m mirret.expand build --mode q2d|hyde|bm25c [--model qwen25_14b] [--name q2d]
         → qexp/<name>/queries_{vi,en,zh}.parquet (+ queries_<lang>_bm25.parquet), samples.md
  python -m mirret.retrieve qencode --variant q2d ; python -m mirret.retrieve run --variant q2d
         → runs/q2d_<lang>_<method>.parquet ; fuse.configs: q2d9, hyde9, q2dlex, bm25c9
"""
import argparse
import difflib
import json
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

from . import config, runs, translate

LANGS = ("vi", "en", "zh")
MODES = ("q2d", "hyde")

SYSTEM = {
    "vi": "Bạn là chuyên gia y khoa. Chỉ viết đoạn văn bằng tiếng Việt, không hỏi lại.",
    "en": "You are a medical expert. Write only the passage, in English. Do not ask questions back.",
    "zh": "你是医学专家。只用简体中文写这段文字，不要反问。",
}
# Cùng một chỉ dẫn cho cả hai chế độ (R2MED: "Please write a passage to help answer the … post"), khác nhau ở ví dụ
INSTRUCT = {
    "vi": "Hãy viết một đoạn văn (khoảng 100–150 từ) giúp trả lời câu hỏi y khoa sau.\nCâu hỏi: {q}\nĐoạn văn:",
    "en": "Please write a passage (about 100–150 words) to help answer the following medical question.\n"
          "Question: {q}\nPassage:",
    "zh": "请写一段文字（约150–200字）帮助回答以下医学问题。\n问题：{q}\n段落：",
}
# Ví dụ cho q2d (R2MED: 2 ví dụ trong miền). Câu hỏi tự soạn, không lấy từ bộ câu hỏi của BTC.
EXAMPLES = [
    {"vi": ("Bị viêm dạ dày nên ăn gì và cần kiêng những gì?",
            "Người bị viêm dạ dày nên ăn các bữa nhỏ, chia nhiều lần trong ngày, ưu tiên thức ăn mềm, dễ tiêu như cháo, "
            "cơm nát, khoai, chuối, sữa chua, thịt cá luộc hoặc hấp. Cần hạn chế đồ cay nóng, chua, nhiều dầu mỡ, đồ "
            "chiên rán, cà phê, rượu bia và nước có ga vì chúng kích thích tiết acid và làm tổn thương niêm mạc dạ dày "
            "nặng hơn. Không nên ăn quá no, ăn khuya hoặc nằm ngay sau ăn. Bỏ thuốc lá và tránh tự dùng thuốc giảm đau "
            "nhóm NSAID. Nếu đau kéo dài, nôn ra máu hoặc đi ngoài phân đen, cần đi khám chuyên khoa tiêu hoá để nội "
            "soi, xét nghiệm vi khuẩn H. pylori và điều trị."),
     "en": ("What should I eat and avoid if I have gastritis?",
            "People with gastritis should eat small, frequent meals and choose soft, easily digestible foods such as "
            "porridge, well-cooked rice, potatoes, bananas, yogurt, and boiled or steamed fish and meat. Spicy, sour, "
            "fatty and fried foods, coffee, alcohol and carbonated drinks should be limited because they stimulate acid "
            "secretion and further irritate the stomach lining. Avoid overeating, late-night meals and lying down right "
            "after eating. Stop smoking and avoid self-medicating with NSAID painkillers. If pain persists or there is "
            "vomiting of blood or black stools, see a gastroenterologist for endoscopy, Helicobacter pylori testing and "
            "treatment."),
     "zh": ("胃炎患者应该吃什么，需要忌口什么？",
            "胃炎患者宜少量多餐，选择柔软、易消化的食物，如粥、软饭、土豆、香蕉、酸奶以及水煮或清蒸的鱼肉。应少吃辛辣、"
            "酸性、油腻和油炸食物，限制咖啡、酒精和碳酸饮料，因为它们会刺激胃酸分泌，加重胃黏膜损伤。避免暴饮暴食、吃夜宵"
            "或饭后立即平躺。应戒烟，避免自行服用非甾体抗炎止痛药（NSAID）。如果疼痛持续，或出现呕血、黑便，应及时到消化科"
            "就诊，进行胃镜检查、幽门螺杆菌检测和治疗。")},
    {"vi": ("Trẻ 2 tuổi sốt 39 độ, khi nào cần đưa đi bệnh viện?",
            "Sốt 39°C ở trẻ 2 tuổi thường do nhiễm virus và có thể chăm sóc tại nhà: cho trẻ uống paracetamol đúng liều "
            "theo cân nặng (10–15 mg/kg mỗi 4–6 giờ), uống nhiều nước, mặc thoáng và lau người bằng nước ấm. Cần đưa trẻ "
            "đi khám ngay nếu trẻ co giật, li bì khó đánh thức, thở nhanh hoặc khó thở, nôn liên tục, không chịu uống, "
            "tiểu ít, nổi ban xuất huyết không mất khi ấn, cổ cứng, hoặc sốt kéo dài quá 2–3 ngày. Trẻ có bệnh nền hoặc "
            "sốt cao không đáp ứng với thuốc hạ sốt cũng nên được bác sĩ thăm khám để tìm nguyên nhân như viêm tai giữa, "
            "viêm phổi, nhiễm trùng tiểu hay sốt xuất huyết."),
     "en": ("My 2-year-old has a fever of 39°C. When should I take them to the hospital?",
            "A 39°C fever in a 2-year-old is usually caused by a viral infection and can often be managed at home: give "
            "paracetamol at the correct weight-based dose (10–15 mg/kg every 4–6 hours), offer plenty of fluids, dress "
            "the child lightly and sponge with lukewarm water. Seek medical care immediately if the child has a seizure, "
            "is unusually drowsy or hard to wake, breathes fast or with difficulty, vomits repeatedly, refuses to drink, "
            "passes little urine, develops a rash that does not fade when pressed, has a stiff neck, or the fever lasts "
            "more than 2–3 days. Children with underlying conditions or a high fever that does not respond to "
            "antipyretics should also be examined for causes such as otitis media, pneumonia, urinary tract infection "
            "or dengue fever."),
     "zh": ("2岁孩子发烧39度，什么时候需要去医院？",
            "2岁儿童发烧39°C多由病毒感染引起，通常可以在家护理：按体重给予正确剂量的对乙酰氨基酚（每次10–15 mg/kg，"
            "间隔4–6小时），多喝水，穿着宽松，并用温水擦身。如果孩子出现抽搐、精神萎靡难以唤醒、呼吸急促或困难、反复呕吐、"
            "拒绝饮水、尿量减少、按压不褪色的皮疹、颈部僵硬，或发烧超过2–3天，应立即就医。有基础疾病或高烧对退烧药无反应"
            "的孩子也应及时就诊，以排查中耳炎、肺炎、尿路感染或登革热等病因。")},
]
_LABEL = re.compile(r"^\s*(?:Đoạn văn|Passage|段落)\s*[:：]\s*", re.I)


def records_dir(cfg, mode, model_key):
    return cfg.path("qexp") / f"{mode}-{model_key}"


def chat(tok, mode, lang, query):
    msgs = [{"role": "system", "content": SYSTEM[lang]}]
    if mode == "q2d":
        for ex in EXAMPLES:
            q, p = ex[lang]
            msgs += [{"role": "user", "content": INSTRUCT[lang].format(q=q)}, {"role": "assistant", "content": p}]
    msgs.append({"role": "user", "content": INSTRUCT[lang].format(q=query)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def clean(text):
    s = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    s = _LABEL.sub("", s).strip().strip('"“”').strip()
    return re.sub(r"\n{2,}", "\n", s)


def looping(text):
    """Lặp câu (lỗi hay gặp khi sinh greedy): ≥ 4 câu mà dưới 60% câu khác nhau."""
    sents = [x.strip() for x in re.split(r"[.!?。！？\n]+", text) if len(x.strip()) >= 4]
    return len(sents) >= 4 and len(set(sents)) < 0.6 * len(sents)


def check(doc, lang, mode):
    """→ (ok, lý do)."""
    if len(doc) < (30 if lang == "zh" else 60):
        return False, "ngắn"
    if lang == "vi":
        if not translate._VI.search(doc) or translate._HAN.search(doc):
            return False, "sai ngôn ngữ"
    elif not translate.script_ok(doc, lang):
        return False, "sai ngôn ngữ"
    if looping(doc):
        return False, "lặp"
    if mode == "q2d" and any(difflib.SequenceMatcher(None, doc, ex[lang][1]).ratio() > 0.8 for ex in EXAMPLES):
        return False, "chép ví dụ"
    return True, ""


def queries(cfg, lang):
    """{qid: câu hỏi} của nhánh lang (qtrans/queries_<lang>.parquet — cùng bản dịch với baseline)."""
    p = cfg.path("qtrans") / f"queries_{lang}.parquet"
    if not p.exists():
        return {}
    t = pq.read_table(p).to_pydict()
    return dict(zip([int(x) for x in t["id"]], t["query"]))


def load_records(cfg, mode, model_key, ok_only=True):
    """{(qid, lang): bản ghi}; bản hợp lệ được ưu tiên, rồi bản sau cùng."""
    done = {}
    for f in sorted(records_dir(cfg, mode, model_key).glob("records*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if ok_only and not r.get("ok"):
                continue
            k = (int(r["id"]), r["lang"])
            if not (k in done and done[k].get("ok") and not r.get("ok")):
                done[k] = r
    return done


# ---------------------------------------------------------------- sinh (GPU)
def generate(cfg, mode, model=None, langs=LANGS, qids=None, shard="0/1", limit=0, batch=None, quant=None, log=print):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    xc = cfg["expand"]
    model = model or xc["model"]
    mid, mkey = translate.model_id(cfg, model), translate.model_key(cfg, model)
    items = []
    for lg in langs:
        qs = queries(cfg, lg)
        if not qs:
            log(f"  [{lg}] thiếu qtrans/queries_{lg}.parquet → bỏ qua")
        items += [(q, lg, t) for q, t in sorted(qs.items()) if qids is None or q in qids]
    if limit:
        keep = sorted({q for q, _, _ in items})[:limit]
        items = [x for x in items if x[0] in keep]
    si, sn = (int(x) for x in shard.split("/"))
    mine = sorted({q for q, _, _ in items})[si::sn]               # chia theo câu hỏi: 3 ngôn ngữ của một câu cùng phần
    items = [x for x in items if x[0] in set(mine)]
    out = records_dir(cfg, mode, mkey) / ("records.jsonl" if sn == 1 else f"records-{si}of{sn}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_records(cfg, mode, mkey)
    tried = set(load_records(cfg, mode, mkey, ok_only=False))
    todo = sorted([x for x in items if (x[0], x[1]) not in done], key=lambda x: -len(x[2]))
    log(f"[{mode}-{mkey}] phần {shard}: {len(items)} lượt ({len(mine)} câu × {len(langs)} ngôn ngữ), "
        f"đã có {len(items) - len(todo)}, còn {len(todo)}")
    if not todo:
        return
    tok, llm = translate.load_llm(mid, quant or cfg["translate"].get("quant"), log)

    def record(item, raw, sample, sec):
        q, lg, t = item
        doc = clean(raw)
        ok, why = check(doc, lg, mode)
        return {"id": q, "lang": lg, "mode": mode, "query": t, "doc": doc, "ok": ok, "why": why, "raw": raw,
                "model": mid, "sampled": sample, "sec": round(sec, 2)}

    generate_loop(tok, llm, todo, out, lambda x: chat(tok, mode, x[1], x[2]), record, int(batch or xc["batch"]),
                  int(xc["max_new_tokens"]), tried=tried, retry_sample=True, log=log)


def generate_loop(tok, llm, todo, out, prompt, record, batch, max_new, tried=(), retry_sample=True,
                  repetition_penalty=1.05, log=print):
    """Sinh theo lô cho các mục todo = [(qid, lang, câu hỏi)], ghi nối từng bản ghi vào out (jsonl) ngay sau mỗi lô.
    prompt(mục) → chuỗi đầu vào; record(mục, đầu ra thô, có lấy mẫu?, s/lượt) → bản ghi (có khoá "ok").
    Sinh tất định (greedy); retry_sample: lượt 2 lấy mẫu lại cho mục hỏng (cả mục hỏng ở phiên trước, tập tried).
    repetition_penalty (lượt greedy; 1.0 = tắt): transformers phạt cả token của PROMPT — hợp với đoạn văn dài (q2d/HyDE,
    chống lặp câu), hại khi cần chép lại từ của câu hỏi (query_expand).
    Hết bộ nhớ GPU → giảm lô một nửa. Dùng chung cho expand (q2d/HyDE) và query_expand (rewrite + thuật ngữ)."""
    import torch
    greedy = {"do_sample": False}
    if repetition_penalty and repetition_penalty != 1.0:
        greedy["repetition_penalty"] = repetition_penalty

    def gen(group, sample):
        enc = tok([prompt(x) for x in group], return_tensors="pt", padding=True).to(llm.device)
        with torch.no_grad():
            g = llm.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                             **({"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample else greedy))
        return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    log(f"  prompt mẫu ~{len(tok(prompt(todo[0]))['input_ids'])} token, tối đa {max_new} token sinh")
    t0, n_ok, n_done, bs = time.time(), 0, 0, batch
    failed_now = []
    passes = [([x for x in todo if (x[0], x[1]) not in tried], False)]
    if retry_sample:                                        # lượt 2: lấy mẫu cho câu hỏng
        passes.append((lambda: [x for x in todo if (x[0], x[1]) in tried] + failed_now, True))
    for group, sample in passes:
        group = group() if callable(group) else group
        if sample and group:
            log(f"  sinh lại có lấy mẫu {len(group)} lượt hỏng")
        k = 0
        while k < len(group):
            b = group[k:k + bs]
            t1 = time.time()
            try:
                texts = gen(b, sample)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log(f"  [CUDA hết bộ nhớ] giảm lô xuống {bs}")
                continue
            sec = (time.time() - t1) / len(b)
            with out.open("a", encoding="utf-8") as f:
                for x, raw in zip(b, texts):
                    r = record(x, raw, sample, sec)
                    n_ok += bool(r["ok"])
                    if not r["ok"] and not sample:
                        failed_now.append(x)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            k += len(b)
            n_done += len(b)
            log(f"  {n_done} lượt, hợp lệ {n_ok}, lô {bs}, {time.time() - t0:.0f}s ({sec:.1f}s/lượt ở lô này)")
    return n_ok


# ---------------------------------------------------------------- dựng file câu hỏi cho retrieve --variant
def write_queries(path, rows, lang):
    pq.write_table(pa.table({"id": pa.array([q for q, _ in rows], pa.int64()), "query": [t for _, t in rows],
                             "lang": [lang] * len(rows)}), path)


def build(cfg, mode, model=None, name=None, use=None, log=print):
    xc = cfg["expand"]
    name = name or mode
    out = cfg.path("qexp") / name
    out.mkdir(parents=True, exist_ok=True)
    samples = [f"# {name}\n"]
    if mode == "bm25c":
        key = translate.resolve_key(cfg, use or cfg["translate"]["use"])
        recs = translate.load_records(cfg, key)
        if not recs:
            raise SystemExit(f"bm25c cần bản ghi dịch (concepts) trong {translate.records_dir(cfg, key)}")
        for lg in LANGS:
            qs = queries(cfg, lg)
            rows = []
            for q, t in sorted(qs.items()):
                terms = [str(c.get(lg, "")).strip() for c in (recs.get(q, {}).get("concepts") or [])]
                rows.append((q, " ".join([t] + [x for x in terms if x])))
            write_queries(out / f"queries_{lg}_bm25.parquet", rows, lg)
            log(f"  {name}/queries_{lg}_bm25: {len(rows)} câu, "
                f"{sum(1 for q in qs if recs.get(q, {}).get('concepts'))} câu có concepts")
            samples += [f"## {lg}"] + [f"- {q}: {t}" for q, t in rows[:3]]
    else:
        model = model or xc["model"]
        mkey = translate.model_key(cfg, model)
        recs = load_records(cfg, mode, mkey)
        if not recs:
            raise SystemExit(f"Chưa có đoạn giả hợp lệ trong {records_dir(cfg, mode, mkey)}")
        qids = sorted({q for q, _ in recs})
        rep = int(xc["bm25_repeat"])
        for lg in LANGS:
            qs = queries(cfg, lg)
            dense, sparse, miss = [], [], []
            for q in qids:
                if q not in qs:
                    continue
                t, r = qs[q], recs.get((q, lg))
                if r is None:                                   # thiếu đoạn giả → dùng câu hỏi (giữ câu trong fusion)
                    miss.append(q)
                    d, s = t, t
                elif mode == "q2d":
                    d, s = f"{t}\n{r['doc']}", " ".join([t] * rep + [r["doc"]])
                else:
                    d, s = r["doc"], r["doc"]
                dense.append((q, d))
                sparse.append((q, s))
            write_queries(out / f"queries_{lg}.parquet", dense, lg)
            write_queries(out / f"queries_{lg}_bm25.parquet", sparse, lg)
            log(f"  {name}/queries_{lg}: {len(dense)} câu" + (f", {len(miss)} câu thiếu đoạn giả (dùng câu hỏi): "
                                                              f"{miss[:8]}" if miss else ""))
            samples.append(f"## {lg}")
            for q in qids[:3]:
                r = recs.get((q, lg))
                if r:
                    samples.append(f"**{q}** {qs.get(q, '')}\n\n> {r['doc']}\n")
    (out / "samples.md").write_text("\n".join(samples), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"mode": mode, "model": model, "bm25_repeat": xc["bm25_repeat"]},
                                              indent=1), encoding="utf-8")
    log(f"→ {out} (xem mẫu: {out / 'samples.md'})")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "build"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", required=True, choices=list(MODES) + ["bm25c"])
    ap.add_argument("--model", default=None, help="tên trong translate.models hoặc id Hugging Face (mặc định expand.model)")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--qids", default=None, help="chỉ các câu này: file qid (.txt/.csv/.parquet) hoặc '1,2,3'")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N câu đầu (chạy thử)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--name", default=None, help="build: tên biến thể (mặc định = mode)")
    ap.add_argument("--use", default=None, help="build bm25c: bản dịch lấy concepts (mặc định translate.use)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "generate":
        if a.mode not in MODES:
            raise SystemExit("bm25c không cần sinh: chạy thẳng `build --mode bm25c`")
        generate(cfg, a.mode, a.model, a.langs.split(","), runs.read_qids(a.qids), a.shard, a.limit, a.batch,
                 a.quant)
    else:
        build(cfg, a.mode, a.model, a.name, a.use)


if __name__ == "__main__":
    main()
