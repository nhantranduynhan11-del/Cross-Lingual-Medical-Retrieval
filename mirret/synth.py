"""Dữ liệu fine-tune reranker không dùng đáp án hay câu thi: câu hỏi tổng hợp + ứng viên khó + nhãn của teacher.

Ràng buộc: teacher > 14B chỉ dùng lúc huấn luyện → chỉ chấm câu hỏi TỔNG HỢP (sinh từ đoạn lấy ngẫu nhiên trong kho),
không chấm 1.200 câu thi (như vậy là chạy teacher lúc suy luận). Không dùng đáp án public, không dùng bộ nhãn 80 câu.

Lựa chọn theo tài liệu (thư mục AI GURU STAGE 3):
  - AR-Med (2025): Qwen3-32B là teacher chính xác nhất trong các model thử (hơn cả 70B); tắt chế độ suy nghĩ (suy nghĩ
    dài gây "overthinking", chấm sai); thang nhiều mức với quy tắc rõ ràng → thang 0/1/2 của docs/EVAL_PLAN.md §2.3.
  - NV-Retriever (2024) / MedCPT (2023) / Hard-negative mining (ACL 2025): âm khó lấy từ đầu danh sách truy hồi nhưng
    phải lọc âm giả (false negative) — ở đây teacher chấm từng ứng viên nên âm = teacher chắc chắn 0; thêm âm "xa"
    (hạng 50–200, như MedCPT re-ranker) và bỏ bản gần trùng của đoạn nguồn (tiêu chí d(PD, D) của bài ACL 2025).

Luồng (Kaggle):
  k8a  python -m mirret.synth sample     → <work>/synth/source.parquet (đoạn nguồn ngẫu nhiên, vi/zh/en theo --n)
       python -m mirret.synth generate   → <work>/synth/gen.jsonl (vLLM, câu hỏi vi + bản dịch en/zh)
       python -m mirret.synth build      → <work>/qexp/synth/queries_{vi,en,zh}.parquet (id từ 100001)
  k7   MODES=['synth'], FUSED=['synth_lin07'], RERANK=False → runs/synth_lin07.parquet
  k8b  python -m mirret.synth candidates → <work>/synth/pairs.parquet
       python -m mirret.synth label      → <work>/synth/labels-<phần>.parquet (P(0), P(1), P(2) từ logprob của teacher)
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config, runs

QID0 = 100001                                                 # câu hỏi tổng hợp không trùng id câu thi (1..1200)
LANG_NAME = {"vi": "tiếng Việt", "en": "tiếng Anh", "zh": "tiếng Trung"}

# phần cố định đặt TRƯỚC (system) để vLLM dùng lại bộ đệm tiền tố cho mọi lượt; phần thay đổi (đoạn, câu hỏi) đặt sau
GEN_SYSTEM = """Bạn tạo dữ liệu huấn luyện cho công cụ tìm kiếm y khoa tiếng Việt. Với mỗi đoạn tài liệu được đưa, hãy \
viết MỘT câu hỏi mà người dùng Việt Nam (bệnh nhân, người nhà) có thể gõ vào công cụ tìm kiếm sức khoẻ, sao cho đoạn đó \
trả lời được câu hỏi.
Yêu cầu:
- Tự nhiên như người thật hỏi: có thể kể ngắn hoàn cảnh (tuổi, triệu chứng, thuốc đang dùng) rồi hỏi; 10–60 từ.
- Hỏi về nội dung chính của đoạn, không hỏi chi tiết vụn vặt; không chép nguyên tiêu đề; không nhắc "đoạn văn", "bài viết".
- Câu tiếng Việt chỉ dùng tiếng Việt (tên thuốc, thuật ngữ quốc tế viết bằng chữ Latinh), không chèn chữ Hán.
BỎ QUA đoạn (trả về {"skip": true}) nếu người bệnh / người nhà sẽ không tìm đọc nó để giải đáp thắc mắc sức khoẻ: tin thời \
sự, chính trị, xã hội, thể thao, giải trí; tin về sự kiện, số liệu trong ngày, nhân sự, chính sách; quảng cáo; nghiên cứu \
trên động vật hoặc tế bào, thí nghiệm quá chuyên sâu mà bệnh nhân không hỏi tới.
Ví dụ về văn phong (không liên quan đến đoạn được đưa):
1. Bé nhà em 2 tuổi bị sốt 39 độ kèm phát ban đỏ ở bụng, có cần đưa đi viện ngay không?
2. Người bị tiểu đường type 2 có ăn được chuối không, ăn bao nhiêu là đủ?
3. Tôi uống thuốc tránh thai khẩn cấp xong bị chậm kinh 1 tuần thì có sao không?
4. Ung thư gan giai đoạn đầu có những phương pháp điều trị nào?
Trả về đúng một dòng JSON: {"vi": "<câu hỏi tiếng Việt>", "en": "<bản dịch tiếng Anh>", "zh": "<bản dịch tiếng Trung giản thể>"} \
hoặc {"skip": true}"""
GEN_PROMPT = """Đoạn tài liệu ({lang}):
Tiêu đề: {ctx}
Nội dung: {text}"""

LABEL_SYSTEM = """Bạn là chuyên gia y khoa đánh giá mức liên quan giữa câu hỏi và đoạn tài liệu cho hệ thống tìm kiếm.
Thang chấm:
2 — trả lời trực tiếp: đoạn chứa câu trả lời / thông tin chính cần tìm, đủ để trả lời ý chính của câu hỏi.
1 — liên quan một phần: chỉ trả lời một ý của câu hỏi nhiều ý; hoặc có thông tin then chốt nhưng chưa đủ; hoặc thông \
tin chung áp dụng được cho trường hợp được hỏi.
0 — không liên quan: khác chủ đề; CÙNG CHỦ ĐỀ NHƯNG KHÔNG TRẢ LỜI câu hỏi; menu, quảng cáo, trang thông báo; đoạn chỉ \
chép lại câu hỏi; danh sách tiêu đề bài khác.
Quy tắc: không chấm theo việc trùng từ khoá; ngôn ngữ của đoạn không ảnh hưởng; câu hỏi nêu đối tượng cụ thể (bà bầu, \
trẻ nhỏ, người suy thận…) mà đoạn nói về đối tượng khác và thông tin không áp dụng được → 0; không chấm độ đúng y khoa; \
không chắc → chọn mức thấp hơn.
Chỉ trả lời một chữ số: 0, 1 hoặc 2."""
LABEL_PROMPT = """Câu hỏi: {q}{qx}

Đoạn tài liệu ({lang}):
{ctx}
{text}"""


def sdir(cfg):
    return Path(cfg["work_dir"]) / "synth"


def cut(text, lang, n_vi=1800, n_zh=900):
    text = (text or "").strip()
    n = n_zh if lang == "zh" else n_vi
    return text if len(text) <= n else text[:n] + "…"


# ---------------------------------------------------------------- k8a
def cmd_sample(cfg, n_spec, seed=42, log=print):
    """Đoạn nguồn: mỗi ngôn ngữ n tài liệu ngẫu nhiên (bỏ bản trùng, bài rỗng / quá ngắn), mỗi tài liệu 1 đoạn cha ≥ 100
    token chọn ngẫu nhiên."""
    want = {k: int(v) for k, v in (x.split(":") for x in n_spec.split(","))}
    rng = np.random.default_rng(seed)
    meta = ds.dataset(str(cfg.path("corpus"))).to_table(
        columns=["id", "lang", "doc_type", "n_chars", "dup_of", "status"]).to_pandas()
    meta = meta[(meta["status"] == "ok") & meta["dup_of"].isna() & meta["n_chars"].between(300, 20000)]
    meta = meta[meta["doc_type"].isin(["knowledge", "news", "qa", "paper"])]
    pick = []
    for lg, n in want.items():
        ids = meta.loc[meta["lang"] == lg, "id"].to_numpy()
        pick += rng.choice(ids, size=min(len(ids), int(n * 1.3)), replace=False).tolist()   # dư 30%: bài không có đoạn hợp lệ
    par = ds.dataset(str(cfg.path("chunks") / "parents")).to_table(
        columns=["unit_id", "doc_id", "lang", "ctx", "text", "n_tokens"], filter=ds.field("doc_id").isin(pick)).to_pandas()
    par = par[par["n_tokens"] >= 100]
    par = par.sample(frac=1.0, random_state=seed).drop_duplicates("doc_id")
    out = pd.concat([par[par["lang"] == lg].head(n) for lg, n in want.items()], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "sid", np.arange(QID0, QID0 + len(out)))
    d = sdir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "source.parquet")
    log(f"→ {d / 'source.parquet'}: {len(out)} đoạn nguồn {out['lang'].value_counts().to_dict()}")


def load_llm(model, tp, max_len, log=print):
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, dtype="half", max_model_len=max_len,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    log(f"nạp {model} (TP={tp}) trong {time.time() - t0:.0f}s")
    return llm, llm.get_tokenizer()


def chat(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse_gen(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, "không có JSON"
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "JSON hỏng"
    if j.get("skip"):
        return None, "bỏ qua (không phải câu hỏi sức khoẻ)"
    vi, en, zh = (str(j.get(k, "")).strip() for k in ("vi", "en", "zh"))
    if not (vi and en and zh):
        return None, "thiếu ngôn ngữ"
    if not re.search(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", vi.lower()):
        return None, "vi không có dấu tiếng Việt"
    if re.search(r"[぀-ヿ一-鿿]", vi):
        return None, "vi có chữ Hán"
    if not re.search(r"[一-鿿]", zh):
        return None, "zh không có chữ Hán"
    if not 6 <= len(vi.split()) <= 90:
        return None, "độ dài vi"
    return {"vi": vi, "en": en, "zh": zh}, ""


def cmd_generate(cfg, model, tp, limit=0, log=print):
    d = sdir(cfg)
    src = pd.read_parquet(d / "source.parquet")
    if limit:
        src = src.head(limit)
    llm, tok = load_llm(model, tp, 4096, log)
    from vllm import SamplingParams
    prompts = [chat(tok, GEN_PROMPT.format(lang=LANG_NAME.get(r.lang, r.lang), ctx=(r.ctx or "").strip(),
                                           text=cut(r.text, r.lang)), GEN_SYSTEM) for r in src.itertuples()]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=300, seed=42))
    dt = time.time() - t0
    ok = 0
    with (d / "gen.jsonl").open("w", encoding="utf-8") as f:
        for r, o in zip(src.itertuples(), outs):
            raw = o.outputs[0].text
            q, why = parse_gen(raw)
            ok += q is not None
            f.write(json.dumps({"sid": int(r.sid), "doc_id": int(r.doc_id), "unit_id": r.unit_id, "lang": r.lang,
                                "q": q, "why": why, "raw": raw}, ensure_ascii=False) + "\n")
    ntok = sum(len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outs)
    log(f"→ {d / 'gen.jsonl'}: {ok}/{len(src)} hợp lệ, {dt:.0f}s ({ntok / dt:.0f} token/s)")
    rows = [json.loads(x) for x in (d / "gen.jsonl").read_text("utf-8").splitlines() if x.strip()]
    st = pd.DataFrame([(r["lang"], r["why"] or "hợp lệ") for r in rows], columns=["lang", "why"])
    log(st.groupby(["lang", "why"]).size().unstack(fill_value=0).to_string())


def cmd_build(cfg, log=print):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = sdir(cfg)
    rows = [json.loads(x) for x in (d / "gen.jsonl").read_text("utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["q"]]
    qd = Path(cfg["work_dir"]) / cfg["expand"]["out_dir"] / "synth"
    qd.mkdir(parents=True, exist_ok=True)
    for lg in ("vi", "en", "zh"):
        pq.write_table(pa.table({"id": pa.array([r["sid"] for r in rows], pa.int64()),
                                 "query": [r["q"][lg] for r in rows], "lang": [lg] * len(rows)}),
                       qd / f"queries_{lg}.parquet")
    with (qd / "samples.md").open("w", encoding="utf-8") as f:
        for r in rows[:30]:
            f.write(f"- [{r['lang']}] {r['q']['vi']}\n")
    log(f"→ {qd}/queries_{{vi,en,zh}}.parquet: {len(rows)} câu hỏi")


# ---------------------------------------------------------------- k8b
def jaccard(a, b, n=3):
    sa = {a[i:i + n] for i in range(max(1, len(a) - n + 1))}
    sb = {b[i:i + n] for i in range(max(1, len(b) - n + 1))}
    return len(sa & sb) / max(1, len(sa | sb))


def cmd_candidates(cfg, run_path, top=12, far=4, far_lo=50, far_hi=200, dup=0.6, seed=42, log=print):
    """Mỗi câu tổng hợp: đoạn nguồn + đoạn tốt nhất của top-`top` tài liệu + `far` tài liệu ngẫu nhiên ở hạng
    [far_lo, far_hi) (âm dễ, như MedCPT re-ranker); bỏ ứng viên gần trùng đoạn nguồn (Jaccard 3-gram ký tự > dup)."""
    from .rerank import unit_rows
    d = sdir(cfg)
    src = pd.read_parquet(d / "source.parquet").set_index("sid")
    r = runs.read(run_path, depth=far_hi * 3)
    r = r[r["qid"].isin(src.index)]
    dr = runs.doc_ranking(r, "max", 400).sort_values(["qid", "drank"])
    rng = np.random.default_rng(seed)
    pairs = []
    for q, g in dr.groupby("qid"):
        s = src.loc[q]
        g = g[g["doc_id"] != s.doc_id]
        head = g[g["drank"] <= top]
        tail = g[(g["drank"] > far_lo) & (g["drank"] <= far_hi)]
        tail = tail.iloc[rng.choice(len(tail), size=min(far, len(tail)), replace=False)] if len(tail) else tail
        pairs.append((q, s.unit_id, int(s.doc_id), 0, "source"))
        pairs += [(q, u, int(dd), int(k), "top") for u, dd, k in zip(head["unit_id"], head["doc_id"], head["drank"])]
        pairs += [(q, u, int(dd), int(k), "far") for u, dd, k in zip(tail["unit_id"], tail["doc_id"], tail["drank"])]
    p = pd.DataFrame(pairs, columns=["qid", "unit_id", "doc_id", "drank", "kind"])
    rows = unit_rows(cfg, p["unit_id"].unique().tolist())
    p = p[p["unit_id"].isin(rows)].copy()
    p["lang"] = [rows[u][0] for u in p["unit_id"]]
    p["ctx"] = [rows[u][1] for u in p["unit_id"]]
    p["text"] = [rows[u][2] for u in p["unit_id"]]
    st = {q: t for q, t in zip(src.index, src["text"])}
    near = [k != "source" and jaccard(t, st[q]) > dup for q, t, k in zip(p["qid"], p["text"], p["kind"])]
    p = p[~np.array(near)].reset_index(drop=True)
    p.to_parquet(d / "pairs.parquet")
    log(f"→ {d / 'pairs.parquet'}: {len(p)} cặp, {p['qid'].nunique()} câu, bỏ {sum(near)} bản gần trùng nguồn; "
        f"{p['kind'].value_counts().to_dict()}; ngôn ngữ {p['lang'].value_counts().to_dict()}")


def cmd_label(cfg, model, tp, shard="0/1", limit=0, log=print):
    from vllm import SamplingParams
    d = sdir(cfg)
    p = pd.read_parquet(d / "pairs.parquet")
    qd = Path(cfg["work_dir"]) / cfg["expand"]["out_dir"] / "synth"
    qs = {lg: dict(pd.read_parquet(qd / f"queries_{lg}.parquet")[["id", "query"]].itertuples(index=False))
          for lg in ("vi", "en", "zh")}
    k, n = (int(x) for x in shard.split("/"))
    mine = sorted(p["qid"].unique())[k::n]
    p = p[p["qid"].isin(set(mine))]
    if limit:
        p = p[p["qid"].isin(set(sorted(p["qid"].unique())[:limit]))]
    p = p.sort_values(["qid", "drank"]).reset_index(drop=True)          # cùng câu liền nhau: prefix caching
    llm, tok = load_llm(model, tp, 3072, log)
    digits = {g: tok.convert_tokens_to_ids(str(g)) for g in (0, 1, 2)}
    prompts = []
    for r in p.itertuples():
        qx = f"\n(Bản dịch {LANG_NAME[r.lang]}: {qs[r.lang][r.qid]})" if r.lang in ("en", "zh") else ""
        prompts.append(chat(tok, LABEL_PROMPT.format(q=qs["vi"][r.qid], qx=qx, lang=LANG_NAME.get(r.lang, r.lang),
                                                     ctx=(r.ctx or "").strip(), text=cut(r.text, r.lang)),
                            LABEL_SYSTEM))
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    dt = time.time() - t0
    P = np.zeros((len(p), 3))
    for i, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        for g, tid in digits.items():
            if tid in lp:
                P[i, g] = np.exp(lp[tid].logprob)
    miss = P.sum(1) == 0
    P[miss] = np.nan
    P = P / P.sum(1, keepdims=True)
    p["p0"], p["p1"], p["p2"] = P[:, 0], P[:, 1], P[:, 2]
    p["grade"] = p["p1"] + 2 * p["p2"]
    out = d / (f"labels-{k}of{n}.parquet" if n > 1 else "labels.parquet")
    p.drop(columns=["ctx", "text"]).to_parquet(out)
    ntok = sum(len(o.prompt_token_ids) for o in outs)
    log(f"→ {out}: {len(p)} cặp trong {dt:.0f}s ({len(p) / dt:.2f} cặp/s, {ntok / dt:.0f} token/s); "
        f"không đọc được {int(miss.sum())}")
    g = p.dropna(subset=["grade"]).groupby("kind")["grade"].agg(["mean", lambda x: (x >= 1.5).mean(),
                                                                  lambda x: (x < 0.5).mean()])
    g.columns = ["điểm TB", "tỉ lệ 2", "tỉ lệ 0"]
    log(g.round(3).to_string())


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "generate", "build", "candidates", "label"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", default="vi:2600,zh:1100,en:1000",
                    help="sample: số đoạn nguồn mỗi ngôn ngữ (dư để bù đoạn bị bỏ qua: vi nhiều tin thời sự)")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel (số GPU)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N mục / N câu đầu (đo tốc độ)")
    ap.add_argument("--run", default=None, help="candidates: run gộp của câu tổng hợp (runs/synth_lin07.parquet)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--far", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "sample":
        cmd_sample(cfg, a.n)
    elif a.cmd == "generate":
        cmd_generate(cfg, a.model, a.tp, a.limit)
    elif a.cmd == "build":
        cmd_build(cfg)
    elif a.cmd == "candidates":
        cmd_candidates(cfg, a.run or cfg.path("runs") / "synth_lin07.parquet", a.top, a.far)
    else:
        cmd_label(cfg, a.model, a.tp, a.shard, a.limit)


if __name__ == "__main__":
    main()
