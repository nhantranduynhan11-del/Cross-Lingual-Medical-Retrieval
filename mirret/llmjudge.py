"""Giám khảo LLM lúc suy luận: Qwen3-14B (mở, ≤ 14B, phát hành 04/2025 → hợp luật) chấm CẢ TÀI LIỆU (tiêu đề + phần đầu
bài + đoạn tốt nhất) cho top-`hi` tài liệu của run ens mỗi câu thi, thang 0/1/2 + prompt như synth.LABEL_SYSTEM_DOC, xác
suất từng mức lấy từ logprob → P(liên quan) = P(2) + ½P(1). Không huấn luyện gì trên câu thi; teacher > 14B không dùng ở đây.

Lý do (nghiên cứu 29/10):
  - TREC NeuCLIR 2024 (tài liệu tiếng Trung): mọi run đứng đầu dùng LLM làm reranker tầng cuối; GPT-4 rerank sau mT5 tăng
    nDCG@20 0,582 → 0,650; gộp nhiều run chỉ thêm rất ít khi đã rerank.
  - TREC RAG 2024 (UMBRELA): đáp án do LLM chấm cho thứ hạng hệ thống khớp người chấm (Kendall τ 0,89–0,94) nhưng rộng tay
    hơn người → đáp án private làm lại cho 1.200 câu × 3 ngôn ngữ nhiều khả năng có LLM chấm (chưa xác nhận); giám khảo LLM
    khớp cách làm đó.
  - Fine-tune từ teacher (k8) thua ở public vì học sinh 4B học câu tổng hợp lệch; ở đây model lớn chấm thẳng câu thật.
  - Qwen3-Reranker cho P(yes) > 0,5 ở ~155 tài liệu/câu trong khi đáp án public TB 72 → điểm reranker phân biệt kém ở vùng
    hạng 50–250, đúng nơi autok cắt.

Mỗi tài khoản Kaggle một phần câu (--shard i/n); trong tài khoản 2 tiến trình, mỗi GPU một bản model (--part j/2):
  python -m mirret.llmjudge pairs --run <lin07_ensi_1200.parquet> --hi 250 --shard 0/6 [--qids …]
                                                              → <work>/judge/pairs-0of6.parquet
  CUDA_VISIBLE_DEVICES=0 python -m mirret.llmjudge label --shard 0/6 --part 0/2   → <work>/judge/labels-0of6-0of2-b*.parquet
  CUDA_VISIBLE_DEVICES=1 python -m mirret.llmjudge label --shard 0/6 --part 1/2   → <work>/judge/labels-0of6-1of2-b*.parquet
Đầu ra theo lô: labels-<phần>-<tiến trình>-b<lô>.parquet. Ghép trên máy: tools/merge_judge.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, runs, synth


def jdir(cfg):
    return Path(cfg["work_dir"]) / "judge"


def split(ids, spec):
    k, n = (int(x) for x in spec.split("/"))
    return set(sorted(ids)[k::n]), f"{k}of{n}"


def cmd_pairs(cfg, run_path, hi, shard, qids=None, log=print):
    """Tài liệu hạng 1..hi theo run (đoạn tốt nhất mỗi tài liệu) + chữ đoạn + phần đầu bài, chỉ các câu của phần."""
    from .rerank import unit_rows
    allq = set(runs.read(run_path, depth=1)["qid"].astype(int))
    mine, tag = split(allq & qids if qids else allq, shard)
    r = runs.read(run_path, depth=max(hi * 4, 1000), qids=mine)
    dr = runs.doc_ranking(r, "max", 400).sort_values(["qid", "drank"])
    p = dr[dr["drank"] <= hi][["qid", "unit_id", "doc_id", "drank"]].reset_index(drop=True)
    rows = unit_rows(cfg, p["unit_id"].unique().tolist())
    p = p[p["unit_id"].isin(rows)].copy()
    p["lang"] = [rows[u][0] for u in p["unit_id"]]
    p["ctx"] = [rows[u][1] for u in p["unit_id"]]
    p["text"] = [rows[u][2] for u in p["unit_id"]]
    p["lead"] = synth.leads(cfg, p, log)
    d = jdir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"pairs-{tag}.parquet"
    p.to_parquet(out)
    log(f"→ {out}: {len(p)} cặp, {p['qid'].nunique()} câu, hạng 1–{hi}; ngôn ngữ {p['lang'].value_counts().to_dict()}")


def cmd_label(cfg, model, tp, shard, part, max_len=3072, limit=0, batch_q=20, log=print):
    """Chấm theo lô `batch_q` câu, mỗi lô ghi một file ngay khi xong → phiên Kaggle bị dừng giữa chừng vẫn giữ các lô đã
    chấm; chạy lại (chép labels cũ vào <work>/judge) thì bỏ qua lô đã có."""
    from .rerank import query_texts
    d = jdir(cfg)
    _, tag = split([0], shard)
    p = pd.read_parquet(d / f"pairs-{tag}.parquet")
    mine, ptag = split(p["qid"].unique(), part)
    qids = sorted(mine)[:limit] if limit else sorted(mine)
    qs = query_texts(cfg)

    def done(b):                                      # lô đã chấm ĐỦ câu (file của lượt LIMIT nhỏ hơn lô → chấm lại)
        f = d / f"labels-{tag}-{ptag}-b{b // batch_q:03d}.parquet"
        return f.exists() and set(pd.read_parquet(f, columns=["qid"])["qid"]) == set(qids[b:b + batch_q])
    todo = [b for b in range(0, len(qids), batch_q) if not done(b)]
    log(f"phần {tag}-{ptag}: {len(qids)} câu, {len(range(0, len(qids), batch_q))} lô, còn {len(todo)} lô chưa chấm")
    llm, tok = synth.load_llm(model, tp, max_len, log) if todo else (None, None)
    t_all, n_new, q_new = time.time(), 0, 0
    for b in todo:
        out = d / f"labels-{tag}-{ptag}-b{b // batch_q:03d}.parquet"
        qb = qids[b:b + batch_q]
        pb = p[p["qid"].isin(set(qb))].sort_values(["qid", "drank"]).reset_index(drop=True)
        label_batch(pb, qs, llm, tok, model, out, log)
        n_new, q_new = n_new + len(pb), q_new + len(qb)
        sec_per_q = (time.time() - t_all) / q_new
        log(f"  tiến độ {b + len(qb)}/{len(qids)} câu; {n_new / (sec_per_q * q_new):.2f} cặp/s; "
            f"còn ~{(len(qids) - b - len(qb)) * sec_per_q / 3600:.1f} giờ")
    lab = pd.concat([pd.read_parquet(f) for f in sorted(d.glob(f"labels-{tag}-{ptag}-b*.parquet"))], ignore_index=True)
    report(lab, log)


def label_batch(p, qs, llm, tok, model, out, log=print):
    from vllm import SamplingParams
    digits = {g: tok.convert_tokens_to_ids(str(g)) for g in (0, 1, 2)}
    prompts = []
    for r in p.itertuples():
        tr = qs.get(r.lang, {}).get(int(r.qid)) if r.lang in ("en", "zh") else None
        qx = f"\n(Bản dịch {synth.LANG_NAME[r.lang]}: {tr})" if tr else ""
        prompts.append(synth.chat(tok, synth.LABEL_PROMPT_DOC.format(
            q=qs["vi"][int(r.qid)], qx=qx, lang=synth.LANG_NAME.get(r.lang, r.lang), ctx=(r.ctx or "").strip(),
            lead=r.lead or "(đoạn liên quan nhất là phần đầu bài)", text=synth.cut(r.text, r.lang)),
            synth.LABEL_SYSTEM_DOC))
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
    p["prel"] = p["p2"] + 0.5 * p["p1"]
    p.drop(columns=["ctx", "text", "lead"]).assign(model=model).to_parquet(out)
    ntok = sum(len(o.prompt_token_ids) for o in outs)
    log(f"→ {out.name}: {len(p)} cặp, {p['qid'].nunique()} câu trong {dt:.0f}s ({len(p) / max(dt, 1):.2f} cặp/s, "
        f"{ntok / max(dt, 1):.0f} token/s, TB {ntok / max(len(p), 1):.0f} token/cặp); không đọc được {int(miss.sum())}")


def report(p, log=print):
    b = pd.cut(p["drank"], [0, 25, 50, 100, 150, 200, 250, 10 ** 6])
    g = p.groupby(b, observed=True).agg(prel=("prel", "mean"), p2=("p2", "mean"), n=("prel", "size"))
    g["Σprel/câu"] = p.groupby(b, observed=True)["prel"].sum() / max(p["qid"].nunique(), 1)
    log("P(liên quan) theo hạng ens:\n" + g.round(3).to_string())
    log(f"Σ P(liên quan) mỗi câu: trung vị {p.groupby('qid')['prel'].sum().median():.1f} "
        f"(so với số tài liệu đáp án public TB ~72)")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "label"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default=None, help="pairs: run ens đã rerank (lin07_ensi_1200.parquet)")
    ap.add_argument("--hi", type=int, default=250, help="pairs: số tài liệu đầu mỗi câu được chấm")
    ap.add_argument("--qids", default=None, help="pairs: chỉ các câu này (vd public600_a.txt khi chạy thử LB)")
    ap.add_argument("--shard", default="0/1", help="phần câu của tài khoản (i/n)")
    ap.add_argument("--part", default="0/1", help="label: phần câu của tiến trình trong tài khoản (j/m, mỗi GPU một phần)")
    ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    ap.add_argument("--tp", type=int, default=1, help="1 = mỗi GPU một bản model (14B AWQ vừa một T4)")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0, help="label: chỉ N câu đầu (đo tốc độ)")
    ap.add_argument("--batch-q", type=int, default=20, help="label: số câu mỗi lô (mỗi lô một file)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "pairs":
        cmd_pairs(cfg, a.run, a.hi, a.shard, runs.read_qids(a.qids))
    else:
        cmd_label(cfg, a.model, a.tp, a.shard, a.part, a.max_len, a.limit, a.batch_q)


if __name__ == "__main__":
    main()
