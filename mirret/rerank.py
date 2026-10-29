"""Stage 2: xếp hạng lại top-N ứng viên (đoạn cha) của run gộp Stage 1.

Phương pháp (rerank.method):
  colbert : multi-vector của BGE-M3, MaxSim như FlagEmbedding: (1/|q|) Σ_i max_j <q_i, d_j> (mir/rerank.py cũ — tốt
            nhất trên leaderboard cũ: 0,1674). Vector ColBERT chỉ tính cho câu hỏi + N ứng viên, không lưu cả kho.
  cross   : cross-encoder (mặc định BAAI/bge-reranker-v2-m3, đa ngôn ngữ): điểm = logit của cặp (câu hỏi, đoạn).
  qwen3   : LLM reranker (rerank.qwen3_model, mặc định Qwen3-Reranker-4B): điểm = P(yes) / (P(yes) + P(no)),
            instruction mặc định của model. qwen3i: như qwen3 với instruction y khoa tiếng Anh.
Ứng viên = ngữ cảnh + "\\n" + đoạn (C). rerank.match_lang: đoạn zh chấm bằng câu hỏi zh, đoạn en bằng câu en (thiếu bản
dịch → câu vi) — dự án cũ: điểm câu vi với đoạn zh đúng chỉ 0,21–0,95, câu dịch zh cho 0,97–0,999.
Kết quả: top-N theo điểm mới, nối thêm phần đuôi (hạng > N) của run gốc → run đủ độ sâu.

  python -m mirret.rerank --run runs/rrf9.parquet [--method cross] [--top-n 400] [--shard i/N] --device cuda:0
     → runs/<run>_<method>/part-i-of-N.parquet ; đủ mọi phần → runs/<run>_<method>.parquet
  Thử nhanh trên bộ nhãn: --qids eval/judge/queries.csv (hoặc "1,2,3") → chỉ các câu đó, tên mặc định
  <run>_<method>_sub (không đè run đủ 1.200 câu); --name đặt tên khác.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config, runs
from .encode import unit_input


def unit_rows(cfg, unit_ids, level="parents"):
    """{unit_id: (lang, ctx, text)} cho các đơn vị cần (đọc lọc theo id, không nạp cả kho)."""
    d = ds.dataset(str(cfg.path("chunks") / level))
    t = d.to_table(columns=["unit_id", "lang", "ctx", "text"], filter=ds.field("unit_id").isin(list(set(unit_ids))))
    x = t.to_pydict()
    return {u: (lg, c, tx) for u, lg, c, tx in zip(x["unit_id"], x["lang"], x["ctx"], x["text"])}


def query_texts(cfg):
    """{lang: {qid: câu hỏi}} từ qtrans/queries_*.parquet."""
    out = {}
    for lg in ("vi", "en", "zh"):
        p = cfg.path("qtrans") / f"queries_{lg}.parquet"
        if p.exists():
            t = ds.dataset(str(p)).to_table().to_pydict()
            out[lg] = dict(zip([int(x) for x in t["id"]], t["query"]))
    return out


def pick_query(qs, qid, doc_lang, match_lang):
    if match_lang and doc_lang in qs and qid in qs[doc_lang]:
        return qs[doc_lang][qid], doc_lang
    return qs["vi"][qid], "vi"


def maxsim(q, d):
    if len(q) == 0 or len(d) == 0:
        return 0.0
    s = q.astype(np.float32) @ d.astype(np.float32).T
    return float(s.max(axis=1).sum() / len(q))


class CrossScorer:
    def __init__(self, name, device="cuda:0", fp16=True, max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = torch.device(device)
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.max_length = max_length

    def score(self, pairs, batch_tokens=16384, max_batch=64):
        torch = self.torch
        enc = self.tok([q for q, _ in pairs], [d for _, d in pairs], truncation="only_second",
                       max_length=self.max_length)["input_ids"]
        order = sorted(range(len(pairs)), key=lambda i: -len(enc[i]))
        out = np.zeros(len(pairs), dtype=np.float32)
        pad = self.tok.pad_token_id
        b = []

        def flush(b):
            L = max(len(enc[i]) for i in b)
            inp = torch.tensor([enc[i] + [pad] * (L - len(enc[i])) for i in b], device=self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=inp, attention_mask=(inp != pad).long()).logits
            out[b] = logits[:, 0].float().cpu().numpy()
        for i in order:
            if b and (len(b) + 1 > max_batch or (len(b) + 1) * len(enc[b[0]]) > batch_tokens):
                flush(b)
                b = []
            b.append(i)
        if b:
            flush(b)
        return out


class Qwen3Scorer:
    """Qwen3-Reranker (LLM yes/no): điểm = P(yes) / (P(yes) + P(no)) ở token cuối, theo định dạng của model card."""
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
              "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
    MED_INSTRUCT = "Given a Vietnamese medical question, retrieve passages that answer the question"

    def __init__(self, name, device="cuda:0", fp16=True, max_length=768, instruct=None, lora=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.device = torch.device(device)
        dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()
        if lora:                                    # adapter của mirret.finetune qwen → gộp vào trọng số (không chậm hơn)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora).merge_and_unload().eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        self.max_length = max_length            # phần <Instruct>/<Query>/<Document>, chưa tính PREFIX/SUFFIX
        self.instruct = instruct or self.DEFAULT_INSTRUCT

    def score(self, pairs, batch_tokens=8192, max_batch=32):
        torch = self.torch
        texts = [f"<Instruct>: {self.instruct}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        body = self.tok(texts, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        enc = [self.pre + b + self.suf for b in body]
        order = sorted(range(len(pairs)), key=lambda i: -len(enc[i]))
        out = np.zeros(len(pairs), dtype=np.float32)
        pad = self.tok.pad_token_id
        b = []

        def flush(b):
            L = max(len(enc[i]) for i in b)
            inp = torch.tensor([[pad] * (L - len(enc[i])) + enc[i] for i in b], device=self.device)
            mask = torch.tensor([[0] * (L - len(enc[i])) + [1] * len(enc[i]) for i in b], device=self.device)
            with torch.inference_mode():
                # logits_to_keep=1: chỉ tính logits token cuối (logits đủ độ dài × 151k từ vựng không vừa T4)
                logits = self.model(input_ids=inp, attention_mask=mask, logits_to_keep=1).logits[:, -1, :]
            yn = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out[b] = torch.softmax(yn, dim=1)[:, 1].cpu().numpy()
        for i in order:
            if b and (len(b) + 1 > max_batch or (len(b) + 1) * len(enc[b[0]]) > batch_tokens):
                flush(b)
                b = []
            b.append(i)
        if b:
            flush(b)
        return out


def rerank(cfg, df, method, device, log=print):
    """df: run đã cắt top-N (đoạn cha). → run mới với điểm của reranker."""
    rc, ec = cfg["rerank"], cfg["encode"]
    from .encode import resolve_device
    device = resolve_device(device)
    qs = query_texts(cfg)
    rows = unit_rows(cfg, df["unit_id"].unique().tolist())
    df = df[df["unit_id"].isin(rows)]
    out = []
    t0 = time.time()
    if method == "colbert":
        from .encode import M3Encoder
        enc = M3Encoder.load(ec["model"], device, ec["fp16"], ec["max_length"], colbert=True)
        qcache = {}
        for n, (q, g) in enumerate(df.groupby("qid", sort=True)):
            ids = g["unit_id"].tolist()
            need = {pick_query(qs, q, rows[u][0], rc["match_lang"]) for u in ids}
            for text, lg in need:
                if (q, lg) not in qcache:
                    qcache[(q, lg)] = enc.colbert([text], rc["query_max_length"], rc["batch_tokens"])[0]
            dv = enc.colbert([unit_input(rows[u][1], rows[u][2]) for u in ids], rc["doc_max_length"],
                             rc["batch_tokens"])
            sc = [maxsim(qcache[(q, pick_query(qs, q, rows[u][0], rc["match_lang"])[1])], d) for u, d in zip(ids, dv)]
            out.append(pd.DataFrame({"qid": q, "unit_id": ids, "doc_id": g["doc_id"].values,
                                     "rank": g["rank"].values, "score": sc}))
            if (n + 1) % 50 == 0:
                log(f"  {n + 1} câu hỏi, {(time.time() - t0) / (n + 1):.2f}s/câu")
    elif method in ("qwen3", "qwen3i"):
        sc = Qwen3Scorer(rc["qwen3_model"], device, True, rc["query_max_length"] + rc["doc_max_length"],
                         Qwen3Scorer.MED_INSTRUCT if method == "qwen3i" else None, rc.get("qwen3_lora"))
        npairs, t0 = 0, time.time()                         # không tính thời gian tải / nạp model
        for n, (q, g) in enumerate(df.groupby("qid", sort=True)):
            ids = g["unit_id"].tolist()
            pairs = [(pick_query(qs, q, rows[u][0], rc["match_lang"])[0], unit_input(rows[u][1], rows[u][2]))
                     for u in ids]
            s = sc.score(pairs, rc["qwen3_batch_tokens"])
            if np.isnan(s).any():
                raise RuntimeError(f"qid {q}: {int(np.isnan(s).sum())}/{len(s)} điểm NaN (fp16 tràn số)")
            npairs += len(pairs)
            out.append(pd.DataFrame({"qid": q, "unit_id": ids, "doc_id": g["doc_id"].values,
                                     "rank": g["rank"].values, "score": s}))
            if (n + 1) % 10 == 0 or n == 0:
                dt = time.time() - t0
                log(f"  {n + 1} câu hỏi, {dt / (n + 1):.2f}s/câu, {1000 * dt / npairs:.1f}ms/cặp")
        if npairs:
            dt = time.time() - t0
            log(f"  xong {n + 1} câu hỏi, {npairs} cặp: {dt / (n + 1):.2f}s/câu, {1000 * dt / npairs:.1f}ms/cặp")
    else:
        sc = CrossScorer(rc["cross_model"], device, True, rc["doc_max_length"])
        for n, (q, g) in enumerate(df.groupby("qid", sort=True)):
            ids = g["unit_id"].tolist()
            pairs = [(pick_query(qs, q, rows[u][0], rc["match_lang"])[0], unit_input(rows[u][1], rows[u][2]))
                     for u in ids]
            s = sc.score(pairs, rc["batch_tokens"])
            out.append(pd.DataFrame({"qid": q, "unit_id": ids, "doc_id": g["doc_id"].values,
                                     "rank": g["rank"].values, "score": s}))
            if (n + 1) % 50 == 0:
                log(f"  {n + 1} câu hỏi, {(time.time() - t0) / (n + 1):.2f}s/câu")
    return runs.rerank_by_score(pd.concat(out, ignore_index=True)) if out else df.iloc[:0]


def reuse_scores(paths, top_n, qids):
    """Điểm reranker đã có: {(qid, unit_id): score} từ các run đã rerank bằng CÙNG phương pháp (mọi cặp đã chấm, kể cả run
    rerank sâu hơn top_n như lin07_cross800; bỏ phần đuôi with_tail có điểm -(hạng + 100000)). Điểm của một cặp không phụ
    thuộc ứng viên khác → dùng lại đúng như chấm lại."""
    parts = []
    for p in paths:
        r = runs.read(p, qids=qids)
        parts.append(r[r["score"] > -1e4][["qid", "unit_id", "score"]])
    c = pd.concat(parts, ignore_index=True).drop_duplicates(["qid", "unit_id"])
    return c.set_index(["qid", "unit_id"])["score"]


def with_tail(reranked, base, top_n):
    """Top-N đã xếp hạng lại + phần đuôi (hạng > N) của run gốc, đánh rank liên tục."""
    tail = base[base["rank"] > top_n].copy()
    tail["rank"] = tail["rank"] + 100000                   # đuôi luôn sau phần đã xếp hạng lại
    tail["score"] = -tail["rank"].astype(float)
    r = reranked.copy()
    r["score"] = r["score"].astype(float)
    allr = pd.concat([r.assign(_o=0), tail.assign(_o=1)], ignore_index=True)
    allr = allr.sort_values(["qid", "_o", "score"], ascending=[True, True, False], kind="stable")
    allr["rank"] = allr.groupby("qid").cumcount() + 1
    return allr.drop(columns="_o").reset_index(drop=True)


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", default=None, choices=[None, "colbert", "cross", "qwen3", "qwen3i"])
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/N theo câu hỏi (mỗi GPU một phần)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="thư mục ghi kết quả (mặc định: cạnh run gốc)")
    ap.add_argument("--qids", default=None, help="chỉ các câu này: file qid (.txt/.csv/.parquet) hoặc '1,2,3'")
    ap.add_argument("--name", default=None, help="tên run kết quả (mặc định <run>_<method>, thêm _sub khi có --qids)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse", nargs="*", default=None,
                    help="run đã rerank bằng cùng phương pháp (vd runs/lin07_qwen3i_1200.parquet): dùng lại điểm các cặp "
                         "(câu, đoạn) đã chấm, chỉ chấm cặp mới")
    ap.add_argument("--pubmed-only", action="store_true",
                    help="chỉ chấm đoạn của bài PubMed (model fine-tune dùng riêng cho tiếng Anh); run kết quả không có "
                         "phần đuôi, chỉ gồm các đoạn PubMed đã chấm")
    a = ap.parse_args()
    cfg = config.load(a.config)
    rc = cfg["rerank"]
    method = a.method or rc["method"]
    top_n = a.top_n or rc["top_n"]
    run_path = Path(a.run)
    qids = runs.read_qids(a.qids)
    tag = a.name or f"{run_path.stem}_{method}" + ("_sub" if qids else "")
    out_dir = Path(a.out) if a.out else run_path.parent
    part_dir = out_dir / tag
    final = out_dir / (tag + ".parquet")
    si, sn = (int(x) for x in a.shard.split("/"))
    part = part_dir / f"part-{si}-of-{sn}.parquet"
    if final.exists() and not a.force:
        print(f"{final} đã có → bỏ qua")
        return
    base = runs.read(run_path, qids=qids)
    if qids:
        print(f"--qids: {base['qid'].nunique()}/{len(qids)} câu có trong run → {tag}")
    if not part.exists() or a.force:
        mine = [q for k, q in enumerate(sorted(base["qid"].unique())) if k % sn == si]
        df = base[base["qid"].isin(mine) & (base["rank"] <= top_n)]
        if a.pubmed_only:
            df = df[df["doc_id"].isin(runs.pubmed_ids(cfg))]
            print(f"  --pubmed-only: {len(df)} đoạn PubMed ({df['qid'].nunique()} câu)")
        reused = 0
        if a.reuse:
            cache = reuse_scores(a.reuse, top_n, set(mine))
            hit = pd.MultiIndex.from_frame(df[["qid", "unit_id"]]).isin(cache.index)
            old = df[hit].drop(columns="score").merge(cache.reset_index(), on=["qid", "unit_id"])
            reused = len(old)
            print(f"  --reuse: dùng lại {reused}/{len(df)} cặp ({reused / max(1, len(df)):.0%}), chấm mới {int((~hit).sum())}")
            new = rerank(cfg, df[~hit], method, a.device) if (~hit).any() else df.iloc[:0]
            cols = ["qid", "unit_id", "doc_id", "rank", "score"]
            res = runs.rerank_by_score(pd.concat([old[cols], new[cols]], ignore_index=True))
        else:
            res = rerank(cfg, df, method, a.device)
        runs.write(res, part, {"run": str(run_path), "method": method, "top_n": top_n, "shard": a.shard,
                               "qids": len(qids) if qids else "all", "rerank": rc, "reused_pairs": reused,
                               "reuse": a.reuse})
        print(f"→ {part}")
    parts = sorted(part_dir.glob(f"part-*-of-{sn}.parquet"))
    if len(parts) == sn:
        rr = pd.concat([runs.read(p) for p in parts], ignore_index=True)
        runs.write(rr if a.pubmed_only else with_tail(rr, base, top_n), final, {"run": str(run_path), "method": method, "top_n": top_n,
                                                      "parts": sn, "qids": len(qids) if qids else "all",
                                                      "tail": "run gốc, hạng > top_n"})
        print(f"đủ {sn} phần → {final}")
    else:
        print(f"đã có {len(parts)}/{sn} phần; chạy các phần còn lại rồi chạy lại lệnh này để gộp")


if __name__ == "__main__":
    main()
