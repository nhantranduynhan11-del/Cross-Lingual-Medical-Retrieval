"""Fine-tune reranker trên nhãn teacher của câu hỏi tổng hợp (mirret/synth.py; không dùng câu thi hay đáp án).

Dữ liệu: <work>/synth/labels-*.parquet (P(0), P(1), P(2) của teacher) + pairs.parquet (chữ) + qexp/synth (câu hỏi 3 ngôn
ngữ). Câu hỏi theo ngôn ngữ đoạn như lúc chấm thật (rerank.match_lang): đoạn zh dùng bản zh, đoạn en dùng bản en.
Nhóm: mỗi câu một nhóm = đoạn nguồn + ứng viên của chính câu (đã chấm 0/1/2) + 2 đoạn của câu khác (điểm 0).
Loss = chưng cất theo danh sách KL(softmax(--beta × điểm teacher) ‖ softmax(điểm model)) (k8b 28/10: teacher chấm ~2/3
ứng viên top-12 là 2 → nhóm "1 dương + âm" sẽ gần như toàn âm dễ) + --bce × BCE(sigmoid(điểm), P(2) + ½P(1)) — phần BCE
kéo P(yes) về mức teacher (chỗ nghẽn hiện tại: Qwen cho P(yes) cao cho quá nhiều đoạn → precision thấp).
5% câu để kiểm tra: độ chính xác cặp (dương > âm) và AUC (P(2) ≥ 0,5) trước và sau khi train.

  python -m mirret.finetune bge  --out <work>/ft/bge   --device cuda:0   # BAAI/bge-reranker-v2-m3, cập nhật toàn bộ
  python -m mirret.finetune qwen --out <work>/ft/qwen  --device cuda:0   # Qwen3-Reranker-4B + LoRA (fp32), base fp16
Dùng ở k5: CROSS_MODEL / QWEN_LORA (rerank.cross_model / rerank.qwen3_lora).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .encode import unit_input


def load_data(cfg, seed=42, variant="synth"):
    d = Path(cfg["work_dir"]) / variant
    lab = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("labels*.parquet"))], ignore_index=True)
    lab = lab.dropna(subset=["p0", "p1", "p2"]).drop_duplicates(["qid", "unit_id"])
    txt = pd.read_parquet(d / "pairs.parquet", columns=["qid", "unit_id", "ctx", "text"])
    lab = lab.merge(txt, on=["qid", "unit_id"])
    qd = Path(cfg["work_dir"]) / cfg["expand"]["out_dir"] / variant
    qs = {lg: dict(pd.read_parquet(qd / f"queries_{lg}.parquet")[["id", "query"]].itertuples(index=False))
          for lg in ("vi", "en", "zh")}
    lab["query"] = [qs[lg if lg in qs else "vi"][q] for q, lg in zip(lab["qid"], lab["lang"])]
    lab["doc"] = [unit_input(c, t) for c, t in zip(lab["ctx"].fillna(""), lab["text"].fillna(""))]
    lab["target"] = lab["p2"] + 0.5 * lab["p1"]
    qids = sorted(lab["qid"].unique())
    random.Random(seed).shuffle(qids)
    dev = set(qids[:max(20, len(qids) // 20)])
    qtext = {q: {lg: qs[lg][q] for lg in qs if q in qs[lg]} for q in qids}
    return (lab[~lab["qid"].isin(dev)].reset_index(drop=True), lab[lab["qid"].isin(dev)].reset_index(drop=True),
            qtext)


def make_groups(lab, qtext, size, n_rand=2, seed=42):
    """Mỗi câu một nhóm `size` mục (câu hỏi, đoạn, nhãn mềm P(2) + ½P(1), điểm teacher P(1) + 2P(2)): đoạn nguồn + ứng
    viên của chính câu (ngẫu nhiên, đã được teacher chấm 0/1/2) + `n_rand` đoạn của câu tổng hợp KHÁC (ghép câu hỏi này
    bằng ngôn ngữ của đoạn, điểm 0). Học theo danh sách (KL tới phân phối điểm teacher) thay vì "1 dương + âm": teacher
    chấm ~2/3 ứng viên top-12 là trả lời được (k8b 28/10) → âm thật chỉ ~2/câu, nhóm "1 dương" sẽ toàn âm dễ."""
    rng = random.Random(seed)
    qid, lang, doc = lab["qid"].to_numpy(), lab["lang"].to_numpy(), lab["doc"].to_numpy()
    item = lambda i: (lab.at[i, "query"], lab.at[i, "doc"], float(lab.at[i, "target"]), float(lab.at[i, "grade"]))
    groups = []
    for q, g in lab.groupby("qid"):
        src = g.index[g["kind"] == "source"].tolist()
        rest = g.index[g["kind"] != "source"].tolist()
        rng.shuffle(rest)
        own = (src + rest)[:size - n_rand]
        if len(own) < 2:
            continue
        items, used = [item(i) for i in own], set(own)
        while len(items) < size:
            j = rng.randrange(len(lab))
            if qid[j] != q and j not in used:
                used.add(j)
                lg = lang[j] if lang[j] in qtext[q] else "vi"
                items.append((qtext[q][lg], doc[j], 0.0, 0.0))
        groups.append(items)
    rng.shuffle(groups)
    return groups


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class BgeModel:
    def __init__(self, name, device, max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device, self.max_length = torch, torch.device(device), max_length
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
        self.model.gradient_checkpointing_enable()

    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, qs, ds):
        enc = self.tok(qs, ds, truncation="only_second", max_length=self.max_length, padding=True,
                       return_tensors="pt").to(self.device)
        return self.model(**enc).logits[:, 0].float()

    def save(self, out):
        self.model.save_pretrained(out)
        self.tok.save_pretrained(out)


class QwenModel:
    """Qwen3-Reranker + LoRA; đầu vào giống hệt rerank.Qwen3Scorer (instruction y khoa), điểm = logit(yes) − logit(no)."""

    def __init__(self, name, device, max_length=768, r=16):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .rerank import Qwen3Scorer
        self.torch, self.device, self.max_length = torch, torch.device(device), max_length
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(self.device)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, cfg)
        for p in self.model.parameters():                     # LoRA fp32 (T4 không có bf16; fp16 thuần dễ tràn)
            if p.requires_grad:
                p.data = p.data.float()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(Qwen3Scorer.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(Qwen3Scorer.SUFFIX, add_special_tokens=False)
        self.instruct = Qwen3Scorer.MED_INSTRUCT

    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, qs, ds):
        torch = self.torch
        texts = [f"<Instruct>: {self.instruct}\n<Query>: {q}\n<Document>: {d}" for q, d in zip(qs, ds)]
        body = self.tok(texts, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        enc = [self.pre + b + self.suf for b in body]
        L = max(len(e) for e in enc)
        pad = self.tok.pad_token_id
        inp = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device=self.device)
        mask = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=self.device)
        logits = self.model(input_ids=inp, attention_mask=mask, logits_to_keep=1).logits[:, -1, :].float()
        return logits[:, self.yes] - logits[:, self.no]

    def save(self, out):
        self.model.save_pretrained(out)


def evaluate(m, dev, batch, log):
    torch = m.torch
    m.model.eval()
    s = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=m.device.type == "cuda"):
        for i in range(0, len(dev), batch):
            b = dev.iloc[i:i + batch]
            s.append(m.forward(b["query"].tolist(), b["doc"].tolist()).cpu().numpy())
    s = np.concatenate(s) if s else np.zeros(0)
    y = (dev["p2"] >= 0.5).astype(int).to_numpy()
    acc, n = 0, 0
    for _, g in dev.assign(s=s).groupby("qid"):
        P, N = g[g["p2"] >= 0.6]["s"].to_numpy(), g[g["p0"] >= 0.7]["s"].to_numpy()
        if len(P) and len(N):
            acc += (P[:, None] > N[None, :]).mean()
            n += 1
    res = {"auc": round(auc(y, s), 4), "pair_acc": round(acc / max(1, n), 4), "n_pairs": int(len(dev))}
    m.model.train()
    log(f"  kiểm tra: {res}")
    return res


def train(kind, cfg, out, device, size, lr, epochs, max_groups, beta, bce, variant="synth", log=print):
    torch = __import__("torch")
    rc = cfg["rerank"]
    tr, dev, qtext = load_data(cfg, variant=variant)
    groups = make_groups(tr, qtext, size)
    if max_groups:
        groups = groups[:max_groups]
    log(f"{kind}: train {tr['qid'].nunique()} câu / {len(tr)} cặp → {len(groups)} nhóm × {size}; "
        f"kiểm tra {dev['qid'].nunique()} câu / {len(dev)} cặp; ngôn ngữ {tr['lang'].value_counts().to_dict()}")
    m = (BgeModel(rc["cross_model"], device) if kind == "bge"
         else QwenModel(rc["qwen3_model"], device, rc["query_max_length"] + rc["doc_max_length"]))
    before = evaluate(m, dev, size, log)
    opt = torch.optim.AdamW(m.params(), lr=lr, weight_decay=0.01)
    steps = len(groups) * epochs
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=m.device.type == "cuda")
    m.model.train()
    t0, k, run_loss = time.time(), 0, []
    for ep in range(epochs):
        for grp in groups:
            qs, ds, ts, gs = zip(*grp)
            with torch.autocast("cuda", dtype=torch.float16, enabled=m.device.type == "cuda"):
                s = m.forward(list(qs), list(ds))
            s = s.float()
            teach = torch.softmax(beta * torch.tensor(gs, dtype=torch.float32, device=s.device), dim=0)
            kl = -(teach * torch.log_softmax(s, dim=0)).sum()        # KL(teacher ‖ model) bỏ hằng số entropy
            tgt = torch.tensor(ts, dtype=torch.float32, device=s.device)
            loss = kl + bce * torch.nn.functional.binary_cross_entropy_with_logits(s, tgt)
            if not torch.isfinite(loss):
                log(f"  bước {k}: loss không hữu hạn → bỏ nhóm")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.params(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            run_loss.append(float(loss))
            k += 1
            if k % 200 == 0:
                log(f"  bước {k}/{steps}: loss {np.mean(run_loss[-200:]):.4f}, {(time.time() - t0) / k:.2f}s/bước")
    after = evaluate(m, dev, size, log)
    Path(out).mkdir(parents=True, exist_ok=True)
    m.save(out)
    meta = {"kind": kind, "base": rc["cross_model"] if kind == "bge" else rc["qwen3_model"], "groups": len(groups),
            "group_size": size, "lr": lr, "epochs": epochs, "beta": beta, "bce": bce,
            "train_queries": int(tr["qid"].nunique()), "dev_before": before, "dev_after": after,
            "minutes": round((time.time() - t0) / 60, 1)}
    (Path(out) / "finetune_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"→ {out}: {meta}")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["bge", "qwen"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--group", type=int, default=None, help="kích thước nhóm (mặc định bge 16, qwen 8)")
    ap.add_argument("--lr", type=float, default=None, help="mặc định bge 2e-5, qwen 1e-4")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--beta", type=float, default=3.0, help="phân phối teacher = softmax(beta × điểm 0–2)")
    ap.add_argument("--bce", type=float, default=1.0)
    ap.add_argument("--variant", default="synth", help="dữ liệu <work>/<variant> (synth = v1, synth2 = v2)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    size = a.group or (16 if a.kind == "bge" else 8)
    lr = a.lr or (2e-5 if a.kind == "bge" else 1e-4)
    train(a.kind, cfg, a.out, a.device, size, lr, a.epochs, a.max_groups, a.beta, a.bce, a.variant)


if __name__ == "__main__":
    main()
