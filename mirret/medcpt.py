"""Nhánh MedCPT cho tài liệu tiếng Anh (toàn bộ là PubMed): medcpt_en_dense.

MedCPT (Jin et al. 2023, NCBI; BERT-base 110M, phát hành 07/2023, mở) học tương phản trên 255 triệu cặp câu hỏi–bài
từ nhật ký tìm kiếm PubMed; zero-shot hơn BM25 + cross-encoder ở 4/5 tác vụ y sinh của BEIR. Câu hỏi: bản dịch tiếng Anh
(qtrans/queries_en.parquet); tài liệu: đoạn cha tiếng Anh (ctx = tiêu đề, text = tóm tắt) mã hoá dạng cặp
[tiêu đề, tóm tắt] như MedCPT-Article-Encoder; điểm = tích vô hướng vector [CLS]. Run một mức (đoạn cha) →
runs/medcpt_en_dense.parquet, gộp vào lin07 như một nhánh dense (fuse.configs lin07m / lin07um).

  python -m mirret.medcpt encode --shard 0/2 --device cuda:0     # mỗi GPU một phần → <work>/emb_medcpt/part-<i>.npz
  python -m mirret.medcpt search --device cuda:0                 # gộp các phần, tìm top-depth → runs/medcpt_en_dense
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config, runs


def out_dir(cfg):
    return Path(cfg["work_dir"]) / cfg["medcpt"]["out_dir"]


def load_model(name, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    if str(device).startswith("cuda"):
        model = model.half()
    return tok, model, torch


def embed(tok, model, torch, inputs, max_length, batch, device, log=print):
    """inputs: list chuỗi (câu hỏi) hoặc cặp [tiêu đề, tóm tắt] → vector [CLS] float16, giữ đúng thứ tự đầu vào."""
    order = np.argsort([-len(x if isinstance(x, str) else x[0] + x[1]) for x in inputs])   # dài trước: ít đệm
    out = np.zeros((len(inputs), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([inputs[i] for i in idx], truncation=True, padding=True, max_length=max_length,
                      return_tensors="pt").to(device)
            out[idx] = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float16)
            if (s // batch) % 500 == 0:
                log(f"  {s + len(idx)}/{len(order)} ({time.time() - t0:.0f}s)")
    return out


def cmd_encode(cfg, shard, device, limit=0, log=print):
    mc = cfg["medcpt"]
    k, n = (int(x) for x in shard.split("/"))
    t = ds.dataset(str(cfg.path("chunks") / "parents")).to_table(
        columns=["unit_id", "doc_id", "ctx", "text"], filter=ds.field("lang") == "en").to_pandas()
    t = t.sort_values("unit_id").reset_index(drop=True)
    t = t.iloc[k::n].reset_index(drop=True)
    if limit:                                                 # chạy thử
        t = t.iloc[:limit]
    log(f"phần {k}/{n}: {len(t)} đoạn tiếng Anh")
    tok, model, torch = load_model(mc["article_model"], device)
    vec = embed(tok, model, torch, [[c or "", x or ""] for c, x in zip(t["ctx"], t["text"])], mc["max_length"],
                mc["batch"], device, log)
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"part-{k}-of-{n}.npz", vec=vec, unit_id=t["unit_id"].to_numpy(dtype=str),
             doc_id=t["doc_id"].to_numpy(dtype=np.int64))
    log(f"→ {d / f'part-{k}-of-{n}.npz'}")


def cmd_search(cfg, device, log=print):
    mc = cfg["medcpt"]
    parts = sorted(out_dir(cfg).glob("part-*.npz"))
    if not parts:
        raise SystemExit(f"chưa có {out_dir(cfg)}/part-*.npz — chạy encode trước")
    z = [np.load(p) for p in parts]
    n_exp = {int(p.stem.split("-of-")[1]) for p in parts}
    if len(n_exp) != 1 or len(parts) != n_exp.pop():
        raise SystemExit(f"thiếu phần: {[p.name for p in parts]}")
    vec = np.concatenate([x["vec"] for x in z])
    uid = np.concatenate([x["unit_id"] for x in z])
    did = np.concatenate([x["doc_id"] for x in z])
    log(f"{len(uid)} đoạn tiếng Anh, {len(set(did.tolist()))} tài liệu")
    q = pd.read_parquet(cfg.path("qtrans") / "queries_en.parquet")
    tok, model, torch = load_model(mc["query_model"], device)
    qv = embed(tok, model, torch, q["query"].tolist(), mc["query_max_length"], 256, device, log)
    D = torch.from_numpy(vec).to(device)
    if not str(device).startswith("cuda"):
        D = D.float()
    depth, rows = mc["depth"], []
    for s in range(0, len(q), 64):
        Q = torch.from_numpy(qv[s:s + 64]).to(device).to(D.dtype)
        sc, ix = torch.topk((Q @ D.T).float(), k=min(depth, D.shape[0]), dim=1)
        sc, ix = sc.cpu().numpy(), ix.cpu().numpy()
        for j, qid in enumerate(q["id"].iloc[s:s + 64]):
            rows.append(pd.DataFrame({"qid": int(qid), "unit_id": uid[ix[j]], "doc_id": did[ix[j]],
                                      "rank": np.arange(1, ix.shape[1] + 1), "score": sc[j]}))
    run = pd.concat(rows, ignore_index=True)
    dst = cfg.path("runs") / "medcpt_en_dense.parquet"
    runs.write(run, dst, {"method": "medcpt", "query_model": mc["query_model"], "article_model": mc["article_model"],
                          "lang": "en", "depth": depth, "units": int(len(uid)), "queries": "qtrans/queries_en.parquet"})
    log(f"→ {dst} ({run['qid'].nunique()} câu hỏi)")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "search"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="encode: chỉ N đoạn đầu mỗi phần (chạy thử)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    from .encode import resolve_device
    dev = resolve_device(a.device)
    if a.cmd == "encode":
        cmd_encode(cfg, a.shard, dev, a.limit)
    else:
        cmd_search(cfg, dev)


if __name__ == "__main__":
    main()
