"""Đoạn neo cho span_chunks từ BẤT KỲ run nào (vd run của model đã fine-tune): chữ của đoạn cha tốt nhất mỗi tài liệu,
lấy bằng cách chia đoạn lại ngay trên máy (mirret.chunk.chunk_doc, cùng cấu hình chunk + tokenizer bge-m3) — không cần
bài nộp có chunk_text tạo trên Kaggle. Đầu ra parquet (qid, doc_id, text) theo thứ tự run → --chunks của span_chunks.

  python tools/run_anchors.py --run results/ft/runs/lin07_ensift_pub.parquet --out results/ft/anchors_ensift.parquet
  --check <neo cũ .parquet>: tỉ lệ neo cũ (chunk_text của bài nộp Kaggle) trùng khít một đoạn cha chia lại (kiểm tái tạo)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402
from mirret.chunk import chunk_doc, load_tokenizer  # noqa: E402


def parents_of(cfg, doc_ids, batch=5000, log=print):
    """{doc_id: {unit_id: text}} — chia đoạn cha lại cho các tài liệu cần (đọc kho theo lô)."""
    cc = dict(cfg["chunk"])
    cc["child"] = dict(cc["child"], enabled=False)           # chỉ cần đoạn cha
    tok = load_tokenizer(cc["tokenizer"])
    corpus = ds.dataset(str(cfg.path("corpus")))
    names = set(corpus.schema.names)
    cols = [c for c in ["id", "lang", "doc_type", "title", "text", "section"] if c in names]
    ids, out = sorted(doc_ids), {}
    for b in range(0, len(ids), batch):
        t = corpus.to_table(columns=cols, filter=ds.field("id").isin(ids[b:b + batch])).to_pydict()
        for i in range(len(t["id"])):
            ps, _ = chunk_doc(int(t["id"][i]), t["lang"][i] or "", t["doc_type"][i] or "", t["title"][i] or "",
                              t["text"][i] or "", (t.get("section") or [None] * len(t["id"]))[i] or "", tok, cc)
            out[int(t["id"][i])] = {p["unit_id"]: p["text"] for p in ps}
        log(f"  chia lại {min(b + batch, len(ids))}/{len(ids)} tài liệu")
    return out


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top-n", type=int, default=400, help="chỉ các đơn vị hạng ≤ N (phần đã rerank)")
    ap.add_argument("--qids", default=None, help="chỉ các câu này (file .txt)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, help="neo cũ để so (chỉ kiểm, không ghi --out)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    qids = runs.read_qids(a.qids) if a.qids else None
    r = runs.read(a.run, depth=a.top_n, qids=qids).sort_values(["qid", "rank"])
    best = r.drop_duplicates(["qid", "doc_id"])                 # đơn vị tốt nhất của mỗi tài liệu, theo thứ tự run
    print(f"{best['qid'].nunique()} câu, {len(best)} cặp (câu, tài liệu), {best['doc_id'].nunique()} tài liệu")
    par = parents_of(cfg, set(best["doc_id"].astype(int)))
    if a.check:
        old = pq.read_table(a.check).to_pandas()
        old = old[old["doc_id"].isin(par)]
        if qids:
            old = old[old["qid"].isin(qids)]
        hit = [t in set(par[d].values()) for d, t in zip(old["doc_id"], old["text"])]
        print(f"kiểm: {sum(hit)}/{len(hit)} neo cũ trùng khít một đoạn cha chia lại ({sum(hit) / max(1, len(hit)):.2%})")
        return
    txt = [par.get(int(d), {}).get(u) for d, u in zip(best["doc_id"], best["unit_id"])]
    out = best.assign(text=txt)
    miss = out["text"].isna()
    print(f"thiếu chữ (đơn vị không khớp khi chia lại): {int(miss.sum())}/{len(out)}")
    out = out[~miss]
    pq.write_table(pa.table({"qid": pa.array(out["qid"].astype(int), pa.int64()),
                             "doc_id": pa.array(out["doc_id"].astype(int), pa.int64()), "text": out["text"].tolist()}),
                   a.out)
    print(f"→ {a.out}: {len(out)} neo")


if __name__ == "__main__":
    main()
