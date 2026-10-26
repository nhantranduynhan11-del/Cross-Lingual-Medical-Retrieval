"""Bài đo recall của tập ứng viên trên LB (không GPU, không đoạn): doc recall của bài đo = phần đáp án nằm trong đúng tập
tài liệu nộp; recall cộng được theo câu vì các tập rời nhau. Precision / F2 của bài đo không có nghĩa.

Lý do (26/10): probe_pool_tail (72 tài liệu/câu của tập 248 tài liệu đã rerank mà chưa nộp) có doc R 0,0273 → reranker
đã giữ ~94% đáp án có trong tập ứng viên; trần recall của tập = 0,4372 + 0,0273 = 0,4645 → hơn nửa đáp án nằm NGOÀI tập
ứng viên → chỗ nghẽn là giai đoạn 1.

  tail: tập ứng viên (top --pool-depth đoạn của --pool) trừ tài liệu đã nộp (--sent)
    python tools/pool_probe.py --pool results/full1200_np/runs/lin07_qwen3i_1200_np.parquet --sent bai.json --name probe_pool_tail
  new : tài liệu của các run --sources (top --source-depth đoạn mỗi run, xếp theo RRF thứ hạng tài liệu) KHÔNG có trong tập
        ứng viên (và không có trong --minus), tối đa --cap tài liệu/câu
    python tools/pool_probe.py --pool results/full1200_np/runs/lin07_qwen3i_1200_np.parquet \
        --sources results/mir2-runs/runs/{vi,en,zh}_{dense,lsr,bm25}.parquet --name probe_branches1000
Bài PubMed luôn bị bỏ (đáp án public không có bài PubMed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs, submit  # noqa: E402


def doc_ranks(path, depth, pm):
    """(qid, doc_id, r): thứ hạng tài liệu (theo đoạn tốt nhất) trong top-depth đoạn; chỉ đọc 3 cột (máy ít RAM)."""
    t = ds.dataset(str(path)).to_table(columns=["qid", "doc_id", "rank"], filter=ds.field("rank") <= depth).to_pandas()
    t = t[~t["doc_id"].isin(pm)].sort_values(["qid", "rank"]).drop_duplicates(["qid", "doc_id"])
    return t.assign(r=t.groupby("qid").cumcount() + 1)[["qid", "doc_id", "r"]]


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--pool", required=True, help="run đã rerank: tập ứng viên = tài liệu trong top --pool-depth đoạn")
    ap.add_argument("--pool-depth", type=int, default=400)
    ap.add_argument("--sent", default=None, help="tail: bài nộp .json (chỉ cần id, relevant_docs)")
    ap.add_argument("--sources", nargs="+", default=None, help="new: các run nguồn")
    ap.add_argument("--source-depth", type=int, default=1000)
    ap.add_argument("--minus", nargs="+", default=[], help="new: bỏ thêm tài liệu của các run này (đo phần riêng)")
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/mix_submit")
    a = ap.parse_args()
    if bool(a.sent) == bool(a.sources):
        raise SystemExit("cần đúng một trong --sent (tail) hoặc --sources (new)")
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    pool = doc_ranks(a.pool, a.pool_depth, pm)
    excl = set(zip(pool["qid"], pool["doc_id"]))
    print(f"tập ứng viên: TB {len(pool) / pool['qid'].nunique():.0f} tài liệu/câu")
    if a.sent:
        sent = {(int(s["id"]), int(d)) for s in json.loads(Path(a.sent).read_text("utf-8")) for d in s["relevant_docs"]}
        keep = pool[[k not in sent for k in zip(pool["qid"], pool["doc_id"])]]
    else:
        for p in a.minus:
            m = doc_ranks(p, a.source_depth, pm)
            excl |= set(zip(m["qid"], m["doc_id"]))
        parts = []
        for p in a.sources:
            d = doc_ranks(p, a.source_depth, pm)
            parts.append(d.assign(s=1.0 / (a.rrf_k + d["r"]))[["qid", "doc_id", "s"]])
            print(f"  {Path(p).name}: TB {len(d) / d['qid'].nunique():.0f} tài liệu/câu")
        g = pd.concat(parts).groupby(["qid", "doc_id"], as_index=False)["s"].sum()
        g = g[[k not in excl for k in zip(g["qid"], g["doc_id"])]]
        n_all = g.groupby("qid").size()
        print(f"tài liệu mới (ngoài tập ứng viên): TB {n_all.mean():.0f}/câu, p50 {n_all.median():.0f}, "
              f"p90 {np.percentile(n_all, 90):.0f}; giữ tối đa {a.cap}")
        keep = g.sort_values(["qid", "s"], ascending=[True, False]).groupby("qid").head(a.cap)
    by_q = keep.groupby("qid")["doc_id"].apply(lambda x: [int(v) for v in x]).to_dict()
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    sub = [{"id": q, "relevant_docs": by_q.get(q, []), "relevant_chunks": []} for q in qids]
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    errs = submit.validate(sub, qids, link_ids)
    if errs:
        raise SystemExit(f"LỖI ({len(errs)}): {errs[:5]}")
    print(f"bài đo: TB {np.mean([len(s['relevant_docs']) for s in sub]):.0f} tài liệu/câu")
    js, zp = submit.write(sub, a.out, a.name)
    (Path(a.out) / f"{a.name}_meta.json").write_text(json.dumps(vars(a), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {zp} ({zp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
