"""Ghép bài nộp: danh sách TÀI LIỆU từ một hệ, danh sách ĐOẠN từ hệ khác (vd tài liệu từ lin07q_cross, đoạn từ lin07_cross).

Tài liệu chỉ cần doc_id → lấy từ file run (.parquet, thứ hạng tài liệu như submit: runs.doc_ranking) hoặc từ bài nộp
(.json). Đoạn cần nguyên văn → lấy từ bài nộp .json có sẵn. Kiểm tra như submit (đủ câu, id nguyên, doc_id có trong
links) rồi ghi <out_dir>/<name>.json + .zip. Chạy trên máy, không cần mir2-chunks.

  python tools/mix_submission.py --docs results/lin07q/runs/lin07q_cross.parquet \
      --chunks results/lin07s/submissions/lin07_cross.json --name mix_lin07qD_lin07C --out results/mix_submit
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs, submit  # noqa: E402


def doc_lists(path, k_doc, agg="max", norm_depth=400):
    p = Path(path)
    if p.suffix == ".json":
        return {int(s["id"]): [int(d) for d in s["relevant_docs"]][:k_doc] for s in json.loads(p.read_text("utf-8"))}
    run = runs.read(p, depth=k_doc * 3)
    dr = runs.doc_ranking(run, agg, norm_depth).sort_values(["qid", "drank"])
    return {int(q): [int(d) for d in g["doc_id"]][:k_doc] for q, g in dr.groupby("qid")}


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--docs", required=True, help="run .parquet hoặc bài nộp .json cho danh sách tài liệu")
    ap.add_argument("--chunks", required=True, help="bài nộp .json cho danh sách đoạn (giữ nguyên)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/mix_submit")
    ap.add_argument("--k-doc", type=int, default=None)
    ap.add_argument("--doc-agg", default="max", choices=list(runs.DOC_AGGS))
    a = ap.parse_args()
    cfg = config.load(a.config)
    k_doc = a.k_doc or cfg["submit"]["k_doc"]
    docs = doc_lists(a.docs, k_doc, a.doc_agg, cfg["rerank"]["top_n"])
    chunks = {int(s["id"]): s["relevant_chunks"] for s in json.loads(Path(a.chunks).read_text("utf-8"))}
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    sub = [{"id": q, "relevant_docs": docs.get(q, []), "relevant_chunks": chunks.get(q, [])} for q in qids]
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    errs = submit.validate(sub, qids, link_ids)
    if errs:
        print(f"LỖI ({len(errs)}), KHÔNG ghi bài nộp:\n  " + "\n  ".join(errs[:30]))
        raise SystemExit(1)
    js, zp = submit.write(sub, a.out, a.name)
    meta = {"docs_from": a.docs, "chunks_from": a.chunks, "k_doc": k_doc, "doc_agg": a.doc_agg,
            "queries": len(sub), "chunks": sum(len(s["relevant_chunks"]) for s in sub), "zip_mb": round(zp.stat().st_size / 1e6, 1)}
    (Path(a.out) / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Hợp lệ: {len(sub)} câu, tài liệu từ {Path(a.docs).name}, đoạn từ {Path(a.chunks).name} → {zp} ({meta['zip_mb']} MB)")


if __name__ == "__main__":
    main()
