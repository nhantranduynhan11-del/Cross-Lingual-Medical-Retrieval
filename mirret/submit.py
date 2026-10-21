"""Tạo bài nộp từ run cuối (đoạn cha).

  relevant_docs  : top k_doc tài liệu (thứ hạng tài liệu theo submit.doc_agg: max = đoạn tốt nhất, sum2/sum3 = tổng
                   điểm 2/3 đoạn tốt nhất — runs.doc_ranking), doc_id là SỐ NGUYÊN (id của links_corpus;
                   dự án cũ: ghi dạng chuỗi → máy chấm coi là PMID → 0 điểm)
  relevant_chunks: top k_chunk đoạn cha {"doc_id", "chunk_text"}; chunk_text = chữ thuần nguyên văn của đoạn (không kèm
                   ngữ cảnh). Đoạn gần trùng (>= 0,8, BTC gộp) trong cùng câu hỏi bị bỏ để nhường chỗ.
Kiểm tra: đủ 1.200 câu, id nguyên, doc_id có trong links, chunk_text khớp chữ thuần của bài (--verify-corpus).
File nén > submit.max_zip_mb → giảm k_chunk 5% mỗi lần tới khi vừa (báo lại).

  python -m mirret.submit --run runs/rrf9_colbert.parquet [--name bai1] [--k-doc 200] [--k-chunk 185] [--doc-agg sum2]
                           [--verify-corpus]
"""
import argparse
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from . import config, runs
from .rerank import unit_rows


def _norm(s):
    return " ".join((s or "").lower().split())


def build(run, qids, k_doc, k_chunk, texts, doc_agg="max", norm_depth=400):
    by_q = {q: g.sort_values("rank") for q, g in run.groupby("qid")}
    dr = runs.doc_ranking(run, doc_agg, norm_depth).sort_values(["qid", "drank"])
    docs_q = {q: [int(d) for d in g["doc_id"]][:k_doc] for q, g in dr.groupby("qid")}
    sub = []
    for q in qids:
        g = by_q.get(q)
        docs, chunks, seen = [], [], set()
        if g is not None:
            docs = docs_q[q]
            for u, d in zip(g["unit_id"], g["doc_id"]):
                if len(chunks) >= k_chunk:
                    break
                t = texts.get(u)
                if not t:
                    continue
                key = (int(d), _norm(t)[:300])
                if key in seen:
                    continue
                seen.add(key)
                chunks.append({"doc_id": int(d), "chunk_text": t})
        sub.append({"id": int(q), "relevant_docs": docs, "relevant_chunks": chunks})
    return sub


def validate(sub, qids, link_ids, plains=None):
    errs = []
    got = [s["id"] for s in sub]
    if sorted(got) != sorted(qids) or len(set(got)) != len(got):
        errs.append(f"id câu hỏi không khớp: có {len(set(got))}, cần {len(qids)}")
    for s in sub:
        if not isinstance(s["id"], int) or isinstance(s["id"], bool):
            errs.append(f"id {s['id']!r} không phải số nguyên")
        for d in s["relevant_docs"]:
            if not isinstance(d, int) or d not in link_ids:
                errs.append(f"câu {s['id']}: doc_id {d!r} không có trong links")
        for c in s["relevant_chunks"]:
            d, t = c["doc_id"], c["chunk_text"]
            if not isinstance(d, int) or d not in link_ids:
                errs.append(f"câu {s['id']}: chunk doc_id {d!r} không có trong links")
            elif not t:
                errs.append(f"câu {s['id']}: chunk_text rỗng")
            elif plains is not None and d in plains and t not in plains[d]:
                errs.append(f"câu {s['id']}: chunk_text không phải chuỗi con của bài {d}")
    return errs


def write(sub, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, zp = out_dir / f"{name}.json", out_dir / f"{name}.zip"
    with js.open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(js, arcname=js.name)                       # đúng một file ở gốc
    return js, zp


def corpus_plains(cfg, doc_ids):
    import pyarrow.dataset as ds
    from .chunk import parse
    t = ds.dataset(str(cfg.path("corpus"))).to_table(columns=["id", "text"],
                                                     filter=ds.field("id").isin(list(doc_ids))).to_pydict()
    return {int(i): parse(x or "")[0] for i, x in zip(t["id"], t["text"])}


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--k-doc", type=int, default=None)
    ap.add_argument("--k-chunk", type=int, default=None)
    ap.add_argument("--doc-agg", default=None, choices=list(runs.DOC_AGGS),
                    help="thứ hạng tài liệu: max (đoạn tốt nhất) | sum2 | sum3 (mặc định submit.doc_agg)")
    ap.add_argument("--verify-corpus", action="store_true", help="đối chiếu chunk_text với chữ thuần của bài (chậm hơn)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    sc = cfg["submit"]
    k_doc, k_chunk = a.k_doc or sc["k_doc"], a.k_chunk or sc["k_chunk"]
    doc_agg = a.doc_agg or sc.get("doc_agg", "max")
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    run = runs.read(a.run, depth=max(k_doc * 3, k_chunk + 100))
    ids = run[run["rank"] <= k_chunk + 100]["unit_id"].unique().tolist()
    texts = {u: r[2] for u, r in unit_rows(cfg, ids).items()}
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    name = a.name or Path(a.run).stem + ("" if doc_agg == "max" else f"_{doc_agg}")
    out_dir = cfg.path("submissions")
    while True:
        sub = build(run, qids, k_doc, k_chunk, texts, doc_agg, cfg["rerank"]["top_n"])
        plains = corpus_plains(cfg, {c["doc_id"] for s in sub for c in s["relevant_chunks"]}) if a.verify_corpus else None
        errs = validate(sub, qids, link_ids, plains)
        if errs:
            print(f"LỖI ({len(errs)}), KHÔNG ghi bài nộp:\n  " + "\n  ".join(errs[:30]))
            raise SystemExit(1)
        js, zp = write(sub, out_dir, name)
        mb = zp.stat().st_size / 1e6
        if mb <= sc["max_zip_mb"] or k_chunk <= 10:
            break
        k_new = int(k_chunk * 0.95)
        print(f"  {zp.name}: {mb:.1f} MB > {sc['max_zip_mb']} MB → giảm k_chunk {k_chunk} → {k_new}")
        k_chunk = k_new
    n_chunks = sum(len(s["relevant_chunks"]) for s in sub)
    (out_dir / f"{name}_meta.json").write_text(json.dumps(
        {"run": a.run, "k_doc": k_doc, "k_chunk": k_chunk, "doc_agg": doc_agg, "zip_mb": round(mb, 1), "queries": len(sub),
         "chunks": n_chunks}, indent=1), encoding="utf-8")
    print(f"Hợp lệ: {len(sub)} câu hỏi, {n_chunks:,} đoạn, k_doc={k_doc}, k_chunk={k_chunk} → {zp} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
