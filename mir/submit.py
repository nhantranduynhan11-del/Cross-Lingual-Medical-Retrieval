"""T8: tạo bài nộp từ run file cuối.

  relevant_docs  : top k_doc tài liệu (điểm tài liệu = thứ hạng đoạn tốt nhất), doc_id dạng chuỗi
  relevant_chunks: top k_chunk đoạn {"doc_id", "chunk_text"}; chunk_text = văn bản đoạn (hoặc đoạn cha nếu run ở cấp cha),
                   KHÔNG kèm tiêu đề, là chuỗi con nguyên văn của bài đã làm sạch.
Kiểm tra trước khi ghi: đủ mọi id câu hỏi (số nguyên), doc_id là chuỗi và có trong links_corpus, mỗi chunk_text là chuỗi
con của corpus_clean.text của tài liệu đó, mức rỗng vẫn là mảng rỗng. Nén ZIP chứa đúng một file ở gốc.

  python -m mir.submit --run runs/structure_G_rr.parquet [--name sub1]
     → submissions/<tên>.json + submissions/<tên>.zip (không ghi đè nếu đã có, trừ --force)
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from . import config, runs, store


def build(run, qids, k_doc, k_chunk, chunk_texts):
    by_q = {q: g.sort_values("rank") for q, g in run.groupby("qid")}
    sub = []
    for q in qids:
        g = by_q.get(q)
        docs, chunks = [], []
        if g is not None:
            docs = list(dict.fromkeys(g["doc_id"]))[:k_doc]
            for c, d in zip(g["chunk_id"], g["doc_id"]):
                if len(chunks) >= k_chunk:
                    break
                if c in chunk_texts:
                    chunks.append({"doc_id": str(d), "chunk_text": chunk_texts[c]})
        sub.append({"id": int(q), "relevant_docs": [str(d) for d in docs], "relevant_chunks": chunks})
    return sub


def validate(sub, qids, link_ids, clean_texts):
    errs = []
    got = [s["id"] for s in sub]
    if sorted(got) != sorted(qids) or len(set(got)) != len(got):
        errs.append(f"id câu hỏi không khớp: có {len(set(got))}, cần {len(qids)}")
    for s in sub:
        if not isinstance(s["id"], int) or isinstance(s["id"], bool):
            errs.append(f"id {s['id']!r} không phải số nguyên")
        if not isinstance(s.get("relevant_docs"), list) or not isinstance(s.get("relevant_chunks"), list):
            errs.append(f"id {s['id']}: thiếu mảng relevant_docs / relevant_chunks")
            continue
        for d in s["relevant_docs"]:
            if not isinstance(d, str) or not d.isdigit() or int(d) not in link_ids:
                errs.append(f"id {s['id']}: doc_id {d!r} không có trong links_corpus")
        for c in s["relevant_chunks"]:
            d, t = c.get("doc_id"), c.get("chunk_text")
            if not isinstance(d, str) or not d.isdigit() or int(d) not in link_ids:
                errs.append(f"id {s['id']}: chunk doc_id {d!r} không có trong links_corpus")
            elif not isinstance(t, str) or not t or t not in clean_texts.get(d, ""):
                errs.append(f"id {s['id']}: chunk_text không phải chuỗi con nguyên văn của bài {d}")
    return errs


def write(sub, out_dir, name, force=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, zp = out_dir / f"{name}.json", out_dir / f"{name}.zip"
    if (js.exists() or zp.exists()) and not force:
        raise FileExistsError(f"{js} / {zp} đã có (dùng --force)")
    js.write_text(json.dumps(sub, ensure_ascii=False, indent=1), encoding="utf-8")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(js, arcname=js.name)                          # đúng một file, ở gốc
    return js, zp


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", default=None, help="tên file bài nộp (mặc định = tên run)")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--clean", default=None)
    ap.add_argument("--k-doc", type=int, default=None)
    ap.add_argument("--k-chunk", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    sc = cfg["submit"]
    k_doc, k_chunk = a.k_doc or sc["k_doc"], a.k_chunk or sc["k_chunk"]
    strategy = a.strategy or cfg["chunk"]["strategy"]
    chunk_dir = cfg.path("chunks", strategy, a.chunks)
    clean_dir = cfg.path("clean", override=a.clean)
    qpath = cfg.path("queries", override=a.queries)
    qids = [int(x) for x in pq.read_table(qpath, columns=["id"]).column("id").to_pylist()]
    run = runs.read(a.run)
    ids = run[run["rank"] <= max(k_chunk, k_doc) * 3]["chunk_id"].unique().tolist()
    texts = store.unit_rows(chunk_dir, ids)
    sub = build(run, qids, k_doc, k_chunk, {k: v["text"] for k, v in texts.items()})
    link_ids = set(pq.read_table(cfg.links_path, columns=["id"]).column("id").to_pylist())
    used = {c["doc_id"] for s in sub for c in s["relevant_chunks"]}
    clean = {k: v["text"] for k, v in store.lookup(clean_dir, "doc_id", list(used), ["text"]).items()}
    errs = validate(sub, qids, link_ids, clean)
    if errs:
        print(f"LỖI ({len(errs)}), KHÔNG ghi bài nộp:\n  " + "\n  ".join(errs[:30]))
        raise SystemExit(1)
    name = a.name or Path(a.run).stem
    out_dir = cfg.path("submissions")
    try:
        js, zp = write(sub, out_dir, name, a.force)
    except FileExistsError as e:
        raise SystemExit(f"Không ghi đè: {e}")
    n_empty = sum(1 for s in sub if not s["relevant_docs"])
    (out_dir / f"{name}_meta.json").write_text(json.dumps(
        {"run": a.run, "k_doc": k_doc, "k_chunk": k_chunk, "queries": len(qids), "empty": n_empty}, indent=1),
        encoding="utf-8")
    print(f"Hợp lệ: {len(sub)} câu hỏi ({n_empty} không có kết quả) → {zp}")


if __name__ == "__main__":
    main()
