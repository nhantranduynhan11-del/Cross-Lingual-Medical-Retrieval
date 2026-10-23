"""Tạo bài nộp từ run cuối (đoạn cha).

  relevant_docs  : top k_doc tài liệu (thứ hạng tài liệu theo submit.doc_agg: max = đoạn tốt nhất, sum2/sum3 = tổng
                   điểm 2/3 đoạn tốt nhất — runs.doc_ranking), doc_id là SỐ NGUYÊN (id của links_corpus;
                   dự án cũ: ghi dạng chuỗi → máy chấm coi là PMID → 0 điểm)
  relevant_chunks: top k_chunk đoạn cha {"doc_id", "chunk_text"}; chunk_text = chữ thuần nguyên văn của đoạn (không kèm
                   ngữ cảnh). Đoạn gần trùng (>= 0,8, BTC gộp) trong cùng câu hỏi bị bỏ để nhường chỗ.
Kiểm tra: đủ 1.200 câu, id nguyên, doc_id có trong links, chunk_text khớp chữ thuần của bài (--verify-corpus).
File nén > submit.max_zip_mb → giảm k_chunk 5% mỗi lần tới khi vừa (báo lại).

Phân bổ lại ngân sách đoạn (--chunk-probs run_qwen3.parquet): tổng số đoạn giữ bằng k_chunk × số câu (giới hạn file nén),
nhưng mỗi câu được k ∈ [--chunk-k-min, --chunk-k-max] đoạn. p(đoạn) = P(yes) của đoạn trong top-N đã rerank (ngoài top-N
= 0), G = Σ p của top-N; tham lam: mỗi lần cấp thêm một đoạn cho câu có mức tăng kỳ vọng F2 ≈ 5·Σ_{i≤k} p_i / (4G + k)
lớn nhất (dừng khi hết ngân sách hoặc không còn mức tăng dương). Câu không có xác suất hợp lệ (vd câu lấp bằng
tools/fill_run.py: điểm ngoài [0, 1]) giữ k_chunk. Không có tham số dò.

  python -m mirret.submit --run runs/rrf9_colbert.parquet [--name bai1] [--k-doc 200] [--k-chunk 185] [--doc-agg sum2]
                           [--chunk-probs runs/lin07_qwen3i.parquet] [--verify-corpus]
"""
import argparse
import heapq
import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import config, runs
from .rerank import unit_rows


def _norm(s):
    return " ".join((s or "").lower().split())


def pick_units(g, k, max_per_doc=0):
    """Đoạn theo thứ hạng; max_per_doc > 0 → mỗi tài liệu tối đa n đoạn (gold ~1 đoạn / tài liệu: nhường chỗ cho tài
    liệu kế tiếp thay vì đoạn thứ 3–7 của cùng tài liệu)."""
    cnt, out = {}, []
    for u, d in zip(g["unit_id"], g["doc_id"]):
        if len(out) >= k:
            break
        if max_per_doc and cnt.get(d, 0) >= max_per_doc:
            continue
        cnt[d] = cnt.get(d, 0) + 1
        out.append((u, d))
    return out


def chunk_probs(path, top_n=400):
    """{qid: ({unit_id: p}, G)} từ run có điểm P(yes); bỏ câu có điểm ngoài [0, 1] trong top-N (câu lấp bằng run khác)."""
    r = runs.read(path, depth=top_n)
    out = {}
    for q, g in r.groupby("qid"):
        s = g["score"].to_numpy()
        if len(s) and s.min() >= 0 and s.max() <= 1:
            out[int(q)] = (dict(zip(g["unit_id"], s)), float(s.sum()))
    return out


def allocate(cands, probs, k_chunk, k_min, k_max):
    """Số đoạn mỗi câu: câu có xác suất chia chung ngân sách k_chunk × số câu đó (tham lam theo mức tăng kỳ vọng F2)."""
    ks = {q: min(k_chunk, len(c)) for q, c in cands.items()}
    pool = [q for q in cands if q in probs]
    if not pool:
        return ks
    cum, G = {}, {}
    for q in pool:
        pu, G[q] = probs[q]
        cum[q] = np.concatenate([[0.0], np.cumsum([pu.get(u, 0.0) for u, _, _ in cands[q][:k_max]])])
    ef2 = lambda q, k: 5 * cum[q][k] / (4 * G[q] + k) if k else 0.0
    budget = k_chunk * len(pool)
    heap = []
    for q in pool:
        ks[q] = min(k_min, len(cum[q]) - 1)
        budget -= ks[q]
        if ks[q] < len(cum[q]) - 1:
            heap.append((-(ef2(q, ks[q] + 1) - ef2(q, ks[q])), q))
    heapq.heapify(heap)
    while budget > 0 and heap:
        gain, q = heapq.heappop(heap)
        if -gain <= 0:
            break
        ks[q] += 1
        budget -= 1
        if ks[q] < len(cum[q]) - 1:
            heapq.heappush(heap, (-(ef2(q, ks[q] + 1) - ef2(q, ks[q])), q))
    k = np.array([ks[q] for q in pool])
    print(f"  phân bổ đoạn ({len(pool)} câu có xác suất): TB {k.mean():.1f}, trung vị {int(np.median(k))}, "
          f"p10–p90 {int(np.percentile(k, 10))}–{int(np.percentile(k, 90))}, ={k_min}: {(k == k_min).mean():.0%}, "
          f"={k_max}: {(k == k_max).mean():.0%}")
    return ks


def build(run, qids, k_doc, k_chunk, texts, doc_agg="max", norm_depth=400, max_per_doc=0, probs=None,
          chunk_k=(100, 400)):
    by_q = {q: g.sort_values("rank") for q, g in run.groupby("qid")}
    dr = runs.doc_ranking(run, doc_agg, norm_depth).sort_values(["qid", "drank"])
    docs_q = {q: [int(d) for d in g["doc_id"]][:k_doc] for q, g in dr.groupby("qid")}
    cap = max(k_chunk, chunk_k[1]) if probs else k_chunk
    cands = {}
    for q in qids:
        g = by_q.get(q)
        chunks, seen, cnt = [], set(), {}
        if g is not None:
            for u, d in zip(g["unit_id"], g["doc_id"]):
                if len(chunks) >= cap:
                    break
                if max_per_doc and cnt.get(d, 0) >= max_per_doc:
                    continue
                t = texts.get(u)
                if not t:
                    continue
                key = (int(d), _norm(t)[:300])
                if key in seen:
                    continue
                seen.add(key)
                cnt[d] = cnt.get(d, 0) + 1
                chunks.append((u, int(d), t))
        cands[q] = chunks
    ks = allocate(cands, probs, k_chunk, *chunk_k) if probs else {q: k_chunk for q in qids}
    return [{"id": int(q), "relevant_docs": docs_q[q] if q in by_q else [],
             "relevant_chunks": [{"doc_id": d, "chunk_text": t} for _, d, t in cands[q][:ks[q]]]} for q in qids]


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
    ap.add_argument("--max-per-doc", type=int, default=None,
                    help="tối đa n đoạn mỗi tài liệu trong danh sách đoạn (0 = không giới hạn; mặc định submit.max_per_doc)")
    ap.add_argument("--chunk-probs", default=None,
                    help="run có điểm P(yes) (qwen3/qwen3i) → phân bổ lại ngân sách đoạn giữa các câu")
    ap.add_argument("--chunk-k-min", type=int, default=100)
    ap.add_argument("--chunk-k-max", type=int, default=400)
    ap.add_argument("--verify-corpus", action="store_true", help="đối chiếu chunk_text với chữ thuần của bài (chậm hơn)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    sc = cfg["submit"]
    k_doc, k_chunk = a.k_doc or sc["k_doc"], a.k_chunk or sc["k_chunk"]
    doc_agg = a.doc_agg or sc.get("doc_agg", "max")
    mpd = a.max_per_doc if a.max_per_doc is not None else int(sc.get("max_per_doc", 0) or 0)
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    probs = chunk_probs(a.chunk_probs, cfg["rerank"]["top_n"]) if a.chunk_probs else None
    if a.chunk_probs:
        print(f"  xác suất đoạn: {len(probs)} câu từ {a.chunk_probs}")
    k_need = max(k_chunk, a.chunk_k_max) if probs else k_chunk
    run = runs.read(a.run, depth=max(k_doc * 3, k_need + 100))
    if mpd:                                               # đoạn được chọn có thể nằm sâu hơn k_chunk + 100
        ids = list({u for _, g in run.sort_values(["qid", "rank"]).groupby("qid")
                     for u, _ in pick_units(g, k_need + 100, mpd)})
    else:
        ids = run[run["rank"] <= k_need + 100]["unit_id"].unique().tolist()
    texts = {u: r[2] for u, r in unit_rows(cfg, ids).items()}
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    name = a.name or (Path(a.run).stem + ("" if doc_agg == "max" else f"_{doc_agg}") + (f"_mpd{mpd}" if mpd else "")
                      + ("_cbud" if probs else ""))
    out_dir = cfg.path("submissions")
    while True:
        sub = build(run, qids, k_doc, k_chunk, texts, doc_agg, cfg["rerank"]["top_n"], mpd, probs,
                    (a.chunk_k_min, a.chunk_k_max))
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
        {"run": a.run, "k_doc": k_doc, "k_chunk": k_chunk, "doc_agg": doc_agg, "max_per_doc": mpd,
         "chunk_probs": a.chunk_probs, "chunk_k": [a.chunk_k_min, a.chunk_k_max] if probs else None, "zip_mb": round(mb, 1), "queries": len(sub),
         "chunks": n_chunks}, indent=1), encoding="utf-8")
    print(f"Hợp lệ: {len(sub)} câu hỏi, {n_chunks:,} đoạn, k_doc={k_doc}, k_chunk={k_chunk} → {zp} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
