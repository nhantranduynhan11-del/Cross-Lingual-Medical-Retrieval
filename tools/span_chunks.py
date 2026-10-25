"""Danh sách đoạn kiểu "mỗi tài liệu một khúc liền mạch" (chạy trên máy: cần corpus_out, không cần mir2-chunks).

Lý do (26/10, số liệu tổng của details.jsonl lin07_cross, máy chấm LCS ≥ 0,4 cả hai phía):
  - đoạn cha của mình ~253 token BTC; cận trên độ dài đáp án (đoạn đúng: token / độ trùng) vi p50 488 (p25 377,
    p75 603), zh p50 271 → đáp án vi dài ~gấp đôi đoạn cha, zh xấp xỉ;
  - precision theo độ dài đoạn tăng dần (200–250 token 3,5% → 350–400 token 9,8%).
Cách làm (cố định trước khi nộp): với mỗi tài liệu theo thứ tự xuất hiện trong danh sách đoạn --chunks (thứ hạng ens),
neo = đoạn đầu tiên của tài liệu đó; khúc = chữ thuần của bài (mirret.chunk.parse — đúng nguồn của chunk_text) mở rộng
ĐỀU hai phía quanh neo tới --target-<lang> ký tự (vi 1.900 ≈ 500 token BTC, zh 420 ≈ 300 token); bài ngắn hơn đích →
cả bài; neo dài hơn đích → giữ neo. Mỗi tài liệu một khúc. Số khúc mỗi câu: phân bổ ngân sách như submit.allocate với
p = P(yes) lớn nhất của tài liệu (run --probs), tổng = --per-query × số câu. Danh sách tài liệu lấy nguyên từ --docs.

  python tools/span_chunks.py --docs results/mix_submit/final1200_cbud200_nopm.json \
      --chunks results/full1200/lin07_ensi_1200_cbud.json --probs results/full1200_np/runs/lin07_qwen3i_1200_np.parquet \
      --name final1200_span --per-query 130
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs, submit  # noqa: E402
from mirret.chunk import parse  # noqa: E402


def span(plain, anchor, target):
    """Khúc liền mạch quanh neo, dài ~target ký tự (chuỗi con của plain)."""
    if len(plain) <= target:
        return plain.strip() or anchor
    a = plain.find(anchor)
    if a < 0 or len(anchor) >= target:
        return anchor
    b = a + len(anchor)
    need = target - (b - a)
    lo, hi = a - need // 2, b + (need - need // 2)
    if lo < 0:
        hi, lo = min(len(plain), hi - lo), 0
    if hi > len(plain):
        lo, hi = max(0, lo - (hi - len(plain))), len(plain)
    if lo > 0 and not plain[lo - 1].isspace():                 # cắt ở ranh giới từ (vi); zh không có khoảng trắng → giữ
        j = next((i for i in range(lo, a) if plain[i - 1].isspace()), None)
        lo = j if j is not None else lo
    if hi < len(plain) and not plain[hi].isspace():
        j = next((i for i in range(hi, b, -1) if plain[i].isspace()), None)
        hi = j if j is not None else hi
    return plain[lo:hi].strip()


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--docs", required=True, help="bài nộp .json lấy nguyên danh sách tài liệu")
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="bài nộp .json có đoạn theo thứ hạng ens (neo = đoạn đầu tiên của tài liệu; nhiều file → hợp)")
    ap.add_argument("--order", default=None, help="run ens: sắp tài liệu theo thứ hạng (cần khi --chunks nhiều file)")
    ap.add_argument("--probs", required=True, help="run Qwen3 có P(yes) (p của tài liệu)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/mix_submit")
    ap.add_argument("--per-query", type=int, default=130, help="số khúc TB mỗi câu (ngân sách)")
    ap.add_argument("--k-min", type=int, default=40)
    ap.add_argument("--k-max", type=int, default=300)
    ap.add_argument("--target-vi", type=int, default=1900)
    ap.add_argument("--target-zh", type=int, default=420)
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    docs = {int(s["id"]): s["relevant_docs"] for s in json.loads(Path(a.docs).read_text("utf-8"))}
    anchors = {}
    for f in a.chunks:
        for s in json.loads(Path(f).read_text("utf-8")):
            seen = anchors.setdefault(int(s["id"]), {})
            for c in s["relevant_chunks"]:
                d = int(c["doc_id"])
                if d not in pm and d not in seen:
                    seen[d] = c["chunk_text"]
    if a.order:                                               # hợp nhiều file → sắp lại theo thứ hạng tài liệu của run ens
        dr = runs.doc_ranking(runs.read(a.order, depth=1000), "max", cfg["rerank"]["top_n"])
        pos = {(int(q), int(d)): int(k) for q, d, k in zip(dr["qid"], dr["doc_id"], dr["drank"])}
        anchors = {q: sorted(v.items(), key=lambda x: pos.get((q, x[0]), 10 ** 9)) for q, v in anchors.items()}
    else:
        anchors = {q: list(v.items()) for q, v in anchors.items()}   # (doc, neo) theo thứ tự ens
    need = sorted({d for v in anchors.values() for d, _ in v})
    print(f"tài liệu cần chữ thuần: {len(need)}")
    t = ds.dataset(str(cfg.path("corpus"))).to_table(columns=["id", "lang", "text"],
                                                     filter=ds.field("id").isin(need)).to_pydict()
    plain = {int(i): parse(x or "")[0] for i, x in zip(t["id"], t["text"])}
    lang = dict(zip([int(i) for i in t["id"]], t["lang"]))
    r = runs.read(a.probs, depth=cfg["rerank"]["top_n"])
    r = r[(r["score"] >= 0) & (r["score"] <= 1)]
    pdoc = r.groupby(["qid", "doc_id"])["score"].max()
    cands, probs, stat = {}, {}, {"whole": 0, "span": 0, "anchor": 0, "miss": 0}
    for q, lst in anchors.items():
        items = []
        for d, anc in lst:
            p = plain.get(d)
            if p is None:
                stat["miss"] += 1
                txt = anc
            else:
                tgt = a.target_zh if lang.get(d) == "zh" else a.target_vi
                txt = span(p, anc, tgt)
                stat["whole" if txt == p.strip() else ("anchor" if txt == anc else "span")] += 1
            items.append((d, d, txt))
        cands[q] = items
        if q in pdoc.index.get_level_values(0):
            pq_ = pdoc.loc[q]
            pu = {d: float(pq_.get(d, 0.0)) for d, _, _ in items}
            probs[q] = (pu, float(pq_.sum()))
    tot = sum(stat.values())
    print("khúc: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in stat.items()))
    ks = submit.allocate(cands, probs, a.per_query, a.k_min, a.k_max)
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    sub = [{"id": q, "relevant_docs": docs.get(q, []),
            "relevant_chunks": [{"doc_id": d, "chunk_text": x} for d, _, x in cands.get(q, [])[:ks.get(q, 0)]]}
           for q in qids]
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    errs = submit.validate(sub, qids, link_ids, {d: plain[d] for d in plain})
    if errs:
        raise SystemExit(f"LỖI ({len(errs)}): {errs[:5]}")
    ln = [len(c["chunk_text"]) for s in sub for c in s["relevant_chunks"]]
    print(f"đoạn: TB {np.mean([len(s['relevant_chunks']) for s in sub]):.0f}/câu, độ dài TB {np.mean(ln):.0f} ký tự")
    js, zp = submit.write(sub, a.out, a.name)
    (Path(a.out) / f"{a.name}_meta.json").write_text(json.dumps(vars(a), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {zp} ({zp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
