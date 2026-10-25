"""Đổi doc_id của bài PubMed (url pubmed.ncbi.nlm.nih.gov/<PMID>/) trong bài nộp sang PMID dạng CHUỖI.

Lý do (25/10): máy chấm phân biệt id kho (số nguyên → "corpus:<id>") với PMID (chuỗi). Trong details.jsonl của
lin07_cross, 9.798 đoạn tiếng Anh (toàn bộ là bài PubMed) không đoạn nào khớp (precision 0,000) dù 64% là cả bài —
giả thuyết: đáp án ghi bài PubMed bằng PMID. Bài nộp này giữ nguyên mọi thứ, chỉ đổi kiểu id của bài PubMed.
KẾT QUẢ LB 25/10: giống hệt bài gốc (0,1804, mọi chỉ số) và luật thi ghi rõ mọi nhãn dùng id số nguyên trong
links_corpus → giả thuyết sai, KHÔNG dùng (giữ lại chỉ để ghi nhận).

  python tools/pmid_submission.py --sub results/mix_submit/final1200_cbud200.json --name final1200_cbud200_pmid
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, submit  # noqa: E402

PM = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sub", required=True, help="bài nộp .json (doc_id số nguyên)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/mix_submit")
    a = ap.parse_args()
    cfg = config.load(a.config)
    t = pq.read_table(cfg.path("links"), columns=["id", "url"]).to_pydict()
    pmid = {}
    for i, u in zip(t["id"], t["url"]):
        m = PM.search(u or "")
        if m:
            pmid[int(i)] = m.group(1)
    print(f"links: {len(t['id'])} id, {len(pmid)} bài PubMed")
    sub = json.loads(Path(a.sub).read_text("utf-8"))
    nd = nc = td = tc = 0
    for s in sub:
        td += len(s["relevant_docs"])
        tc += len(s["relevant_chunks"])
        docs = []
        for d in s["relevant_docs"]:
            docs.append(pmid.get(d, d))
            nd += d in pmid
        s["relevant_docs"] = docs
        for c in s["relevant_chunks"]:
            if c["doc_id"] in pmid:
                c["doc_id"] = pmid[c["doc_id"]]
                nc += 1
    print(f"đổi sang PMID: {nd}/{td} tài liệu ({nd / max(1, td):.1%}), {nc}/{tc} đoạn ({nc / max(1, tc):.1%})")
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    errs = [e for e in submit.validate(sub, qids, set(t["id"])) if "không có trong links" not in e]
    bad = [d for s in sub for d in s["relevant_docs"] if not (isinstance(d, int) or (isinstance(d, str) and d.isdigit()))]
    if errs or bad:
        raise SystemExit(f"LỖI: {errs[:5]} {bad[:5]}")
    js, zp = submit.write(sub, a.out, a.name)
    print(f"→ {zp} ({zp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
