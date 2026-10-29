"""Bài nộp ghép: lấy --base, thay nguyên mục (tài liệu + đoạn) của các câu trong --qids bằng mục của --over. Dùng khi chạy
thử một thành phần trên một phần câu (vd giám khảo LLM trên public600_a): chênh lệch LB so với --base chỉ đến từ các câu đó.

  python tools/splice_sub.py --base results/mix_submit/final1200_vi2300.json \
      --over results/mix_submit/judge_np_full.json --qids results/qids/public600_a.txt --name final1200_judge_pa
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs, submit  # noqa: E402


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--over", required=True)
    ap.add_argument("--qids", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/mix_submit")
    a = ap.parse_args()
    cfg = config.load(a.config)
    qs = runs.read_qids(a.qids)
    base = {int(s["id"]): s for s in json.loads(Path(a.base).read_text("utf-8"))}
    over = {int(s["id"]): s for s in json.loads(Path(a.over).read_text("utf-8"))}
    lost = sorted(q for q in qs if q not in over)
    if lost:
        raise SystemExit(f"{len(lost)} câu của --qids không có trong --over, vd {lost[:5]}")
    sub = [over[q] if q in qs else s for q, s in base.items()]
    qids = [int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()]
    link_ids = set(pq.read_table(cfg.path("links"), columns=["id"]).column("id").to_pylist())
    errs = submit.validate(sub, qids, link_ids)
    if errs:
        raise SystemExit(f"LỖI ({len(errs)}): {errs[:5]}")
    js, zp = submit.write(sub, a.out, a.name)
    n = sum(q in qs for q in base)
    print(f"thay {n} câu từ {a.over}; → {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    (Path(a.out) / f"{a.name}_meta.json").write_text(json.dumps(vars(a), indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
