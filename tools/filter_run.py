"""Bỏ các bài PubMed khỏi run (giữ nguyên số hạng của các dòng còn lại để rank <= top_n vẫn đúng là phần đã rerank).

LB 25/10: bỏ mọi bài PubMed khỏi bài 0,1804 (final1200_cbud200_nopm) → doc R 0,4372 và chunk R 0,1538 KHÔNG đổi,
precision tăng (FINAL 0,1946) → đáp án public không có bài PubMed nào (toàn bộ tài liệu tiếng Anh của kho là PubMed).

  python tools/filter_run.py --runs results/full1200/runs/lin07_qwen3i_1200.parquet results/full1200/runs/lin07_ensi_1200.parquet \
      --out results/full1200_np/runs --suffix _np
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="_np")
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.runs:
        r = runs.read(p)
        keep = r[~r["doc_id"].isin(pm)].reset_index(drop=True)
        dst = out / f"{Path(p).stem}{a.suffix}.parquet"
        runs.write(keep, dst, {"from": str(p), "filter": "bỏ bài PubMed", "rows_before": len(r), "rows_after": len(keep)})
        top = r[r["rank"] <= 400]
        print(f"{Path(p).name}: bỏ {len(r) - len(keep)}/{len(r)} dòng ({1 - len(keep) / len(r):.1%}; top-400: "
              f"{top['doc_id'].isin(pm).mean():.1%}) → {dst}")


if __name__ == "__main__":
    main()
