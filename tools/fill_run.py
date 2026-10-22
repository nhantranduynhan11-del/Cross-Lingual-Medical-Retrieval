"""Lấp đủ 1.200 câu cho một run chỉ chạy trên tập con (vd 600 câu public của leaderboard) bằng run nền, để tạo bài nộp
hợp lệ. Câu nào có trong --run giữ nguyên; câu còn lại lấy từ --base (không được chấm ở public nên không ảnh hưởng điểm).

  python tools/fill_run.py --run lin07_qwen3i.parquet --base lin07_cross.parquet --out runs/lin07_qwen3i_full.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run trên tập con (giữ nguyên)")
    ap.add_argument("--base", required=True, help="run đủ câu dùng để lấp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, base = runs.read(a.run), runs.read(a.base)
    mine = set(run["qid"])
    fill = base[~base["qid"].isin(mine)]
    df = pd.concat([run[["qid", "unit_id", "doc_id", "rank", "score"]], fill[["qid", "unit_id", "doc_id", "rank", "score"]]],
                   ignore_index=True).sort_values(["qid", "rank"]).reset_index(drop=True)
    runs.write(df, Path(a.out), {"run": a.run, "base_fill": a.base, "qids_run": len(mine), "qids_fill": fill["qid"].nunique()})
    print(f"→ {a.out}: {len(mine)} câu từ run, {fill['qid'].nunique()} câu lấp từ nền, tổng {df['qid'].nunique()} câu")


if __name__ == "__main__":
    main()
