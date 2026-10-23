"""Ghép run Qwen3 chạy theo hai nửa (600 câu public + 600 câu còn lại) thành run đủ 1.200 câu, rồi tạo ens (bge + Qwen3)
trên đủ 1.200 câu. Chạy trên máy (CPU), không cần mir2-chunks.

  python tools/assemble_full.py --parts results/lin07_qwen3i/runs/lin07_qwen3i_sub.parquet results/lin07_qwen3i_h600/runs/lin07_qwen3i_sub.parquet \
      --base results/lin_fusion/runs/lin07.parquet --bge results/lin07s/runs/lin07_cross.parquet --out results/full1200/runs
  → <out>/lin07_qwen3i_1200.parquet, <out>/lin07_ensi_1200.parquet
Kiểm tra: các phần không trùng câu, đủ mọi câu trong queries; phần public của run mới trùng khớp run cũ (--check).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402


def same_ranks(a, b, qids):
    m = runs.read(a, qids=qids).merge(runs.read(b, qids=qids), on=["qid", "unit_id"], how="outer", indicator=True)
    return (m["_merge"] == "both").all() and (m["rank_x"] == m["rank_y"]).all()


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--parts", nargs="+", required=True, help="các run Qwen3 trên tập con câu hỏi (không trùng nhau)")
    ap.add_argument("--base", required=True, help="run Stage 1 mà Qwen3 / bge đã rerank (vd lin07.parquet)")
    ap.add_argument("--bge", required=True, help="run bge (cross) đủ câu trên cùng run gốc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default="lin07", help="tên gốc của run kết quả: <stem>_qwen3i_1200, <stem>_ensi_1200")
    ap.add_argument("--check", nargs=2, metavar=("QWEN_OLD", "ENS_OLD"), default=None,
                    help="run cũ (vd lb_qwen/runs/lin07_qwen3i.parquet, lin07_ensi.parquet) để đối chiếu phần câu đã có")
    ap.add_argument("--allow-partial", action="store_true", help="cho phép thiếu câu (chỉ để thử)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = [runs.read(p) for p in a.parts]
    sets = [set(p["qid"].unique()) for p in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise SystemExit(f"{a.parts[i]} và {a.parts[j]} trùng {len(sets[i] & sets[j])} câu")
    allq = {int(x) for x in pq.read_table(cfg.path("queries"), columns=["id"]).column("id").to_pylist()}
    have = set().union(*sets)
    miss = allq - have
    print(f"Qwen3: {' + '.join(str(len(s)) for s in sets)} = {len(have)} câu; thiếu {len(miss)} / {len(allq)}")
    if miss and not a.allow_partial:
        raise SystemExit("Thiếu câu → chưa ghép (thêm phần còn thiếu hoặc --allow-partial)")
    qpath = out / f"{a.stem}_qwen3i_1200.parquet"
    df = pd.concat([p[["qid", "unit_id", "doc_id", "rank", "score"]] for p in parts], ignore_index=True)
    runs.write(df.sort_values(["qid", "rank"]).reset_index(drop=True), qpath,
               {"method": "qwen3i", "parts": [str(p) for p in a.parts], "qids": len(have)})
    print(f"→ {qpath}")

    subprocess.run([sys.executable, "-m", "mirret.ensemble", "--base", a.base, "--bge", a.bge, "--qwen", str(qpath),
                    "--out", str(out), "--suffix", "i_1200"] + (["--config", a.config] if a.config else []),
                   check=True, cwd=Path(__file__).resolve().parents[1])
    for lam in ("ip08", "ip06"):                          # ensemble.py luôn ghi thêm hai run nội suy — không dùng
        for f in out.glob(f"{Path(a.base).stem}_{lam}i_1200.*"):
            f.unlink()
    epath = out / f"{Path(a.base).stem}_ensi_1200.parquet"
    if not epath.exists():
        raise SystemExit(f"ensemble không tạo {epath}")

    if a.check:
        old = set(runs.read(a.check[0])["qid"].unique()) & set(runs.read(a.check[1])["qid"].unique())
        common = have & sets[0]                             # phần đã có (phần đầu, vd 600 câu public)
        print(f"đối chiếu {len(common)} câu đã có: qwen3i khớp {same_ranks(qpath, a.check[0], common)}, "
              f"ens khớp {same_ranks(epath, a.check[1], common & old)}")


if __name__ == "__main__":
    main()
