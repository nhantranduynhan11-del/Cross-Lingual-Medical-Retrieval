"""So các bài nộp leaderboard theo TỪNG CÂU từ details.jsonl của BTC: FINAL = (doc F2 + chunk F2) / 2 (macro, bỏ câu không
có gold — tái tạo đúng FINAL_SCORE). Chênh lệch cặp với bài gốc: KTC 95% bootstrap, p hoán vị dấu, hai nửa câu hỏi ngẫu
nhiên (cố định seed), số câu tăng/giảm. Chỉ dùng điểm F2 tổng mỗi câu — không đọc cờ đúng/sai từng tài liệu.

  python tools/lb_compare.py --base lin07_cross=path/details.jsonl --run lin07q_cross=path/details.jsonl [--run …]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"id": int(r["id"]),
                         "doc": r["document"]["f2"] if r["document"]["gold"] else np.nan,
                         "chunk": r["chunk"]["f2"] if r["chunk"]["gold"] else np.nan})
    d = pd.DataFrame(rows)
    d["final"] = (d["doc"] + d["chunk"]) / 2
    return d


def compare(base, run, seed=0, n=5000):
    m = run.merge(base, on="id", suffixes=("", "_b")).dropna()
    rng = np.random.default_rng(seed)
    half = np.random.default_rng(42).permutation(len(m)) < len(m) // 2
    out = {}
    for c in ("final", "doc", "chunk"):
        d = (m[c] - m[f"{c}_b"]).to_numpy()
        bs = rng.choice(d, (n, len(d))).mean(1)
        flips = (rng.choice([-1, 1], (n, len(d))) * d).mean(1)
        out[c] = {"delta": d.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                  "p": float((np.abs(flips) >= abs(d.mean()) - 1e-12).mean()),
                  "h1": d[half].mean(), "h2": d[~half].mean(), "up": int((d > 1e-9).sum()), "down": int((d < -1e-9).sum())}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="tên=đường dẫn details.jsonl của bài gốc")
    ap.add_argument("--run", action="append", required=True, help="tên=đường dẫn details.jsonl (lặp lại được)")
    ap.add_argument("--min-delta", type=float, default=0.0, help="mức tăng FINAL tối thiểu để nhận (luật đăng ký trước)")
    a = ap.parse_args()
    bn, bp = a.base.split("=", 1)
    base = load(bp)
    print(f"{bn}: FINAL {np.nanmean(base['doc']) / 2 + np.nanmean(base['chunk']) / 2:.4f} ({base['final'].notna().sum()} câu có gold)")
    for spec in a.run:
        n, p = spec.split("=", 1)
        r = load(p)
        fin = np.nanmean(r["doc"]) / 2 + np.nanmean(r["chunk"]) / 2
        res = compare(base, r)
        f = res["final"]
        ok = f["lo"] > 0 and f["h1"] > 0 and f["h2"] > 0 and f["delta"] >= a.min_delta
        print(f"\n{n}: FINAL {fin:.4f} | doc {np.nanmean(r['doc']):.4f} | chunk {np.nanmean(r['chunk']):.4f}  → "
              f"{'ĐẠT' if ok else 'KHÔNG đạt'} luật (Δ>0, KTC không chứa 0, 2 nửa cùng dấu, Δ ≥ {a.min_delta})")
        for c, s in res.items():
            print(f"  Δ {c:5s} {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] p={s['p']:.4f} | 2 nửa {s['h1']:+.4f} / "
                  f"{s['h2']:+.4f} | câu tăng/giảm {s['up']}/{s['down']}")


if __name__ == "__main__":
    main()
