"""Đóng gói dataset Kaggle cho pipeline truy hồi.

  python tools/pack_kaggle_ret.py code --user <tên_kaggle>   → build/mir2_code/: mir2_code.zip (mirret/, mircrawl/,
                                                               configs/, VERSION.txt) + dataset-metadata.json
  python tools/pack_kaggle_ret.py data --user <tên_kaggle>   → build/mir2_data/: data/query.parquet, data/links…,
                                                               corpus_out/corpus/*.parquet (hard link nếu cùng ổ) + VERSION.txt
  python tools/pack_kaggle_ret.py qtrans --user <tên_kaggle> [--src results/mir-qtrans] → build/mir2_qtrans/ (bản dịch k2)
  kaggle datasets create  -p build/mir2_code                  (lần đầu; lần sau: kaggle datasets version -p … -m "…")
  kaggle datasets create  -p build/mir2_data --dir-mode zip

Notebook tìm dữ liệu bằng đường dẫn tương đối (data/query.parquet, corpus_out/corpus) ở mọi cấp trong /kaggle/input,
nên Kaggle giải nén hay giữ thư mục con đều được. VERSION.txt in ở ô (1) của notebook để biết đang chạy đúng bản.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def place(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def meta(out, user, slug, title):
    (out / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": f"{user}/{slug}",
                                                           "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


def pack_code(user, out, config=None):
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    h = hashlib.sha1()
    files = []
    for pkg in ("mirret", "mircrawl"):
        for p in sorted((ROOT / pkg).rglob("*.py")):
            if "__pycache__" not in p.parts:
                files.append(p)
    cfg_src = Path(config) if config else ROOT / "configs" / "retrieval.yaml"
    for p in files + [cfg_src] + sorted((ROOT / "configs").glob("*.tsv")) + sorted((ROOT / "configs").glob("*.txt")):
        h.update(p.read_bytes())
    version = f"{datetime.datetime.now():%Y-%m-%d %H:%M} sha1 {h.hexdigest()[:10]}"
    with zipfile.ZipFile(out / "mir2_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(cfg_src, "configs/retrieval.yaml")
        for extra in sorted((ROOT / "configs").glob("*.tsv")) + sorted((ROOT / "configs").glob("*.txt")):  # bảng thuật ngữ, userdict jieba
            z.write(extra, f"configs/{extra.name}")
        z.writestr("VERSION.txt", version + "\n")
    if user:
        meta(out, user, "mir2-code", "mir2-code")
    print(f"→ {out / 'mir2_code.zip'} ({len(files)} file .py) | VERSION {version}")


def pack_data(user, out):
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for rel in ("data/query.parquet", "data/links_corpus_vi_en_zh.parquet"):
        place(ROOT / rel, out / rel)
    corpus = sorted((ROOT / "corpus_out" / "corpus").glob("*.parquet"))
    if not corpus:
        sys.exit("Chưa có corpus_out/corpus (chạy python -m mircrawl merge …)")
    for p in corpus:
        place(p, out / "corpus_out" / "corpus" / p.name)
    summ = ROOT / "corpus_out" / "summary.json"
    info = json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else {}
    (out / "VERSION.txt").write_text(f"{datetime.datetime.now():%Y-%m-%d %H:%M} | corpus {len(corpus)} file, "
                                     f"{info.get('n_corpus', '?')} tài liệu, built_at {info.get('built_at', '?')}\n",
                                     encoding="utf-8")
    if user:
        meta(out, user, "mir2-data", "mir2-data")
    gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    print(f"→ {out}: {len(corpus)} file corpus, {gb:.2f} GB")


def pack_qtrans(user, out, srcs):
    """Output k2 đã tải về (các thư mục có qtrans/<khoá>/records*.jsonl, gộp được nhiều lần chạy) → dataset mir2-qtrans:
    chỉ bản ghi + thư mục so sánh (phiếu, khoá). Trùng đường dẫn: lấy bản ở thư mục ghi sau."""
    pairs = {}
    for src in srcs:
        src = Path(src)
        for f in sorted(src.glob("qtrans/*/records*.jsonl")) + sorted(src.glob("qtrans/compare_*/*")):
            pairs[f.relative_to(src)] = f
    files = list(pairs.values())
    if not any(f.name.startswith("records") for f in files):
        sys.exit(f"Không thấy qtrans/<model>/records*.jsonl trong {srcs}")
    shutil.rmtree(out, ignore_errors=True)
    for rel, f in pairs.items():
        place(f, out / rel)
    if user:
        meta(out, user, "mir2-qtrans", "mir2-qtrans")
    for d in sorted({f.parent.name for f in files}):
        print(f"  {d}: {sum(1 for f in files if f.parent.name == d)} file")
    print(f"→ {out}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # console Windows (cp1252)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["code", "data", "qtrans"])
    ap.add_argument("--user", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None, help="code: dùng file cấu hình khác thay cho configs/retrieval.yaml")
    ap.add_argument("--src", default="results/mir-qtrans", help="qtrans: thư mục output k2 đã tải về (phẩy ngăn cách)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "build" / f"mir2_{a.what}"
    if a.what == "code":
        pack_code(a.user, out, a.config)
    elif a.what == "qtrans":
        pack_qtrans(a.user, out, [ROOT / x for x in a.src.split(",")])
    else:
        pack_data(a.user, out)


if __name__ == "__main__":
    main()
