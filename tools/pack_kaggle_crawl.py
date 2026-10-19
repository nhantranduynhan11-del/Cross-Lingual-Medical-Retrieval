"""Đóng gói code + danh sách URL thành thư mục dataset để tải lên Kaggle.

    python tools/pack_kaggle_crawl.py --user <tên_kaggle> [--slug vibiomir-crawl]
    kaggle datasets create  -p build/kaggle_crawl            # lần đầu
    kaggle datasets version -p build/kaggle_crawl -m "…"     # khi sửa code (notebook tự chép bản mới)

Hoặc tải thủ công 2 file trong build/kaggle_crawl/ lên trang Kaggle → Datasets → New Dataset.
Code được nén thành mircrawl.zip (notebook tự giải nén) để không phụ thuộc cách Kaggle xử lý thư mục con.
"""
import argparse
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="tên người dùng Kaggle (chủ dataset)")
    ap.add_argument("--slug", default="vibiomir-crawl")
    ap.add_argument("--links", default=os.path.join(ROOT, "data", "links_corpus_vi_en_zh.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "kaggle_crawl"))
    a = ap.parse_args()
    if not os.path.exists(a.links):
        sys.exit(f"Không thấy {a.links}")
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    src = os.path.join(ROOT, "mircrawl")
    with zipfile.ZipFile(os.path.join(a.out, "mircrawl.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(src):
            if "__pycache__" in d:
                continue
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(d, f)
                    z.write(p, os.path.relpath(p, ROOT))
    shutil.copyfile(a.links, os.path.join(a.out, os.path.basename(a.links)))
    meta = {"title": a.slug, "id": f"{a.user}/{a.slug}", "licenses": [{"name": "other"}]}
    with open(os.path.join(a.out, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"→ {a.out}: mircrawl.zip, {os.path.basename(a.links)}, dataset-metadata.json")
    print(f"  kaggle datasets create -p \"{a.out}\"   (hoặc: kaggle datasets version -p \"{a.out}\" -m \"cập nhật\")")


if __name__ == "__main__":
    main()
