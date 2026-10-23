"""Công cụ kiểm tra, dọn dẹp và hợp nhất các khối vector embedding mE5 (Role C).

Sử dụng:
  # 1. Kiểm tra hiện trạng thư mục emb_e5 đã tải về:
  python tools/verify_emb.py --dir ../path/to/dest/emb_e5

  # 2. Dọn dẹp các khối .tmp lỗi do tràn đĩa:
  python tools/verify_emb.py --dir ../path/to/dest/emb_e5 --clean

  # 3. Hợp nhất các khối mới chạy xong từ Kaggle (Part 2) vào bộ dữ liệu chính:
  python tools/verify_emb.py --dir ../path/to/dest/emb_e5 --merge-from /path/to/downloaded_part2/emb_e5
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def inspect_level(root_dir, level, expected_blocks, total_units, fp_expected=None, clean=False):
    level_dir = Path(root_dir) / level
    if not level_dir.exists():
        return {
            "level": level, "status": "MISSING", "total_blocks": 0, "expected_blocks": expected_blocks,
            "missing_blocks": list(range(expected_blocks)), "valid_blocks": [], "tmp_cleaned": []
        }

    valid = []
    corrupted = []
    tmp_folders = []

    for item in sorted(level_dir.iterdir()):
        if item.is_dir():
            if item.name.endswith(".tmp"):
                tmp_folders.append(item)
                if clean:
                    shutil.rmtree(item)
                continue
            done_file = item / "_DONE.json"
            dense_file = item / "dense.npy"
            ids_file = item / "ids.parquet"
            if done_file.exists() and dense_file.exists() and ids_file.exists():
                try:
                    meta = json.loads(done_file.read_text(encoding="utf-8"))
                    blk_idx = int(meta.get("block", item.name.split("-")[-1]))
                    valid.append((blk_idx, item.name, meta))
                except Exception:
                    corrupted.append(item.name)
            else:
                corrupted.append(item.name)

    valid_dict = {b[0]: b for b in valid}
    missing = [i for i in range(expected_blocks) if i not in valid_dict]
    
    total_encoded_units = sum(b[2].get("n", 100000) for b in valid)
    pct = (total_encoded_units / total_units) * 100 if total_units else 0

    return {
        "level": level,
        "valid_count": len(valid),
        "expected_blocks": expected_blocks,
        "missing_blocks": missing,
        "missing_count": len(missing),
        "corrupted": corrupted,
        "tmp_folders": [p.name for p in tmp_folders],
        "total_units_encoded": total_encoded_units,
        "total_units_expected": total_units,
        "completion_pct": pct,
        "size_gb": sum(f.stat().st_size for f in level_dir.rglob("*") if f.is_file()) / (1024**3)
    }


def merge(dst_dir, src_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)
    print(f"\n-> Tiến hành hợp nhất từ {src} vào {dst}...")
    copied = 0
    for level in ("parents", "children"):
        sl = src / level
        dl = dst / level
        if not sl.exists():
            continue
        dl.mkdir(parents=True, exist_ok=True)
        for item in sl.iterdir():
            if item.is_dir() and not item.name.endswith(".tmp"):
                target = dl / item.name
                if (target / "_DONE.json").exists():
                    continue
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
                copied += 1
                print(f"  + Đã chép khối {level}/{item.name}")
    print(f"-> Hoàn tất hợp nhất: Đã bổ sung {copied} khối mới.\n")


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Kiểm tra & Hợp nhất khối Embedding mE5 Role C")
    ap.add_argument("--dir", default="../path/to/dest/emb_e5", help="Đường dẫn thư mục emb_e5 cần kiểm tra")
    ap.add_argument("--clean", action="store_true", help="Xoá các thư mục .tmp dở dang")
    ap.add_argument("--merge-from", default=None, help="Đường dẫn thư mục chứa các khối mới để hợp nhất vào --dir")
    args = ap.parse_args()

    target_dir = Path(args.dir).resolve()

    if args.merge_from:
        merge(target_dir, args.merge_from)

    print("=" * 72)
    print(f"BÁO CÁO KIỂM TRA EMBEDDING mE5: {target_dir}")
    print("=" * 72)

    p_info = inspect_level(target_dir, "parents", 34, 3332632, clean=args.clean)
    c_info = inspect_level(target_dir, "children", 87, 8676196, clean=args.clean)

    for info in (p_info, c_info):
        lv = info["level"].upper()
        print(f"\n[KHỐI {lv}]")
        print(f"  - Số khối hoàn chỉnh : {info['valid_count']} / {info['expected_blocks']} khối")
        print(f"  - Số đơn vị văn bản  : {info['total_units_encoded']:,} / {info['total_units_expected']:,} ({info['completion_pct']:.2f}%)")
        print(f"  - Dung lượng         : {info['size_gb']:.2f} GB")
        if info["tmp_folders"]:
            status_tmp = "đã xoá" if args.clean else "còn tồn tại (chạy lại với --clean để xoá)"
            print(f"  - Thư mục dở dang .tmp: {info['tmp_folders']} ({status_tmp})")
        if info["missing_blocks"]:
            print(f"  - Số khối còn thiếu  : {info['missing_count']} khối")
            print(f"  - Danh sách khối thiếu: {info['missing_blocks']}")
        else:
            print("  - Trạng thái         : >>> 100% HOÀN THÀNH TRỌN VẸN <<<")

    total_gb = p_info["size_gb"] + c_info["size_gb"]
    print("\n" + "-" * 72)
    print(f"TỔNG DUNG LƯỢNG HIỆN CÓ: {total_gb:.2f} GB")
    if not p_info["missing_blocks"] and not c_info["missing_blocks"]:
        print("TÌNH TRẠNG CHUNG: ĐÃ ĐẦY ĐỦ 100% SẴN SÀNG LẬP CHỈ MỤC (INDEXING) VÀ NỘP BÀI!")
    else:
        print(f"TÌNH TRẠNG CHUNG: Cần chạy bổ sung {c_info['missing_count']} khối children.")
    print("=" * 72)


if __name__ == "__main__":
    main()
