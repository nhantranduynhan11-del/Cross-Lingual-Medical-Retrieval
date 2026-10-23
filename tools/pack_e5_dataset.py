"""Công cụ đóng gói và chuẩn bị metadata để đăng tải Dataset mE5 Embeddings lên Kaggle.

Cách dùng:
  # 1. Khởi tạo metadata và kiểm tra tính toàn vẹn (mặc định kiểm tra d:/BKAi/ViBioMIR/path/to/dest/emb_e5):
  python tools/pack_e5_dataset.py --dir ../path/to/dest/emb_e5 --user huyphong09

  # 2. Tự động chạy lệnh upload qua Kaggle CLI (nếu đã cấu hình kaggle.json):
  python tools/pack_e5_dataset.py --dir ../path/to/dest/emb_e5 --user huyphong09 --upload
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]


def verify_emb_dir(emb_dir: Path):
    emb_dir = Path(emb_dir).resolve()
    if not emb_dir.exists():
        sys.exit(f"LỖI: Thư mục {emb_dir} không tồn tại!")

    parents_dir = emb_dir / "parents"
    children_dir = emb_dir / "children"

    if not parents_dir.exists() or not children_dir.exists():
        sys.exit(f"LỖI: Thiếu thư mục con 'parents' hoặc 'children' trong {emb_dir}!")

    p_blocks = [b for b in parents_dir.iterdir() if b.is_dir() and (b / "_DONE.json").exists()]
    c_blocks = [b for b in children_dir.iterdir() if b.is_dir() and (b / "_DONE.json").exists()]

    total_bytes = sum(f.stat().st_size for f in emb_dir.rglob("*") if f.is_file())
    gb = total_bytes / 1e9

    print("=" * 60)
    print(f"KIỂM TRA DỮ LIỆU EMBEDDINGS mE5 TẠI: {emb_dir}")
    print(f"  • Khối Parents:  {len(p_blocks)} / 34 khối hoàn chỉnh")
    print(f"  • Khối Children: {len(c_blocks)} / 87 khối hoàn chỉnh")
    print(f"  • Tổng dung lượng: {gb:.2f} GB ({len(p_blocks) + len(c_blocks)} / 121 khối)")
    print("=" * 60)

    if len(p_blocks) < 34 or len(c_blocks) < 87:
        print("[CẢNH BÁO] Số khối chưa đạt 121! Hãy đảm bảo đã merge đủ dữ liệu trước khi upload.")
    else:
        print(">>> DỮ LIỆU ĐẦY ĐỦ 100% CHO TASK C2 <<<")

    return len(p_blocks), len(c_blocks), gb


def create_metadata(emb_dir: Path, user: str, title: str = "mir2-emb-e5"):
    emb_dir = Path(emb_dir).resolve()
    meta_path = emb_dir / "dataset-metadata.json"
    meta = {
        "title": title,
        "id": f"{user}/{title}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"→ Đã tạo {meta_path} (ID: {user}/{title})")

    ver_path = emb_dir / "VERSION.txt"
    ver_text = (
        f"ViBioMIR 2026 - Multilingual-E5-Large Embeddings\n"
        f"Built: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Model: intfloat/multilingual-e5-large (dim=1024, max_len=512, L2 norm)\n"
        f"Parents: 34 blocks (3,332,632 units)\n"
        f"Children: 87 blocks (8,676,196 units)\n"
        f"Total: 121 blocks (12,008,828 units)\n"
    )
    ver_path.write_text(ver_text, encoding="utf-8")
    print(f"→ Đã tạo {ver_path}")


def main():
    parser = argparse.ArgumentParser(description="Đóng gói và upload Dataset mE5 lên Kaggle.")
    parser.add_argument("--dir", default=str(ROOT.parent / "path" / "to" / "dest" / "emb_e5"),
                        help="Đường dẫn tới thư mục emb_e5")
    parser.add_argument("--user", default="huyphong09", help="Kaggle username sở hữu dataset")
    parser.add_argument("--title", default="mir2-emb-e5", help="Tên dataset trên Kaggle")
    parser.add_argument("--upload", action="store_true", help="Tự động gọi kaggle datasets create")
    args = parser.parse_args()

    emb_dir = Path(args.dir).resolve()
    np, nc, gb = verify_emb_dir(emb_dir)
    create_metadata(emb_dir, args.user, args.title)

    print("\nHƯỚNG DẪN TẢI LÊN KAGGLE:")
    print("1. Nếu tải lên qua Kaggle CLI:")
    print(f"   kaggle datasets create -p \"{emb_dir}\" --dir-mode tar")
    print("   (hoặc nếu cập nhật: kaggle datasets version -p \"{emb_dir}\" -m \"Full 121 blocks mE5\" --dir-mode tar)")
    print("\n2. Nếu tải lên qua giao diện Web Kaggle:")
    print("   - Vào https://www.kaggle.com/datasets -> Nhấn 'New Dataset'")
    print(f"   - Đặt tên: {args.title}")
    print(f"   - Kéo thả thư mục: {emb_dir}")
    print("=" * 60)

    if args.upload:
        print("\nĐang thực thi lệnh Kaggle CLI...")
        cmd = ["kaggle", "datasets", "create", "-p", str(emb_dir), "--dir-mode", "tar"]
        try:
            res = subprocess.run(cmd, check=True)
            print("Đăng tải dataset thành công!")
        except Exception as e:
            print(f"Lệnh tự động upload thất bại: {e}")
            print("Hãy thử chạy thủ công lệnh trên trong terminal.")


if __name__ == "__main__":
    main()
