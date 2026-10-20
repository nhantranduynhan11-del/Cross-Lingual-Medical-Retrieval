"""Công cụ hỗ trợ đóng gói và đẩy code / notebook lên Kaggle cho Task C1 (mE5-large).

Cách dùng:
  1. Đăng nhập Kaggle CLI (nếu chưa):
     kaggle auth login
     (hoặc đặt file kaggle.json vào C:\\Users\\<user>\\.kaggle\\kaggle.json)

  2. Đóng gói mã nguồn & đẩy notebook lên Kaggle:
     python tools/deploy_kaggle.py --user <ten_kaggle> [--push-code] [--push-kernel]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]


def check_auth():
    """Kiểm tra xem Kaggle CLI đã được xác thực hay chưa."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        return True
    try:
        res = subprocess.run([sys.executable, "-m", "kaggle", "auth", "print-access-token"],
                             capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return True
    except Exception:
        pass
    return False


def pack_code(user):
    """Đóng gói mã nguồn nhánh hiện tại bằng tools/pack_kaggle_ret.py."""
    print("\n[1/3] Đóng gói mã nguồn mir2-code...")
    cmd = [sys.executable, str(ROOT / "tools" / "pack_kaggle_ret.py"), "code", "--user", user]
    subprocess.run(cmd, check=True)


def push_dataset_code(user):
    """Tải / Cập nhật dataset mir2-code lên Kaggle."""
    code_dir = ROOT / "build" / "mir2_code"
    if not (code_dir / "mir2_code.zip").exists():
        pack_code(user)

    print("\n[2/3] Đẩy dataset mir2-code lên Kaggle...")
    # Thử version trước, nếu chưa có dataset thì create
    res = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(code_dir),
                          "-m", "Update code for Task C1 mE5-large"], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  -> Cập nhật phiên bản mới thành công cho dataset {user}/mir2-code")
    else:
        print(f"  -> Chưa có dataset, tiến hành tạo mới dataset {user}/mir2-code...")
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(code_dir)], check=True)


def prepare_and_push_kernel(user, chunks_dataset=None, push=False):
    """Chuẩn bị metadata và đẩy notebook k3_encode_e5 lên Kaggle."""
    print("\n[3/3] Chuẩn bị notebook k3_encode_e5.ipynb...")
    kernel_dir = ROOT / "build" / "kernel_k3_e5"
    shutil.rmtree(kernel_dir, ignore_errors=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    src_nb = ROOT / "notebooks" / "k3_encode_e5.ipynb"
    dst_nb = kernel_dir / "k3_encode_e5.ipynb"
    shutil.copy2(src_nb, dst_nb)

    chunks_source = chunks_dataset or "nhntrnduy/mir2-chunks"
    meta = {
        "id": f"{user}/vibiomir-k3-me5-large-embedding-pipeline",
        "title": "ViBioMIR k3 - mE5 Large Embedding Pipeline",
        "code_file": "k3_encode_e5.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            f"{user}/mir2-code",
            chunks_source,
            "nhntrnduy/mir2-data"
        ],
        "kernel_sources": []
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Đã tạo thư mục kernel: {kernel_dir}")
    print(f"  -> Kernel Slug: {meta['id']}")
    print(f"  -> Datasets gắn kèm: {meta['dataset_sources']}")

    if push:
        print("\nĐang đẩy Notebook lên Kaggle (kaggle kernels push)...")
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)
        print(f"  -> Notebook đã được đẩy lên: https://www.kaggle.com/code/{meta['id']}")
    else:
        print(f"\n[Gợi ý]: Chạy lệnh sau để đẩy lên Kaggle:")
        print(f"  python -m kaggle kernels push -p build/kernel_k3_e5")


def main():
    ap = argparse.ArgumentParser(description="Triển khai Task C1 lên Kaggle qua Kaggle CLI")
    ap.add_argument("--user", required=True, help="Tên tài khoản Kaggle của bạn (username)")
    ap.add_argument("--chunks", default=None, help="Dataset chunks (mặc định: <user>/mir2-chunks)")
    ap.add_argument("--push-code", action="store_true", help="Tự động đẩy dataset mir2-code lên Kaggle")
    ap.add_argument("--push-kernel", action="store_true", help="Tự động đẩy notebook lên Kaggle")
    a = ap.parse_args()

    is_authed = check_auth()
    if not is_authed:
        print("=" * 70)
        print("[LƯU Ý] Bạn chưa xác thực Kaggle CLI.")
        print("Cách 1 (Khuyên dùng): Chạy lệnh sau trên terminal để đăng nhập trình duyệt:")
        print("    python -m kaggle auth login")
        print("Cách 2: Đặt file kaggle.json tải từ https://www.kaggle.com/settings/api vào:")
        print(f"    {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("=" * 70)
        if a.push_code or a.push_kernel:
            sys.exit(1)

    pack_code(a.user)
    if a.push_code:
        push_dataset_code(a.user)
    prepare_and_push_kernel(a.user, chunks_dataset=a.chunks, push=a.push_kernel)


if __name__ == "__main__":
    main()
