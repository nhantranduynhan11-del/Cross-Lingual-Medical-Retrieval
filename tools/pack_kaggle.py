"""Đóng gói thư mục để tải lên Kaggle làm dataset (chạy từ thư mục gốc repo).

  py tools/pack_kaggle.py code              → kaggle/mir-code/: mir/, configs/, VERSION.txt
  py tools/pack_kaggle.py data [--no-full]  → kaggle/mir-data/: Data/, sample/, corpus_clean/, chunks/<strategy>/,
                                              VERSION.txt (--no-full: chỉ Data/ và sample/, đủ cho notebook e2e)
Thư mục đích được xoá và tạo lại mỗi lần chạy. File dữ liệu được tạo bằng hard link (không tốn thêm dung lượng) nếu
cùng ổ đĩa, nếu không thì chép. VERSION.txt được notebook in ra ở ô (1): so với dòng in ở đây để biết Kaggle đang
dùng đúng bản.

--user <tên tài khoản Kaggle>: ghi thêm dataset-metadata.json để tải lên bằng Kaggle CLI:
  kaggle datasets create  -p kaggle/mir-code --dir-mode zip                 (lần đầu)
  kaggle datasets version -p kaggle/mir-code --dir-mode zip -m "ghi chú"     (các lần sau)
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mir import config  # noqa: E402
from mir.encode import chunk_parts, fingerprint  # noqa: E402

OUT = ROOT / "kaggle"


def place(src, dst):
    """Hard link src → dst (cùng ổ đĩa), nếu không được thì chép."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def place_tree(src, dst, patterns=("*",)):
    n = 0
    for pat in patterns:
        for f in sorted(Path(src).rglob(pat)):
            if f.is_file() and "__pycache__" not in f.parts:
                place(f, dst / f.relative_to(src))
                n += 1
    return n


def git_version():
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    head = run("rev-parse", "--short", "HEAD") or "không có git"
    dirty = run("status", "--porcelain", "--", "mir", "configs")
    return head + (" + sửa chưa commit" if dirty else "")


def content_hash(files):
    h = hashlib.sha1()
    for f in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))     # cùng mã trên Windows (CRLF) và Linux
    return h.hexdigest()[:10]


def fresh(name, user):
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    if user:
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": name, "id": f"{user}/{name}", "licenses": [{"name": "CC0-1.0"}]}, indent=1), encoding="utf-8")
    return d


def pack_code(user):
    d = fresh("mir-code", user)
    files = [f for f in (ROOT / "mir").glob("*.py")] + list((ROOT / "configs").glob("*.yaml"))
    for f in files:
        place(f, d / f.relative_to(ROOT))
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    version = f"{now} | git {git_version()} | mã {content_hash(files)}"
    (d / "VERSION.txt").write_text(version + "\n", encoding="utf-8")
    print(f"→ {d} ({len(files)} file)\nVERSION: {version}")


def pack_data(user, full, strategy):
    cfg = config.load(ROOT / "configs" / "baseline.yaml")
    d = fresh("mir-data", user)
    work = cfg.work_dir
    for f in (cfg.queries_path, cfg.links_path):
        if not f.exists():
            raise SystemExit(f"Thiếu {f}: tải dữ liệu BTC trước (xem README).")
        place(f, d / "Data" / f.name)
    lines = [f"mir-data {datetime.datetime.now():%d/%m/%Y %H:%M}"]
    s = work / cfg["sample"]["dir"]
    n = sum(place_tree(s / sub, d / "sample" / sub) for sub in ("corpus_clean", "chunks")) if s.exists() else 0
    if (s / "query.parquet").exists():
        place(s / "query.parquet", d / "sample" / "query.parquet")
    lines.append(f"sample: {n} file")
    if full:
        clean, chunks = cfg.path("clean"), cfg.path("chunks", strategy)
        if not any(clean.glob("part-*.parquet")) or not any(chunks.glob("part-*.parquet")):
            raise SystemExit(f"Chưa có {clean} hoặc {chunks}: chạy T1, T2 trước, hoặc dùng --no-full.")
        place_tree(clean, d / "corpus_clean", ("part-*.parquet", "_meta.json"))
        place_tree(chunks, d / "chunks" / strategy, ("part-*.parquet", "parents-*.parquet", "_config.json"))
        parts = chunk_parts(chunks)
        lines.append(f"chunks/{strategy}: {sum(n for _, n in parts):,} đoạn, {len(parts)} part, "
                     f"dấu vân tay {fingerprint(parts)}")
    (d / "VERSION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {d}\n" + "\n".join(lines))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["code", "data"])
    ap.add_argument("--user", default=None, help="tên tài khoản Kaggle (để ghi dataset-metadata.json cho Kaggle CLI)")
    ap.add_argument("--no-full", action="store_true", help="data: bỏ corpus_clean/ và chunks/ của toàn kho")
    ap.add_argument("--strategy", default="structure")
    a = ap.parse_args()
    if a.what == "code":
        pack_code(a.user)
    else:
        pack_data(a.user, not a.no_full, a.strategy)


if __name__ == "__main__":
    main()
