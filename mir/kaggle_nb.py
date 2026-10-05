"""Tiện ích dùng chung cho các notebook Kaggle (notebooks/*.ipynb).

Ô (1) của mọi notebook chép mir-code mới nhất vào /kaggle/working/code rồi gọi setup():
  - in phiên bản notebook và phiên bản mir-code (VERSION.txt do tools/pack_kaggle.py ghi) để biết có chạy nhầm bản cũ;
  - tìm dataset dữ liệu (thư mục có Data/query.parquet) ở bất kỳ đâu trong /kaggle/input;
  - viết cấu hình làm việc /kaggle/working/config.yaml = configs/baseline.yaml + các giá trị ghi đè. Đường dẫn tới
    corpus_clean/, chunks/ trong dataset được ghi thẳng vào cấu hình, không tạo symlink trong /kaggle/working.
sh() / par(): chạy lệnh (par: song song, mỗi GPU một lệnh), in log, cộng thời gian từng bước vào T.

Chạy thử trên máy cá nhân: đặt biến môi trường KAGGLE_ROOT tới một thư mục giả lập /kaggle (có input/, working/).
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("KAGGLE_ROOT", "/kaggle"))
INPUT, WORK = ROOT / "input", ROOT / "working"
LOCAL = "KAGGLE_ROOT" in os.environ          # chạy thử ngoài Kaggle: không pip install
T = {}                                       # thời gian từng bước (giây)


# ---------------- tìm dữ liệu đầu vào ----------------
def locate_all(rel):
    """Mọi thư mục D trong /kaggle/input sao cho D/rel tồn tại (rel: file hoặc thư mục, vd "emb/structure")."""
    rel = rel.strip("/")
    n = len(Path(rel).parts)
    return sorted({Path(h).parents[n - 1] for h in glob.glob(f"{INPUT}/**/{rel}", recursive=True)})


def locate(rel):
    """Như locate_all nhưng trả về một thư mục (đường dẫn ngắn nhất); không có → lỗi kèm gợi ý."""
    hits = locate_all(rel)
    if not hits:
        raise FileNotFoundError(f"Không tìm thấy '{rel}' trong {INPUT}: kiểm tra đã Add Input đúng dataset chưa.")
    return min(hits, key=lambda p: (len(str(p)), str(p)))


# ---------------- cấu hình ----------------
def write_config(src, dst, overrides):
    """src (YAML) + ghi đè {"mục.khoá": giá trị} → dst. Khoá không có trong src → lỗi (tránh gõ nhầm mà không biết)."""
    cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    for key, val in overrides.items():
        *parents, last = key.split(".")
        d = cfg
        for k in parents:
            d = d[k]
        if last not in d:
            raise KeyError(f"cấu hình không có khoá '{key}'")
        d[last] = val
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dst


class Env:
    """Đường dẫn của phiên Kaggle: code, data, work, cfg (cấu hình làm việc). set() thêm giá trị ghi đè."""

    def __init__(self, code, data, overrides):
        self.code, self.data, self.work = Path(code), Path(data), WORK
        self.cfg = WORK / "config.yaml"
        self.overrides = {}
        self.set(overrides)

    def set(self, overrides):
        self.overrides.update(overrides or {})
        write_config(self.code / "configs" / "baseline.yaml", self.cfg, self.overrides)
        print("cấu hình:", self.cfg, "| ghi đè:", json.dumps(self.overrides, ensure_ascii=False))


def setup(nb_version, overrides=None, corpus="full", pip="pyvi jieba faiss-cpu"):
    """Gọi ở ô (1), sau khi đã chép code vào /kaggle/working/code và chdir vào đó.
    corpus: "full" → corpus_clean/, chunks/ ở gốc dataset dữ liệu; "sample" → sample/corpus_clean, sample/chunks
    (kho mẫu cho notebook e2e); None → không cần kho."""
    code = Path.cwd()
    print("NOTEBOOK", nb_version)
    v = code / "VERSION.txt"
    print("mir-code", v.read_text(encoding="utf-8").strip() if v.exists() else
          "(không có VERSION.txt: bản đóng gói cũ, nên tạo New Version bằng tools/pack_kaggle.py)")
    data = locate("Data/query.parquet")
    dv = data / "VERSION.txt"
    print("mir-data", data, "|", dv.read_text(encoding="utf-8").strip().replace("\n", " | ") if dv.exists() else "")
    os.environ.update(MIR_DATA_DIR=str(data), MIR_WORK_DIR=str(WORK), PYTHONIOENCODING="utf-8",
                      HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    ov = {}
    if corpus:
        base = data / "sample" if corpus == "sample" else data
        for key, sub in (("clean.out_dir", "corpus_clean"), ("chunk.out_dir", "chunks")):
            if (base / sub).is_dir():
                ov[key] = str(base / sub)
            else:
                print(f"[LƯU Ý] dataset dữ liệu chưa có {base / sub}")
    ov.update(overrides or {})
    env = Env(code, data, ov)
    if pip and not LOCAL:
        sh(f"python -m pip install -q {pip}")
    gpu_info()
    return env


# ---------------- chạy lệnh ----------------
def _py(cmd):
    """Lệnh bắt đầu bằng python → chạy bằng đúng trình thông dịch của notebook (cùng môi trường đã pip install)."""
    return re.sub(r"^python(?=\s)", lambda _: f'"{sys.executable}"', cmd.strip())


def sh(cmd, step=None, check=True):
    """Chạy một lệnh, in lệnh và toàn bộ đầu ra; thời gian cộng vào T[step]. Lỗi → dừng notebook (nếu check)."""
    cmd = _py(cmd)
    print("$", cmd, flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
                         errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    if step:
        T[step] = T.get(step, 0) + time.time() - t0
    if check and code != 0:
        raise RuntimeError(f"LỖI (mã {code}): {cmd}")
    return code


def tail(path, n=1500):
    return Path(path).read_text(encoding="utf-8", errors="replace")[-n:] if Path(path).exists() else ""


def par(cmds, step):
    """Chạy song song các lệnh (vd mỗi GPU một lệnh), mỗi lệnh một file log trong /kaggle/working; in đuôi log.
    Có lệnh lỗi → dừng notebook."""
    tag = re.sub(r"[^A-Za-z0-9]+", "_", step)
    logs = [WORK / f"log_{tag}_{i}.txt" for i in range(len(cmds))]
    t0 = time.time()
    ps = []
    for c, lg in zip(cmds, logs):
        c = _py(c)
        print("$", c, ">", lg, flush=True)
        ps.append(subprocess.Popen(f'{c} > "{lg}" 2>&1', shell=True))
    codes = [p.wait() for p in ps]
    T[step] = T.get(step, 0) + time.time() - t0
    for i, lg in enumerate(logs):
        print(f"--- {step} #{i} (mã {codes[i]})\n{tail(lg)}")
    if any(codes):
        raise RuntimeError(f"LỖI ở {step}: xem {[str(x) for x in logs]}")


# ---------------- thông tin ----------------
def gpu_info():
    try:
        import torch
        n = torch.cuda.device_count()
        print("torch", torch.__version__, "| GPU:", n, [torch.cuda.get_device_name(i) for i in range(n)])
    except ImportError:
        print("torch: chưa cài")


def size_gb(path):
    """Tổng dung lượng file trong path (GB), không đi theo symlink."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e9
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9


def report(name):
    """In thời gian từng bước và dung lượng /kaggle/working (output tối đa ~20 GB); ghi results/timing_<name>.json."""
    for k, v in T.items():
        print(f"  {k:28s} {v / 60:7.1f} phút")
    sizes = {d.name: round(size_gb(d), 2) for d in sorted(WORK.iterdir()) if d.is_dir() and d.name != "code"}
    print("dung lượng (GB):", sizes, "| tổng", round(sum(sizes.values()), 2))
    out = WORK / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"timing_{name}.json").write_text(json.dumps({"minutes": {k: round(v / 60, 2) for k, v in T.items()},
                                                          "sizes_gb": sizes}, indent=1), encoding="utf-8")
