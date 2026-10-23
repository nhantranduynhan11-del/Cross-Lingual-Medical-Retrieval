"""Tiện ích cho các notebook Kaggle (notebooks/k*.ipynb) — dựa trên mir/kaggle_nb.py cũ.

Ô (1) của mọi notebook chép mã (dataset mir2-code) vào /kaggle/working/code rồi gọi setup():
  - tìm dataset dữ liệu (thư mục có data/query.parquet) và các output phiên trước trong /kaggle/input;
  - viết cấu hình làm việc /kaggle/working/config.yaml = configs/retrieval.yaml + ghi đè; đường dẫn tới dữ liệu chỉ đọc
    (chunks, emb, index… nằm trong /kaggle/input) được ghi thẳng vào cấu hình (đường dẫn tuyệt đối);
  - MIR_DATA_DIR = dataset dữ liệu, MIR_WORK_DIR = /kaggle/working.
sh(): chạy một lệnh, in log. par(): chạy song song (mỗi GPU một lệnh), log riêng từng lệnh.
Chạy thử ngoài Kaggle: đặt biến môi trường KAGGLE_ROOT tới thư mục giả lập (có input/, working/).
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
LOCAL = "KAGGLE_ROOT" in os.environ
T = {}


def locate_all(rel):
    """Mọi thư mục D trong /kaggle/input sao cho D/rel tồn tại."""
    rel = rel.strip("/")
    n = len(Path(rel).parts)
    return sorted({Path(h).parents[n - 1] for h in glob.glob(f"{INPUT}/**/{rel}", recursive=True)})


def locate(rel, required=True):
    hits = locate_all(rel)
    if not hits:
        if required:
            raise FileNotFoundError(f"Không tìm thấy '{rel}' trong {INPUT}: kiểm tra đã Add Input đúng dataset chưa.")
        return None
    return min(hits, key=lambda p: (len(str(p)), str(p)))


def write_config(src, dst, overrides):
    cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    for key, val in overrides.items():
        *parents, last = key.split(".")
        d = cfg
        for k in parents:
            d = d[k]
        if last not in d:
            raise KeyError(f"cấu hình không có khoá '{key}'")
        d[last] = val
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return Path(dst)


class Env:
    def __init__(self, code, data, overrides):
        self.code, self.data, self.work = Path(code), Path(data), WORK
        self.cfg = WORK / "config.yaml"
        self.overrides = {}
        self.set(overrides)

    def set(self, overrides):
        self.overrides.update(overrides or {})
        write_config(self.code / "configs" / "retrieval.yaml", self.cfg, self.overrides)
        print("cấu hình:", self.cfg, "| ghi đè:", json.dumps(self.overrides, ensure_ascii=False))


def setup(nb_version, overrides=None, pip="pyvi jieba faiss-cpu", inputs=("chunks", "index", "qtrans", "runs")):
    """inputs: các thư mục kết quả của phiên/notebook trước cần đọc từ /kaggle/input (nếu có) — ghi vào cấu hình."""
    code = Path.cwd()
    print("NOTEBOOK", nb_version)
    v = code / "VERSION.txt"
    print("mir2-code", v.read_text(encoding="utf-8").strip() if v.exists() else "(không có VERSION.txt)")
    data = locate("data/query.parquet")
    print("mir2-data", data)
    os.environ.update(MIR_DATA_DIR=str(data), MIR_WORK_DIR=str(WORK), PYTHONIOENCODING="utf-8",
                      HF_HUB_DISABLE_SYMLINKS_WARNING="1", TOKENIZERS_PARALLELISM="false",
                      PYTHONPATH=f"{code}:{os.environ.get('PYTHONPATH', '')}")
    ov = {}
    keys = {"chunks": ("chunk.out_dir", "chunks/parents"), "index": ("index.out_dir", "index"),
            "qtrans": ("translate.out_dir", "qtrans/queries_vi.parquet"), "runs": ("retrieve.out_dir", "runs")}
    for name in inputs:
        key, probe = keys[name]
        hit = locate(probe, required=False)
        if hit is not None:
            ov[key] = str(hit / Path(probe).parts[0])
            print(f"  {name}: dùng {ov[key]} (output phiên trước)")
        else:
            in_dirs = [p.name for p in INPUT.iterdir() if p.is_dir()] if INPUT.exists() else []
            print(f"  [CẢNH BÁO] Không tìm thấy '{name}' (probe '{probe}') trong {INPUT}!")
            print(f"             Các dataset hiện có trong {INPUT}: {in_dirs}")
            print(f"             -> Hãy bấm '+ Add Input' để thêm dataset chứa '{name}' (vd: mir2-{name})!")
    ov.update(overrides or {})
    env = Env(code, data, ov)
    if pip and not LOCAL:
        sh(f"python -m pip install -q {pip}")
    gpu_info()
    return env


def _py(cmd):
    return re.sub(r"^python(?=\s)", lambda _: f'"{sys.executable}"', cmd.strip())


def sh(cmd, step=None, check=True):
    cmd = _py(cmd)
    print("$", cmd, flush=True)
    t0 = time.time()
    code_dir = WORK / "code"
    run_cwd = str(code_dir) if code_dir.exists() else None
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
                         errors="replace", cwd=run_cwd)
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    if step:
        T[step] = T.get(step, 0) + time.time() - t0
    if check and code != 0:
        raise RuntimeError(f"LỖI (mã {code}): {cmd}")
    return code


def tail(path, n=2000):
    return Path(path).read_text(encoding="utf-8", errors="replace")[-n:] if Path(path).exists() else ""


def par(cmds, step):
    """Chạy song song (vd mỗi GPU một lệnh: tiền tố CUDA_VISIBLE_DEVICES=i), log riêng; lệnh lỗi → dừng notebook."""
    tag = re.sub(r"[^A-Za-z0-9]+", "_", step)
    logs = [WORK / f"log_{tag}_{i}.txt" for i in range(len(cmds))]
    t0 = time.time()
    ps = []
    for c, lg in zip(cmds, logs):
        m = re.match(r"^((?:[A-Z_][A-Z0-9_]*=\S+\s+)*)(.*)$", c.strip(), re.S)   # "VAR=x python …" → env (chạy cả Windows)
        env = dict(os.environ, **dict(kv.split("=", 1) for kv in m.group(1).split()))
        c = _py(m.group(2))
        print("$", m.group(1) + c, ">", lg, flush=True)
        code_dir = WORK / "code"
        run_cwd = str(code_dir) if code_dir.exists() else None
        with open(lg, "w", encoding="utf-8") as fh:
            ps.append(subprocess.Popen(c, shell=True, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=run_cwd))
    codes = [p.wait() for p in ps]
    T[step] = T.get(step, 0) + time.time() - t0
    for i, lg in enumerate(logs):
        print(f"--- {step} #{i} (mã {codes[i]})\n{tail(lg)}")
    if any(codes):
        raise RuntimeError(f"LỖI ở {step}: xem {[str(x) for x in logs]}")


def gpu_info():
    try:
        import torch
        n = torch.cuda.device_count()
        print("torch", torch.__version__, "| GPU:", n, [torch.cuda.get_device_name(i) for i in range(n)])
    except ImportError:
        print("torch: chưa cài")


def size_gb(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e9
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9


def report(name):
    for k, v in T.items():
        print(f"  {k:28s} {v / 60:7.1f} phút")
    sizes = {d.name: round(size_gb(d), 2) for d in sorted(WORK.iterdir()) if d.is_dir() and d.name != "code"}
    print("dung lượng /kaggle/working (GB):", sizes, "| tổng", round(sum(sizes.values()), 2), "(giới hạn ~20 GB)")
    out = WORK / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"timing_{name}.json").write_text(json.dumps({"minutes": {k: round(v / 60, 2) for k, v in T.items()},
                                                          "sizes_gb": sizes}, indent=1), encoding="utf-8")
