"""Nạp cấu hình YAML; đường dẫn lấy từ biến môi trường MIR_DATA_DIR / MIR_WORK_DIR nếu có."""
import os
from pathlib import Path
import yaml


class Cfg(dict):
    def __getattr__(self, k):
        return self[k]


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    data = Path(os.environ.get("MIR_DATA_DIR", p["data_dir"]))
    work = Path(os.environ.get("MIR_WORK_DIR", p["work_dir"]))
    cfg["data_dir"], cfg["work_dir"] = data, work
    cfg["queries_path"] = data / p["queries"]
    cfg["links_path"] = data / p["links"]
    cfg["crawl_path"] = data / p["crawl_dir"]
    return Cfg(cfg)
