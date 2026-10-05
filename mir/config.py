"""Nạp cấu hình YAML và quy ước đường dẫn chung cho mọi bước.

data_dir / work_dir trong `paths`: đường dẫn tương đối tính từ thư mục gốc repo (thư mục chứa mir/). Biến môi trường
MIR_DATA_DIR / MIR_WORK_DIR, nếu có, ghi đè hai giá trị này (dùng trên Kaggle).
Thư mục kết quả của từng bước lấy bằng cfg.path(...), ví dụ cfg.path("chunks", "structure") → <work>/chunks/structure.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# tên → (mục, khoá) trong cấu hình chứa tên thư mục con của work_dir
_SUBDIR = {"clean": ("clean", "out_dir"), "chunks": ("chunk", "out_dir"), "emb": ("encode", "out_dir"),
           "index": ("index", "out_dir"), "runs": ("retrieve", "out_dir"), "submissions": ("submit", "out_dir")}
_FIXED = ("results", "qemb")                   # thư mục con của work_dir có tên cố định


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def path(self, name, strategy=None, override=None):
        """Đường dẫn chuẩn: clean | chunks | emb | index | runs | submissions | results | qemb | queries.
        override (giá trị truyền qua dòng lệnh) được ưu tiên; strategy thêm thư mục con theo cách chia đoạn."""
        if override:
            return Path(override)
        if name == "queries":
            return self.data_dir / self["retrieve"]["queries"]
        if name in _FIXED:
            p = self.work_dir / name
        else:
            sec, key = _SUBDIR[name]
            p = self.work_dir / self[sec][key]
        return p / strategy if strategy else p

    def results(self):
        """<work>/results, tạo nếu chưa có."""
        d = self.path("results")
        d.mkdir(parents=True, exist_ok=True)
        return d


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    data = Path(os.environ.get("MIR_DATA_DIR") or ROOT / p["data_dir"])
    work = Path(os.environ.get("MIR_WORK_DIR") or ROOT / p["work_dir"])
    cfg["data_dir"], cfg["work_dir"] = data, work
    cfg["queries_path"] = data / p["queries"]
    cfg["links_path"] = data / p["links"]
    cfg["crawl_path"] = data / p["crawl_dir"]
    return Cfg(cfg)
