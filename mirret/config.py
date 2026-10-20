"""Nạp cấu hình YAML + quy ước đường dẫn.

data_dir / work_dir trong `paths` là đường dẫn tương đối từ thư mục gốc repo; biến môi trường MIR_DATA_DIR /
MIR_WORK_DIR (nếu có) ghi đè (dùng trên Kaggle). cfg.path("chunks") → <work>/chunks, cfg.path("corpus") → corpus.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "retrieval.yaml"

_WORK = {"chunks": ("chunk", "out_dir"), "emb": ("encode", "out_dir"), "index": ("index", "out_dir"),
         "qtrans": ("translate", "out_dir"), "runs": ("retrieve", "out_dir"), "submissions": ("submit", "out_dir"),
         "eval": ("eval", "out_dir"), "qexp": ("expand", "out_dir")}
_FIXED = ("qemb", "results")


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def path(self, name, override=None):
        """corpus | queries | links (dưới data_dir) hoặc chunks | emb | index | qtrans | qexp | qemb | runs | eval |
        submissions | results (dưới work_dir). override (giá trị dòng lệnh) được ưu tiên."""
        if override:
            return Path(override)
        if name in ("corpus", "queries", "links"):
            return self.data_dir / self["paths"][name]
        if name in _FIXED:
            return self.work_dir / name
        sec, key = _WORK[name]
        return self.work_dir / self[sec][key]


def load(path=None):
    path = Path(path or DEFAULT)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    cfg["data_dir"] = Path(os.environ.get("MIR_DATA_DIR") or ROOT / p["data_dir"])
    cfg["work_dir"] = Path(os.environ.get("MIR_WORK_DIR") or ROOT / p["work_dir"])
    cfg["config_path"] = str(path)
    return Cfg(cfg)


def stdout_utf8():
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
