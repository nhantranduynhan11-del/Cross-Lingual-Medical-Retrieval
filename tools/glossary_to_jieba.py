"""Bảng thuật ngữ y khoa của prompt dịch t2 (configs/med_glossary_vi.tsv, cột zh) → userdict cho jieba (bm25.zh_userdict).

Ô zh có thể gồm nhiều thuật ngữ: "腺样体；腺样体炎", "心悸（心慌）", "非甾体抗炎药（NSAID）" → tách theo ；;、，,/ và
ngoặc; chỉ giữ cụm có chữ Hán (cho phép lẫn chữ/số Latin, như jieba), dài ≥ 2, bỏ trùng. Không ghi tần suất: jieba tự
chọn tần suất đủ để cụm được giữ nguyên khi tách.

  python tools/glossary_to_jieba.py [--glossary configs/med_glossary_vi.tsv] [--out configs/jieba_med_userdict.txt]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, translate  # noqa: E402

_SPLIT = re.compile(r"[；;、，,/（）()【】\[\]\s]+")
_HAN = re.compile(r"[一-鿿]")
_OK = re.compile(r"^[一-鿿A-Za-z0-9\-·]+$")


def zh_terms(glossary):
    out, seen = [], set()
    for g in glossary:
        for t in _SPLIT.split(g[2]):
            t = t.strip("·-")
            if len(t) >= 2 and _HAN.search(t) and _OK.match(t) and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", default="configs/med_glossary_vi.tsv")
    ap.add_argument("--out", default=str(config.ROOT / "configs" / "jieba_med_userdict.txt"))
    a = ap.parse_args()
    gl = translate.load_glossary(a.glossary)
    terms = zh_terms(gl)
    Path(a.out).write_text("\n".join(terms) + "\n", encoding="utf-8")
    print(f"→ {a.out}: {len(terms)} thuật ngữ zh từ {len(gl)} mục bảng thuật ngữ")


if __name__ == "__main__":
    main()
