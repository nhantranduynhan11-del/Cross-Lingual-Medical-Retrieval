"""Đọc kết quả crawl (out/part-*.jsonl.gz), chịu được file bị cụt và nhiều thư mục shard.

Một id có thể xuất hiện nhiều lần (tải lại bằng --redo-stale, trích lại bằng --reextract): lấy bản ghi có
"xv" (phiên bản bộ trích chữ) lớn nhất; cùng xv thì lấy bản xuất hiện sau. Bản ghi không có "xv"
(crawl.py trước 3/10/2026) coi là xv = 1 và được chuẩn hoá nhẹ cho khớp phiên bản mới
(NFC, giải mã thực thể HTML còn sót, nhận diện lại ngôn ngữ).
"""
import gzip
import html
import json
import re
import unicodedata
import zlib
from pathlib import Path

_ID = re.compile(r'^\{"id":\s*(-?\d+)')
_XV = re.compile(r'"xv":\s*(\d+)\}\s*$')


def part_files(crawl_dir):
    return sorted(Path(crawl_dir).rglob("part-*.jsonl.gz"))


def _lines(fp):
    """Các dòng của một file gz; file cụt (crawl bị ngắt) → giữ phần đọc được."""
    try:
        with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
    except (EOFError, zlib.error, OSError):
        return


# Giữ đồng bộ với crawl.detect_lang
_CJK = re.compile(r"[一-鿿]")
_VI = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđ]", re.I)
_VI_HOSTS = {"hellobacsi.com", "benhvienvietduc.org", "pmc-ecm-healthblog.beta.pharmacity.io"}


def detect_lang(text, host=""):
    s = text[:3000]
    cjk, vi = len(_CJK.findall(s)), len(_VI.findall(s))
    if cjk >= 10 and cjk > 2 * vi:
        return "zh"
    if vi >= 5:
        return "vi"
    return "vi" if host.endswith(".vn") or host in _VI_HOSTS else "zh" if cjk else "vi"


_ENT = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


def _upgrade(d):
    """Bản ghi cũ (không có xv) → cùng quy ước với bộ trích chữ mới ở các điểm tính lại được từ chữ."""
    if "xv" in d:
        return d
    t = d.get("text", "")
    if _ENT.search(t):
        t = html.unescape(t)
    d["text"] = unicodedata.normalize("NFC", t)
    d["title"] = unicodedata.normalize("NFC", html.unescape(d.get("title", "")))
    d["lang"] = detect_lang(d["text"], d.get("host", ""))
    d["xv"] = 1
    return d


def iter_docs(crawl_dir):
    """Sinh từng bản ghi {"id","url","host","lang","title","text","xv"} — mỗi id đúng một lần (bản mới nhất).
    Dòng JSON hỏng / file gz bị cụt: bỏ phần hỏng, giữ phần đọc được."""
    files = part_files(crawl_dir)
    # Lượt 1 (nhẹ, không giải mã JSON): chọn vị trí bản ghi tốt nhất cho mỗi id
    best, dup = {}, False
    for fi, fp in enumerate(files):
        for li, line in enumerate(_lines(fp)):
            m = _ID.match(line)
            if not m:
                continue
            i = int(m.group(1))
            xm = _XV.search(line)
            key = (int(xm.group(1)) if xm else 1, fi, li)
            if i in best:
                dup = True
                if key > best[i]:
                    best[i] = key
            else:
                best[i] = key
    # Lượt 2: chỉ sinh bản ghi được chọn
    for fi, fp in enumerate(files):
        for li, line in enumerate(_lines(fp)):
            m = _ID.match(line)
            if not m:
                continue
            k = best.get(int(m.group(1)))
            if dup and (k is None or k[1] != fi or k[2] != li):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield _upgrade(d)


def read_progress(crawl_dir):
    """id -> trạng thái (dòng sau ghi đè dòng trước), gộp mọi progress.tsv tìm thấy.
    progress.tsv: id <tab> trạng thái [<tab> xv]."""
    st = {}
    for fp in sorted(Path(crawl_dir).rglob("progress.tsv")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0].lstrip("-").isdigit():
                    st[int(p[0])] = p[1]
    return st
