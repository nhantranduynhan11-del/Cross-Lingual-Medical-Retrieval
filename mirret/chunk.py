"""Chia đoạn kho tài liệu: A (đoạn cha = đơn vị nộp) + B (đoạn con = đơn vị tìm kiếm mức mịn) + C (ngữ cảnh).

Đầu vào : corpus_out/corpus/part-*.parquet (kết quả `python -m mircrawl merge`), cột text có dấu cấu trúc
          ("# " tiêu đề mục, "- " mục danh sách, "a | b" hàng bảng, "---" ranh giới hỏi/đáp).
Chữ nộp : CHỮ THUẦN của bài = mircrawl.htmlx.to_plain(text) (bỏ dấu cấu trúc). Mọi char_start/char_end tính trên chữ
          thuần; bất biến: plain[char_start:char_end] == text của đoạn → chunk_text là chuỗi con nguyên văn.

A. Đoạn cha (chunks/parents): bài <= parent.max token → 1 đoạn. Bài dài: duyệt các khối (đoạn văn / danh sách / bảng;
   tiêu đề dính vào khối ngay sau nó), gom liên tiếp tới ~parent.target; mở đoạn mới ở mỗi tiêu đề mục hoặc ranh giới
   hỏi/đáp nếu đoạn hiện tại đã >= parent.min; không vượt parent.max; khối quá dài tách theo câu. Không chồng lấn.
B. Đoạn con (chunks/children): trong mỗi đoạn cha dài hơn child.min_parent token, tách theo câu thành các mảnh
   <= child.max token (không chồng lấn). Chỉ dùng để tìm; điểm dồn về đoạn cha (mirret.retrieve).
C. Ngữ cảnh (cột ctx): "tiêu đề bài > tiêu đề mục cha > tiêu đề mục" tại vị trí đầu đoạn. Ghép "ctx\\n" + text khi mã
   hoá (dense, LSR, ColBERT, cross-encoder) và khi tách từ BM25; KHÔNG nộp.
Độ dài tính bằng token BGE-M3 (không tính <s></s>).

  python -m mirret.chunk run    [--config ...] [--limit-parts N] [--workers 4]
  python -m mirret.chunk report [--config ...]          # phân bố độ dài, kiểm tra bất biến → results/chunk_report.md
"""
import argparse
import bisect
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

PARENT_SCHEMA = pa.schema([
    ("unit_id", pa.string()), ("doc_id", pa.int64()), ("lang", pa.string()), ("doc_type", pa.string()),
    ("char_start", pa.int32()), ("char_end", pa.int32()), ("text", pa.string()), ("ctx", pa.string()),
    ("n_tokens", pa.int32()), ("h", pa.string())])
CHILD_SCHEMA = PARENT_SCHEMA.append(pa.field("parent_id", pa.string()))


# ---------------------------------------------------------------- tokenizer
_TOK = {}


def load_tokenizer(name):
    """Tokenizer BGE-M3 (thư viện tokenizers): id Hugging Face, thư mục hoặc đường dẫn tokenizer.json."""
    if name not in _TOK:
        from tokenizers import Tokenizer
        p = Path(name)
        if p.is_file():
            t = Tokenizer.from_file(str(p))
        elif (p / "tokenizer.json").is_file():
            t = Tokenizer.from_file(str(p / "tokenizer.json"))
        else:
            from huggingface_hub import hf_hub_download
            t = Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))
        t.no_truncation()
        t.no_padding()
        _TOK[name] = t
    return _TOK[name]


class Doc:
    """Chữ thuần + vị trí bắt đầu từng token → đếm token của khoảng [s, e) bằng tìm kiếm nhị phân."""

    def __init__(self, text, offsets):
        self.text = text
        self.starts = [a for a, b in offsets]

    def ntok(self, s, e):
        return bisect.bisect_left(self.starts, e) - bisect.bisect_left(self.starts, s)

    def trim(self, s, e):
        t = self.text
        while s < e and t[s].isspace():
            s += 1
        while e > s and t[e - 1].isspace():
            e -= 1
        return s, e


# ---------------------------------------------------------------- văn bản có dấu cấu trúc → chữ thuần + các dòng
_HEAD = re.compile(r"^(#{1,6}) ")


def parse(marked):
    """→ (chữ thuần, các dòng [(loại, cấp tiêu đề, s, e, nội dung)], tập chỉ số dòng đứng ngay sau "---").
    Chữ thuần trùng mircrawl.htmlx.to_plain(marked)."""
    lines, seps, parts, pos, sep = [], set(), [], 0, False
    for raw in (marked or "").split("\n"):
        if raw == "---":
            sep = True
            continue
        m = _HEAD.match(raw)
        if m:
            kind, lvl, c = "head", len(m.group(1)), raw[m.end():]
        elif raw.startswith("- "):
            kind, lvl, c = "li", 0, raw[2:]
        elif " | " in raw:
            kind, lvl, c = "row", 0, raw
        else:
            kind, lvl, c = "para", 0, raw
        if parts:
            pos += 1                                       # "\n" giữa hai dòng
        if sep:
            seps.add(len(lines))
            sep = False
        lines.append((kind, lvl, pos, pos + len(c), c))
        parts.append(c)
        pos += len(c)
    plain = "\n".join(parts)
    lead = len(plain) - len(plain.lstrip())               # to_plain() có .strip() ở cuối
    if lead or plain != plain.rstrip():
        plain = plain.strip()
        lines = [(k, lv, max(0, s - lead), max(0, e - lead), c) for k, lv, s, e, c in lines]
    return plain, lines, seps


def blocks(lines, seps):
    """Các khối: chuỗi dòng li liên tiếp, chuỗi dòng row liên tiếp, một đoạn văn, hoặc một tiêu đề.
    → [(chỉ số dòng đầu, chỉ số dòng cuối, loại)]"""
    out, i, n = [], 0, len(lines)
    while i < n:
        kind = lines[i][0]
        j = i + 1
        if kind in ("li", "row"):
            while j < n and lines[j][0] == kind and j not in seps:
                j += 1
        out.append((i, j - 1, kind))
        i = j
    return out


def atoms(lines, seps):
    """Khối cuối cùng để gom: tiêu đề (cùng các tiêu đề liền sau) dính vào khối nội dung ngay sau nó.
    → [(dòng đầu, dòng cuối, cấp tiêu đề ở đầu (0 nếu không), có ranh giới "---" ngay trước)]"""
    bl = blocks(lines, seps)
    out, k = [], 0
    while k < len(bl):
        a, b, kind = bl[k]
        lvl = lines[a][1] if kind == "head" else 0
        k2 = k
        while bl[k2][2] == "head" and k2 + 1 < len(bl) and bl[k2 + 1][0] not in seps:
            k2 += 1                                        # tiêu đề + (tiêu đề…) + khối nội dung kế tiếp
        out.append((a, bl[k2][1], lvl, a in seps))
        k = k2 + 1
    return out


# ---------------------------------------------------------------- tách khối dài (theo dòng → câu → mệnh đề → khoảng trắng)
_SEPS = [re.compile(r"\n"),
         re.compile(r"[。！？!?；;…]+[”」』）)\"']*|\.(?=\s)"),
         re.compile(r"[，,、：:]"),
         re.compile(r"\s+")]


def _pieces(doc, s, e, level):
    cuts, out, a = [], [], s
    for m in _SEPS[level].finditer(doc.text, s, e):
        if s < m.end() < e:
            cuts.append(m.end())
    for c in cuts:
        out.append((a, c))
        a = c
    out.append((a, e))
    return out


def _hard_cut(doc, s, e, max_t):
    i0, i1 = bisect.bisect_left(doc.starts, s), bisect.bisect_left(doc.starts, e)
    out, a = [], s
    for k in range(i0 + max_t, i1, max_t):
        out.append((a, doc.starts[k]))
        a = doc.starts[k]
    out.append((a, e))
    return out


def rsplit(doc, s, e, max_t, level=0):
    """Các khoảng liền nhau phủ [s, e), mỗi khoảng <= max_t token, gom tham lam (từ mir/chunk.py cũ)."""
    if doc.ntok(s, e) <= max_t:
        return [(s, e)]
    if level >= len(_SEPS):
        return _hard_cut(doc, s, e, max_t)
    ps = _pieces(doc, s, e, level)
    if len(ps) <= 1:
        return rsplit(doc, s, e, max_t, level + 1)
    out, cur = [], None
    for a, b in ps:
        if doc.ntok(a, b) > max_t:
            if cur:
                out.append(cur)
                cur = None
            out += rsplit(doc, a, b, max_t, level + 1)
        elif cur is None:
            cur = (a, b)
        elif doc.ntok(cur[0], b) <= max_t:
            cur = (cur[0], b)
        else:
            out.append(cur)
            cur = (a, b)
    if cur:
        out.append(cur)
    return out


def merge_tail(doc, spans, min_t, max_t):
    """Mảnh cuối ngắn hơn min_t → gộp vào mảnh trước nếu không vượt max_t."""
    spans = [doc.trim(a, b) for a, b in spans]
    spans = [(a, b) for a, b in spans if a < b]
    if len(spans) >= 2 and doc.ntok(*spans[-1]) < min_t and doc.ntok(spans[-2][0], spans[-1][1]) <= max_t:
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    return spans


# ---------------------------------------------------------------- A. đoạn cha
def parent_spans(doc, lines, seps, pc):
    """→ [(s, e)] đoạn cha trên chữ thuần."""
    tgt, mn, mx = pc["target"], pc["min"], pc["max"]
    if not lines:
        return []
    if doc.ntok(0, len(doc.text)) <= mx:
        return [doc.trim(0, len(doc.text))]
    out, cur = [], None                                    # cur = (s, e)

    def close():
        nonlocal cur
        if cur:
            out.append(cur)
            cur = None

    for a, b, lvl, sep in atoms(lines, seps):
        s, e = lines[a][2], lines[b][3]
        if cur and (lvl or sep) and doc.ntok(*cur) >= mn:
            close()                                        # tiêu đề mục / ranh giới hỏi-đáp: mở đoạn mới
        t = doc.ntok(s, e)
        if t > mx:                                         # khối quá dài: tách theo câu, mỗi mảnh ~target
            close()
            out += merge_tail(doc, rsplit(doc, s, e, tgt), mn, mx)
            continue
        if cur is None:
            cur = (s, e)
        elif doc.ntok(cur[0], e) <= tgt or (doc.ntok(*cur) < mn and doc.ntok(cur[0], e) <= mx):
            cur = (cur[0], e)
        else:
            close()
            cur = (s, e)
    close()
    out = [doc.trim(a, b) for a, b in out]
    out = [(a, b) for a, b in out if a < b]
    # đoạn ngắn hơn min: gộp vào đoạn trước (hoặc sau) nếu không vượt max
    k = 0
    while k < len(out):
        if doc.ntok(*out[k]) < mn and len(out) > 1:
            if k > 0 and doc.ntok(out[k - 1][0], out[k][1]) <= mx:
                out[k - 1:k + 1] = [(out[k - 1][0], out[k][1])]
                continue
            if k + 1 < len(out) and doc.ntok(out[k][0], out[k + 1][1]) <= mx:
                out[k:k + 2] = [(out[k][0], out[k + 1][1])]
                continue
        k += 1
    return out


# ---------------------------------------------------------------- C. ngữ cảnh
def heading_paths(lines):
    """Đường dẫn tiêu đề có hiệu lực NGAY TRƯỚC mỗi dòng → list[list[(cấp, tiêu đề)]]."""
    stack, out = [], []
    for kind, lvl, s, e, c in lines:
        out.append(list(stack))
        if kind == "head":
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, c))
    return out


def make_ctx(title, path, section, cc):
    parts = []
    if cc.get("use_section") and section:
        parts.append(section)
    if title:
        parts.append(title)
    # "tiêu đề" dài thực chất là sapo đặt trong thẻ h2 (báo Việt: "## (Dân trí) - …") → không đưa vào ngữ cảnh
    parts += [p for p in path if p and p != title and len(p) <= 100]
    s = " > ".join(parts)
    return s[:cc["max_chars"]].rstrip()


# ---------------------------------------------------------------- một bài
def chunk_doc(doc_id, lang, doc_type, title, marked, section, tok, cc):
    """→ (dòng đoạn cha, dòng đoạn con)."""
    plain, lines, seps = parse(marked)
    if not plain.strip():
        return [], []
    enc = tok.encode(plain, add_special_tokens=False)
    doc = Doc(plain, enc.offsets)
    paths = heading_paths(lines)
    line_starts = [ln[2] for ln in lines]

    def ctx_at(pos):
        i = max(0, bisect.bisect_right(line_starts, pos) - 1)
        path = paths[i]
        if lines[i][0] == "head" and pos <= lines[i][2]:   # đoạn mở đầu bằng tiêu đề cấp L: bỏ tiêu đề cũ cùng cấp/thấp hơn
            path = [x for x in path if x[0] < lines[i][1]]
        return make_ctx((title or "").strip(), [c for _, c in path], section, cc["ctx"])

    def h(ctx, text):
        return hashlib.md5((ctx + "\x1f" + text).encode("utf-8")).hexdigest()[:12]

    parents, children = [], []
    chc = cc["child"]
    for k, (a, b) in enumerate(parent_spans(doc, lines, seps, cc["parent"])):
        text = plain[a:b]
        ctx = ctx_at(a)
        pid = f"{doc_id}_p{k}"
        nt = doc.ntok(a, b)
        parents.append({"unit_id": pid, "doc_id": doc_id, "lang": lang, "doc_type": doc_type, "char_start": a,
                        "char_end": b, "text": text, "ctx": ctx, "n_tokens": nt, "h": h(ctx, text)})
        if chc["enabled"] and nt > chc["min_parent"]:
            for j, (s, e) in enumerate(merge_tail(doc, rsplit(doc, a, b, chc["max"]), chc["min"], chc["max"])):
                t = plain[s:e]
                cx = ctx_at(s)
                children.append({"unit_id": f"{pid}_c{j}", "doc_id": doc_id, "lang": lang, "doc_type": doc_type,
                                 "char_start": s, "char_end": e, "text": t, "ctx": cx, "n_tokens": doc.ntok(s, e),
                                 "h": h(cx, t), "parent_id": pid})
    return parents, children


# ---------------------------------------------------------------- chạy trên kho
_COLS = ["id", "lang", "doc_type", "title", "text", "dup_of", "section"]


def _job(args):
    src, out_root, cc, force = args
    src = Path(src)
    dp, dc = Path(out_root) / "parents" / src.name, Path(out_root) / "children" / src.name
    if dp.exists() and (dc.exists() or not cc["child"]["enabled"]) and not force:
        return src.name, None
    tok = load_tokenizer(cc["tokenizer"])
    names = set(pq.read_schema(src).names)
    t = pq.read_table(src, columns=[c for c in _COLS if c in names]).to_pydict()
    n = len(t["id"])
    P, C, skipped_dup, empty = [], [], 0, 0
    t0 = time.time()
    for i in range(n):
        if cc["skip_dups"] and t.get("dup_of") and t["dup_of"][i] is not None:
            skipped_dup += 1
            continue
        ps, cs = chunk_doc(int(t["id"][i]), t["lang"][i] or "", t["doc_type"][i] or "", t["title"][i] or "",
                              t["text"][i] or "", (t.get("section") or [None] * n)[i] or "", tok, cc)
        if not ps:
            empty += 1
        P += ps
        C += cs
    for d, rows, schema in ((dp, P, PARENT_SCHEMA), (dc, C, CHILD_SCHEMA)):
        if d is dc and not cc["child"]["enabled"]:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
        tmp.replace(d)
    return src.name, {"docs": n, "skipped_dup": skipped_dup, "no_text": empty, "parents": len(P),
                      "children": len(C), "seconds": round(time.time() - t0, 1)}


def run(cfg, corpus=None, out=None, workers=None, limit_parts=0, force=False, log=print):
    cc = cfg["chunk"]
    src = cfg.path("corpus", corpus)
    parts = sorted(Path(src).glob("*.parquet"))
    if limit_parts:
        parts = parts[:limit_parts]
    if not parts:
        raise SystemExit(f"Không có file parquet trong {src}")
    out_root = cfg.path("chunks", out)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(out_root), cc, force) for p in parts]
    workers = workers or cc["workers"]
    stats = {}
    log(f"chia đoạn {len(parts)} file từ {src} → {out_root} ({workers} tiến trình)")
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for name, s in ex.map(_job, jobs):
                stats[name] = s
                log(f"  {name}: {'đã có → bỏ qua' if s is None else s}")
    else:
        for j in jobs:
            name, s = _job(j)
            stats[name] = s
            log(f"  {name}: {'đã có → bỏ qua' if s is None else s}")
    (out_root / "_config.json").write_text(json.dumps({"chunk": cc, "corpus": str(src), "parts": stats},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- báo cáo
def _q(xs):
    import numpy as np
    if len(xs) == 0:
        return "-"
    p = np.percentile(xs, [5, 25, 50, 75, 95])
    return " / ".join(f"{v:.0f}" for v in p) + f" (max {max(xs)})"


def report(cfg, out=None, corpus=None, check_docs=3000, log=print):
    import pandas as pd
    root = cfg.path("chunks", out)
    P = pd.read_parquet(root / "parents", columns=["unit_id", "doc_id", "lang", "doc_type", "n_tokens"])
    L = ["# Báo cáo chia đoạn", "", f"Thư mục: `{root}`. Độ dài = token BGE-M3. Phân vị 5 / 25 / 50 / 75 / 95.", "",
         "## A. Đoạn cha (đơn vị nộp)", "",
         f"- {len(P):,} đoạn cha / {P['doc_id'].nunique():,} tài liệu "
         f"({len(P) / max(1, P['doc_id'].nunique()):.2f} đoạn/tài liệu)",
         f"- Token: {_q(P['n_tokens'].to_numpy())}",
         f"- Tài liệu chỉ có 1 đoạn: {(P.groupby('doc_id').size() == 1).mean() * 100:.1f}%",
         f"- Đoạn < 150 token (thường là cả bài ngắn): {(P['n_tokens'] < 150).mean() * 100:.1f}% · "
         f"> 400: {(P['n_tokens'] > 400).mean() * 100:.2f}%", "",
         "| ngôn ngữ | loại | số đoạn | token (p5/p25/p50/p75/p95) |", "|---|---|---:|---|"]
    for (lg, dt), g in P.groupby(["lang", "doc_type"]):
        if len(g) >= 100:
            L.append(f"| {lg} | {dt} | {len(g):,} | {_q(g['n_tokens'].to_numpy())} |")
    cdir = root / "children"
    if cdir.exists() and any(cdir.glob("*.parquet")):
        C = pd.read_parquet(cdir, columns=["unit_id", "parent_id", "n_tokens"])
        L += ["", "## B. Đoạn con (đơn vị tìm kiếm)", "",
              f"- {len(C):,} đoạn con trong {C['parent_id'].nunique():,} đoạn cha "
              f"({C['parent_id'].nunique() / max(1, len(P)) * 100:.1f}% số đoạn cha có con)",
              f"- Token: {_q(C['n_tokens'].to_numpy())}"]
    # bất biến: chữ đoạn == chữ thuần của bài cắt theo vị trí; và chữ thuần == mircrawl to_plain
    try:
        from mircrawl.htmlx import to_plain
    except ImportError:
        to_plain = None
    src = cfg.path("corpus", corpus)
    f0 = sorted(Path(src).glob("*.parquet"))[0]
    docs = pq.read_table(f0, columns=["id", "text"]).to_pandas().head(check_docs)
    sub = pd.read_parquet(root / "parents" / f0.name, columns=["doc_id", "char_start", "char_end", "text", "ctx"])
    sub = sub[sub["doc_id"].isin(set(docs["id"]))]
    plains = {int(i): parse(t or "")[0] for i, t in zip(docs["id"], docs["text"])}
    bad = sum(1 for d, a, b, t in zip(sub["doc_id"], sub["char_start"], sub["char_end"], sub["text"])
              if plains[int(d)][a:b] != t)
    tp_bad = sum(1 for i, t in zip(docs["id"], docs["text"]) if to_plain and parse(t or "")[0] != to_plain(t or ""))
    L += ["", "## Kiểm tra", "",
          f"- Bất biến plain[char_start:char_end] == text: {bad} lỗi / {len(sub):,} đoạn cha ({f0.name})",
          f"- Chữ thuần trùng mircrawl to_plain(): {tp_bad if to_plain else 'không kiểm (thiếu mircrawl)'} lỗi / "
          f"{len(docs):,} bài",
          "", "## Ví dụ ngữ cảnh (C)", ""]
    for c, t in list(zip(sub["ctx"], sub["text"]))[:8]:
        L.append(f"- `{c[:120]}` ← {t[:80].replace(chr(10), ' / ')}…")
    md = "\n".join(L) + "\n"
    res = cfg.path("results")
    res.mkdir(parents=True, exist_ok=True)
    (res / "chunk_report.md").write_text(md, encoding="utf-8")
    log(md)
    return md


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=None, help="thư mục corpus (mặc định paths.corpus)")
    ap.add_argument("--out", default=None, help="mặc định <work>/chunks")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit-parts", type=int, default=0, help="chỉ N file đầu (chạy thử)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "run":
        run(cfg, a.corpus, a.out, a.workers, a.limit_parts, a.force)
    else:
        report(cfg, a.out, a.corpus)


if __name__ == "__main__":
    main()
