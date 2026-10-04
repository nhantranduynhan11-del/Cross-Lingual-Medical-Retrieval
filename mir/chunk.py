"""T2: chia đoạn. Ba chiến lược (chọn bằng --strategy, mặc định chunk.strategy trong cấu hình):

  recursive    tách dần theo dòng → câu → mệnh đề → khoảng trắng → token, gộp tham lam tới max_tokens.
  structure    nhận diện tiêu đề mục (【处方】, "1.", "一、", "Nguyên nhân:"...), xếp nguyên mục vào đoạn, không cắt
               giữa mục trừ khi một mục dài hơn max_tokens (khi đó tách mục đó theo recursive).
  parent_child đoạn cha = structure với parent_max_tokens (đơn vị nộp bài / xếp hạng lại);
               đoạn con = recursive trong từng đoạn cha với child_max_tokens (đơn vị tìm kiếm), có cột parent_id.

Độ dài tính bằng token BGE-M3 (không tính <s></s>). Bất biến: text == corpus_clean.text[char_start:char_end].

  python -m mir.chunk --config configs/baseline.yaml [--strategy structure]          # corpus_clean/ → chunks/<strategy>/
  python -m mir.chunk --config configs/baseline.yaml --input sample/corpus_clean --out sample/chunks --strategy all
  python -m mir.chunk --config configs/baseline.yaml --report --input sample/corpus_clean --out sample/chunks
Chạy lại: bỏ qua file part đã có (dùng --force để làm lại).
"""
import argparse
import bisect
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config

STRATEGIES = ("recursive", "structure", "parent_child")
CHUNK_SCHEMA = pa.schema([("chunk_id", pa.string()), ("doc_id", pa.string()), ("lang", pa.string()),
                          ("char_start", pa.int32()), ("char_end", pa.int32()), ("text", pa.string()),
                          ("n_tokens", pa.int32())])
CHILD_SCHEMA = CHUNK_SCHEMA.append(pa.field("parent_id", pa.string()))
PARENT_SCHEMA = pa.schema([("parent_id", pa.string()), ("doc_id", pa.string()), ("lang", pa.string()),
                           ("char_start", pa.int32()), ("char_end", pa.int32()), ("text", pa.string()),
                           ("n_tokens", pa.int32())])

# ---------------- tokenizer ----------------
_TOK = {}


def load_tokenizer(name):
    """Tokenizer BGE-M3 (thư viện tokenizers). name: id Hugging Face, thư mục hoặc đường dẫn tokenizer.json."""
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
    """Văn bản + vị trí ký tự bắt đầu của từng token → đếm số token của một khoảng [s, e) bằng tìm kiếm nhị phân."""

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


# ---------------- recursive ----------------
# Mức tách: dòng → câu → mệnh đề → khoảng trắng. Mỗi mảnh giữ dấu tách ở cuối, các mảnh phủ liền khoảng [s, e).
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
    """Các khoảng liền nhau phủ [s, e), mỗi khoảng <= max_t token."""
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


# ---------------- structure ----------------
_HEAD = re.compile(r"^(【[^】]{1,20}】|第[一二三四五六七八九十百\d]+[章节条部分篇]|[一二三四五六七八九十]+[、．.]|"
                   r"[（(][一二三四五六七八九十\d]+[）)]|\d+(\.\d+)*[.、．)]\s*\S|[IVX]+\.\s)")
_END = tuple("。.!?！？;；…，,、")


def is_heading(line):
    """Dòng mở đầu một mục: đánh số / 【...】 / nhãn ngắn kết thúc bằng dấu hai chấm / dòng ngắn không có dấu câu cuối."""
    l = line.strip()
    if not l:
        return False
    if l.startswith("【") or (len(l) <= 80 and _HEAD.match(l)):
        return True
    if len(l) <= 30 and l.endswith(("：", ":")):
        return True
    return len(l) <= 25 and not l.endswith(_END) and not l.isdigit()


def sections(doc):
    """Chia bài thành các mục: mỗi mục bắt đầu ở một dòng tiêu đề (phần trước tiêu đề đầu tiên là mục 0)."""
    t, starts, pos = doc.text, [], 0
    for line in t.split("\n"):
        if pos == 0 or is_heading(line):
            starts.append(pos)
        pos += len(line) + 1
    ends = starts[1:] + [len(t)]
    return [(a, b) for a, b in zip(starts, ends) if a < b]


def structure_split(doc, max_t):
    out, cur = [], None
    for a, b in sections(doc):
        if doc.ntok(a, b) > max_t:                 # mục quá dài: tách riêng theo recursive (bắt đầu từ mức dòng)
            if cur:
                out.append(cur)
                cur = None
            out += rsplit(doc, a, b, max_t)
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


# ---------------- tiện ích chung ----------------
def finish(doc, spans, max_t, min_t, overlap=0):
    """Cắt khoảng trắng hai đầu, gộp đoạn cuối quá ngắn vào đoạn trước, thêm chồng lấn (nếu có)."""
    spans = [doc.trim(a, b) for a, b in spans]
    spans = [(a, b) for a, b in spans if a < b]
    if len(spans) >= 2 and doc.ntok(*spans[-1]) < min_t and doc.ntok(spans[-2][0], spans[-1][1]) <= max_t:
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    if overlap > 0 and len(spans) > 1:                 # lùi điểm đầu mỗi đoạn (trừ đoạn đầu) thêm tối đa `overlap` token
        out = [spans[0]]
        for (prev_a, _), (a, b) in zip(spans, spans[1:]):
            i = bisect.bisect_left(doc.starts, a)
            j = max(bisect.bisect_left(doc.starts, prev_a), i - overlap)
            out.append(doc.trim(doc.starts[j], b) if j < i else (a, b))
        spans = out
    return spans


def chunk_doc(doc, strategy, cc):
    """→ (đoạn: [(s, e)], cha: [(s, e)] | None, cha_của_đoạn: [chỉ số cha] | None)."""
    mx, mn, ov = cc["max_tokens"], cc["min_tokens"], cc["overlap_tokens"]
    if strategy == "recursive":
        return finish(doc, rsplit(doc, 0, len(doc.text), mx), mx, mn, ov), None, None
    if strategy == "structure":
        return finish(doc, structure_split(doc, mx), mx, mn), None, None
    pc = cc["parent_child"]
    pmx, cmx = pc["parent_max_tokens"], pc["child_max_tokens"]
    parents = finish(doc, structure_split(doc, pmx), pmx, mn)
    kids, owner = [], []
    for j, (a, b) in enumerate(parents):
        for s in finish(doc, rsplit(doc, a, b, cmx), cmx, min(mn, cmx // 4), ov):
            kids.append(s)
            owner.append(j)
    return kids, parents, owner


# ---------------- chạy trên corpus_clean ----------------
def process_table(t, strategy, cc, tok, batch=512):
    rows, prow = [], []
    d = t.to_pydict()
    n = len(d["doc_id"])
    for k0 in range(0, n, batch):
        texts = d["text"][k0:k0 + batch]
        encs = tok.encode_batch(texts, add_special_tokens=False)
        for k, (text, enc) in enumerate(zip(texts, encs)):
            i = k0 + k
            doc = Doc(text, enc.offsets)
            spans, parents, owner = chunk_doc(doc, strategy, cc)
            did, lang = d["doc_id"][i], d["lang"][i]
            for c, (a, b) in enumerate(spans):
                r = {"chunk_id": f"{did}_{c}", "doc_id": did, "lang": lang, "char_start": a, "char_end": b,
                     "text": text[a:b], "n_tokens": doc.ntok(a, b)}
                if parents is not None:
                    r["parent_id"] = f"{did}_p{owner[c]}"
                rows.append(r)
            for j, (a, b) in enumerate(parents or []):
                prow.append({"parent_id": f"{did}_p{j}", "doc_id": did, "lang": lang, "char_start": a,
                             "char_end": b, "text": text[a:b], "n_tokens": doc.ntok(a, b)})
    return rows, prow


def run(in_dir, out_root, strategy, cc, force=False, shard="0/1"):
    tok = load_tokenizer(cc["tokenizer"])
    out = Path(out_root) / strategy
    out.mkdir(parents=True, exist_ok=True)
    si, sn = (int(x) for x in shard.split("/"))
    parts = sorted(Path(in_dir).glob("part-*.parquet"))
    for k, p in enumerate(parts):
        if k % sn != si:
            continue
        dst = out / p.name
        if dst.exists() and not force:
            print(f"  {dst} đã có → bỏ qua")
            continue
        rows, prow = process_table(pq.read_table(p), strategy, cc, tok)
        tmp = dst.with_suffix(".tmp")                       # ghi tạm rồi đổi tên: part dở dang không bị coi là xong
        if prow:
            pq.write_table(pa.Table.from_pylist(prow, schema=PARENT_SCHEMA), out / p.name.replace("part-", "parents-"))
        pq.write_table(pa.Table.from_pylist(rows, schema=CHILD_SCHEMA if strategy == "parent_child" else CHUNK_SCHEMA),
                       tmp)
        tmp.replace(dst)
        print(f"  {strategy}: {p.name}: {len(rows):,} đoạn" + (f", {len(prow):,} đoạn cha" if prow else ""))
    (out / "_config.json").write_text(json.dumps({"strategy": strategy, "chunk": cc, "input": str(in_dir)},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------- báo cáo ----------------
def _q(xs, ps=(5, 25, 50, 75, 95, 99)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{p}": xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in ps} | {"max": xs[-1]}


def _read_parts(d, prefix):
    """Đọc các file <prefix>*.parquet (thư mục parent_child có cả part- lẫn parents-, khác lược đồ)."""
    return pa.concat_tables([pq.read_table(x) for x in sorted(Path(d).glob(prefix + "*.parquet"))]).to_pydict()


def report(in_dir, out_root, cc, est_docs=None):
    tok = load_tokenizer(cc["tokenizer"])
    docs = pq.read_table(in_dir).to_pydict()
    text = dict(zip(docs["doc_id"], docs["text"]))
    title = dict(zip(docs["doc_id"], docs["title"]))
    lang = dict(zip(docs["doc_id"], docs["lang"]))
    nd = len(text)
    L = ["# T2 — Chia đoạn: báo cáo", "",
         f"Đầu vào: `{in_dir}` ({nd:,} bài). Độ dài = token BGE-M3. T3 mã hoá `tiêu đề + \\n + đoạn` với max_length 512.",
         f"Cấu hình: max_tokens={cc['max_tokens']}, min_tokens={cc['min_tokens']}, overlap={cc['overlap_tokens']}, "
         f"parent={cc['parent_child']['parent_max_tokens']}, child={cc['parent_child']['child_max_tokens']}.", ""]
    summary = {}
    for strategy in STRATEGIES:
        d = Path(out_root) / strategy
        if not any(d.glob("part-*.parquet")):
            continue
        t = _read_parts(d, "part-")
        n = len(t["chunk_id"])
        bad = sum(1 for i in range(n) if text[t["doc_id"][i]][t["char_start"][i]:t["char_end"][i]] != t["text"][i])
        exact = [len(e.ids) for e in tok.encode_batch(t["text"], add_special_tokens=False)]
        inp = [len(e.ids) for e in tok.encode_batch(
            [(title[t["doc_id"][i]] + "\n" + t["text"][i]) for i in range(n)], add_special_tokens=True)]
        over = sum(1 for x in inp if x > 512)
        by_lang = {}
        for lg in ("vi", "zh"):
            xs = [exact[i] for i in range(n) if lang[t["doc_id"][i]] == lg]
            by_lang[lg] = (len(xs), _q(xs))
        per_doc = n / max(1, len({x for x in t["doc_id"]}))
        s = {"chunks": n, "chunks_per_doc": round(per_doc, 2), "invariant_errors": bad,
             "tokens": _q(exact), "input_over_512": over}
        L += [f"## {strategy}", "",
              f"- Số đoạn: **{n:,}** ({per_doc:.2f} đoạn/bài); bất biến char_start/char_end sai: **{bad}**",
              f"- Token mỗi đoạn: " + ", ".join(f"{k}={v}" for k, v in _q(exact).items()),
              *[f"  - {lg} ({c:,} đoạn): " + ", ".join(f"{k}={v}" for k, v in q.items()) for lg, (c, q) in by_lang.items()],
              f"- `tiêu đề + đoạn` vượt 512 token (bị cắt ở T3): **{over:,} ({100 * over / max(1, n):.2f}%)**"]
        if strategy == "parent_child":
            pt = _read_parts(d, "parents-")
            pex = [len(e.ids) for e in tok.encode_batch(pt["text"], add_special_tokens=False)]
            pbad = sum(1 for i in range(len(pt["parent_id"]))
                       if text[pt["doc_id"][i]][pt["char_start"][i]:pt["char_end"][i]] != pt["text"][i])
            s |= {"parents": len(pt["parent_id"]), "parent_tokens": _q(pex), "parent_invariant_errors": pbad}
            L.append(f"- Đoạn cha: {len(pt['parent_id']):,} ({len(pt['parent_id']) / nd:.2f}/bài), token: "
                     + ", ".join(f"{k}={v}" for k, v in _q(pex).items()) + f"; bất biến sai: {pbad}")
        if est_docs:
            L.append(f"- Ước lượng toàn kho (~{est_docs:,} bài × {per_doc:.2f}): **~{int(est_docs * per_doc):,} đoạn**")
        L.append("")
        summary[strategy] = s
    return "\n".join(L) + "\n", summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--strategy", default=None, help="recursive | structure | parent_child | all")
    ap.add_argument("--input", default=None, help="thư mục corpus_clean (mặc định <work>/corpus_clean)")
    ap.add_argument("--out", default=None, help="thư mục gốc kết quả (mặc định <work>/chunks)")
    ap.add_argument("--shard", default="0/1", help="i/N: chỉ xử lý file part thứ k với k %% N == i")
    ap.add_argument("--report", action="store_true", help="đo kết quả đã có, viết results/t2_report.md")
    ap.add_argument("--est-docs", type=int, default=None, help="số bài ước lượng của toàn kho (cho báo cáo)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    cc = cfg["chunk"]
    in_dir = Path(a.input) if a.input else cfg.work_dir / cfg["clean"]["out_dir"]
    out_root = Path(a.out) if a.out else cfg.work_dir / cc["out_dir"]
    if a.report:
        md, summary = report(in_dir, out_root, cc, a.est_docs)
        res = cfg.work_dir / "results"
        res.mkdir(parents=True, exist_ok=True)
        (res / "t2_report.md").write_text(md, encoding="utf-8")
        (res / "t2_report.json").write_text(json.dumps({"summary": summary, "chunk": cc, "input": str(in_dir)},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(md)
        return
    strategies = STRATEGIES if a.strategy == "all" else [a.strategy or cc["strategy"]]
    for s in strategies:
        if s not in STRATEGIES:
            raise SystemExit(f"chiến lược không hợp lệ: {s}")
        run(in_dir, out_root, s, cc, a.force, a.shard)


if __name__ == "__main__":
    main()
