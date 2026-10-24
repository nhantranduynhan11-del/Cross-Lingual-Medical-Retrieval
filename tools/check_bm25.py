"""Kiểm tra mirret/bm25.py trên mẫu nhỏ (không đụng chỉ mục thật).

Nguồn đoạn mẫu: --chunks chunks/parents/part-00000.parquet (unit_id, lang, ctx, text) hoặc --sheets "<judge>/sheet_*.csv"
(context, chunk_text, doc_lang — các sheet chấm nhãn). Câu hỏi: --queries <thư mục có queries_{vi,en,zh}.parquet>
(qtrans), không có thì dùng ngữ cảnh (tiêu đề) của các đoạn mẫu.

  python tools/check_bm25.py tok   --sheets "...judge/sheet_*.csv" [--rev HEAD] [--n 300]
         (a) cấu hình mặc định tách từ giống hệt bản bm25.py ở commit --rev
  python tools/check_bm25.py ref   --sheets ... [--queries qtrans] [--n 300]
         (b) dựng chỉ mục nhỏ (2 part, qua tokenize_parts + build + QueryEncoder + sparse_index.search) cho từng cấu hình
         tách từ, so điểm với bm25s_reference_scores (thiếu bm25s → công thức Lucene tự cài trong file này); kèm kiểm tra
         trọng số câu hỏi được nhân vào điểm (câu hỏi lặp từ → điểm từ đó nhân đôi)
  python tools/check_bm25.py trad  --chunks .../part-00000.parquet   tỉ lệ chữ phồn thể trong đoạn zh (cần opencc)
"""
import argparse
import glob
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import bm25, config, sparse_index  # noqa: E402

LANGS = ("vi", "en", "zh")
CONFIGS = {                                      # tên → tuỳ chọn tách từ (tok_spec)
    "default": {},
    "vi_hybrid": {"vi_mode": "hybrid"},
    "vi_syllable": {"vi_mode": "syllable"},
    "en_stem": {"en_stem": True},
    "zh_t2s": {"zh_t2s": True},
    "zh_userdict": {"zh_userdict": str(config.ROOT / "configs" / "jieba_med_userdict.txt")},
}
NEEDS = {"en_stem": "snowballstemmer", "zh_t2s": "opencc"}


def load_samples(a):
    """→ [(unit_id, lang, ctx, text)]."""
    rows = []
    if a.chunks:
        for p in glob.glob(a.chunks):
            t = pq.read_table(p, columns=["unit_id", "lang", "ctx", "text"]).to_pydict()
            rows += list(zip(t["unit_id"], t["lang"], t["ctx"], t["text"]))
    if a.sheets:
        import pandas as pd
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.sheets))]).drop_duplicates("chunk_text")
        rows += [(f"s{i}", r.doc_lang, r.context if isinstance(r.context, str) else "", r.chunk_text)
                 for i, r in enumerate(d.itertuples()) if isinstance(r.chunk_text, str)]
    if not rows:
        raise SystemExit("cần --chunks hoặc --sheets")
    return rows


def per_lang(rows, n):
    return {lg: [r for r in rows if r[1] == lg][:n] for lg in LANGS}


def unit_text(ctx, text):
    return (ctx + "\n" if ctx else "") + text


def available(name):
    mod = NEEDS.get(name)
    if not mod:
        return True
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- (a) tách từ mặc định giống bản cũ
def old_module(rev):
    src = subprocess.run(["git", "show", f"{rev}:mirret/bm25.py"], cwd=config.ROOT, capture_output=True,
                         check=True).stdout.decode("utf-8")
    m = types.ModuleType("mirret._bm25_old")
    m.__package__ = "mirret"
    exec(compile(src, f"{rev}:mirret/bm25.py", "exec"), m.__dict__)
    return m


def cmd_tok(a):
    old = old_module(a.rev)
    bad = 0
    for lg, rows in per_lang(load_samples(a), a.n).items():
        modes = ["pyvi", "syllable"] if lg == "vi" else ["pyvi"]
        for vm in modes:
            diff = [r[0] for r in rows if old.tokenize(unit_text(r[2], r[3]), lg, vm)
                    != bm25.tokenize(unit_text(r[2], r[3]), lg, vm)]
            # QueryEncoder của chỉ mục cũ (stats.json không có "tokenizer") cũng phải ra cùng tham số
            kw = bm25.index_tok(Path(tempfile.gettempdir()) / "khong_co_chi_muc", vm)
            diff_q = [r[0] for r in rows if old.tokenize(r[3], lg, vm) != bm25.tokenize(r[3], lg, **kw)]
            bad += len(diff) + len(diff_q)
            print(f"  {lg} vi_mode={vm}: {len(rows)} đoạn, khác bản {a.rev}: {len(diff)} (tokenize), "
                  f"{len(diff_q)} (tham số chỉ mục cũ) {diff[:5]}")
    print("→ (a) " + ("ĐẠT: mặc định giống hệt bản cũ" if not bad else f"KHÔNG ĐẠT: {bad} khác biệt"))
    return bad == 0


# ---------------------------------------------------------------- (b) điểm khớp bm25s
def lucene_scores(token_lists, query_tokens, k1=1.5, b=0.75):
    """Công thức bm25s method="lucene" (dự phòng khi chưa cài bm25s): Σ_t∈q idf(t)·tf/(tf + k1·(1−b+b·dl/avgdl))."""
    n = len(token_lists)
    dl = np.array([len(d) for d in token_lists], dtype=np.float64)
    avgdl = dl.sum() / max(1, n)
    tfs = []
    df = {}
    for d in token_lists:
        c = {}
        for t in d:
            c[t] = c.get(t, 0) + 1
        tfs.append(c)
        for t in c:
            df[t] = df.get(t, 0) + 1
    out = np.zeros(n)
    for t in query_tokens:
        if t not in df:
            continue
        idf = np.log1p((n - df[t] + 0.5) / (df[t] + 0.5))
        tf = np.array([c.get(t, 0) for c in tfs], dtype=np.float64)
        out += idf * tf / (tf + k1 * ((1 - b) + b * dl / avgdl))
    return out


def reference(token_lists, query_tokens):
    try:
        import bm25s  # noqa: F401
        return bm25.bm25s_reference_scores(token_lists, query_tokens), "bm25s"
    except ImportError:
        return lucene_scores(token_lists, query_tokens), "lucene tự cài (chưa có bm25s)"


def build_small(tmp, lang, rows, opts):
    """Chỉ mục nhỏ qua đúng đường mirret: chunks 2 part → tokenize_parts → build."""
    cdir, idir = Path(tmp) / "chunks", Path(tmp) / "index"
    cdir.mkdir(parents=True, exist_ok=True)
    half = len(rows) // 2
    for k, part in enumerate((rows[:half], rows[half:])):
        pq.write_table(pa.table({"unit_id": [r[0] for r in part], "lang": [lang] * len(part),
                                 "ctx": [r[2] for r in part], "text": [r[3] for r in part]}),
                       cdir / f"part-{k:05d}.parquet")
    bm25.tokenize_parts(cdir, idir, lang, workers=1, log=lambda *_: None, **opts)
    bm25.build(idir, log=lambda *_: None)
    return idir


def queries_for(a, lang, rows):
    if a.queries and (Path(a.queries) / f"queries_{lang}.parquet").exists():
        return pq.read_table(Path(a.queries) / f"queries_{lang}.parquet").column("query").to_pylist()[:a.nq]
    return [r[2] for r in rows if r[2]][:a.nq]


def cmd_ref(a):
    samples = per_lang(load_samples(a), a.n)
    ok_all = True
    for name, extra in CONFIGS.items():
        if not available(name):
            print(f"[{name}] BỎ QUA: chưa cài {NEEDS[name]}")
            continue
        langs = [lg for lg in LANGS if name == "default" or name.startswith(lg)]
        for lg in langs:
            rows = samples[lg]
            opts = {k: v for k, v in extra.items()}
            with tempfile.TemporaryDirectory() as tmp:
                idir = build_small(tmp, lg, rows, opts)
                qe = bm25.QueryEncoder(idir)
                ids_all = pq.read_table(idir / "shard-0000_ids.parquet").column("unit_id").to_pylist() + \
                    pq.read_table(idir / "shard-0001_ids.parquet").column("unit_id").to_pylist()
                order = {u: i for i, u in enumerate(r[0] for r in rows)}
                docs = [qe.tokens(unit_text(r[2], r[3]), lg) for r in rows]
                qs = queries_for(a, lg, rows)
                res = sparse_index.search(idir, [qe(q, lg) for q in qs], len(rows))
                worst, src = 0.0, ""
                for q, (ids, sc) in zip(qs, res):
                    ref, src = reference(docs, qe.tokens(q, lg))
                    got = np.zeros(len(rows))
                    got[[order[u] for u in ids]] = sc
                    worst = max(worst, float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))))
                # trọng số câu hỏi: lặp câu hỏi 2 lần → (id, 2·w) → điểm gấp đôi
                t, w = qe(qs[0], lg)
                r1 = sparse_index.search(idir, [(t, w)], 5)[0]
                r2 = sparse_index.search(idir, [qe(qs[0] + " " + qs[0], lg)], 5)[0]
                wt_ok = r1[0] == r2[0] and np.allclose(2 * np.array(r1[1]), r2[1], rtol=1e-5)
                ok = worst < 2e-3 and wt_ok and len(ids_all) == len(rows)
                ok_all &= ok
                spec = bm25.json.loads((idir / "stats.json").read_text(encoding="utf-8")).get("tokenizer")
                print(f"[{name}] {lg}: {len(rows)} đoạn, {len(qs)} câu hỏi, sai lệch tương đối lớn nhất {worst:.2e} "
                      f"(so {src}; ma trận lưu float16), trọng số câu hỏi {'đúng' if wt_ok else 'SAI'} "
                      f"→ {'ĐẠT' if ok else 'KHÔNG ĐẠT'}; stats.json tokenizer={spec}")
                if a.show:
                    print(f"    vd: {qs[0][:80]!r} → {qe.tokens(qs[0], lg)[:25]}")
    print("→ (b) " + ("ĐẠT" if ok_all else "KHÔNG ĐẠT"))
    return ok_all


# ---------------------------------------------------------------- tỉ lệ phồn thể
def cmd_trad(a):
    import re
    han = re.compile(r"[一-鿿]")
    rows = [r for r in load_samples(a) if r[1] == "zh"]
    n_ch = n_tr = n_doc = 0
    ex = []
    for r in rows:
        s = unit_text(r[2], r[3])
        t = bm25._t2s(s)
        chs = [(x, y) for x, y in zip(s, t) if han.match(x)] if len(s) == len(t) else []
        k = sum(x != y for x, y in chs)
        n_ch += len(chs)
        n_tr += k
        if k:
            n_doc += 1
            if len(ex) < 5:
                ex.append("".join(sorted({x for x, y in chs if x != y}))[:20])
    print(f"zh: {len(rows):,} đoạn, {n_ch:,} chữ Hán; chữ đổi khi t2s: {n_tr:,} ({n_tr / max(1, n_ch):.3%}); "
          f"đoạn có ≥1 chữ phồn thể: {n_doc:,} ({n_doc / max(1, len(rows)):.1%}). Ví dụ: {ex}")


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tok", "ref", "trad"])
    ap.add_argument("--chunks", default=None, help="glob part-*.parquet của chunks")
    ap.add_argument("--sheets", default=None, help="glob sheet_*.csv chấm nhãn")
    ap.add_argument("--queries", default=None, help="thư mục qtrans có queries_{vi,en,zh}.parquet")
    ap.add_argument("--n", type=int, default=300, help="số đoạn mỗi ngôn ngữ")
    ap.add_argument("--nq", type=int, default=20, help="số câu hỏi mỗi ngôn ngữ (ref)")
    ap.add_argument("--rev", default="HEAD", help="(tok) commit chứa bm25.py bản cũ")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    ok = {"tok": cmd_tok, "ref": cmd_ref, "trad": cmd_trad}[a.cmd](a)
    sys.exit(0 if ok in (True, None) else 1)


if __name__ == "__main__":
    main()
