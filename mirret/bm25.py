"""BM25 (biến thể Lucene, cùng công thức bm25s) với thống kê toàn cục, chia shard — MỘT chỉ mục cho mỗi
(mức đơn vị, ngôn ngữ tài liệu): index/bm25/{parents,children}_{vi,en,zh}/.

Nhánh BM25 của câu hỏi ngôn ngữ L tìm trong chỉ mục tài liệu ngôn ngữ L (câu hỏi zh ↔ tài liệu zh …): từ của ngôn ngữ
khác hầu như không khớp nên tách riêng vừa nhẹ hơn vừa cho IDF đúng của từng ngôn ngữ.
Văn bản đơn vị = ngữ cảnh + "\\n" + đoạn (C). Tách từ: vi pyvi (hoặc theo âm tiết), zh jieba, en chữ thường theo từ.
Phần tf/build/QueryEncoder dùng lại từ mir/bm25.py (đã kiểm chứng trùng điểm với bm25s method="lucene").

Tuỳ chọn tách từ (mặc định tắt — mặc định = đúng hành vi cũ): vi_mode="hybrid" (từ ghép pyvi + âm tiết rời), en_stem
(Snowball), zh_t2s (OpenCC phồn → giản), zh_userdict (từ điển riêng cho jieba). Cấu hình được ghi vào stats.json
("tokenizer") khi dựng; QueryEncoder đọc lại để tách câu hỏi y hệt lúc dựng. Chỉ mục cũ không có khoá này = mặc định cũ.
"""
import functools
import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import sparse_index

_WORD = re.compile(r"\w", re.U)
_EN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_JIEBA = None
_JIEBA_USER = {}             # đường dẫn userdict → jieba.Tokenizer riêng (không đụng từ điển toàn cục của jieba)
_STEM = None
_T2S = None

TOK_DEFAULT = {"vi_mode": "pyvi", "en_stem": False, "zh_t2s": False, "zh_userdict": None}
USERDICT = "zh_userdict.txt"  # bản sao userdict trong thư mục chỉ mục (tf/ khi tách, gốc chỉ mục sau build)


def _jieba(userdict=None):
    global _JIEBA
    if _JIEBA is None:
        import logging
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _JIEBA = jieba
    if not userdict:
        return _JIEBA
    key = str(userdict)
    if key not in _JIEBA_USER:
        t = _JIEBA.Tokenizer()
        t.load_userdict(key)                              # nạp một lần cho mỗi tiến trình
        _JIEBA_USER[key] = t
    return _JIEBA_USER[key]


def _stem(word):
    global _STEM
    if _STEM is None:
        import snowballstemmer
        _STEM = functools.lru_cache(maxsize=1 << 18)(snowballstemmer.stemmer("english").stemWord)
    return _STEM(word)


def _t2s(text):
    global _T2S
    if _T2S is None:
        import opencc
        try:
            _T2S = opencc.OpenCC("t2s")
        except Exception:
            _T2S = opencc.OpenCC("t2s.json")              # bản opencc chính thức cũ cần đuôi .json
    return _T2S.convert(text)


def tokenize(text, lang, vi_mode="pyvi", en_stem=False, zh_t2s=False, zh_userdict=None):
    text = text or ""
    if lang == "zh":
        if zh_t2s:
            text = _t2s(text)
        toks = _jieba(zh_userdict).lcut(text)
    elif lang == "en":
        toks = _EN.findall(text.lower())
        return [_stem(t) for t in toks] if en_stem else toks
    elif vi_mode in ("pyvi", "hybrid"):
        from pyvi import ViTokenizer
        toks = ViTokenizer.tokenize(text).split()
        if vi_mode == "hybrid":                           # đái_tháo_đường → đái_tháo_đường, đái, tháo, đường
            toks = [y for x in toks for y in ([x] + x.split("_") if "_" in x else [x])]
    else:
        toks = text.split()
    return [t.lower() for t in (x.strip() for x in toks) if t and _WORD.search(t)]


def tok_spec(lang, vi_mode="pyvi", en_stem=False, zh_t2s=False, zh_userdict=None):
    """Cấu hình tách từ của một chỉ mục ngôn ngữ lang (tuỳ chọn của ngôn ngữ khác về mặc định). zh_userdict ở đây là
    đường dẫn nguồn; khi ghi vào chỉ mục thay bằng tên bản sao USERDICT + sha1."""
    s = dict(TOK_DEFAULT)
    if lang == "vi":
        s["vi_mode"] = vi_mode
    elif lang == "en":
        s["en_stem"] = bool(en_stem)
    elif lang == "zh":
        s["zh_t2s"] = bool(zh_t2s)
        s["zh_userdict"] = str(zh_userdict) if zh_userdict else None
    return s


def tok_kwargs(spec, base_dir):
    """Cấu hình đã ghi (tokenizer.json / stats.json) → tham số cho tokenize(); kiểm tra bản sao userdict."""
    kw = {k: spec.get(k, v) for k, v in TOK_DEFAULT.items()}
    if kw["zh_userdict"]:
        p = Path(base_dir) / kw["zh_userdict"]
        if not p.exists():
            raise SystemExit(f"Chỉ mục BM25 cần userdict {p} (thiếu)")
        if spec.get("zh_userdict_sha1") and _sha1(p) != spec["zh_userdict_sha1"]:
            raise SystemExit(f"{p} khác bản đã dùng khi tách từ (sha1) → câu hỏi sẽ tách khác kho")
        kw["zh_userdict"] = str(p)
    return kw


def _sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _tok_batch(args):
    texts, lang, kw = args
    return [tokenize(t, lang, **kw) for t in texts]


def tf_matrix(token_lists):
    import scipy.sparse as sp
    vocab, indptr, indices, data, lens = {}, [0], [], [], []
    for toks in token_lists:
        c = {}
        for t in toks:
            j = vocab.setdefault(t, len(vocab))
            c[j] = c.get(j, 0) + 1
        ks = sorted(c)
        indices += ks
        data += [min(c[k], 65535) for k in ks]
        indptr.append(len(indices))
        lens.append(len(toks))
    m = sp.csr_matrix((np.array(data, dtype=np.uint16), np.array(indices, dtype=np.int32), np.array(indptr)),
                      shape=(len(token_lists), len(vocab)))
    words = [None] * len(vocab)
    for w, j in vocab.items():
        words[j] = w
    return m, words, np.array(lens, dtype=np.int32)


def _prepare_tf(out, spec, force):
    """Ghi <out>/tf/tokenizer.json (+ bản sao userdict) → tham số tokenize(). Không cho trộn hai cấu hình trong một tf/."""
    rec, src = dict(spec), None
    if spec["zh_userdict"]:
        src = Path(spec["zh_userdict"])
        if not src.exists():
            raise SystemExit(f"Không thấy userdict {src}")
        rec["zh_userdict"], rec["zh_userdict_sha1"] = USERDICT, _sha1(src)
    f = out / "tokenizer.json"
    done = any(out.glob("part-*_ids.parquet"))
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old != rec and done and not force:
            raise SystemExit(f"{out} đã tách từ với cấu hình khác {old} ≠ {rec}: dùng thư mục chỉ mục khác "
                             f"(bm25.index_dir) hoặc --force")
    elif done and not force:
        new = rec["vi_mode"] == "hybrid" or any(rec[k] != TOK_DEFAULT[k] for k in ("en_stem", "zh_t2s", "zh_userdict"))
        if new:
            raise SystemExit(f"{out} có kết quả tách từ cũ (không ghi cấu hình): tuỳ chọn mới cần thư mục chỉ mục "
                             f"khác (bm25.index_dir) hoặc --force")
        return tok_kwargs(rec, out)                       # tf cũ: giữ nguyên, build ghi stats.json như trước
    if src is not None and src.resolve() != (out / USERDICT).resolve():
        shutil.copyfile(src, out / USERDICT)
    f.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return tok_kwargs(rec, out)


def tokenize_parts(chunk_dir, out_dir, lang, shard="0/1", workers=4, vi_mode="pyvi", force=False, log=print,
                   en_stem=False, zh_t2s=False, zh_userdict=None):
    """Bước 1: mỗi file chunks/<mức>/part-*.parquet → <out>/tf/part-*.npz cho các đơn vị có lang == lang.
    Cấu hình tách từ ghi ở <out>/tf/tokenizer.json (build chép sang stats.json)."""
    out = Path(out_dir) / "tf"
    out.mkdir(parents=True, exist_ok=True)
    kw = _prepare_tf(out, tok_spec(lang, vi_mode, en_stem, zh_t2s, zh_userdict), force)
    si, sn = (int(x) for x in shard.split("/"))
    parts = sorted(Path(chunk_dir).glob("part-*.parquet"))
    for k, p in enumerate(parts):
        if k % sn != si:
            continue
        stem = out / p.stem
        if Path(str(stem) + "_ids.parquet").exists() and not force:
            continue
        t = pq.read_table(p, columns=["unit_id", "lang", "ctx", "text"])
        t = t.filter(pc.equal(t.column("lang"), lang)).to_pydict()
        n = len(t["unit_id"])
        texts = [(c + "\n" if c else "") + x for c, x in zip(t["ctx"], t["text"])]
        step = 2000
        jobs = [(texts[i:i + step], lang, kw) for i in range(0, n, step)]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(workers) as ex:
                toks = [x for b in ex.map(_tok_batch, jobs) for x in b]
        else:
            toks = [x for j in jobs for x in _tok_batch(j)]
        m, words, lens = tf_matrix(toks)
        np.savez(str(stem) + ".npz", indptr=m.indptr.astype(np.int64), indices=m.indices, data=m.data,
                 shape=np.array(m.shape), lens=lens)
        pq.write_table(pa.table({"term": pa.array(words, pa.string())}), str(stem) + "_vocab.parquet")
        pq.write_table(pa.table({"unit_id": pa.array(t["unit_id"], pa.string())}), str(stem) + "_ids.parquet")
        log(f"  bm25 {lang} {p.name}: {n:,} đơn vị, {len(words):,} từ cục bộ")


def _load_tf(stem):
    import scipy.sparse as sp
    z = np.load(str(stem) + ".npz")
    m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    words = pq.read_table(str(stem) + "_vocab.parquet").column("term").to_pylist()
    ids = pq.read_table(str(stem) + "_ids.parquet").column("unit_id").to_pylist()
    return m, words, z["lens"], ids


def idf_lucene(df, n):
    return np.log1p((n - df + 0.5) / (df + 0.5))


def build(out_dir, k1=1.5, b=0.75, force=False, log=print):
    """Bước 2: thống kê toàn cục + ma trận BM25 mỗi part → <out>/shard-XXXX.npz. Trọng số tính bằng float32, khi ghi
    sparse_index.save_csc lưu CSC float16 (sai số tương đối ~1e-3 so với bm25s). Cấu hình tách từ (tf/tokenizer.json,
    nếu có) chép vào stats.json["tokenizer"]."""
    import scipy.sparse as sp
    out = Path(out_dir)
    stems = sorted(p.with_name(p.name[:-len("_ids.parquet")]) for p in (out / "tf").glob("part-*_ids.parquet"))
    if not stems:
        raise SystemExit(f"Chưa có kết quả tách từ trong {out / 'tf'}")
    gvocab, df_parts, n_docs, total_len = {}, [], 0, 0
    for s in stems:
        m, words, lens, _ = _load_tf(s)
        gid = np.array([gvocab.setdefault(w, len(gvocab)) for w in words], dtype=np.int64)
        df_parts.append((gid, np.bincount(m.indices, minlength=m.shape[1])))
        n_docs += m.shape[0]
        total_len += int(lens.sum())
    df = np.zeros(len(gvocab), dtype=np.int64)
    for gid, d in df_parts:
        np.add.at(df, gid, d)
    avgdl = total_len / max(1, n_docs)
    idf = idf_lucene(df.astype(np.float64), n_docs)
    terms = [None] * len(gvocab)
    for w, j in gvocab.items():
        terms[j] = w
    pq.write_table(pa.table({"term": pa.array(terms, pa.string()), "df": df}), out / "vocab.parquet")
    stats = {"n_docs": n_docs, "avgdl": avgdl, "k1": k1, "b": b, "vocab": len(terms), "method": "lucene"}
    tf_spec = out / "tf" / "tokenizer.json"
    if tf_spec.exists():
        stats["tokenizer"] = json.loads(tf_spec.read_text(encoding="utf-8"))
        if stats["tokenizer"].get("zh_userdict"):
            shutil.copyfile(out / "tf" / stats["tokenizer"]["zh_userdict"], out / stats["tokenizer"]["zh_userdict"])
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, s in enumerate(stems):
        if (out / f"shard-{k:04d}.npz").exists() and not force:
            continue
        m, words, lens, ids = _load_tf(s)
        if m.shape[0] == 0:
            continue
        gid = np.array([gvocab[w] for w in words], dtype=np.int64)
        m = m.tocoo()
        tf = m.data.astype(np.float64)
        norm = k1 * ((1 - b) + b * lens[m.row] / avgdl)
        w = idf[gid[m.col]] * tf / (norm + tf)
        X = sp.csr_matrix((w.astype(np.float32), (m.row, gid[m.col])), shape=(m.shape[0], len(terms)))
        sparse_index.write_shard(out, k, X, ids)
    log(f"BM25 {out.name}: {n_docs:,} đơn vị, {len(terms):,} từ, độ dài trung bình {avgdl:.1f}")


def index_tok(out_dir, vi_mode="pyvi"):
    """Tham số tokenize() của một chỉ mục đã dựng: theo stats.json["tokenizer"]; chỉ mục cũ (không có khoá) = mặc định
    cũ với vi_mode truyền vào (bm25.vi_tokenizer)."""
    out_dir = Path(out_dir)
    p = out_dir / "stats.json"
    spec = json.loads(p.read_text(encoding="utf-8")).get("tokenizer") if p.exists() else None
    if spec is None:
        return dict(TOK_DEFAULT, vi_mode=vi_mode)
    return tok_kwargs(spec, out_dir)


class QueryEncoder:
    """Câu hỏi → (id từ, số lần) theo từ vựng toàn cục của một chỉ mục; từ không có trong kho bị bỏ.
    Tách từ theo cấu hình ghi trong stats.json của chỉ mục (vi_mode chỉ dùng cho chỉ mục cũ không ghi cấu hình)."""

    def __init__(self, out_dir, vi_mode="pyvi"):
        terms = pq.read_table(Path(out_dir) / "vocab.parquet").column("term").to_pylist()
        self.vocab = {w: i for i, w in enumerate(terms)}
        self.tok = index_tok(out_dir, vi_mode)
        self.vi_mode = self.tok["vi_mode"]

    def tokens(self, text, lang):
        return tokenize(text, lang, **self.tok)

    def __call__(self, text, lang):
        c = {}
        for t in self.tokens(text, lang):
            if t in self.vocab:
                c[self.vocab[t]] = c.get(self.vocab[t], 0) + 1
        ks = sorted(c)
        return np.array(ks, dtype=np.int64), np.array([c[k] for k in ks], dtype=np.float32)


def bm25s_reference_scores(token_lists, query_tokens, k1=1.5, b=0.75):
    """Điểm của thư viện bm25s (để kiểm chứng) — chỉ dùng cho tập nhỏ."""
    import bm25s
    vocab = {}
    ids = [[vocab.setdefault(t, len(vocab)) for t in d] for d in token_lists]
    r = bm25s.BM25(k1=k1, b=b, method="lucene")
    r.index(bm25s.tokenization.Tokenized(ids=ids, vocab=vocab), show_progress=False)
    q = [t for t in query_tokens if t in vocab]
    return r.get_scores(q) if q else np.zeros(len(token_lists))
