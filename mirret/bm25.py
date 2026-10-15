"""BM25 (biến thể Lucene, cùng công thức bm25s) với thống kê toàn cục, chia shard — MỘT chỉ mục cho mỗi
(mức đơn vị, ngôn ngữ tài liệu): index/bm25/{parents,children}_{vi,en,zh}/.

Nhánh BM25 của câu hỏi ngôn ngữ L tìm trong chỉ mục tài liệu ngôn ngữ L (câu hỏi zh ↔ tài liệu zh …): từ của ngôn ngữ
khác hầu như không khớp nên tách riêng vừa nhẹ hơn vừa cho IDF đúng của từng ngôn ngữ.
Văn bản đơn vị = ngữ cảnh + "\\n" + đoạn (C). Tách từ: vi pyvi (hoặc theo âm tiết), zh jieba, en chữ thường theo từ.
Phần tf/build/QueryEncoder dùng lại từ mir/bm25.py (đã kiểm chứng trùng điểm với bm25s method="lucene").
"""
import json
import re
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


def tokenize(text, lang, vi_mode="pyvi"):
    text = text or ""
    if lang == "zh":
        global _JIEBA
        if _JIEBA is None:
            import logging
            import jieba
            jieba.setLogLevel(logging.WARNING)
            _JIEBA = jieba
        toks = _JIEBA.lcut(text)
    elif lang == "en":
        return _EN.findall(text.lower())
    elif vi_mode == "pyvi":
        from pyvi import ViTokenizer
        toks = ViTokenizer.tokenize(text).split()
    else:
        toks = text.split()
    return [t.lower() for t in (x.strip() for x in toks) if t and _WORD.search(t)]


def _tok_batch(args):
    texts, lang, vi_mode = args
    return [tokenize(t, lang, vi_mode) for t in texts]


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


def tokenize_parts(chunk_dir, out_dir, lang, shard="0/1", workers=4, vi_mode="pyvi", force=False, log=print):
    """Bước 1: mỗi file chunks/<mức>/part-*.parquet → <out>/tf/part-*.npz cho các đơn vị có lang == lang."""
    out = Path(out_dir) / "tf"
    out.mkdir(parents=True, exist_ok=True)
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
        jobs = [(texts[i:i + step], lang, vi_mode) for i in range(0, n, step)]
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
    """Bước 2: thống kê toàn cục + ma trận BM25 (CSC float16) mỗi part → <out>/shard-XXXX.npz."""
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
    (out / "stats.json").write_text(json.dumps({"n_docs": n_docs, "avgdl": avgdl, "k1": k1, "b": b,
                                                "vocab": len(terms), "method": "lucene"}, indent=1), encoding="utf-8")
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


class QueryEncoder:
    """Câu hỏi → (id từ, số lần) theo từ vựng toàn cục của một chỉ mục; từ không có trong kho bị bỏ."""

    def __init__(self, out_dir, vi_mode="pyvi"):
        terms = pq.read_table(Path(out_dir) / "vocab.parquet").column("term").to_pylist()
        self.vocab = {w: i for i, w in enumerate(terms)}
        self.vi_mode = vi_mode

    def __call__(self, text, lang):
        c = {}
        for t in tokenize(text, lang, self.vi_mode):
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
