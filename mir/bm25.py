"""BM25 (biến thể Lucene, cùng công thức với thư viện bm25s) với thống kê TOÀN CỤC, chia shard.

Vì sao không dùng thẳng bm25s: với ~8,6 triệu đoạn, bm25s dựng một chỉ mục trong bộ nhớ vượt ~30 GB; chia shard
bằng bm25s thì mỗi shard có IDF riêng nên điểm giữa các shard không so được. Ở đây:
  1) tokenize: mỗi file chunks/<strategy>/part-*.parquet → ma trận tần suất (TF) với từ vựng cục bộ  [chia --shard được]
  2) build: gộp từ vựng + df + độ dài trung bình toàn kho → mỗi part thành ma trận trọng số BM25 (CSC) với IDF chung
  3) truy vấn: câu hỏi tách từ bằng pyvi → id từ → tổng trọng số (mir.sparse_index.search)
Điểm trùng với bm25s(method="lucene") — xem tests/test_t4.py.

Tách từ: tiếng Việt bằng pyvi (ViTokenizer, ghép từ ghép bằng "_"), tiếng Trung bằng jieba; chữ thường, bỏ token
không có chữ/số. Mỗi đoạn tách theo ngôn ngữ của đoạn; câu hỏi (tiếng Việt) tách bằng pyvi.
"""
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import sparse_index

_WORD = re.compile(r"\w", re.U)
_JIEBA = None


def tokenize(text, lang):
    if lang == "zh":
        global _JIEBA
        if _JIEBA is None:
            import logging
            import jieba
            jieba.setLogLevel(logging.WARNING)
            _JIEBA = jieba
        toks = _JIEBA.lcut(text)
    else:
        from pyvi import ViTokenizer
        toks = ViTokenizer.tokenize(text).split()
    return [t.lower() for t in (x.strip() for x in toks) if t and _WORD.search(t)]


def _tok_batch(args):
    texts, langs = args
    return [tokenize(t, l) for t, l in zip(texts, langs)]


def tf_matrix(token_lists):
    """→ (CSR tần suất uint16 [n, |V cục bộ|], từ vựng cục bộ list[str], độ dài đoạn np.int32)."""
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


def tokenize_parts(chunk_dir, out_dir, shard="0/1", workers=4, force=False, log=print):
    """Bước 1: mỗi part → <out>/tf/part-XXXXX.npz (+ _vocab.parquet, _ids.parquet)."""
    out = Path(out_dir) / "tf"
    out.mkdir(parents=True, exist_ok=True)
    si, sn = (int(x) for x in shard.split("/"))
    parts = sorted(Path(chunk_dir).glob("part-*.parquet"))
    for k, p in enumerate(parts):
        if k % sn != si:
            continue
        stem = out / p.stem
        if Path(str(stem) + "_ids.parquet").exists() and not force:
            log(f"  {p.name}: đã tách từ → bỏ qua")
            continue
        t = pq.read_table(p, columns=["chunk_id", "lang", "text"]).to_pydict()
        n = len(t["chunk_id"])
        step = 2000
        jobs = [(t["text"][i:i + step], t["lang"][i:i + step]) for i in range(0, n, step)]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(workers) as ex:
                toks = [x for b in ex.map(_tok_batch, jobs) for x in b]
        else:
            toks = [x for j in jobs for x in _tok_batch(j)]
        m, words, lens = tf_matrix(toks)
        np.savez(str(stem) + ".npz", indptr=m.indptr.astype(np.int64), indices=m.indices, data=m.data,
                 shape=np.array(m.shape), lens=lens)
        pq.write_table(pa.table({"term": words}), str(stem) + "_vocab.parquet")
        pq.write_table(pa.table({"chunk_id": t["chunk_id"]}), str(stem) + "_ids.parquet")   # ghi sau cùng = xong
        log(f"  {p.name}: {n:,} đoạn, {len(words):,} từ cục bộ")


def _load_tf(stem):
    import scipy.sparse as sp
    z = np.load(str(stem) + ".npz")
    m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    words = pq.read_table(str(stem) + "_vocab.parquet").column("term").to_pylist()
    ids = pq.read_table(str(stem) + "_ids.parquet").column("chunk_id").to_pylist()
    return m, words, z["lens"], ids


def idf_lucene(df, n):
    return np.log1p((n - df + 0.5) / (df + 0.5))


def build(out_dir, k1=1.5, b=0.75, force=False, log=print):
    """Bước 2: thống kê toàn cục + ma trận BM25 (CSC float16) mỗi part → <out>/shard-XXXX.npz."""
    out = Path(out_dir)
    stems = sorted(p.with_name(p.name[:-len("_ids.parquet")]) for p in (out / "tf").glob("part-*_ids.parquet"))
    if not stems:
        raise SystemExit(f"Chưa có kết quả tách từ trong {out / 'tf'}")
    gvocab, df_parts, n_docs, total_len = {}, [], 0, 0
    for s in stems:                                       # lượt 1: từ vựng chung, df, tổng độ dài
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
    pq.write_table(pa.table({"term": terms, "df": df}), out / "vocab.parquet")
    (out / "stats.json").write_text(json.dumps({"n_docs": n_docs, "avgdl": avgdl, "k1": k1, "b": b,
                                                "vocab": len(terms), "method": "lucene"}, indent=1), encoding="utf-8")
    import scipy.sparse as sp
    for k, s in enumerate(stems):                         # lượt 2: trọng số BM25 với thống kê chung
        if (out / f"shard-{k:04d}.npz").exists() and not force:
            continue
        m, words, lens, ids = _load_tf(s)
        gid = np.array([gvocab[w] for w in words], dtype=np.int64)
        m = m.tocoo()
        tf = m.data.astype(np.float64)
        norm = k1 * ((1 - b) + b * lens[m.row] / avgdl)
        w = idf[gid[m.col]] * tf / (norm + tf)
        X = sp.csr_matrix((w.astype(np.float32), (m.row, gid[m.col])), shape=(m.shape[0], len(terms)))
        sparse_index.write_shard(out, k, X, ids)
        log(f"  shard {k}: {m.shape[0]:,} đoạn")
    log(f"BM25: {n_docs:,} đoạn, {len(terms):,} từ, độ dài trung bình {avgdl:.1f}")


class QueryEncoder:
    """Câu hỏi → (id từ, số lần xuất hiện) theo từ vựng toàn cục; từ không có trong kho bị bỏ."""

    def __init__(self, out_dir):
        terms = pq.read_table(Path(out_dir) / "vocab.parquet").column("term").to_pylist()
        self.vocab = {w: i for i, w in enumerate(terms)}

    def __call__(self, text, lang="vi"):
        c = {}
        for t in tokenize(text, lang):
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

