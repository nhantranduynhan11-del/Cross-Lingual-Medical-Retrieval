"""T3: mã hoá BGE-M3 — vector dense (CLS, chuẩn hoá L2) + lexical weights (sparse). ColBERT để dành cho T6.

Cài đặt lại đúng BGEM3 của FlagEmbedding (đã đối chiếu trong notebooks/t3_encode_kaggle.ipynb):
  dense  = last_hidden[:, 0] chuẩn hoá L2
  sparse = relu(sparse_linear(last_hidden)), bỏ token đặc biệt (<s> </s> <pad> <unk>), mỗi token id lấy giá trị lớn nhất.
Đầu vào mỗi đoạn: tiêu đề + "\\n" + đoạn, cắt ở max_length (tính cả <s></s>).

Chia việc: toàn bộ đoạn (đọc các chunks/<strategy>/part-*.parquet theo thứ tự) được cắt thành khối shard_size đoạn;
khối k → emb/<strategy>/shard-{k:04d}/ gồm dense.npy (float16 [n,1024]), sparse.npz (CSR float16, đọc bằng
load_sparse()), chunk_ids.parquet, _DONE.json. --shard i/N làm các khối k % N == i; khối đã có _DONE.json bị bỏ qua.

  python -m mir.encode --config configs/baseline.yaml --shard 0/6 --device cuda:0
  python -m mir.encode --config configs/baseline.yaml --bench 2000 --device cuda:0      # đo tốc độ, ước lượng toàn kho
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config


# ---------------- sparse: lưu / đọc ----------------
def save_sparse(path, indptr, indices, data, n_cols):
    """CSR với data float16 (scipy.sparse không hỗ trợ float16 → lưu từng mảng)."""
    np.savez(path, format=np.array("csr"), shape=np.array([len(indptr) - 1, n_cols]),
             indptr=np.asarray(indptr, dtype=np.int64), indices=np.asarray(indices, dtype=np.int32),
             data=np.asarray(data, dtype=np.float16))


def load_sparse(path, dtype=np.float32):
    """→ scipy.sparse.csr_matrix (float32 mặc định)."""
    import scipy.sparse as sp
    z = np.load(path)
    return sp.csr_matrix((z["data"].astype(dtype), z["indices"], z["indptr"]), shape=tuple(z["shape"]))


# ---------------- model ----------------
class M3Encoder:
    def __init__(self, model, tokenizer, sparse_linear, device="cpu", fp16=False, max_length=512, colbert_linear=None):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.half = fp16 and self.device.type == "cuda"
        self.model = model.to(self.device).eval()
        self.sparse_linear = sparse_linear.to(self.device).eval()
        self.colbert_linear = colbert_linear.to(self.device).eval() if colbert_linear is not None else None
        if self.half:
            self.model.half()
            self.sparse_linear.half()
            if self.colbert_linear is not None:
                self.colbert_linear.half()
        self.tok = tokenizer
        self.max_length = max_length
        self.vocab = len(tokenizer)
        self.unused = {tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id}

    @classmethod
    def load(cls, name, device="cpu", fp16=True, max_length=512, colbert=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        H = model.config.hidden_size

        def head(fname, out_dim):
            p = Path(name) / fname
            if not p.is_file():
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(name, fname)
            lin = torch.nn.Linear(H, out_dim)
            lin.load_state_dict(torch.load(p, map_location="cpu"))
            return lin
        return cls(model, tok, head("sparse_linear.pt", 1), device, fp16, max_length,
                   head("colbert_linear.pt", H) if colbert else None)

    def colbert(self, texts, max_length=None, batch_tokens=16384, max_batch=128):
        """Vector ColBERT như FlagEmbedding: colbert_linear(h[:, 1:]) (bỏ <s>, bỏ đệm), chuẩn hoá L2.
        → list mảng float16 [số token - 1, H] theo đúng thứ tự texts."""
        torch = self.torch
        out = [None] * len(texts)
        pad = self.tok.pad_token_id
        with torch.inference_mode():
            for idx, ids in self.batches(texts, batch_tokens, max_batch, max_length):
                L = max(map(len, ids))
                inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=self.device)
                mask = (inp != pad).long()
                h = self.model(input_ids=inp, attention_mask=mask).last_hidden_state
                v = self.colbert_linear(h[:, 1:]) * mask[:, 1:, None].to(h.dtype)
                v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy().astype(np.float16)
                for k, i in enumerate(idx):
                    out[i] = v[k, :len(ids[k]) - 1]
        return out

    def batches(self, texts, batch_tokens, max_batch, max_length=None):
        """Sắp theo độ dài để giảm đệm; trả về (chỉ số gốc, input) từng batch."""
        enc = self.tok(texts, truncation=True, max_length=max_length or self.max_length,
                       add_special_tokens=True)["input_ids"]
        order = sorted(range(len(texts)), key=lambda i: -len(enc[i]))
        b = []
        for i in order:
            if b and (len(b) + 1 > max_batch or (len(b) + 1) * len(enc[b[0]]) > batch_tokens):
                yield b, [enc[j] for j in b]
                b = []
            b.append(i)
        if b:
            yield b, [enc[j] for j in b]

    def encode(self, texts, batch_tokens=16384, max_batch=128, min_weight=0.0):
        """→ dense float16 [n, H]; sparse: list[(token_ids int32 tăng dần, weights float32)]; số đoạn bị cắt."""
        torch = self.torch
        n = len(texts)
        dense = np.zeros((n, self.model.config.hidden_size), dtype=np.float16)
        sparse = [None] * n
        truncated = 0
        unused = np.array(sorted(self.unused), dtype=np.int64)
        V = np.int64(self.vocab)
        pad = self.tok.pad_token_id
        with torch.inference_mode():
            for idx, ids in self.batches(texts, batch_tokens, max_batch):
                L = max(map(len, ids))
                truncated += sum(1 for x in ids if len(x) >= self.max_length)
                ids_np = np.full((len(ids), L), pad, dtype=np.int64)
                for k, x in enumerate(ids):
                    ids_np[k, :len(x)] = x
                inp = torch.from_numpy(ids_np).to(self.device)
                mask = (inp != pad).long()
                h = self.model(input_ids=inp, attention_mask=mask).last_hidden_state
                d = torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu().numpy().astype(np.float16)
                w = torch.relu(self.sparse_linear(h)).squeeze(-1).float().cpu().numpy()
                dense[idx] = d
                # lexical weights, vector hoá cho cả batch: bỏ token đặc biệt/đệm, lấy max theo (dòng, token id)
                rows, cols = np.nonzero(~np.isin(ids_np, unused) & (w > min_weight))
                keys = rows.astype(np.int64) * V + ids_np[rows, cols]
                vals = w[rows, cols]
                order = np.argsort(keys, kind="stable")
                keys, vals = keys[order], vals[order]
                uk, start = np.unique(keys, return_index=True)
                mx = np.maximum.reduceat(vals, start) if len(vals) else vals
                r, t = uk // V, (uk % V).astype(np.int32)
                bounds = np.searchsorted(r, np.arange(len(ids) + 1))
                for k, i in enumerate(idx):
                    sparse[i] = (t[bounds[k]:bounds[k + 1]], mx[bounds[k]:bounds[k + 1]])
        return dense, sparse, truncated


# ---------------- dữ liệu ----------------
def chunk_parts(chunk_dir):
    parts = sorted(Path(chunk_dir).glob("part-*.parquet"))
    return [(p, pq.ParquetFile(p).metadata.num_rows) for p in parts]


def block_rows(parts, k, size):
    """Khối k = các đoạn [k*size, (k+1)*size) của chuỗi nối mọi part → [(part, đầu, cuối)]."""
    lo, hi, pos, out = k * size, (k + 1) * size, 0, []
    for p, n in parts:
        a, b = max(lo, pos), min(hi, pos + n)
        if a < b:
            out.append((p, a - pos, b - pos))
        pos += n
    return out


_CACHE = {}


def read_block(parts, k, size):
    cols = ["chunk_id", "doc_id", "text"]
    tabs = []
    for p, a, b in block_rows(parts, k, size):
        if _CACHE.get("path") != p:                          # giữ 1 part trong bộ nhớ (các khối liên tiếp dùng chung)
            _CACHE.clear()
            _CACHE.update(path=p, table=pq.read_table(p, columns=cols))
        tabs.append(_CACHE["table"].slice(a, b - a))
    return pa.concat_tables(tabs).to_pydict() if tabs else {c: [] for c in cols}


def load_titles(clean_dir):
    t = pq.read_table(clean_dir, columns=["doc_id", "title"])
    return dict(zip(t.column("doc_id").to_pylist(), t.column("title").to_pylist()))


def make_inputs(block, titles):
    return [((titles.get(d) or "").strip() + "\n" + x) for d, x in zip(block["doc_id"], block["text"])]


def write_shard(out, ids, dense, sparse, vocab, meta):
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "dense.npy", dense)
    indptr = np.zeros(len(sparse) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(t) for t, _ in sparse])
    indices = np.concatenate([t for t, _ in sparse]) if sparse else np.zeros(0, np.int32)
    data = np.concatenate([v for _, v in sparse]) if sparse else np.zeros(0, np.float32)
    save_sparse(tmp / "sparse.npz", indptr, indices, data, vocab)
    pq.write_table(pa.table({"chunk_id": ids}), tmp / "chunk_ids.parquet")
    (tmp / "_DONE.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if out.exists():                                          # thư mục dở dang của lần chạy trước (không có _DONE)
        import shutil
        shutil.rmtree(out)
    tmp.rename(out)


def run(enc, chunk_dir, clean_dir, out_root, ec, shard="0/1", force=False, log=print):
    parts = chunk_parts(chunk_dir)
    total = sum(n for _, n in parts)
    n_blocks = (total + ec["shard_size"] - 1) // ec["shard_size"]
    si, sn = (int(x) for x in shard.split("/"))
    mine = [k for k in range(n_blocks) if k % sn == si]
    todo = [k for k in mine if force or not (Path(out_root) / f"shard-{k:04d}" / "_DONE.json").exists()]
    log(f"{total:,} đoạn → {n_blocks} khối; shard {shard}: {len(mine)} khối, còn {len(todo)}")
    titles = load_titles(clean_dir) if todo else {}
    for k in todo:
        t0 = time.time()
        b = read_block(parts, k, ec["shard_size"])
        dense, sparse, trunc = enc.encode(make_inputs(b, titles), ec["batch_tokens"], ec["max_batch"],
                                          ec["sparse_min_weight"])
        sec = time.time() - t0
        meta = {"block": k, "n": len(b["chunk_id"]), "seconds": round(sec, 1), "truncated": trunc,
                "nnz": int(sum(len(t) for t, _ in sparse)), "model": ec["model"], "max_length": ec["max_length"],
                "fp16": enc.half, "chunk_dir": str(chunk_dir)}
        write_shard(Path(out_root) / f"shard-{k:04d}", b["chunk_id"], dense, sparse, enc.vocab, meta)
        log(f"  khối {k}: {meta['n']:,} đoạn, {sec:.0f}s ({meta['n'] / max(sec, 1e-9):.0f} đoạn/s), "
            f"bị cắt {trunc}, nnz/đoạn {meta['nnz'] / max(1, meta['n']):.0f}")


def bench(enc, chunk_dir, clean_dir, ec, n, est_chunks=None, log=print):
    """Đo trên n đoạn đầu (sau 1 batch khởi động) → tốc độ và ước lượng thời gian, dung lượng cho est_chunks đoạn."""
    parts = chunk_parts(chunk_dir)
    b = read_block(parts, 0, n)
    texts = make_inputs(b, load_titles(clean_dir))
    enc.encode(texts[:64], ec["batch_tokens"], ec["max_batch"])          # khởi động (cuDNN, cấp phát bộ nhớ)
    torch = enc.torch
    if enc.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    dense, sparse, trunc = enc.encode(texts, ec["batch_tokens"], ec["max_batch"], ec["sparse_min_weight"])
    if enc.device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    nnz = sum(len(t) for t, _ in sparse) / len(texts)
    rate = len(texts) / sec
    res = {"device": str(enc.device), "fp16": enc.half, "n": len(texts), "seconds": round(sec, 1),
           "chunks_per_s": round(rate, 1), "truncated": trunc, "nnz_per_chunk": round(nnz, 1),
           "dense_bytes_per_chunk": dense.shape[1] * 2, "sparse_bytes_per_chunk": round(nnz * 6, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if enc.device.type == "cuda" else None}
    if est_chunks:
        res["est_chunks"] = est_chunks
        res["est_gpu_hours_1gpu"] = round(est_chunks / rate / 3600, 1)
        res["est_storage_gb"] = round(est_chunks * (res["dense_bytes_per_chunk"] + res["sparse_bytes_per_chunk"]) / 1e9, 1)
    log(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--strategy", default=None, help="mặc định chunk.strategy")
    ap.add_argument("--input", default=None, help="thư mục chunks/<strategy> (mặc định <work>/chunks/<strategy>)")
    ap.add_argument("--clean", default=None, help="thư mục corpus_clean (lấy tiêu đề; mặc định <work>/corpus_clean)")
    ap.add_argument("--out", default=None, help="mặc định <work>/emb/<strategy>")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bench", type=int, default=0, help="chỉ đo tốc độ trên N đoạn")
    ap.add_argument("--est-chunks", type=int, default=None, help="tổng số đoạn toàn kho (để ước lượng khi --bench)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ec = cfg["encode"]
    strategy = a.strategy or cfg["chunk"]["strategy"]
    chunk_dir = Path(a.input) if a.input else cfg.work_dir / cfg["chunk"]["out_dir"] / strategy
    clean_dir = Path(a.clean) if a.clean else cfg.work_dir / cfg["clean"]["out_dir"]
    out_root = Path(a.out) if a.out else cfg.work_dir / ec["out_dir"] / strategy
    enc = M3Encoder.load(ec["model"], a.device, ec["fp16"], ec["max_length"])
    if a.bench:
        res = bench(enc, chunk_dir, clean_dir, ec, a.bench, a.est_chunks)
        d = cfg.work_dir / "results"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"t3_bench_{a.device.replace(':', '')}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    run(enc, chunk_dir, clean_dir, out_root, ec, a.shard, a.force)


if __name__ == "__main__":
    main()
