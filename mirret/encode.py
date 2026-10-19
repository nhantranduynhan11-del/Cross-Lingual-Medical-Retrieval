"""Mã hoá BGE-M3 cho đoạn cha / đoạn con: dense (CLS chuẩn hoá L2) + lexical weights (LSR). ColBERT chỉ dùng khi
xếp hạng lại (mirret.rerank), không lưu cho cả kho.

M3Encoder lấy nguyên từ mir/encode.py (đã đối chiếu với FlagEmbedding BGEM3 bằng trọng số thật trên Kaggle):
  dense  = last_hidden[:, 0] chuẩn hoá L2
  sparse = relu(sparse_linear(last_hidden)), bỏ token đặc biệt, mỗi token id lấy giá trị lớn nhất
  colbert= normalize(colbert_linear(last_hidden[:, 1:]))
Đầu vào mỗi đơn vị: ngữ cảnh + "\\n" + đoạn (C), cắt ở encode.max_length (tính cả <s></s>).

Chia việc & mã hoá bổ sung:
  - Danh sách cần mã hoá = mọi đơn vị (theo thứ tự file part, rồi thứ tự dòng) TRỪ các cặp (unit_id, h) đã có trong
    các thư mục --skip (kết quả mã hoá trước đó). h = hash(ngữ cảnh + đoạn): đoạn đổi nội dung thì được mã hoá lại.
  - Danh sách chia thành khối encode.block đơn vị; khối k thuộc phần k % N. Mọi máy phải có CÙNG bản chunks và CÙNG
    --skip thì danh sách mới giống nhau: dấu vân tay (fp) in ra ở đầu và ghi trong tên khối "<fp>-<k>".
  - Khối xong: emb/<mức>/<fp>-<k>/{dense.npy (float16), sparse.npz (CSR float16), ids.parquet (unit_id, h), _DONE.json}.
    Chạy lại = bỏ qua khối đã có _DONE; --resume <thư mục>: chép khối đã xong từ output phiên trước.

  python -m mirret.encode plan  --level parents [--skip emb_cũ]
  python -m mirret.encode run   --level parents --shard 0/6 --device cuda:0 [--skip …] [--resume …]
  python -m mirret.encode bench --level parents --n 2000 --device cuda:0
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .sparse_index import save_csr


# ---------------------------------------------------------------- model (từ mir/encode.py)
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
        try:
            model = AutoModel.from_pretrained(name, attn_implementation="sdpa")
        except Exception:
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

    def batches(self, texts, batch_tokens, max_batch, max_length=None):
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

    def colbert(self, texts, max_length=None, batch_tokens=16384, max_batch=128):
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

    def encode(self, texts, batch_tokens=16384, max_batch=128, min_weight=0.0):
        """→ dense float16 [n, H]; sparse list[(token_ids int32 tăng dần, weights float32)]; số đơn vị bị cắt."""
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


def resolve_device(device):
    """cuda:* khi không có GPU (chạy thử trên máy) → cpu, kèm cảnh báo."""
    import torch
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[CẢNH BÁO] không có GPU → chạy trên CPU thay cho {device}")
        return "cpu"
    return device


def unit_input(ctx, text):
    return (ctx + "\n" + text) if ctx else text


# ---------------------------------------------------------------- khối emb
def done_blocks(dirs, level=None):
    """Mọi thư mục khối đã xong (_DONE.json) trong các thư mục (tìm đệ quy; lọc theo mức nếu có)."""
    out = []
    for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs or []):
        for f in Path(d).glob("**/_DONE.json"):
            if (f.parent / "ids.parquet").exists():
                m = json.loads(f.read_text(encoding="utf-8"))
                if level is None or m.get("level") == level:
                    out.append(f.parent)
    return sorted(set(out), key=lambda p: p.name)


def encoded_pairs(dirs, level):
    """{"unit_id:h"} đã mã hoá trong các thư mục emb."""
    s = set()
    for b in done_blocks(dirs, level):
        t = pq.read_table(b / "ids.parquet", columns=["unit_id", "h"])
        s.update(f"{u}:{h}" for u, h in zip(t.column("unit_id").to_pylist(), t.column("h").to_pylist()))
    return s


def plan(chunk_dir, skip=None):
    """→ (parts, rows_per_part [np.int32 chỉ số dòng cần mã hoá], tổng, fp)."""
    parts = sorted(Path(chunk_dir).glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"Không có chunks trong {chunk_dir}")
    hasher = hashlib.sha1()
    rows, total = [], 0
    for p in parts:
        t = pq.read_table(p, columns=["unit_id", "h"])
        keys = [f"{u}:{h}" for u, h in zip(t.column("unit_id").to_pylist(), t.column("h").to_pylist())]
        keep = np.array([i for i, k in enumerate(keys) if not skip or k not in skip], dtype=np.int32)
        for i in keep.tolist():
            hasher.update(keys[i].encode("utf-8") + b";")
        rows.append(keep)
        total += len(keep)
    return parts, rows, total, hasher.hexdigest()[:10]


_CACHE = {}


def read_block(parts, rows, k, size):
    """Khối k = vị trí [k*size, (k+1)*size) của chuỗi nối các dòng cần mã hoá → dict cột."""
    lo, hi, pos, tabs = k * size, (k + 1) * size, 0, []
    for p, r in zip(parts, rows):
        a, b = max(lo, pos), min(hi, pos + len(r))
        if a < b:
            if _CACHE.get("path") != p:
                _CACHE.clear()
                _CACHE.update(path=p, table=pq.read_table(p, columns=["unit_id", "h", "ctx", "text"]))
            tabs.append(_CACHE["table"].take(pa.array(r[a - pos:b - pos])))
        pos += len(r)
    if not tabs:
        return {c: [] for c in ("unit_id", "h", "ctx", "text")}
    return pa.concat_tables(tabs).to_pydict()


def write_block(out, ids, hs, dense, sparse, vocab, meta):
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "dense.npy", dense)
    indptr = np.zeros(len(sparse) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(t) for t, _ in sparse])
    indices = np.concatenate([t for t, _ in sparse]) if sparse else np.zeros(0, np.int32)
    data = np.concatenate([v for _, v in sparse]) if sparse else np.zeros(0, np.float32)
    save_csr(tmp / "sparse.npz", indptr, indices, data, vocab)
    pq.write_table(pa.table({"unit_id": pa.array(ids, pa.string()), "h": pa.array(hs, pa.string())}),
                   tmp / "ids.parquet")
    (tmp / "_DONE.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def run(enc, cfg, level, shard="0/1", skip=None, resume=None, log=print):
    ec = cfg["encode"]
    chunk_dir = cfg.path("chunks") / level
    out_root = cfg.path("emb") / level
    out_root.mkdir(parents=True, exist_ok=True)
    skipset = encoded_pairs(skip, level) if skip else None
    parts, rows, total, fp = plan(chunk_dir, skipset)
    size = ec["block"]
    n_blocks = -(-total // size)
    si, sn = (int(x) for x in shard.split("/"))
    mine = [k for k in range(n_blocks) if k % sn == si]
    log(f"[{level}] cần mã hoá {total:,} đơn vị (đã bỏ {len(skipset or ()):,} có sẵn) → {n_blocks} khối, "
        f"dấu vân tay {fp}; phần {shard}: {len(mine)} khối")
    prev = {b.name: b for b in done_blocks(resume, level)} if resume else {}
    for k in mine:
        name = f"{fp}-{k:04d}"
        out = out_root / name
        if (out / "_DONE.json").exists():
            continue
        if name in prev:
            shutil.copytree(prev[name], out)
            log(f"  chép khối đã xong {name} từ {prev[name]}")
            continue
        t0 = time.time()
        b = read_block(parts, rows, k, size)
        dense, sparse, trunc = enc.encode([unit_input(c, t) for c, t in zip(b["ctx"], b["text"])],
                                          ec["batch_tokens"], ec["max_batch"])
        sec = time.time() - t0
        meta = {"level": level, "fp": fp, "block": k, "n_blocks": n_blocks, "n": len(b["unit_id"]),
                "seconds": round(sec, 1), "truncated": trunc, "nnz": int(sum(len(t) for t, _ in sparse)),
                "model": ec["model"], "max_length": ec["max_length"], "fp16": enc.half}
        write_block(out, b["unit_id"], b["h"], dense, sparse, enc.vocab, meta)
        log(f"  khối {name}: {meta['n']:,} đơn vị, {sec:.0f}s ({meta['n'] / max(sec, 1e-9):.0f}/s), "
            f"bị cắt {trunc}, nnz/đơn vị {meta['nnz'] / max(1, meta['n']):.0f}")


def encode_texts(enc, ids, texts, path, batch_tokens=16384):
    """Câu hỏi → npz: qids, dense float32, sparse CSR (indptr, indices, data)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dense, sparse, trunc = enc.encode(texts, batch_tokens=batch_tokens)
    indptr = np.concatenate([[0], np.cumsum([len(t) for t, _ in sparse])]).astype(np.int64)
    np.savez(path, qids=np.array(ids, dtype=np.int64), dense=dense.astype(np.float32), indptr=indptr,
             indices=np.concatenate([t for t, _ in sparse] + [np.zeros(0, np.int32)]).astype(np.int32),
             data=np.concatenate([w for _, w in sparse] + [np.zeros(0, np.float32)]).astype(np.float32),
             truncated=np.array(trunc))
    return path


def load_qemb(path):
    z = np.load(path)
    sp = [(z["indices"][a:b], z["data"][a:b]) for a, b in zip(z["indptr"][:-1], z["indptr"][1:])]
    return [int(q) for q in z["qids"]], z["dense"], sp


def bench(enc, cfg, level, n=2000, log=print):
    ec = cfg["encode"]
    parts, rows, total, fp = plan(cfg.path("chunks") / level)
    b = read_block(parts, rows, 0, n)
    texts = [unit_input(c, t) for c, t in zip(b["ctx"], b["text"])]
    enc.encode(texts[:64], ec["batch_tokens"], ec["max_batch"])
    torch = enc.torch
    if enc.device.type == "cuda":
        torch.cuda.synchronize(enc.device)
    t0 = time.time()
    dense, sparse, trunc = enc.encode(texts, ec["batch_tokens"], ec["max_batch"])
    if enc.device.type == "cuda":
        torch.cuda.synchronize(enc.device)
    sec = time.time() - t0
    rate = len(texts) / sec
    nnz = sum(len(t) for t, _ in sparse) / len(texts)
    res = {"level": level, "n": len(texts), "per_s": round(rate, 1), "truncated": trunc,
           "total_units": total, "est_gpu_hours_1gpu": round(total / rate / 3600, 2),
           "est_storage_gb": round(total * (dense.shape[1] * 2 + nnz * 6) / 1e9, 1)}
    log(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "bench"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--level", default="parents", choices=["parents", "children"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--skip", default="", help="thư mục emb đã có (phẩy ngăn cách): chỉ mã hoá phần mới/đổi")
    ap.add_argument("--resume", default="", help="thư mục output phiên trước: chép khối đã xong")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    cfg = config.load(a.config)
    skip = [x for x in a.skip.split(",") if x]
    resume = [x for x in a.resume.split(",") if x]
    if a.cmd == "plan":
        _, _, total, fp = plan(cfg.path("chunks") / a.level, encoded_pairs(skip, a.level) if skip else None)
        print(json.dumps({"level": a.level, "units": total, "fp": fp,
                          "blocks": -(-total // cfg["encode"]["block"])}, indent=1))
        return
    ec = cfg["encode"]
    enc = M3Encoder.load(ec["model"], resolve_device(a.device), ec["fp16"], ec["max_length"])
    if a.cmd == "bench":
        bench(enc, cfg, a.level, a.n)
    else:
        run(enc, cfg, a.level, a.shard, skip, resume)


if __name__ == "__main__":
    main()
