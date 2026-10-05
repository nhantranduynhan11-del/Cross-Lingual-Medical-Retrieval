"""Kiểm thử T3 bằng model XLM-R rất nhỏ, trọng số ngẫu nhiên (không tải BGE-M3), dùng tokenizer BGE-M3 thật."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from mir import encode  # noqa: E402

try:
    from huggingface_hub import hf_hub_download
    TOKJSON = hf_hub_download("BAAI/bge-m3", "tokenizer.json")
except Exception:
    TOKJSON = None
pytestmark = pytest.mark.skipif(TOKJSON is None, reason="chưa có tokenizer BGE-M3")

EC = {"model": "tiny-test", "max_length": 64, "fp16": False, "batch_tokens": 256, "max_batch": 8,
      "shard_size": 7, "sparse_min_weight": 0.0, "out_dir": "emb"}
TEXTS = ["Tiêu đề\nBệnh tăng huyết áp cần giảm muối, tập thể dục.", "标题\n高血压患者应该少吃盐，多运动，按时服药。",
         "t\n" + "dài " * 200, "x\nngắn", "y\n中医 中医 中医 治疗 治疗"]


@pytest.fixture(scope="module")
def enc():
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKJSON, bos_token="<s>", eos_token="</s>", cls_token="<s>",
                                  sep_token="</s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    torch.manual_seed(0)
    cfg = XLMRobertaConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=80, pad_token_id=tok.pad_token_id)
    sl = torch.nn.Linear(16, 1)
    return encode.M3Encoder(XLMRobertaModel(cfg), tok, sl, "cpu", False, EC["max_length"])


def reference(enc, text):
    """Cách FlagEmbedding làm: từng đoạn một, vòng lặp theo token."""
    ids = enc.tok(text, truncation=True, max_length=enc.max_length)["input_ids"]
    with torch.inference_mode():
        h = enc.model(input_ids=torch.tensor([ids])).last_hidden_state
        d = torch.nn.functional.normalize(h[:, 0], dim=-1)[0].numpy()
        w = torch.relu(enc.sparse_linear(h)).squeeze(-1)[0].numpy()
    lw = {}
    for t, x in zip(ids, w):
        if t not in enc.unused and x > 0 and x > lw.get(t, 0):
            lw[t] = float(x)
    return d, lw


def test_encode_matches_reference_and_batching(enc):
    d1, s1, tr1 = enc.encode(TEXTS, batch_tokens=10_000, max_batch=64)
    d2, s2, _ = enc.encode(TEXTS, batch_tokens=64, max_batch=2)          # chia batch khác → kết quả như nhau
    assert tr1 == 1                                                        # đoạn "dài " * 200 bị cắt ở 64 token
    for i, t in enumerate(TEXTS):
        rd, rl = reference(enc, t)
        assert np.allclose(d1[i].astype(np.float32), rd, atol=2e-3) and np.allclose(d1[i], d2[i], atol=2e-3)
        assert abs(np.linalg.norm(d1[i].astype(np.float32)) - 1) < 1e-2
        ids, ws = s1[i]
        assert list(ids) == sorted(rl) and np.allclose(ws, [rl[k] for k in sorted(rl)], atol=1e-4)
        assert not (set(ids.tolist()) & enc.unused)
        assert np.array_equal(ids, s2[i][0]) and np.allclose(ws, s2[i][1], atol=1e-4)


def _write_inputs(tmp_path):
    clean, chunks = tmp_path / "clean", tmp_path / "chunks"
    clean.mkdir()
    chunks.mkdir()
    pq.write_table(pa.table({"doc_id": ["1", "2"], "host": ["a", "b"], "lang": ["vi", "zh"],
                             "title": ["Tiêu đề", "标题"], "text": ["x", "y"]}), clean / "part-00000.parquet")
    rows = [{"chunk_id": f"{d}_{k}", "doc_id": d, "text": TEXTS[k % len(TEXTS)]} for d in "12" for k in range(5)]
    pq.write_table(pa.Table.from_pylist(rows[:6]), chunks / "part-00000.parquet")      # khối 0 vắt qua 2 part
    pq.write_table(pa.Table.from_pylist(rows[6:]), chunks / "part-00001.parquet")
    return clean, chunks, [r["chunk_id"] for r in rows]


def test_run_shards_resume(enc, tmp_path):
    clean, chunks, ids = _write_inputs(tmp_path)
    out = tmp_path / "emb"
    assert encode.block_rows(encode.chunk_parts(chunks), 0, 7)[-1][2] == 1           # 6 dòng part 0 + 1 dòng part 1
    encode.run(enc, chunks, clean, out, EC, shard="1/2")                               # chỉ khối 1
    assert not (out / "shard-0000").exists() and (out / "shard-0001" / "_DONE.json").exists()
    (out / "shard-0000").mkdir()                                                       # thư mục dở dang (không _DONE)
    encode.run(enc, chunks, clean, out, EC, shard="0/2")
    got = []
    for k in (0, 1):
        d = out / f"shard-{k:04d}"
        cid = pq.read_table(d / "chunk_ids.parquet").column("chunk_id").to_pylist()
        dense = np.load(d / "dense.npy")
        sp = encode.load_sparse(d / "sparse.npz")
        assert dense.dtype == np.float16 and dense.shape == (len(cid), 16) and sp.shape == (len(cid), len(enc.tok))
        assert json.loads((d / "_DONE.json").read_text(encoding="utf-8"))["n"] == len(cid)
        got += cid
    assert got == ids
    m = (out / "shard-0001" / "dense.npy").stat().st_mtime_ns
    encode.run(enc, chunks, clean, out, EC, shard="0/1")                               # đã xong → bỏ qua
    assert (out / "shard-0001" / "dense.npy").stat().st_mtime_ns == m
    meta = encode.done_meta(out / "shard-0000")
    assert meta["chunks_total"] == 10 and meta["shard_size"] == 7
    assert meta["chunks_fp"] == encode.done_meta(out / "shard-0001")["chunks_fp"]
    # đầu vào có tiêu đề: đoạn của doc 1 mã hoá "Tiêu đề\n" + text
    sp0 = encode.load_sparse(out / "shard-0000" / "sparse.npz")
    _, rl = reference(enc, "Tiêu đề\n" + TEXTS[0])
    assert set(sp0[0].indices.tolist()) == set(rl)


def test_run_refuses_other_chunks_version(enc, tmp_path):
    clean, chunks, _ = _write_inputs(tmp_path)
    out = tmp_path / "emb"
    encode.run(enc, chunks, clean, out, EC, shard="0/2")
    pq.write_table(pa.Table.from_pylist([{"chunk_id": "3_0", "doc_id": "1", "text": "thêm"}]),
                   chunks / "part-00002.parquet")                                    # bản chunks khác
    with pytest.raises(SystemExit, match="bản chunks khác"):
        encode.run(enc, chunks, clean, out, EC, shard="1/2")
