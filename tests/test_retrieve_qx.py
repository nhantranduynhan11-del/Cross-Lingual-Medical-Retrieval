"""Test truy hồi có rewrite / thuật ngữ (retrieve qx, fuse --linear) trên bộ dữ liệu giả lập nhỏ — không cần GPU / mô hình.

Dựng trong thư mục tạm: chunks (đoạn cha), chỉ mục BM25 thật, khối emb (dense + LSR ngẫu nhiên) + FAISS Flat, câu hỏi
qtrans, qemb câu gốc / rewrite / rewrite + thuật ngữ, cache query_expand → build.

  python -m unittest tests.test_retrieve_qx -v
"""
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirret import bm25, config, encode, expand, index, retrieve, runs  # noqa: E402
from mirret import query_expand as qx  # noqa: E402
from mirret.kaggle_nb import write_config  # noqa: E402

LANGS = ("vi", "en", "zh")
DOCS = {
    "vi": ["đái tháo đường type 2 dùng metformin", "tăng huyết áp và thuốc ức chế men chuyển", "viêm dạ dày do vi khuẩn",
           "sốt xuất huyết ở trẻ em", "đột quỵ não và phục hồi chức năng", "viêm phổi do vi khuẩn ở người già",
           "rụng tóc sau sinh khi cho con bú", "đau đầu migraine và cách điều trị"],
    "en": ["type 2 diabetes treated with metformin", "hypertension and ACE inhibitors", "gastritis caused by bacteria",
           "dengue fever in children", "stroke and rehabilitation", "bacterial pneumonia in the elderly",
           "postpartum hair loss while breastfeeding", "migraine headache treatment"],
    "zh": ["二型糖尿病使用二甲双胍", "高血压和血管紧张素转换酶抑制剂", "细菌引起的胃炎", "儿童登革热", "脑卒中和康复训练",
           "老年人细菌性肺炎", "哺乳期产后脱发", "偏头痛的治疗"],
}
QUERIES = {"vi": ["thuốc tiểu đường", "đau đầu", "sốt ở trẻ"], "en": ["diabetes drug", "headache", "fever child"],
           "zh": ["糖尿病药物", "头痛", "儿童发烧"]}
DIM, VOCAB = 8, 40
OLD_REF = "19f44cf7b94d191d6090db0a2b737d1665fc6ef8"      # retrieve.py trước khi thêm `qx` (B1/B2, nhánh b-q2d)


def old_retrieve():
    """retrieve.py ở OLD_REF (trước thay đổi) để so kết quả `run`."""
    src = subprocess.run(["git", "show", f"{OLD_REF}:mirret/retrieve.py"], cwd=ROOT, capture_output=True,
                         check=True).stdout.decode("utf-8")
    m = types.ModuleType("mirret._retrieve_old")
    m.__package__ = "mirret"
    exec(compile(src, f"{OLD_REF}:mirret/retrieve.py", "exec"), m.__dict__)
    return m


def build_env(tmp, overrides=None):
    tmp = Path(tmp)
    rng = np.random.default_rng(0)
    write_config(ROOT / "configs" / "retrieval.yaml", tmp / "cfg.yaml",
                 {"paths.work_dir": str(tmp / "work"), "paths.data_dir": str(tmp), "retrieve.levels": ["parents"],
                  "retrieve.depth": 50, "fuse.depth": 50, **(overrides or {})})
    cfg = config.load(tmp / "cfg.yaml")
    rows = [(f"{100 * k + i}_p0", lg, "", t) for k, lg in enumerate(LANGS) for i, t in enumerate(DOCS[lg])]
    cdir = cfg.path("chunks") / "parents"
    cdir.mkdir(parents=True)
    pq.write_table(pa.table({"unit_id": [r[0] for r in rows], "lang": [r[1] for r in rows], "ctx": [r[2] for r in rows],
                             "text": [r[3] for r in rows], "h": [f"h{i}" for i in range(len(rows))]}),
                   cdir / "part-00000.parquet")
    for lg in LANGS:
        d = index.bm25_dir(cfg, "parents", lg)
        bm25.tokenize_parts(cdir, d, lg, workers=1, log=lambda *_: None)
        bm25.build(d, log=lambda *_: None)
    dense = rng.normal(size=(len(rows), DIM)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    sparse = [(np.sort(rng.choice(VOCAB, 5, replace=False)).astype(np.int32), rng.random(5).astype(np.float32))
              for _ in rows]
    encode.write_block(cfg.path("emb") / "parents" / "b0000", [r[0] for r in rows], [f"h{i}" for i in range(len(rows))],
                       dense, sparse, VOCAB, {"level": "parents"})
    index.build_dense(cfg, "parents", log=lambda *_: None)
    qt = cfg.path("qtrans")
    qt.mkdir(parents=True)
    for lg in LANGS:
        expand.write_queries(qt / f"queries_{lg}.parquet", list(enumerate(QUERIES[lg], 1)), lg)

    def qemb(name, seed):
        r = np.random.default_rng(seed)
        for lg in LANGS:
            n = len(QUERIES[lg])
            d = r.normal(size=(n, DIM)).astype(np.float32)
            sp = [(np.sort(r.choice(VOCAB, 4, replace=False)).astype(np.int32), r.random(4).astype(np.float32))
                  for _ in range(n)]
            p = retrieve.qemb_path(cfg, lg, name)
            p.parent.mkdir(parents=True, exist_ok=True)
            indptr = np.concatenate([[0], np.cumsum([4] * n)]).astype(np.int64)
            np.savez(p, qids=np.arange(1, n + 1), dense=d, indptr=indptr, indices=np.concatenate([t for t, _ in sp]),
                     data=np.concatenate([w for _, w in sp]), truncated=np.zeros(n, bool))
    qemb(None, 1)
    qemb("rw", 2)
    qemb("rwt", 3)
    out = expand.records_dir(cfg, qx.MODE, "qwen25_14b") / "records.jsonl"
    out.parent.mkdir(parents=True)
    extra = {"vi": ["đái tháo đường", "metformin"], "en": ["diabetes mellitus", "metformin"], "zh": ["二甲双胍"]}
    with out.open("w", encoding="utf-8") as f:
        for lg in LANGS:
            for q, t in enumerate(QUERIES[lg], 1):
                raw = json.dumps({"rewrite": t + (" ?" if lg != "zh" else "？"), "terms": extra[lg] if q == 1 else [],
                                  "candidates": []}, ensure_ascii=False)
                f.write(json.dumps(qx.to_record(q, lg, t, raw), ensure_ascii=False) + "\n")
    qx.build(cfg, log=lambda *_: None)
    return cfg


def read_run(cfg, name):
    return runs.read(cfg.path("runs") / f"{name}.parquet").sort_values(["qid", "rank"]).reset_index(drop=True)


class FlagsOff(unittest.TestCase):
    def test_run_identical_to_head_and_qx_noop(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            new, old = build_env(t1), build_env(t2)
            retrieve.run_all(new, list(LANGS), ["dense", "lsr", "bm25"], log=lambda *_: None)
            old_retrieve().run_all(old, list(LANGS), ["dense", "lsr", "bm25"], log=lambda *_: None)
            names = sorted(p.stem for p in new.path("runs").glob("*.parquet"))
            self.assertEqual(names, sorted(p.stem for p in old.path("runs").glob("*.parquet")))
            self.assertEqual(len(names), 18)                         # 9 nhánh × (nhánh + @parents)
            for n in names:
                pd.testing.assert_frame_equal(read_run(new, n), read_run(old, n))
            msgs = []
            retrieve.run_qx(new, list(LANGS), ["dense", "lsr", "bm25"], log=msgs.append)
            self.assertIn("mọi cờ đang tắt", msgs[0])
            self.assertFalse(list(new.path("runs").glob("qx_*")))


class FlagsOn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # kho 8 đoạn / ngôn ngữ: mọi từ có df/n ≥ 1/8 > 0.05 → nới ngưỡng để thuật ngữ không bị lọc hết
        cls.cfg = build_env(cls.tmp.name, {"qexpand.use_rewrite_dense": True, "qexpand.use_rewrite_sparse": True,
                                           "qexpand.use_terms": True, "qexpand.max_df_ratio": 0.5})
        retrieve.run_all(cls.cfg, list(LANGS), ["dense", "lsr", "bm25"], log=lambda *_: None)
        retrieve.run_qx(cls.cfg, list(LANGS), ["dense", "lsr", "bm25"], log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs(self):
        for lg in LANGS:
            for me in ("dense", "lsr", "bm25"):
                for suf in ("", "@parents"):
                    df = read_run(self.cfg, f"qx_{lg}_{me}{suf}")
                    base = set(read_run(self.cfg, f"{lg}_{me}{suf}")["qid"])
                    # dense/LSR: đủ câu; BM25: ít nhất các câu baseline tìm được (câu không khớp từ nào thì không có kết quả)
                    self.assertTrue(set(df["qid"]) >= base and (me == "bm25" or set(df["qid"]) == {1, 2, 3}), (lg, me))
                    self.assertTrue((df.groupby("qid")["rank"].min() == 1).all())
        meta = json.loads((self.cfg.path("runs") / "qx_vi_lsr@parents.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["variants"], [None, "rw", "rwt"])

    def test_dense_is_fusion_of_variants(self):
        base = runs.read(self.cfg.path("runs") / "vi_dense@parents.parquet")
        from mirret.index import dense_search
        _, d, _ = encode.load_qemb(retrieve.qemb_path(self.cfg, "vi", "rw"))
        res = dense_search(self.cfg, "parents", d, 50)
        rw = runs.from_topk([1, 2, 3], [r[0] for r in res], [r[1] for r in res])
        want = qx.fuse_variants([base, rw], "max", depth=50).sort_values(["qid", "rank"]).reset_index(drop=True)
        got = read_run(self.cfg, "qx_vi_dense@parents")
        self.assertEqual(list(got["unit_id"]), list(want["unit_id"]))

    def test_bm25_terms_change_weights(self):
        base = read_run(self.cfg, "vi_bm25@parents")
        got = read_run(self.cfg, "qx_vi_bm25@parents")
        b1, g1 = base[base.qid == 1], got[got.qid == 1]
        # câu 1 "thuốc tiểu đường": đoạn "đái tháo đường type 2 dùng metformin" chỉ khớp nhờ thuật ngữ mở rộng
        self.assertNotIn("0_p0", set(b1["unit_id"]))
        self.assertIn("0_p0", set(g1["unit_id"]))
        # câu 2, 3 không có thuật ngữ → chỉ thêm rewrite (= câu gốc + "?") → thứ tự giữ nguyên, điểm gấp đôi
        for q in (2, 3):
            b, g = base[base.qid == q], got[got.qid == q]
            self.assertEqual(list(b["unit_id"]), list(g["unit_id"]))
            np.testing.assert_allclose(2 * b["score"].to_numpy(), g["score"].to_numpy(), rtol=1e-5)

    def test_bm25_zero_weights_equals_baseline(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = build_env(t, {"qexpand.use_terms": True, "qexpand.w_terms": 0.0, "qexpand.w_rewrite": 0.0})
            retrieve.run_all(cfg, ["vi"], ["bm25"], log=lambda *_: None)
            retrieve.run_qx(cfg, ["vi"], ["dense", "lsr", "bm25"], log=lambda *_: None)
            a, b = read_run(cfg, "vi_bm25"), read_run(cfg, "qx_vi_bm25")
            self.assertEqual(list(a["unit_id"]), list(b["unit_id"]))
            np.testing.assert_allclose(a["score"], b["score"], rtol=1e-6)
            self.assertFalse((cfg.path("runs") / "qx_vi_dense.parquet").exists())   # use_rewrite_dense tắt

    def test_fuse_linear_cli(self):
        items = ["qx_vi_dense", "vi_lsr", "qx_vi_bm25", "en_dense", "en_lsr*0.5", "en_bm25*0.5"]
        r = subprocess.run([sys.executable, "-m", "mirret.fuse", "--config", str(self.cfg["config_path"]),
                            "--runs", ",".join(items), "--linear", "0.7", "--name", "lin07qx"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        from mirret import fuse
        scored = [(n, w, fuse.branch_scores(self.cfg.path("runs"), n, 50)) for n, w in map(fuse.parse_item, items)]
        want = fuse.linear(scored, 0.7, 50).sort_values(["qid", "rank"]).reset_index(drop=True)
        got = read_run(self.cfg, "lin07qx")
        self.assertEqual(list(got["unit_id"]), list(want["unit_id"]))
        meta = json.loads((self.cfg.path("runs") / "lin07qx.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["fusion"], meta["alpha"]), ("linear", 0.7))
        bad = subprocess.run([sys.executable, "-m", "mirret.fuse", "--config", str(self.cfg["config_path"]),
                              "--runs", "khong_co,vi_lsr", "--linear", "0.7"], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
