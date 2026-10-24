"""Unit test cho mirret/query_expand.py (+ vòng sinh dùng chung expand.generate_loop). Không cần GPU / dữ liệu thật.

  python -m unittest tests.test_query_expand -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import bm25, expand, sparse_index  # noqa: E402
from mirret import query_expand as qx  # noqa: E402

DOCS_EN = [
    "Metformin is the first-line drug for type 2 diabetes mellitus.",
    "Hypertension treatment includes ACE inhibitors and lifestyle changes.",
    "Amoxicillin clavulanate (Augmentin) treats bacterial sinusitis.",
    "Diabetes patients should monitor blood glucose and HbA1c regularly.",
    "The patient has a fever and cough; pneumonia is suspected.",
    "Children with fever should drink plenty of fluids.",
    "Gastritis causes stomach pain; avoid spicy food.",
    "Helicobacter pylori infection is a common cause of gastritis.",
] * 3                                                        # 24 đơn vị: "the", "and" có df cao


def small_index(tmp, docs, lang="en"):
    c, i = Path(tmp) / "chunks", Path(tmp) / "index"
    c.mkdir()
    pq.write_table(pa.table({"unit_id": [f"{k}_p0" for k in range(len(docs))], "lang": [lang] * len(docs),
                             "ctx": [""] * len(docs), "text": docs}), c / "part-00000.parquet")
    bm25.tokenize_parts(c, i, lang, workers=1, log=lambda *_: None)
    bm25.build(i, log=lambda *_: None)
    return i


class ParseJson(unittest.TestCase):
    def test_valid(self):
        d = qx.parse_json('{"rewrite": "Điều trị viêm dạ dày?", "terms": ["viêm dạ dày", "gastritis"]}')
        self.assertEqual(d["rewrite"], "Điều trị viêm dạ dày?")
        self.assertEqual(d["terms"], ["viêm dạ dày", "gastritis"])

    def test_extra_text(self):
        raw = 'Đây là kết quả:\n```json\n{"rewrite": "Sốt ở trẻ 2 tuổi?", "terms": ["sốt"]}\n```\nHy vọng giúp ích {x}'
        self.assertEqual(qx.parse_json(raw), {"rewrite": "Sốt ở trẻ 2 tuổi?", "terms": ["sốt"], "candidates": None})

    def test_think_and_other_braces(self):
        raw = '<think>{"nháp": 1}</think> {"a": 1} {"rewrite": "Q?", "terms": []}'
        self.assertEqual(qx.parse_json(raw)["rewrite"], "Q?")

    def test_broken_regex_fallback(self):
        raw = '{"rewrite": "Đau đầu kéo dài?", "terms": ["đau đầu", "migraine", '      # cắt giữa chừng
        d = qx.parse_json(raw)
        self.assertEqual(d["rewrite"], "Đau đầu kéo dài?")
        self.assertEqual(d["terms"], ["đau đầu", "migraine"])

    def test_garbage(self):
        self.assertIsNone(qx.parse_json("Xin lỗi, tôi không thể trả lời."))
        self.assertIsNone(qx.parse_json(""))
        self.assertIsNone(qx.parse_json(None))

    def test_record_fallback(self):
        r = qx.to_record(7, "en", "Base question?", "no json here")
        self.assertFalse(r["ok"])
        self.assertEqual((r["rewrite"], r["terms"], r["why"]), ("Base question?", [], "JSON hỏng"))
        r = qx.to_record(7, "zh", "原问题", '{"rewrite": "What is it?", "terms": ["x"]}')
        self.assertEqual((r["ok"], r["why"], r["rewrite"]), (False, "sai ngôn ngữ", "原问题"))
        r = qx.to_record(7, "en", "Base?", '{"rewrite": " Fever in kids? ", "terms": "fever; pyrexia"}')
        self.assertTrue(r["ok"])
        self.assertEqual((r["rewrite"], r["terms"]), ("Fever in kids?", ["fever", "pyrexia"]))


class CleanTerms(unittest.TestCase):
    def test_clean(self):
        terms = ["  Tiểu đường ", "tiểu đường", "", "x" * 61, None, 5, "- đái tháo đường", "\"HbA1c\""] + \
                [f"t{k}" for k in range(10)]
        out = qx.clean_terms(terms)
        self.assertEqual(out[:4], ["Tiểu đường", "5", "đái tháo đường", "HbA1c"])
        self.assertEqual(len(out), 8)
        self.assertTrue(all(0 < len(t) <= 60 for t in out))

    def test_string_input(self):
        self.assertEqual(qx.clean_terms("a, b；b\nc"), ["a", "b", "c"])

    def test_split_pairs(self):
        self.assertEqual(qx.clean_terms(["脑卒中 ↔ 中风", "VA = adenoids", "a<->b", "脑卒中"]),
                         ["脑卒中", "中风", "VA", "adenoids", "a", "b"])


class V3(unittest.TestCase):
    """Bản rw3: candidates tách khỏi terms, lọc term sai chữ viết, thêm tên từ bảng thuật ngữ."""

    def test_parse_candidates(self):
        d = qx.parse_json('{"rewrite": "Q?", "terms": ["a"], "candidates": ["sởi", "rubella"]}')
        self.assertEqual(d["candidates"], ["sởi", "rubella"])
        self.assertEqual(qx.parse_json('{"rewrite": "Q?", "terms": ["a"]}')["candidates"], None)
        d = qx.parse_json('{"rewrite": "Q?", "terms": ["a", "b"], "candidates": ["c", ')      # hỏng
        self.assertEqual((d["terms"], d["candidates"]), (["a", "b"], ["c"]))

    def test_filter_terms(self):
        self.assertEqual(qx.filter_terms(["phì đại cơ咬合肌肉肥大", "hở间隙", "đái tháo đường"], "vi"),
                         ["đái tháo đường"])
        self.assertEqual(qx.filter_terms(["colic哭闹", "đau bụng", "colic"], "en"), ["colic"])
        self.assertEqual(qx.filter_terms(["ALT", "胃炎", "37.5", "HbA1c"], "zh"), ["ALT", "胃炎", "HbA1c"])

    def test_drop_artifacts(self):
        ts = ["hở van tim", "viêm phế quản", "Hở bím", "hởi"]
        self.assertEqual(qx.drop_artifacts(ts, "vi", "Ho kéo dài chữa thế nào?"), ["viêm phế quản", "hởi"])
        self.assertEqual(qx.drop_artifacts(ts, "vi", "Hở van tim 2 lá có nguy hiểm?"), ts)   # câu hỏi về hở van
        self.assertEqual(qx.drop_artifacts(["hở x"], "en", "q"), ["hở x"])
        r = qx.to_record(1, "vi", "Ho lâu ngày", '{"rewrite": "Ho kéo dài?", "terms": ["ho mạn tính", "hở van tim"], '
                                                 '"candidates": ["hở van hai lá", "viêm phế quản"]}')
        self.assertEqual((r["terms"], r["candidates"]), (["ho mạn tính"], ["viêm phế quản"]))

    def test_record_v3(self):
        raw = '{"rewrite": "Đau bụng?", "terms": ["viêm dạ dày", "胃炎", "x"], "candidates": ["a", "b", "c", "d"]}'
        r = qx.to_record(1, "vi", "Đau bụng là bệnh gì?", raw)
        self.assertTrue(r["ok"])
        self.assertEqual(r["terms"], ["viêm dạ dày", "x"])                  # term chữ Hán bị bỏ, rewrite vẫn giữ
        self.assertEqual(r["candidates"], ["a", "b", "c"])                  # tối đa 3
        self.assertEqual(qx.to_record(1, "vi", "B", "hỏng")["candidates"], [])

    def test_glossary_terms(self):
        from mirret import translate
        gl = translate.load_glossary("configs/med_glossary_vi.tsv")
        al = qx.glossary_aliases("configs/med_glossary_vi.tsv")
        self.assertEqual(len(al), len(gl))
        q = "Mẹ em bị đột quỵ và viêm amidan"
        vi = qx.glossary_terms(q, "vi", gl, al)
        self.assertIn("tai biến mạch máu não", vi)
        self.assertNotIn("đột quỵ", vi)                                       # đã có trong câu
        self.assertIn("viêm a-mi-đan", vi)
        self.assertEqual(set(qx.glossary_terms(q, "en", gl, al)) >= {"stroke", "tonsillitis"}, True)
        self.assertTrue({"脑卒中", "中风", "扁桃体炎"} <= set(qx.glossary_terms(q, "zh", gl, al)))
        en_va = qx.glossary_terms("Con bị viêm VA", "en", gl, al)
        self.assertIn("adenoiditis", en_va)
        self.assertFalse([t for t in en_va if "viêm" in t])                 # cụm tiếng Việt trong cột en bị lọc


class Prompt(unittest.TestCase):
    class Tok:
        def apply_chat_template(self, msgs, **kw):
            return msgs

    def test_example_and_rules(self):
        for lg in qx.LANGS:
            msgs = qx.chat(self.Tok(), lg, "CÂU HỎI THẬT", "\nGỢI Ý", "\nGỢI Ý VÍ DỤ")
            self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
            self.assertIn(qx.EXAMPLE[lg][0], msgs[1]["content"])
            self.assertIn("GỢI Ý VÍ DỤ", msgs[1]["content"])
            self.assertEqual(json.loads(msgs[2]["content"]), qx.EXAMPLE[lg][1])
            self.assertTrue(msgs[3]["content"].endswith("CÂU HỎI THẬT") and "GỢI Ý" in msgs[3]["content"])
        self.assertIn("không thêm tuổi", qx.INSTRUCT["vi"])
        self.assertIn("do not add age", qx.INSTRUCT["en"])
        self.assertIn("不要添加年龄", qx.INSTRUCT["zh"])

    def test_example_terms_follow_rules(self):
        for lg, (q, a) in qx.EXAMPLE.items():
            self.assertEqual(qx.clean_terms(a["terms"]), a["terms"])
            self.assertFalse([t for t in a["terms"] if t.casefold() in q.casefold()], lg)   # tên khác, không chép câu
            self.assertTrue(qx.to_record(1, lg, q, json.dumps(a, ensure_ascii=False))["ok"])


class WeightedBM25(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.idx = small_index(cls.tmp.name, DOCS_EN)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def enc(self, **kw):
        return qx.WeightedQueryEncoder(self.idx, **kw)

    def test_base_only_equals_query_encoder(self):
        qe = bm25.QueryEncoder(self.idx)
        e = self.enc()
        for q in DOCS_EN[:8] + ["fever fever and the diabetes"]:
            a, b = qe(q, "en"), e("en", q)
            self.assertTrue(np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1]))

    def test_accumulate(self):
        e = self.enc(w_base=1.0, w_rewrite=1.0, w_terms=0.5, max_df_ratio=None)
        c = e.weights("en", "diabetes fever", "diabetes treatment", ["diabetes mellitus", "fever"])[0]
        w = {t: c.get(e.qe.vocab[t], 0) for t in ("diabetes", "fever", "treatment", "mellitus")}
        self.assertEqual(w, {"diabetes": 1 + 1 + 0.5, "fever": 1 + 0.5, "treatment": 1.0, "mellitus": 0.5})

    def test_df_filter_terms_only(self):
        # "and": 2/8 đơn vị mẫu có → df/n = 0.25 > 0.05; "mellitus": 1/8 = 0.125 → cũng bị lọc với ngưỡng 0.05
        e = self.enc(max_df_ratio=0.2)
        c, _, dropped = e.weights("en", "fever and cough", None, ["and", "mellitus"])
        v = e.qe.vocab
        self.assertEqual(c[v["and"]], 1.0)                   # token của base giữ nguyên dù df cao
        self.assertEqual(c[v["mellitus"]], 0.5)              # df/n = 0.125 ≤ 0.2 → giữ
        self.assertIn("and", [t for t, _ in dropped])

    def test_weights_reach_scores(self):
        e = self.enc(max_df_ratio=None)
        q1 = e("en", "gastritis")
        q2 = e("en", "gastritis", "gastritis")               # rewrite trùng → trọng số 2 → điểm gấp đôi
        r1 = sparse_index.search(self.idx, [q1], 5)[0]
        r2 = sparse_index.search(self.idx, [q2], 5)[0]
        self.assertEqual(r1[0], r2[0])
        self.assertTrue(np.allclose(2 * np.array(r1[1]), r2[1], rtol=1e-5))

    def test_explain(self):
        tab, dropped = self.enc().explain("en", "fever", "fever in children", ["pyrexia", "the"])
        self.assertEqual(float(tab.loc[tab["token"] == "fever", "trước"].iloc[0]), 1.0)
        self.assertEqual(float(tab.loc[tab["token"] == "fever", "sau"].iloc[0]), 2.0)


class Variants(unittest.TestCase):
    def test_query_variants(self):
        self.assertEqual(qx.query_variants("Sốt cao?", "sốt cao"), ["Sốt cao?"])     # trùng (bỏ dấu câu, hoa thường)
        self.assertEqual(qx.query_variants("A dài", "B ngắn"), ["A dài", "B ngắn"])
        self.assertEqual(qx.query_variants("A", "B", ["x", "y"], with_terms=True), ["A", "B", "B ; x ; y"])
        self.assertEqual(qx.query_variants("A", "", ["x"], with_terms=True), ["A", "A ; x"])

    def run_df(self, rows):
        return pd.DataFrame(rows, columns=["qid", "unit_id", "doc_id", "rank", "score"])

    def test_fuse_max_and_mean(self):
        a = self.run_df([(1, "1_p0", 1, 1, 10.0), (1, "2_p0", 2, 2, 5.0), (1, "3_p0", 3, 3, 0.0)])
        b = self.run_df([(1, "3_p0", 3, 1, 0.9), (1, "4_p0", 4, 2, 0.5), (1, "1_p0", 1, 3, 0.1)])
        m = qx.fuse_variants([a, b], "max").set_index("unit_id")["score"]
        self.assertAlmostEqual(m["1_p0"], 1.0)
        self.assertAlmostEqual(m["3_p0"], 1.0)
        self.assertAlmostEqual(m["4_p0"], 0.5)
        self.assertAlmostEqual(m["2_p0"], 0.5)
        f = qx.fuse_variants([a, b], "max")
        self.assertEqual(list(f["rank"]), [1, 2, 3, 4])
        self.assertEqual(list(f.columns), ["qid", "unit_id", "doc_id", "rank", "score"])
        mean = qx.fuse_variants([a, b], "mean", weights=[3, 1]).set_index("unit_id")["score"]
        self.assertAlmostEqual(mean["1_p0"], 0.75 * 1.0 + 0.25 * 0.0)
        self.assertAlmostEqual(mean["4_p0"], 0.25 * 0.5)

    def test_fuse_single_is_identity_order(self):
        a = self.run_df([(1, "1_p0", 1, 1, 3.0), (1, "2_p0", 2, 2, 2.0), (2, "3_p0", 3, 1, 7.0)])
        f = qx.fuse_variants([a])
        self.assertEqual(list(f["unit_id"]), list(a["unit_id"]))


class GenerateLoop(unittest.TestCase):
    """Vòng sinh dùng chung với mô hình giả: ghi cache, lượt lỗi → ok=False, retry_sample chỉ khi bật."""

    def fake(self, outputs):
        import torch

        class Enc(dict):
            def to(self, _):
                return self

        class Tok:
            pad_token_id = 0

            def __call__(self, texts, return_tensors=None, padding=None):
                if isinstance(texts, str):
                    return {"input_ids": [0] * 5}
                return Enc(input_ids=torch.zeros((len(texts), 3), dtype=torch.long))

            def batch_decode(self, g, skip_special_tokens=True):
                return [outputs.pop(0) for _ in range(g.shape[0])]

        class LLM:
            device = "cpu"
            calls = []

            def generate(self, input_ids, **kw):
                LLM.calls.append(kw)
                return torch.zeros((input_ids.shape[0], 6), dtype=torch.long)
        return Tok(), LLM()

    def test_rw_loop(self):
        todo = [(1, "en", "Base one?"), (2, "en", "Base two?"), (3, "en", "Base three?")]
        tok, llm = self.fake(['{"rewrite": "One?", "terms": ["a"]}', "hỏng", 'x {"rewrite": "Three?", "terms": []}'])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "records.jsonl"
            n = expand.generate_loop(tok, llm, todo, out, lambda x: x[2],
                                     lambda x, raw, s, sec: qx.to_record(x[0], x[1], x[2], raw, "m", sec), 2, 64,
                                     retry_sample=False, log=lambda *_: None)
            recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(n, 2)
        self.assertEqual([r["ok"] for r in recs], [True, False, True])
        self.assertEqual(recs[1]["rewrite"], "Base two?")
        self.assertEqual(recs[1]["raw"], "hỏng")
        self.assertTrue(all(not c["do_sample"] for c in llm.calls))    # greedy, không lượt lấy mẫu
        self.assertTrue(all(c.get("repetition_penalty") == 1.05 for c in llm.calls))   # mặc định cũ (q2d/HyDE)

    def test_no_repetition_penalty(self):
        tok, llm = self.fake(['{"rewrite": "One?", "terms": []}'])
        with tempfile.TemporaryDirectory() as tmp:
            expand.generate_loop(tok, llm, [(1, "en", "q")], Path(tmp) / "r.jsonl", lambda x: x[2],
                                 lambda x, raw, s, sec: qx.to_record(x[0], x[1], x[2], raw), 4, 8,
                                 retry_sample=False, repetition_penalty=1.0, log=lambda *_: None)
        self.assertNotIn("repetition_penalty", llm.calls[-1])
        self.assertFalse(llm.calls[-1]["do_sample"])

    def test_expand_retry_sample(self):
        todo = [(1, "en", "q")]
        tok, llm = self.fake(["bad", "good"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.jsonl"
            expand.generate_loop(tok, llm, todo, out, lambda x: x[2],
                                 lambda x, raw, s, sec: {"ok": raw == "good", "sampled": s}, 4, 8,
                                 retry_sample=True, log=lambda *_: None)
            recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(recs, [{"ok": False, "sampled": False}, {"ok": True, "sampled": True}])


if __name__ == "__main__":
    unittest.main()
