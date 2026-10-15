"""Run file (kết quả truy hồi): parquet với cột qid, unit_id, doc_id, rank, score (+ <tên>.json cấu hình).

unit_id: đoạn cha "{doc_id}_p{k}" hoặc đoạn con "{doc_id}_p{k}_c{j}". rank bắt đầu từ 1; score lớn hơn = tốt hơn.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([("qid", pa.int64()), ("unit_id", pa.string()), ("doc_id", pa.int64()),
                    ("rank", pa.int32()), ("score", pa.float32())])


def doc_of(unit_id):
    return int(unit_id.split("_", 1)[0])


def parent_of(unit_id):
    """"123_p4_c2" → "123_p4"; đoạn cha giữ nguyên."""
    head, sep, tail = unit_id.rpartition("_c")
    return head if sep and tail.isdigit() else unit_id


def from_topk(qids, ids_per_q, scores_per_q):
    """[(qid, [unit_id], [score] giảm dần)] → DataFrame run."""
    q, u, r, s = [], [], [], []
    for qid, ids, sc in zip(qids, ids_per_q, scores_per_q):
        n = len(ids)
        q += [int(qid)] * n
        u += list(ids)
        r += list(range(1, n + 1))
        s += [float(x) for x in sc]
    df = pd.DataFrame({"qid": q, "unit_id": u, "rank": r, "score": s})
    df["doc_id"] = df["unit_id"].map(doc_of)
    return df[["qid", "unit_id", "doc_id", "rank", "score"]]


def rerank_by_score(df, depth=None):
    """Sắp lại theo score giảm dần trong mỗi qid, đánh rank lại từ 1 (giữ thứ tự cũ khi bằng điểm)."""
    df = df.sort_values(["qid", "score", "rank"], ascending=[True, False, True], kind="stable").copy()
    df["rank"] = df.groupby("qid").cumcount() + 1
    if depth:
        df = df[df["rank"] <= depth]
    return df.reset_index(drop=True)


def to_parents(df):
    """Run đoạn con → run đoạn cha: mỗi đoạn cha lấy thứ hạng (và điểm) của đoạn con tốt nhất."""
    df = df.copy()
    df["unit_id"] = df["unit_id"].map(parent_of)
    df = df.sort_values(["qid", "rank"]).drop_duplicates(["qid", "unit_id"], keep="first")
    df["rank"] = df.groupby("qid").cumcount() + 1
    return df.reset_index(drop=True)


def write(df, path, meta=None, force=True):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} đã có")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.astype({"qid": np.int64, "doc_id": np.int64, "rank": np.int32, "score": np.float32})
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")    # ghi tạm rồi đổi tên: 2 tiến trình cùng gộp không hỏng file
    pq.write_table(pa.Table.from_pandas(df[["qid", "unit_id", "doc_id", "rank", "score"]], schema=SCHEMA,
                                        preserve_index=False), tmp)
    os.replace(tmp, path)
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")


def read(path, depth=None, qids=None):
    import pyarrow.dataset as ds
    f = None
    if depth:
        f = ds.field("rank") <= depth
    if qids is not None:
        g = ds.field("qid").isin(list(qids))
        f = g if f is None else (f & g)
    return ds.dataset(str(path)).to_table(filter=f).to_pandas()


def doc_ranking(df):
    """Run đoạn → thứ hạng tài liệu (điểm tài liệu = đoạn tốt nhất). Thêm cột drank."""
    d = df.sort_values(["qid", "rank"]).drop_duplicates(["qid", "doc_id"], keep="first").copy()
    d["drank"] = d.groupby("qid").cumcount() + 1
    return d


def topk_sets(df, k, level="doc"):
    """{qid: tập doc_id (level='doc', theo thứ hạng tài liệu) hoặc unit_id (level='unit') trong top-k}."""
    if level == "doc":
        d = doc_ranking(df)
        d = d[d["drank"] <= k]
        return {q: set(g["doc_id"]) for q, g in d.groupby("qid")}
    d = df[df["rank"] <= k]
    return {q: set(g["unit_id"]) for q, g in d.groupby("qid")}


def topk_lists(df, k, level="doc"):
    if level == "doc":
        d = doc_ranking(df)
        d = d[d["drank"] <= k].sort_values(["qid", "drank"])
        return {q: list(g["doc_id"]) for q, g in d.groupby("qid")}
    d = df[df["rank"] <= k].sort_values(["qid", "rank"])
    return {q: list(g["unit_id"]) for q, g in d.groupby("qid")}
