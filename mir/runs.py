"""Run file (kết quả truy hồi): runs/<tên>.parquet với cột qid, chunk_id, doc_id, rank, score (+ <tên>.json cấu hình).

rank bắt đầu từ 1, score lớn hơn = tốt hơn. chunk_id dạng "{doc_id}_{k}" (đoạn) hoặc "{doc_id}_p{k}" (đoạn cha).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([("qid", pa.int64()), ("chunk_id", pa.string()), ("doc_id", pa.string()),
                    ("rank", pa.int32()), ("score", pa.float32())])


def doc_of(chunk_id):
    return chunk_id.rsplit("_", 1)[0]


def from_topk(qids, ids_per_q, scores_per_q):
    """Danh sách (qid, [chunk_id], [score] đã sắp giảm dần) → DataFrame run."""
    rows_q, rows_c, rows_r, rows_s = [], [], [], []
    for q, ids, sc in zip(qids, ids_per_q, scores_per_q):
        n = len(ids)
        rows_q += [int(q)] * n
        rows_c += list(ids)
        rows_r += list(range(1, n + 1))
        rows_s += [float(x) for x in sc]
    df = pd.DataFrame({"qid": rows_q, "chunk_id": rows_c, "rank": rows_r, "score": rows_s})
    df["doc_id"] = df["chunk_id"].map(doc_of)
    return df[["qid", "chunk_id", "doc_id", "rank", "score"]]


def rerank_by_score(df, depth=None):
    """Sắp lại theo score giảm dần trong mỗi qid, đánh rank lại từ 1 (giữ thứ tự cũ khi bằng điểm)."""
    df = df.sort_values(["qid", "score", "rank"], ascending=[True, False, True], kind="stable")
    df["rank"] = df.groupby("qid").cumcount() + 1
    if depth:
        df = df[df["rank"] <= depth]
    return df.reset_index(drop=True)


def write(df, path, meta=None, force=False):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} đã có (dùng --force để ghi đè)")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.astype({"qid": np.int64, "rank": np.int32, "score": np.float32})
    pq.write_table(pa.Table.from_pandas(df[["qid", "chunk_id", "doc_id", "rank", "score"]], schema=SCHEMA,
                                        preserve_index=False), path)
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")


def read(path):
    return pq.read_table(path).to_pandas()


def doc_ranking(df):
    """Run đoạn → thứ hạng tài liệu: điểm tài liệu = thứ hạng đoạn tốt nhất (rank nhỏ nhất)."""
    d = df.sort_values(["qid", "rank"]).drop_duplicates(["qid", "doc_id"], keep="first").copy()
    d["drank"] = d.groupby("qid").cumcount() + 1
    return d
