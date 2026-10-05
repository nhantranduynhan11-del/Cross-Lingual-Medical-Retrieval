"""Đọc nhanh một số dòng theo khoá từ thư mục parquet (corpus_clean, chunks, parents) — không nạp cả kho."""
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds


def lookup(dirpath, key, keys, columns, prefix="part-"):
    """→ {khoá: {cột: giá trị}} cho các dòng có `key` thuộc `keys`, trong các file <prefix>*.parquet."""
    files = sorted(str(p) for p in Path(dirpath).glob(prefix + "*.parquet"))
    if not files or not keys:
        return {}
    t = ds.dataset(files, format="parquet").to_table(columns=list({key, *columns}),
                                                     filter=pc.field(key).isin(list(set(keys))))
    d = t.to_pydict()
    return {k: {c: d[c][i] for c in columns} for i, k in enumerate(d[key])}


def is_parent_id(unit_id):
    """Id đoạn cha có dạng "{doc_id}_p{k}"; id đoạn "{doc_id}_{k}"."""
    return unit_id.rsplit("_", 1)[1].startswith("p")


def unit_rows(chunk_dir, ids, columns=("text",)):
    """Dòng của các đơn vị trong một run: đoạn (part-*.parquet) hoặc đoạn cha (parents-*.parquet, chiến lược
    parent_child), nhận theo dạng id. → {id: {cột: giá trị}}"""
    ids = list(ids)
    if any(is_parent_id(c) for c in ids[:50]):
        return lookup(chunk_dir, "parent_id", ids, columns, prefix="parents-")
    return lookup(chunk_dir, "chunk_id", ids, columns)
