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
