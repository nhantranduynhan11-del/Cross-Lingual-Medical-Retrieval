"""Lưu kết quả thành các file parquet "phần" (part), mỗi file ghi trọn vẹn rồi mới đổi tên → phiên Kaggle bị ngắt
giữa chừng cũng không làm hỏng file nào (chỉ mất phần đang đệm trong bộ nhớ, tối đa vài phút).

Thư mục kết quả của một lần chạy:
  pages/<phiên>-00000.parquet   mọi URL đã thử: trạng thái + nội dung (nếu có). Một id có thể có nhiều dòng
                                (thử lại, trích lại) → merge chọn dòng tốt nhất.
  raw/<phiên>-00000.parquet     HTML gọn của các trang tải được (để trích lại không cần mạng).
  runs/<phiên>.json             thống kê của phiên.
Tên phiên = "<shard>-<thời điểm>", vd "s0of4-20261012T101500".
"""
import glob
import os
import shutil
import time

import pyarrow as pa
import pyarrow.parquet as pq

PAGE_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()), ("host", pa.string()),
    ("status", pa.string()), ("lang", pa.string()), ("doc_type", pa.string()),
    ("title", pa.string()), ("text", pa.string()), ("description", pa.string()), ("author", pa.string()),
    ("published", pa.string()), ("modified", pa.string()), ("section", pa.string()), ("keywords", pa.string()),
    ("site_name", pa.string()), ("extra", pa.string()), ("extractor", pa.string()), ("xv", pa.int16()),
    ("n_chars", pa.int32()), ("text_hash", pa.string()), ("fetched_at", pa.string()),
    ("processed_at", pa.string()), ("session", pa.string()),
])
RAW_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()), ("host", pa.string()),
    ("charset", pa.string()), ("fetched_at", pa.string()), ("html", pa.string()),
])

# Trạng thái coi như XONG (không tự thử lại). Các trạng thái khác (err_*, http_5xx…) được thử lại ở phiên sau cho tới
# khi đủ --max-attempts lần; http_429 và blocked_* không tính vào số lần thử (thử lại mãi tới khi trang cho phép).
DONE = {"ok", "empty", "notice", "gone", "not_html", "http_400", "http_404", "http_410", "pm_missing", "err_too_big"}
# Thứ tự ưu tiên khi một id có nhiều dòng (số nhỏ = tốt hơn).
STATUS_RANK = {"ok": 0, "empty": 1, "notice": 2, "gone": 3, "not_html": 3, "http_404": 3, "http_410": 3,
               "http_400": 3, "pm_missing": 3, "err_too_big": 3}


def status_rank(s):
    return STATUS_RANK.get(s, 5)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PartWriter:
    """Đệm bản ghi trong bộ nhớ; ghi ra một file parquet mới khi đủ rows_per_part dòng hoặc quá max_age giây."""

    def __init__(self, folder, prefix, schema, rows_per_part=5000, max_age=600, compression="zstd", level=6):
        os.makedirs(folder, exist_ok=True)
        self.folder, self.prefix, self.schema = folder, prefix, schema
        self.rows_per_part, self.max_age = rows_per_part, max_age
        self.compression, self.level = compression, level
        self.buf, self.t_last, self.n_written = [], time.time(), 0
        existing = glob.glob(os.path.join(folder, f"{prefix}-*.parquet"))
        self.k = max([int(os.path.basename(f)[len(prefix) + 1:-8]) for f in existing], default=-1) + 1

    def add(self, row):
        self.buf.append(row)

    def due(self):
        return len(self.buf) >= self.rows_per_part or (self.buf and time.time() - self.t_last >= self.max_age)

    def take(self):
        """Lấy phần đệm hiện tại (để ghi ở luồng khác) → (rows, path) hoặc None."""
        if not self.buf:
            return None
        rows, self.buf, self.t_last = self.buf, [], time.time()
        path = os.path.join(self.folder, f"{self.prefix}-{self.k:05d}.parquet")
        self.k += 1
        return rows, path

    def write(self, job):
        if job is None:
            return
        rows, path = job
        tbl = pa.Table.from_pylist(rows, schema=self.schema)
        tmp = path + ".tmp"
        pq.write_table(tbl, tmp, compression=self.compression, compression_level=self.level)
        os.replace(tmp, path)
        self.n_written += len(rows)

    def flush(self):
        self.write(self.take())


def find_parts(dirs, sub):
    """Mọi file <dir>/**/<sub>/*.parquet (tìm đệ quy: thư mục input của Kaggle có thể lồng thêm cấp)."""
    out = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        out += glob.glob(os.path.join(d, "**", sub, "*.parquet"), recursive=True)
    return sorted(set(os.path.normpath(f) for f in out))


def read_state(files, keep_ids=None):
    """id → (đã xong?, số lần thử). Chỉ đọc 2 cột nên nhanh."""
    state = {}
    for f in files:
        try:
            t = pq.read_table(f, columns=["id", "status"])
        except Exception as e:                             # file hỏng (hiếm: chỉ khi bị ngắt lúc đang đổi tên)
            print(f"[CẢNH BÁO] bỏ qua {f}: {e}")
            continue
        for i, s in zip(t.column("id").to_pylist(), t.column("status").to_pylist()):
            if keep_ids is not None and i not in keep_ids:
                continue
            done, n = state.get(i, (False, 0))
            # bị giới hạn tốc độ / bị chặn là lỗi của phía mình gọi, không phải của trang → không tính là một lần thử
            state[i] = (done or s in DONE, n + (0 if s == "http_429" or s.startswith("blocked_") else 1))
    return state


def carry_over(resume_dirs, out_dir, tag, subs=("pages", "raw", "runs")):
    """Chép các file của phiên trước (cùng shard `tag`) từ thư mục input sang thư mục kết quả, để output của phiên
    cuối luôn chứa ĐỦ dữ liệu của shard (chỉ cần Add Input output mới nhất cho phiên sau)."""
    n = 0
    for sub in subs:
        pat = "*.json" if sub == "runs" else "*.parquet"
        for d in resume_dirs:
            if not d or not os.path.isdir(d) or os.path.abspath(d) == os.path.abspath(out_dir):
                continue
            for f in glob.glob(os.path.join(d, "**", sub, pat), recursive=True):
                name = os.path.basename(f)
                if not (name.startswith(tag + "-") or name.startswith("rx-" + tag + "-")):
                    continue
                dst = os.path.join(out_dir, sub, name)
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(f, dst)
                    n += 1
    return n
