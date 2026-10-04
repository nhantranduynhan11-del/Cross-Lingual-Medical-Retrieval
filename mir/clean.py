"""T1: làm sạch kho — loại dòng giao diện lặp lại theo nhóm tên miền, bỏ bài quá ngắn.

Hai bước, có điểm duyệt ở giữa:
  python -m mir.clean --config configs/baseline.yaml --stats
      → <work>/clean_lines.json (dòng bị loại theo nhóm) + results/t1_review.md (để duyệt)
  python -m mir.clean --config configs/baseline.yaml --apply
      → <work>/corpus_clean/part-*.parquet: doc_id, host, lang, title, text (dùng ĐÚNG clean_lines.json đã duyệt)

Làm sạch tập mẫu bằng danh sách dòng của toàn kho:
  python -m mir.clean --config configs/baseline.yaml --apply --input sample/corpus_raw.parquet --out sample/corpus_clean

Quy tắc:
- Nhóm = tên miền; các tên miền con khai báo trong clean.groups được gộp (vd các kênh tin 39.net).
- Dòng (so khớp nguyên dòng) xuất hiện ở >= repeat_line_ratio số bài của nhóm → bị loại, trừ khi là tiêu đề mục
  (ngắn, thường đứng trước một dòng nội dung, ít khi ở cuối bài) — xem clean.heading; force_drop/force_keep ghi đè.
- Nhóm có < min_group_docs bài: không lọc. Thống kê trên mẫu ngẫu nhiên tối đa max_stats_docs bài mỗi nhóm.
- Bài còn < min_chars ký tự sau khi làm sạch: bỏ. Bài trùng nội dung: KHÔNG gộp (đáp án có thể trỏ tới id bất kỳ), chỉ báo cáo.
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .crawlio import iter_docs

SCHEMA = pa.schema([("doc_id", pa.string()), ("host", pa.string()), ("lang", pa.string()),
                    ("title", pa.string()), ("text", pa.string())])


def group_of(host, groups):
    for g in groups or []:
        s = g["suffix"]
        if host not in set(g.get("exclude", [])) and (host == s or host.endswith("." + s)):
            return g["name"]
    return host


def read_input(cfg, path=None):
    """Sinh bản ghi {"id","host","lang","title","text"} từ out/ (mặc định) hoặc từ một file parquet (vd sample)."""
    if path:
        t = pq.read_table(path)
        cols = t.column_names
        for b in t.to_batches(10000):
            d = b.to_pydict()
            for k in range(b.num_rows):
                yield {c: d[c][k] for c in cols}
    else:
        yield from iter_docs(cfg.crawl_path)


# ---------------- bước 1: thống kê ----------------
def compute_lines(docs, cc, seed):
    """docs: iterable bản ghi. Trả về (lines: {nhóm: {"drop": [...], "keep_heading": [...]}}, stats: {nhóm: ...})."""
    rng = random.Random(seed)
    cap = cc["max_stats_docs"]
    res = {}                                       # nhóm -> [số bài đã thấy, mẫu (list các danh sách dòng)]
    for d in docs:
        g = group_of(d["host"], cc.get("groups"))
        seen, sample = res.setdefault(g, [0, []])
        lines = d["text"].split("\n")
        if len(sample) < cap:
            sample.append(lines)
        else:                                      # lấy mẫu hồ chứa: mỗi bài có xác suất như nhau
            j = rng.randrange(seen + 1)
            if j < cap:
                sample[j] = lines
        res[g][0] = seen + 1
    h = cc["heading"]
    out, stats = {}, {}
    for g, (n_total, sample) in sorted(res.items(), key=lambda x: -x[1][0]):
        n = len(sample)
        df = collections.Counter()
        for lines in sample:
            df.update(set(lines))
        rep = {l for l, c in df.items() if c >= cc["repeat_line_ratio"] * n}
        st = {"docs": n_total, "stats_docs": n, "filtered": n_total >= cc["min_group_docs"]}
        stats[g] = st
        if not st["filtered"]:
            continue
        # tiêu đề mục: xét vị trí và dòng ngay sau ở mỗi lần xuất hiện
        follow, at_end, at_start, occ = (collections.Counter() for _ in range(4))
        for lines in sample:
            L = len(lines)
            for k, l in enumerate(lines):
                if l in rep:
                    occ[l] += 1
                    nxt = lines[k + 1] if k + 1 < L else ""
                    if len(nxt) >= 20 and nxt not in rep:
                        follow[l] += 1
                    if k >= L - 2:
                        at_end[l] += 1
                    if k == 0:
                        at_start[l] += 1
        fd = set((cc.get("force_drop") or {}).get(g, []))
        fk = set((cc.get("force_keep") or {}).get(g, []))
        drop, keep = [], []
        for l in sorted(rep, key=lambda x: -df[x]):
            is_heading = (len(l) <= h["max_len"] and occ[l] and follow[l] / occ[l] >= h["min_followed_by_content"]
                          and at_end[l] / occ[l] <= h["max_at_end"] and at_start[l] / occ[l] <= h["max_at_start"])
            item = {"line": l, "df": df[l], "ratio": round(df[l] / n, 3),
                    "followed_by_content": round(follow[l] / max(1, occ[l]), 2),
                    "at_end": round(at_end[l] / max(1, occ[l]), 2), "at_start": round(at_start[l] / max(1, occ[l]), 2)}
            if l in fk or (is_heading and l not in fd):
                keep.append(item)
            else:
                drop.append(item)
        for l in fd - rep:                         # bỏ theo danh sách tay dù chưa tới ngưỡng
            drop.append({"line": l, "df": df[l], "ratio": round(df[l] / n, 3), "forced": True})
        out[g] = {"drop": drop, "keep_heading": keep}
    return out, stats


def review_md(lines, stats, cc, top=6):
    L = ["# T1 — Dòng lặp bị loại (để duyệt)", "",
         f"Ngưỡng: dòng ở >= {cc['repeat_line_ratio']:.0%} số bài của nhóm; nhóm cần >= {cc['min_group_docs']} bài; "
         f"thống kê trên tối đa {cc['max_stats_docs']} bài/nhóm.", ""]
    small = [g for g, s in stats.items() if s["filtered"] and s["docs"] < cc["min_docs_for_stats"]]
    if small:
        L.append(f"Cảnh báo: {len(small)} nhóm có < {cc['min_docs_for_stats']} bài (thống kê kém ổn định): "
                 + ", ".join(f"{g} ({stats[g]['docs']})" for g in small[:12]) + ("…" if len(small) > 12 else ""))
        L.append("")
    L += ["| nhóm | bài | lọc? | dòng bị loại | tiêu đề mục giữ lại |", "|---|---:|---|---:|---:|"]
    for g, s in stats.items():
        x = lines.get(g, {"drop": [], "keep_heading": []})
        L.append(f"| {g} | {s['docs']:,} | {'có' if s['filtered'] else 'không (ít bài)'} | {len(x['drop'])} | {len(x['keep_heading'])} |")
    for g in [g for g in stats if stats[g]["filtered"]][:top]:
        x = lines[g]
        L += ["", f"## {g} ({stats[g]['docs']:,} bài)", "", "Bị loại (tỷ lệ bài chứa dòng):", ""]
        L += [f"- `{i['ratio']:.0%}` {i['line'][:160]}" for i in x["drop"]] or ["- (không có)"]
        L += ["", "Giữ lại vì là tiêu đề mục (tỷ lệ | sau nó là nội dung | ở cuối bài):", ""]
        L += [f"- `{i['ratio']:.0%} | {i['followed_by_content']:.0%} | {i['at_end']:.0%}` {i['line'][:160]}"
              for i in x["keep_heading"]] or ["- (không có)"]
    return "\n".join(L) + "\n"


# ---------------- bước 2: áp dụng ----------------
def clean_text(text, drop_set):
    return "\n".join(l for l in text.split("\n") if l not in drop_set)


def apply(docs, lines, cc, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop = {g: {i["line"] for i in x["drop"]} for g, x in lines.items()}
    buf, k = [], 0
    st = collections.Counter()
    hashes = collections.Counter()

    def flush():
        nonlocal buf, k
        if buf:
            pq.write_table(pa.Table.from_pylist(buf, schema=SCHEMA), out_dir / f"part-{k:05d}.parquet")
            k += 1
            buf = []

    for d in docs:
        g = group_of(d["host"], cc.get("groups"))
        st["in"] += 1
        st["chars_in"] += len(d["text"])
        text = clean_text(d["text"], drop.get(g, ()))
        if len(text) < cc["min_chars"]:
            st["dropped_short"] += 1
            continue
        st["out"] += 1
        st["chars_out"] += len(text)
        hashes[hashlib.md5(text.encode("utf-8")).hexdigest()] += 1
        buf.append({"doc_id": str(d["id"]), "host": d["host"], "lang": d["lang"], "title": d.get("title") or "",
                    "text": text})
        if len(buf) >= cc["part_size"]:
            flush()
    flush()
    st["dup_extra_copies"] = sum(c - 1 for c in hashes.values() if c > 1)
    return dict(st)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stats", action="store_true", help="bước 1: tính dòng lặp, viết danh sách để duyệt")
    g.add_argument("--apply", action="store_true", help="bước 2: làm sạch theo danh sách đã duyệt")
    ap.add_argument("--input", default=None, help="file parquet thay cho out/ (vd sample/corpus_raw.parquet)")
    ap.add_argument("--lines", default=None, help="file danh sách dòng (mặc định <work>/clean_lines.json)")
    ap.add_argument("--out", default=None, help="thư mục corpus_clean (mặc định <work>/corpus_clean)")
    ap.add_argument("--force", action="store_true", help="cho phép ghi đè kết quả đã có")
    a = ap.parse_args()
    cfg = config.load(a.config)
    cc = cfg["clean"]
    work = cfg.work_dir
    lines_path = Path(a.lines) if a.lines else work / cc["lines_file"]
    res_dir = work / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if a.stats:
        if lines_path.exists() and not a.force:
            print(f"{lines_path} đã có (có thể đã được duyệt) → không ghi đè; dùng --force để tính lại.")
            return
        lines, stats = compute_lines(read_input(cfg, a.input), cc, cfg["seed"])
        lines_path.write_text(json.dumps({"config": cc, "seed": cfg["seed"], "input": a.input or str(cfg.crawl_path),
                                          "stats": stats, "lines": lines}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        md = review_md(lines, stats, cc)
        (res_dir / "t1_review.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"→ {lines_path}\n→ {res_dir / 't1_review.md'}")
        return

    out_dir = Path(a.out) if a.out else work / cc["out_dir"]
    if out_dir.exists() and any(out_dir.glob("part-*.parquet")) and not a.force:
        print(f"{out_dir} đã có kết quả → không ghi đè; dùng --force để làm lại.")
        return
    if not lines_path.exists():
        print(f"Chưa có {lines_path}: chạy --stats và duyệt trước.")
        return
    if a.force:
        for f in out_dir.glob("part-*.parquet"):
            f.unlink()
    saved = json.loads(lines_path.read_text(encoding="utf-8"))
    cc_apply = dict(cc, groups=saved["config"].get("groups", cc.get("groups")))
    st = apply(read_input(cfg, a.input), saved["lines"], cc_apply, out_dir)
    st["lines_file"] = str(lines_path)
    st["input"] = a.input or str(cfg.crawl_path)
    (out_dir / "_meta.json").write_text(json.dumps({"stats": st, "config": cc_apply}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
