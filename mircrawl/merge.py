"""Sau crawl: gộp, kiểm tra chất lượng, báo cáo, trích lại.

merge     : output của mọi shard / phiên → corpus/*.parquet (mỗi id một dòng, dòng tốt nhất) + status.parquet
            + report.tsv. Kiểm tra chất lượng:
              - "boilerplate": cùng nội dung lặp ở ≥ --boiler-min trang của một tên miền (trang lỗi / thông báo
                theo mẫu) → loại khỏi corpus;
              - "dup": nội dung trùng hệt một tài liệu khác (giữ cả hai, dup_of = id nhỏ nhất của nhóm — nhãn của BTC
                có thể trỏ vào bất kỳ id nào trong nhóm);
              - hậu tố tên trang lặp lại ở tiêu đề (" | Sức Khỏe Cộng Đồng", "_有问必答_快速问医生") → bỏ;
              - "generic_title": tiêu đề lặp ở rất nhiều trang cùng tên miền (chỉ còn tên trang…) → để trống;
              - tác giả mặc định của cả trang ("VnExpress", "Báo Đà Nẵng") và từ khoá SEO lặp ở ≥30% số bài của
                một tên miền ("Tin tức; Thế giới; …") → bỏ;
              - "short": tài liệu chỉ có tiêu đề / chữ rất ngắn (status "empty") — vẫn đưa vào corpus (trừ --no-short).
report    : thống kê theo tên miền (đã xong / lỗi / chưa thử) — dùng sau --probe hoặc giữa các phiên.
reextract : chạy lại bộ trích hiện tại trên raw/*.parquet (không cần mạng) → pages/rx-*.parquet (xv mới).
"""
import collections
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from . import XV
from .store import PAGE_SCHEMA, find_parts, now_iso, status_rank

_SEPS = (" | ", " • ", " - ", " – ", " — ", " _ ", "_", " :: ", " » ", " · ", "--", "-", "|")
_P1 = ["id", "url", "host", "status", "xv", "processed_at", "n_chars", "text_hash", "title", "lang", "doc_type",
       "published", "author", "keywords", "site_name"]


def _unique(files):
    seen = {}
    for f in files:
        seen.setdefault(os.path.basename(f), f)
    return sorted(seen.values())


def load_winners(inputs):
    """Đọc các cột nhỏ của mọi pages/*.parquet → DataFrame mỗi id một dòng (dòng tốt nhất) + số lần thử."""
    import pandas as pd
    files = _unique(find_parts(inputs, "pages"))
    if not files:
        raise SystemExit(f"Không thấy pages/*.parquet trong {inputs}")
    frames = []
    for fi, f in enumerate(files):
        names = set(pq.read_schema(f).names)
        t = pq.read_table(f, columns=[c for c in _P1 if c in names]).to_pandas()
        t["_f"], t["_r"] = fi, range(len(t))
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df["_rank"] = df["status"].map(status_rank)
    df["xv"] = df["xv"].fillna(0)
    df = df.sort_values(["id", "_rank", "xv", "processed_at"], ascending=[True, True, False, False])
    attempts = df.groupby("id").size()
    win = df.drop_duplicates("id", keep="first").copy()
    win["attempts"] = win["id"].map(attempts).astype("int32")
    return files, win.reset_index(drop=True)


def _site_suffixes(titles, min_docs=20, min_frac=0.3):
    """Các hậu tố (dấu ngăn, phần đuôi) lặp ở ≥ min_frac số tiêu đề của một tên miền."""
    cnt = collections.Counter()
    n = 0
    for t in titles:
        if not t:
            continue
        n += 1
        for sep in _SEPS:
            if sep in t:
                cnt[(sep, t.rpartition(sep)[2].strip())] += 1
    need = max(min_docs, min_frac * n)
    return {k for k, c in cnt.items() if c >= need and k[1]}


def _strip(t, sufs):
    for _ in range(2):                                     # "X - Chuyên mục | Tên trang": bỏ tối đa 2 tầng
        for sep, tail in sufs:
            if t.endswith(sep + tail) and len(t) > len(sep + tail):
                t = t[: -len(sep + tail)].strip()
    return t


def quality(win, boiler_min=20):
    """Thêm cột title_new, flags, dup_of; đổi status thành 'boilerplate' cho trang mẫu lặp."""
    import pandas as pd
    win["title_new"] = win["title"].fillna("")
    win["author_new"] = win["author"].fillna("")
    win["keywords_new"] = win["keywords"].fillna("")
    win["flags"] = ""
    content = win["status"].isin(["ok", "empty"]) & win["text_hash"].fillna("").ne("")
    # trang mẫu lặp lại (cùng nội dung ở rất nhiều URL của một tên miền)
    grp = win[content].groupby(["host", "text_hash"])["id"].transform("size")
    boiler = grp[grp >= boiler_min].index
    win.loc[boiler, "status"] = "boilerplate"
    # bản trùng nội dung (mọi tên miền)
    c2 = win["status"].isin(["ok", "empty"]) & win["text_hash"].fillna("").ne("")
    first = win[c2].groupby("text_hash")["id"].transform("min")
    win["dup_of"] = pd.Series(pd.NA, index=win.index, dtype="Int64")
    d = first[first != win.loc[first.index, "id"]]
    win.loc[d.index, "dup_of"] = d.astype("int64")
    win.loc[d.index, "flags"] += "dup,"
    # tiêu đề: bỏ hậu tố tên trang lặp lại; tiêu đề lặp quá nhiều → để trống
    stripped_hosts = {}
    for host, idx in win[win["status"].isin(["ok", "empty"])].groupby("host").groups.items():
        titles = win.loc[idx, "title_new"]
        sufs = _site_suffixes(titles.tolist())
        if sufs:
            stripped_hosts[host] = sorted(t for _, t in sufs)
            win.loc[idx, "title_new"] = titles.map(lambda t: _strip(t, sufs))
        titles = win.loc[idx, "title_new"]
        vc = titles[titles.ne("")].value_counts()
        tails = {t for _, t in sufs}
        site = set(win.loc[idx, "site_name"].dropna().unique()) if "site_name" in win else set()
        generic = set(vc[vc >= max(50, 0.05 * len(idx))].index) | (set(vc.index) & (tails | site))
        if generic:
            g = titles.isin(generic)
            win.loc[g[g].index, "title_new"] = ""
            win.loc[g[g].index, "flags"] += "generic_title,"
        need = max(20, 0.3 * len(idx))
        au = win.loc[idx, "author_new"]
        vc = au[au.ne("")].value_counts()
        site_auth = set(vc[vc >= need].index)
        if site_auth:
            a = au.isin(site_auth)
            win.loc[a[a].index, "author_new"] = ""
        kw = win.loc[idx, "keywords_new"]
        kc = collections.Counter(k for v in kw for k in set(v.split("; ")) if k)
        common = {k for k, c in kc.items() if c >= need}
        if common:
            win.loc[idx, "keywords_new"] = kw.map(lambda v: "; ".join(k for k in v.split("; ") if k and k not in common))
    win.loc[win["status"].eq("empty"), "flags"] += "short,"
    win["flags"] = win["flags"].str.rstrip(",")
    return win, stripped_hosts


def host_table(win, links=None, shard=None):
    """Bảng theo tên miền: tổng URL (theo danh sách gốc nếu có), ok, rỗng, thông báo, đã gỡ, lỗi, chưa thử."""
    import pandas as pd
    from urllib.parse import urlparse
    w = win[["id", "host", "status", "n_chars", "lang", "published", "author"]].copy()
    if links is not None:
        ids = set(w["id"])
        si, sn = shard if shard else (0, 1)
        miss = [(i, urlparse(u).netloc.lower()) for i, u in links if i % sn == si and i not in ids]
        if miss:
            w = pd.concat([w, pd.DataFrame({"id": [i for i, _ in miss], "host": [h for _, h in miss],
                                            "status": "pending"})], ignore_index=True)
    st = w["status"].fillna("")
    w["kind"] = "error"
    for k, cond in (("ok", st.eq("ok")), ("short", st.eq("empty")), ("notice", st.isin(["notice", "boilerplate"])),
                    ("gone", st.isin(["gone", "http_404", "http_410", "http_400", "not_html", "pm_missing"])),
                    ("pending", st.eq("pending"))):
        w.loc[cond, "kind"] = k
    t = pd.crosstab(w["host"], w["kind"])
    for k in ("ok", "short", "notice", "gone", "error", "pending"):
        if k not in t:
            t[k] = 0
    t["total"] = t[["ok", "short", "notice", "gone", "error", "pending"]].sum(axis=1)
    t["ok_pct"] = (100 * t["ok"] / t["total"]).round(1)
    okw = w[w["kind"].eq("ok")]
    t["median_chars"] = okw.groupby("host")["n_chars"].median().reindex(t.index).fillna(0).astype(int)
    t["pct_date"] = (100 * okw.groupby("host")["published"].apply(lambda s: s.fillna("").ne("").mean())
                     ).reindex(t.index).fillna(0).round(0)
    t["pct_author"] = (100 * okw.groupby("host")["author"].apply(lambda s: s.fillna("").ne("").mean())
                       ).reindex(t.index).fillna(0).round(0)
    top = w[w["kind"].eq("error")].groupby("host")["status"].agg(lambda s: ", ".join(
        f"{k}:{v}" for k, v in s.value_counts().head(3).items()))
    t["top_errors"] = top.reindex(t.index).fillna("")
    cols = ["total", "ok", "ok_pct", "short", "notice", "gone", "error", "pending", "median_chars", "pct_date",
            "pct_author", "top_errors"]
    return t[cols].sort_values("total", ascending=False)


def print_table(t, n=60):
    import pandas as pd
    with pd.option_context("display.max_rows", n, "display.width", 250, "display.max_colwidth", 60):
        print(t.head(n).to_string())
    s = t[["total", "ok", "short", "notice", "gone", "error", "pending"]].sum()
    print(f"\nTỔNG: {int(s['total']):,} URL | ok {int(s['ok']):,} ({100 * s['ok'] / max(1, s['total']):.1f}%) | "
          f"ngắn {int(s['short']):,} | thông báo {int(s['notice']):,} | đã gỡ {int(s['gone']):,} | "
          f"lỗi {int(s['error']):,} | chưa thử {int(s['pending']):,}")


def report(inputs, links_path=None, shard=None, out=None):
    from .crawl import load_links, parse_shard
    _, win = load_winners(inputs)
    links = load_links(links_path) if links_path else None
    t = host_table(win, links, parse_shard(shard) if shard else None)
    print_table(t)
    by_lang = win[win["status"].eq("ok")].groupby("lang")["n_chars"].agg(["count", "median"])
    print("\nTài liệu ok theo ngôn ngữ:\n" + by_lang.to_string())
    if out:
        os.makedirs(out, exist_ok=True)
        t.to_csv(os.path.join(out, "report.tsv"), sep="\t", encoding="utf-8-sig")
        print(f"→ {os.path.join(out, 'report.tsv')}")
    return t


def merge(inputs, out, links_path=None, include_short=True, boiler_min=20, part_rows=50000):
    import pandas as pd
    from .crawl import load_links
    t0 = time.time()
    files, win = load_winners(inputs)
    print(f"{len(files)} file, {len(win):,} id ({time.time() - t0:.0f}s)")
    win, stripped = quality(win, boiler_min)
    keep = win["status"].eq("ok") | (include_short & win["status"].eq("empty") & win["title_new"].ne(""))
    os.makedirs(os.path.join(out, "corpus"), exist_ok=True)
    extra_cols = ["dup_of", "flags", "attempts"]
    schema = pa.schema(list(PAGE_SCHEMA) + [("dup_of", pa.int64()), ("flags", pa.string()), ("attempts", pa.int32())])
    sel = win[keep].set_index(["_f", "_r"])
    sel_files = set(sel.index.get_level_values(0))
    buf, k, n_out = [], 0, 0

    def dump(rows):
        nonlocal k
        tbl = pa.Table.from_pandas(pd.concat(rows, ignore_index=True), schema=schema, preserve_index=False)
        pq.write_table(tbl, os.path.join(out, "corpus", f"part-{k:05d}.parquet"), compression="zstd",
                       compression_level=6)
        k += 1

    for fi, f in enumerate(files):
        if fi not in sel_files:
            continue
        part = sel.loc[fi]
        t = pq.read_table(f).to_pandas()
        for c in PAGE_SCHEMA.names:                        # file cũ thiếu cột → thêm cột rỗng
            if c not in t:
                t[c] = None
        t = t.iloc[part.index.to_numpy()].reset_index(drop=True)
        t["title"] = part["title_new"].to_numpy()
        t["author"] = part["author_new"].to_numpy()
        t["keywords"] = part["keywords_new"].to_numpy()
        t["status"] = part["status"].to_numpy()
        for c in extra_cols:
            t[c] = part[c].to_numpy()
        buf.append(t[[f.name for f in schema]])
        if sum(len(x) for x in buf) >= part_rows:
            dump(buf)
            n_out += sum(len(x) for x in buf)
            buf = []
    if buf:
        dump(buf)
        n_out += sum(len(x) for x in buf)
    st = win[["id", "url", "host", "status", "attempts", "lang", "doc_type", "n_chars", "dup_of", "flags"]]
    st.to_parquet(os.path.join(out, "status.parquet"), index=False)
    links = load_links(links_path) if links_path else None
    t = host_table(win, links)
    t.to_csv(os.path.join(out, "report.tsv"), sep="\t", encoding="utf-8-sig")
    summary = {
        "built_at": now_iso(), "xv": XV, "inputs": inputs, "n_ids": int(len(win)), "n_corpus": int(n_out),
        "status": {k: int(v) for k, v in win["status"].value_counts().items()},
        "corpus_by_lang": {k: int(v) for k, v in win[keep]["lang"].value_counts().items()},
        "corpus_by_doc_type": {k: int(v) for k, v in win[keep]["doc_type"].value_counts().items()},
        "n_dup": int(win["dup_of"].notna().sum()), "title_suffixes_stripped": stripped,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print_table(t, 40)
    print(f"\ncorpus: {n_out:,} tài liệu → {out}/corpus/ ({k} file) | theo ngôn ngữ {summary['corpus_by_lang']}")
    print(f"trạng thái: {summary['status']} | bản trùng: {summary['n_dup']:,} | {time.time() - t0:.0f}s")
    return summary


# ---------------------------------------------------------------- trích lại từ HTML đã lưu
def _rx_job(args):
    from .extract import extract_job
    html, host, url, charset, mc, mcz = args
    return extract_job(html, host, url, charset, False, mc, mcz)


def reextract(inputs, out, hosts=None, workers=None, min_chars=30, min_chars_zh=10, batch=2000):
    from concurrent.futures import ProcessPoolExecutor
    files = _unique(find_parts(inputs, "raw"))
    if not files:
        raise SystemExit(f"Không thấy raw/*.parquet trong {inputs} (cần crawl với lưu HTML).")
    hosts = {h.strip().lower() for h in (hosts or "").split(",") if h.strip()}
    os.makedirs(os.path.join(out, "pages"), exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        for f in files:
            name = f"rx-{os.path.basename(f)[:-8]}-x{XV}.parquet"
            dst = os.path.join(out, "pages", name)
            if os.path.exists(dst):
                continue
            rows = []
            for b in pq.ParquetFile(f).iter_batches(batch_size=batch):
                d = b.to_pydict()
                idx = [j for j, h in enumerate(d["host"])
                       if not hosts or any(h == x or h.endswith("." + x) for x in hosts)]
                cs = d.get("charset") or [None] * len(d["id"])
                jobs = [(d["html"][j], d["host"][j], d["url"][j], cs[j], min_chars, min_chars_zh) for j in idx]
                for j, (rec, _) in zip(idx, ex.map(_rx_job, jobs, chunksize=20)):
                    row = dict.fromkeys(PAGE_SCHEMA.names)
                    row.update(rec)
                    row.update(id=d["id"][j], url=d["url"][j], final_url=d["final_url"][j], host=d["host"][j],
                               fetched_at=d["fetched_at"][j], processed_at=now_iso(), session="rx-" + name[3:-8])
                    rows.append(row)
            if rows:
                tmp = dst + ".tmp"
                pq.write_table(pa.Table.from_pylist(rows, schema=PAGE_SCHEMA), tmp, compression="zstd")
                os.replace(tmp, dst)
                n += len(rows)
            print(f"[{(time.time() - t0) / 60:5.1f} phút] {os.path.basename(f)} → {name}: {len(rows):,} trang "
                  f"(tổng {n:,})", flush=True)
    print(f"Xong reextract: {n:,} trang → {out}/pages/")
