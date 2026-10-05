"""Lấy bộ HTML đánh giá cho crawl.py: mỗi tên miền vài trang, rải theo các kiểu đường dẫn khác nhau.

    py crawl_eval/fetch_eval.py                      # mặc định: crawl_eval/html/
    py crawl_eval/fetch_eval.py --hosts cnkang.com   # chỉ vài tên miền

Chạy lại: bỏ qua trang đã tải. HTML lưu nguyên bản (gzip) để thử lại bộ trích chữ không cần gọi mạng.
"""
import argparse
import asyncio
import collections
import gzip
import json
import os
import random
import re
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq  # noqa: E402

import crawl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def template(url):
    """Kiểu đường dẫn: đoạn đầu giữ nguyên, các đoạn sau thay chữ số bằng #, chữ bằng a."""
    p = urlparse(url)
    segs = [s for s in p.path.split("/") if s]
    if not segs:
        return "/"
    head = segs[0] if not re.search(r"\d", segs[0]) else re.sub(r"\d+", "#", segs[0])
    rest = [re.sub(r"[^\W\d_]+", "a", re.sub(r"\d+", "#", s)) for s in segs[1:]]
    return "/" + "/".join([head] + rest) + ("?q" if p.query else "")


def pick(urls, k, rng):
    """Chọn k URL: phủ các kiểu đường dẫn phổ biến nhất, trong mỗi kiểu lấy rải theo id (cũ → mới)."""
    by = collections.defaultdict(list)
    for i, u in urls:
        by[template(u)].append((i, u))
    if len(by) > 100:                    # URL dạng slug: "kiểu đường dẫn" vô nghĩa → rải đều theo id
        lst = sorted(urls)
        return [lst[int((j + rng.random()) * len(lst) / k)] for j in range(min(k, len(lst)))]
    tpls = sorted(by, key=lambda t: -len(by[t]))
    out, round_ = [], 0
    while len(out) < k and round_ < 4:
        for t in tpls:
            lst = sorted(by[t])
            if len(lst) > round_:
                j = int((round_ + 0.5) / 4 * len(lst)) if len(lst) >= 4 else round_
                cand = lst[min(j, len(lst) - 1)]
                if cand not in out:
                    out.append(cand)
            if len(out) >= k:
                break
        round_ += 1
    rest = [x for x in urls if x not in out]
    while len(out) < k and rest:
        out.append(rest.pop(rng.randrange(len(rest))))
    return out


async def main_async(a):
    os.makedirs(os.path.join(a.out, "html"), exist_ok=True)
    idx_path = os.path.join(a.out, "index.jsonl")
    done = set()
    if os.path.exists(idx_path):
        for line in open(idx_path, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] == "ok" or r["status"] in crawl.BENIGN:     # trang lỗi tạm thời: lần sau thử lại
                done.add(r["id"])
    t = pq.read_table(a.input).to_pydict()
    by_host = collections.defaultdict(list)
    for i, u in zip(t["id"], t["url"]):
        by_host[urlparse(u).netloc.lower()].append((i, u))
    only = [h.strip().lower() for h in a.hosts.split(",") if h.strip()]
    rng = random.Random(0)
    groups = collections.defaultdict(list)
    for h, urls in by_host.items():
        if only and not any(h == x or h.endswith("." + x) for x in only):
            continue
        n = len(urls)
        k = a.big if n > 50_000 else a.mid if n > 5_000 else a.small
        for i, u in pick(urls, k, rng):
            if i not in done:
                groups[crawl.rate_group(h)[0]].append((i, u, h))
    total = sum(len(v) for v in groups.values())
    print(f"Cần tải {total} trang, {len(groups)} nhóm tên miền")
    idx = open(idx_path, "a", encoding="utf-8")
    from curl_cffi.requests import AsyncSession
    stats = collections.Counter()
    async with AsyncSession(impersonate="chrome", max_clients=32,
                            headers={"Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}) as s:
        async def worker(g, q):
            rule = crawl.rate_group(q[0][2])[1]
            delay = max(a.delay, rule["delay"] if rule else 0)
            timeout = a.timeout or (rule or {}).get("timeout") or 25
            fails = 0
            for i, u, h in q:
                body, st = await crawl.fetch(s, "curl", u, timeout, 1)
                stats[st] += 1
                rec = {"id": i, "url": u, "host": h, "status": st}
                if body is not None:
                    safe = re.sub(r"[^A-Za-z0-9.\-]", "_", h)
                    with gzip.open(os.path.join(a.out, "html", f"{safe}__{i}.html.gz"), "wb") as f:
                        f.write(body)
                    fails = 0
                elif st not in crawl.BENIGN:
                    fails += 1
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                idx.flush()
                if fails >= 4 or st == "blocked_captcha":          # đừng gọi tiếp tên miền đang từ chối
                    print(f"  dừng nhóm {g}: {st}")
                    return
                await asyncio.sleep(delay * (0.5 + random.random()))
        await asyncio.gather(*[worker(g, q) for g, q in groups.items()])
    idx.close()
    print("Xong:", dict(stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(os.path.dirname(HERE), "Data", "links_corpus.parquet"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--hosts", default="")
    ap.add_argument("--big", type=int, default=16, help="số trang / tên miền > 50k URL")
    ap.add_argument("--mid", type=int, default=10, help="số trang / tên miền 5k–50k URL")
    ap.add_argument("--small", type=int, default=6, help="số trang / tên miền nhỏ")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=0, help="ghi đè thời gian chờ (giây)")
    a = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
