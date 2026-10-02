#!/usr/bin/env python3
"""
Crawl nội dung bài viết từ links_corpus.parquet (ViBioMIR) — có thể dừng/chạy tiếp.

Cài đặt:   pip install aiohttp trafilatura pyarrow tqdm
Chạy thử:  python crawl.py --limit 300 --out out_test
Chạy thật: python crawl.py --out out --shard 0/3        (mỗi thành viên một shard khác nhau)

Kết quả (thư mục --out):
  part-00000.jsonl.gz ...  mỗi dòng: {"id","url","host","lang","title","text"}
  progress.tsv             id <tab> trạng thái (ok / http_404 / err_timeout ...), dùng để chạy tiếp
Chạy lại cùng lệnh = tiếp tục từ chỗ dừng. Thêm --retry-failed để thử lại các URL bị lỗi tạm thời.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # tránh lỗi "OpenBLAS memory allocation failed" khi chạy nhiều process
import argparse, asyncio, collections, gzip, json, random, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlparse

import aiohttp
import pyarrow.parquet as pq

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CJK = re.compile(r"[\u4e00-\u9fff]")
FINAL_HTTP = {400, 401, 403, 404, 410, 451}      # không thử lại
RETRY_HTTP = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


# ---------- trích nội dung (chạy ở process con) ----------
def extract(html_bytes: bytes):
    import trafilatura
    text = trafilatura.extract(html_bytes, include_comments=False, include_tables=True,
                               favor_recall=True, deduplicate=False)
    title = ""
    try:
        meta = trafilatura.extract_metadata(html_bytes)
        title = (meta.title or "") if meta else ""
    except Exception:
        pass
    text = "\n".join(l.strip() for l in (text or "").split("\n") if l.strip())
    cjk = len(CJK.findall(text[:2000]))
    lang = "zh" if cjk > 0.2 * max(1, len(text[:2000])) else "vi"
    return title.strip(), text.strip(), lang


# ---------- tải một URL ----------
BLOCKED = {"http_401", "http_403", "http_451", "blocked_cf", "blocked_tiny"}


async def _get(session, engine, url, timeout):
    if engine == "curl":
        r = await session.get(url, timeout=timeout, allow_redirects=True, max_redirects=5)
        return r.status_code, r.headers, r.content
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                           allow_redirects=True, max_redirects=5) as r:
        return r.status, r.headers, await r.read()


JS_COOKIE = re.compile(rb'''document\.cookie\s*=\s*["']([A-Za-z0-9_\-]+)=([^"';]+)["']''')


def set_cookie(session, engine, url, name, value):
    host = urlparse(url).hostname
    if engine == "curl":
        session.cookies.set(name, value, domain=host, path="/")
    else:
        from yarl import URL
        session.cookie_jar.update_cookies({name: value}, response_url=URL(url))


async def fetch(session, engine, url, timeout, retries):
    last = "err_unknown"
    cookie_tried = False
    attempt = -1
    while attempt < retries:
        attempt += 1
        try:
            status, hdr, body = await _get(session, engine, url, timeout)
            if hdr.get("cf-mitigated"):                     # Cloudflare đang đòi thử thách (challenge)
                return None, "blocked_cf"
            if status == 200:
                if len(body) > 8_000_000:
                    return None, "err_too_big"
                if len(body) < 600:
                    # Một số trang (vd laodong.vn) trả về trang nhỏ chỉ để đặt cookie rồi tải lại:
                    #   <script>document.cookie="D1N=...";window.location.reload(true);</script>
                    # Trình duyệt nào cũng tự làm việc này; ta đặt cookie đó rồi yêu cầu lại một lần.
                    m = JS_COOKIE.search(body)
                    if m and not cookie_tried:
                        cookie_tried = True
                        set_cookie(session, engine, url, m.group(1).decode(), m.group(2).decode())
                        attempt -= 1                         # lần gọi lại này không tính là một lần thử lỗi
                        continue
                    return None, "blocked_tiny"
                return body, "ok"
            if status in FINAL_HTTP:
                return None, f"http_{status}"
            last = f"http_{status}"
            if status == 429:
                ra = hdr.get("Retry-After", "")
                await asyncio.sleep(min(60, int(ra)) if str(ra).isdigit() else 10 * (attempt + 1))
        except asyncio.TimeoutError:
            last = "err_timeout"
        except Exception as e:
            last = "err_" + type(e).__name__
        if attempt < retries:
            await asyncio.sleep(2 ** attempt + random.random())
    return None, last


class Writer:
    def __init__(self, out, rec_per_part):
        self.out, self.n_per, self.k, self.cnt = out, rec_per_part, 0, 0
        self.fh = None
        self.prog = open(os.path.join(out, "progress.tsv"), "a", encoding="utf-8")
        existing = [f for f in os.listdir(out) if f.startswith("part-") and f.endswith(".jsonl.gz")]
        self.k = max([int(f[5:10]) for f in existing], default=-1) + 1   # luôn mở part mới khi chạy lại

    def write(self, rec, status):
        if rec is not None:
            if self.fh is None or self.cnt >= self.n_per:
                if self.fh:
                    self.fh.close()
                self.fh = gzip.open(os.path.join(self.out, f"part-{self.k:05d}.jsonl.gz"), "at", encoding="utf-8")
                self.k += 1
                self.cnt = 0
            self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cnt += 1
        self.prog.write(f"{rec['id'] if rec else status[0]}\t{status[2]}\n")

    def flush(self):
        if self.fh:
            self.fh.flush()
        self.prog.flush()

    def close(self):
        self.flush()
        if self.fh:
            self.fh.close()
        self.prog.close()


async def main_async(a):
    os.makedirs(a.out, exist_ok=True)
    tbl = pq.read_table(a.input).to_pydict()
    ids, urls = tbl["id"], tbl["url"]

    # shard i/N
    si, sn = (int(x) for x in a.shard.split("/"))
    items = [(i, u) for i, u in zip(ids, urls) if i % sn == si]

    # bỏ qua những id đã xong
    done = {}
    pf = os.path.join(a.out, "progress.tsv")
    if os.path.exists(pf):
        for line in open(pf, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                done[int(p[0])] = p[1]
    retry_status = {f"http_{c}" for c in RETRY_HTTP}
    def finished(i):
        s = done.get(i)
        if s is None:
            return False
        if a.retry_failed and (s.startswith("err_") or s in retry_status):
            return False
        return True
    items = [(i, u) for i, u in items if not finished(i)]
    skip = {h.strip().lower() for h in a.skip_hosts.split(",") if h.strip()}
    if skip:
        n0 = len(items)
        items = [(i, u) for i, u in items if urlparse(u).netloc.lower() not in skip]
        print(f"Bỏ qua {n0 - len(items):,} URL thuộc: {sorted(skip)}")
    if a.limit:
        random.Random(0).shuffle(items)
        items = items[:a.limit]

    by_host = collections.defaultdict(collections.deque)
    for i, u in items:
        by_host[urlparse(u).netloc.lower()].append((i, u))
    total = len(items)
    print(f"Shard {a.shard}: {total:,} URL cần crawl, {len(by_host)} tên miền, đã xong trước đó: {len(done):,}")
    if not total:
        return

    writer = Writer(a.out, a.records_per_part)
    pool = ProcessPoolExecutor(max_workers=a.cpu)
    loop = asyncio.get_running_loop()
    glob = asyncio.Semaphore(a.concurrency)
    stats = collections.Counter()
    hstats = collections.defaultdict(collections.Counter)
    seen_err = set()
    html_saved = collections.Counter()
    t0 = time.time()
    stop = asyncio.Event()

    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}

    state = collections.defaultdict(lambda: {"consec": 0, "dead": False})

    async def host_worker(host, q):
        while q and not stop.is_set():
            if state[host]["dead"]:
                return
            i, u = q.popleft()
            async with glob:
                body, st = await fetch(session, engine, u, a.timeout, a.retries)
            if body is None:
                writer.write(None, (i, u, st)); stats[st] += 1; hstats[host][st] += 1
                if st not in {"http_404", "http_410", "http_400"}:
                    state[host]["consec"] += 1
                    if state[host]["consec"] >= a.block_threshold and not state[host]["dead"]:
                        state[host]["dead"] = True
                        print(f"[BỊ CHẶN] {host}: {a.block_threshold} URL liên tiếp lỗi ({st}) → dừng tên miền này. "
                              f"Còn {len(q):,} URL chưa thử (sẽ không ghi vào progress).", flush=True)
                else:
                    state[host]["consec"] = 0
            else:
                state[host]["consec"] = 0
                if html_saved[host] < a.save_html:
                    html_saved[host] += 1
                    os.makedirs(os.path.join(a.out, "html_samples"), exist_ok=True)
                    safe = re.sub(r"[^A-Za-z0-9.\-]", "_", host)
                    with gzip.open(os.path.join(a.out, "html_samples", f"{safe}__{i}.html.gz"), "wb") as hf:
                        hf.write(body)
                try:
                    title, text, lang = await loop.run_in_executor(pool, extract, body)
                except Exception as e:
                    key = type(e).__name__
                    if key not in seen_err:
                        seen_err.add(key)
                        print(f"[CẢNH BÁO] lỗi trích nội dung ({key}): {str(e)[:300]} | url={u}", flush=True)
                    writer.write(None, (i, u, "err_extract")); stats["err_extract"] += 1; hstats[host]["err_extract"] += 1
                    continue
                else:
                    if len(text) < a.min_chars:
                        writer.write(None, (i, u, "empty")); stats["empty"] += 1; hstats[host]["empty"] += 1
                    else:
                        writer.write({"id": i, "url": u, "host": host, "lang": lang,
                                      "title": title, "text": text}, ("", "", "ok"))
                        stats["ok"] += 1; hstats[host]["ok"] += 1
            await asyncio.sleep(a.delay * (0.5 + random.random()))

    async def reporter():
        while True:
            await asyncio.sleep(30)
            writer.flush()
            n = sum(stats.values()); el = time.time() - t0
            eta = (total - n) / max(n / el, 1e-9) / 3600
            print(f"[{el/60:6.1f} phút] {n:,}/{total:,} ({100*n/total:.1f}%) "
                  f"{n/el:.1f} trang/s | ok={stats['ok']:,} lỗi={n-stats['ok']:,} | còn ~{eta:.1f} giờ", flush=True)

    engine = a.engine
    if engine == "auto":
        try:
            import curl_cffi  # noqa: F401
            engine = "curl"
        except ImportError:
            engine = "aiohttp"
            print("[CẢNH BÁO] chưa cài curl_cffi → dùng aiohttp, nhiều trang sẽ trả 403. Chạy: pip install curl_cffi")
    print("Engine tải trang:", engine)
    if engine == "curl":
        from curl_cffi.requests import AsyncSession
        sess_cm = AsyncSession(impersonate="chrome", max_clients=a.concurrency,
                               headers={"Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"})
    else:
        sess_cm = aiohttp.ClientSession(connector=aiohttp.TCPConnector(
            limit=a.concurrency, limit_per_host=max(a.per_host_big, a.per_host), ttl_dns_cache=600, ssl=False), headers=headers,
            cookie_jar=aiohttp.CookieJar(unsafe=True))
    async with sess_cm as session:
        tasks = []
        for host, q in by_host.items():
            k = a.per_host_big if len(q) > 50_000 else a.per_host
            tasks += [asyncio.create_task(host_worker(host, q)) for _ in range(k)]
        rep = asyncio.create_task(reporter())
        try:
            await asyncio.gather(*tasks)
        except (asyncio.CancelledError, KeyboardInterrupt):
            stop.set()
        finally:
            rep.cancel()
            writer.close()
            pool.shutdown(wait=False, cancel_futures=True)
    print("Xong. Thống kê:", dict(stats))
    with open(os.path.join(a.out, "host_stats.tsv"), "w", encoding="utf-8") as f:
        f.write("host\ttotal\tok\tlỗi_chính\n")
        for h, c in sorted(hstats.items(), key=lambda x: -sum(x[1].values())):
            bad = [(k, v) for k, v in c.most_common() if k != "ok"]
            f.write(f"{h}\t{sum(c.values())}\t{c['ok']}\t{bad[0][0] + ':' + str(bad[0][1]) if bad else ''}\n")
    print("Tên miền có nhiều lỗi nhất (xem đủ ở host_stats.tsv):")
    for h, c in sorted(hstats.items(), key=lambda x: -(sum(x[1].values()) - x[1]["ok"]))[:12]:
        tot = sum(c.values())
        if tot - c["ok"]:
            print(f"  {h:40s} ok {c['ok']}/{tot}  lỗi: {dict((k, v) for k, v in c.items() if k != 'ok')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=r"E:\Medical Retrieval\Data\links_corpus.parquet")
    p.add_argument("--out", default="out")
    p.add_argument("--shard", default="0/1", help="i/N: chỉ lấy id % N == i (chia việc cho nhiều máy)")
    p.add_argument("--limit", type=int, default=0, help="chỉ crawl ngẫu nhiên N URL để chạy thử")
    p.add_argument("--concurrency", type=int, default=64, help="tổng số kết nối đồng thời")
    p.add_argument("--per-host", type=int, default=2, help="kết nối đồng thời / tên miền nhỏ")
    p.add_argument("--per-host-big", type=int, default=6, help="kết nối đồng thời / tên miền >50k URL")
    p.add_argument("--delay", type=float, default=0.3, help="nghỉ (giây) giữa 2 request của mỗi worker")
    p.add_argument("--timeout", type=int, default=25)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--records-per-part", type=int, default=20000)
    p.add_argument("--cpu", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)), help="số process trích nội dung")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--engine", choices=["auto", "curl", "aiohttp"], default="auto",
                   help="curl = curl_cffi (giống trình duyệt, ít bị 403 hơn)")
    p.add_argument("--save-html", type=int, default=3, help="lưu HTML gốc của N trang đầu mỗi tên miền (để kiểm tra bộ trích chữ)")
    p.add_argument("--block-threshold", type=int, default=25,
                   help="dừng một tên miền nếu lỗi từng này URL liên tiếp (bị chặn, máy chủ sập, quá thời gian)")
    p.add_argument("--skip-hosts", default="", help="các tên miền bỏ qua, cách nhau dấu phẩy, vd: zysjonline.com,laodong.vn")
    a = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        print("\nĐã dừng. Chạy lại cùng lệnh để tiếp tục.")


if __name__ == "__main__":
    main()
