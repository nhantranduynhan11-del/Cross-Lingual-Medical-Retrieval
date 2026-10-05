#!/usr/bin/env python3
"""
Crawl nội dung bài viết từ links_corpus.parquet (ViBioMIR) — có thể dừng/chạy tiếp.

Cài đặt:   pip install aiohttp trafilatura pyarrow tqdm
Thăm dò:   python crawl.py --probe 3                    (vài trang mỗi tên miền, in bảng lỗi theo từng trang)
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
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "section", "article", "table", "ul", "ol"}

# Quy tắc riêng cho các trang mà bộ trích chung (trafilatura) làm mất nội dung.
# keep: các vùng chứa nội dung chính; drop: phần giao diện nằm trong vùng đó; stop: các dòng bỏ đi.
SITE_RULES = [
    {"suffix": "120ask.com",
     "keep": ['//div[contains(@class,"b_askbox")]',
              '//div[contains(@class,"b_answerli") and not(contains(@class,"zhenshiAD"))]'],
     "drop": './/*[contains(@class,"b_answerarea") or contains(@class,"b_askbot_ad") or contains(@class,"tousu") '
             'or contains(@class,"codeposition") or contains(@class,"b_askbox_btn") or contains(@class,"b_bico") or @id="area"]',
     "title": '//h1', "stop": set()},
    {"suffix": "pmphai.com",
     "keep": ['//div[contains(@class,"gu_det_left_info")]'], "drop": None, "title": None,
     "stop": {"收藏", "已收藏"}},
]


_WS = re.compile(r"[ \t\u3000]{2,}")


def _lines(t):
    return "\n".join(_WS.sub(" ", l.strip()) for l in (t or "").replace("\xa0", " ").split("\n") if l.strip())


def _block_text(el):
    import copy
    el = copy.deepcopy(el)
    for bad in el.xpath(".//script|.//style|.//noscript"):
        bad.drop_tree()
    for e in el.iter():
        if isinstance(e.tag, str) and e.tag.lower() in _BLOCK:
            e.tail = "\n" + (e.tail or "")
            if e.tag.lower() != "br":
                e.text = "\n" + (e.text or "")
    return _lines(el.text_content())


def _site_extract(doc, rule):
    import copy
    parts = []
    for xp in rule["keep"]:
        for el in doc.xpath(xp):
            if rule["drop"]:
                el = copy.deepcopy(el)
                for bad in el.xpath(rule["drop"]):
                    bad.drop_tree()
            t = "\n".join(l for l in _block_text(el).split("\n") if l not in rule["stop"])
            if t:
                parts.append(t)
    title = ""
    if rule["title"]:
        h = doc.xpath(rule["title"])
        title = _lines(h[0].text_content()) if h else ""
    return title, "\n".join(parts)


def _fallback_extract(doc):
    """Khi bộ trích chung trả về rỗng: lấy vùng có nhiều chữ trong thẻ <p> nhất (kiểu readability)."""
    # không bỏ <form>: nhiều trang (ASP.NET) bọc cả bài viết trong một thẻ form
    for bad in doc.xpath("//script|//style|//noscript|//nav|//footer|//aside"):
        bad.drop_tree()
    score = collections.Counter()
    for p in doc.iter("p"):
        n = len((p.text_content() or "").strip())
        if n < 25:
            continue
        par = p.getparent()
        if par is not None:
            score[par] += n
            g = par.getparent()
            if g is not None:
                score[g] += n // 2
    if not score:
        return ""
    best = max(score, key=score.get)
    return _block_text(best)


def extract(html_bytes: bytes, host: str = ""):
    import copy
    import trafilatura
    import lxml.html
    # Trang khai báo bảng mã GB (gb2312/gbk) ở thẻ meta đầu tiên: tự giải mã để tránh chữ bị vỡ.
    m = re.search(rb'charset\s*=\s*["\']?\s*([\w-]+)', html_bytes[:4000], re.I)
    src = html_bytes
    if m and m.group(1).lower().startswith(b"gb"):
        src = html_bytes.decode("gb18030", "replace")
    title, text = "", ""
    cache = {}

    def get_doc():
        if "d" not in cache:
            if isinstance(src, str):
                raw = src
            else:
                try:
                    raw = html_bytes.decode(m.group(1).decode() if m else "utf-8", "replace")
                except LookupError:
                    raw = html_bytes.decode("utf-8", "replace")
            cache["d"] = lxml.html.fromstring(raw)
        return cache["d"]

    rule = next((r for r in SITE_RULES if host == r["suffix"] or host.endswith("." + r["suffix"])), None)
    if rule:
        try:
            title, text = _site_extract(get_doc(), rule)
        except Exception:
            title, text = "", ""
    if len(text) < 30:                                     # không có quy tắc riêng, hoặc quy tắc không khớp bố cục
        text = _lines(trafilatura.extract(src, include_comments=False, include_tables=True,
                                          favor_recall=True, deduplicate=False))
    if not title:
        try:
            meta = trafilatura.extract_metadata(src)
            title = (meta.title or "") if meta else ""
        except Exception:
            pass
    if len(text) < 50:                                     # bộ trích chung trả về rỗng
        try:
            # HTML hỏng (vd thẻ <html> đóng ngay đầu trang) làm bộ trích chung không thấy nội dung:
            # lấy lại nhánh <body>/<html> chứa nhiều chữ nhất rồi trích lần nữa.
            d = get_doc()
            root = max(d.xpath("//body") + d.xpath("//html") or [d], key=lambda e: len(e.text_content() or ""))
            text = _lines(trafilatura.extract(lxml.html.tostring(root, encoding="unicode"), include_comments=False,
                                              include_tables=True, favor_recall=True, deduplicate=False))
            if len(text) < 50:                             # vẫn rỗng → lấy vùng nhiều chữ nhất
                text = _fallback_extract(copy.deepcopy(root))
            h = d.xpath("//h1")
            if len(h) == 1 and _lines(h[0].text_content()):
                title = _lines(h[0].text_content())
        except Exception:
            pass
    text = _lines(text)
    cjk = len(CJK.findall(text[:2000]))
    lang = "zh" if cjk > 0.2 * max(1, len(text[:2000])) else "vi"
    return title.strip(), text.strip(), lang


# ---------- tải một URL ----------
BLOCKED = {"http_401", "http_403", "http_451", "blocked_cf", "blocked_tiny"}

# Quy tắc riêng theo tên miền: gộp mọi tên miền con vào MỘT hàng đợi, với số kết nối / nhịp nghỉ / thời gian chờ riêng.
SLOW_RULES = [
    # 39.net (các kênh tin): gọi dày là bật CAPTCHA kéo ghép hình → 1 kết nối, nghỉ lâu.
    # Đã thử 120 lần gọi ở nhịp này mà không bị chặn.
    {"suffix": "39.net", "exclude": {"ask.39.net", "jbk.39.net"}, "group": "39.net (các kênh tin)",
     "workers": 1, "delay": 2.0, "timeout": None, "retries": None},
    # youlai.cn: khoảng một nửa số trang treo tới hết thời gian chờ (không phụ thuộc tốc độ gọi), nửa còn lại trả về tức thì
    # → chờ ngắn, không thử lại ngay; các trang treo để --retry-failed xử lý sau.
    {"suffix": "youlai.cn", "exclude": set(), "group": "youlai.cn",
     "workers": 4, "delay": 0.3, "timeout": 8, "retries": 0},
    {"suffix": "qihuangzhishu.com", "exclude": set(), "group": "qihuangzhishu.com",
     "workers": 1, "delay": 1.0, "timeout": None, "retries": 1},
]
# Trang xác minh người dùng (CAPTCHA). Script KHÔNG giải CAPTCHA: gặp là dừng nhóm tên miền đó.
CAPTCHA = re.compile(rb"AJ-Captcha|slideVerify|\xe6\xbb\x91\xe5\x8a\xa8\xe6\x8b\xbc\xe5\x9b\xbe\xe9\xaa\x8c\xe8\xaf\x81|cf-turnstile|g-recaptcha|h-captcha", re.I)


def rate_group(host):
    """Trả về (tên nhóm, quy tắc hoặc None)."""
    for r in SLOW_RULES:
        suf = r["suffix"]
        if host not in r["exclude"] and (host == suf or host.endswith("." + suf)):
            return r["group"], r
    return host, None


LOGIN_PATH = re.compile(r"login|signin|sign-in|dang-nhap|passport", re.I)
BENIGN = {"http_400", "http_404", "http_410", "not_html", "gone_redirect"}   # trang không còn, không phải bị chặn


HTTP1_HOSTS = set()      # tên miền từng báo lỗi khung HTTP/2 (curl 16/92) → từ đó gọi bằng HTTP/1.1


async def _get(session, engine, url, timeout):
    if engine == "curl":
        kw = {}
        if urlparse(url).netloc.lower() in HTTP1_HOSTS:
            try:
                from curl_cffi.const import CurlHttpVersion
                kw["http_version"] = CurlHttpVersion.V1_1
            except Exception:
                kw["http_version"] = 2                       # giá trị số của HTTP/1.1 trong libcurl
        r = await session.get(url, timeout=timeout, allow_redirects=True, max_redirects=5, **kw)
        return r.status_code, r.headers, r.content, str(r.url)
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                           allow_redirects=True, max_redirects=5) as r:
        return r.status, r.headers, await r.read(), str(r.url)


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
    h2_tried = False
    attempt = -1
    while attempt < retries:
        attempt += 1
        try:
            status, hdr, body, final = await _get(session, engine, url, timeout)
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
                if len(body) < 8000 and CAPTCHA.search(body):
                    return None, "blocked_captcha"
                ctype = str(hdr.get("content-type", "")).lower()
                if (ctype and not any(x in ctype for x in ("html", "xml", "text/plain"))) \
                        or body[:3] == b"\xff\xd8\xff" or body[:4] in (b"%PDF", b"\x89PNG", b"GIF8"):
                    return None, "not_html"                  # URL trỏ tới ảnh / PDF, không phải bài viết
                o, f = urlparse(url), urlparse(final)
                if o.path.strip("/") and f.path != o.path and (not f.path.strip("/") or LOGIN_PATH.search(f.path)):
                    return None, "gone_redirect"             # bài đã gỡ: trang chuyển về trang chủ / trang đăng nhập
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
            code = getattr(e, "code", None)                  # curl_cffi: mã lỗi libcurl (28 = quá thời gian, 56 = đứt kết nối...)
            last = "err_" + type(e).__name__ + (f"_curl{code}" if isinstance(code, int) and code else "")
            if code == 28:
                last = "err_timeout"
            host = urlparse(url).netloc.lower()
            if code in (16, 92) and host not in HTTP1_HOSTS and not h2_tried:
                # máy chủ trả khung HTTP/2 hỏng: chuyển tên miền này sang HTTP/1.1 và gọi lại ngay, không tính là lần thử
                HTTP1_HOSTS.add(host)
                h2_tried = True
                attempt -= 1
                continue
            if code in (16, 92) and not h2_tried:            # tên miền đã ở HTTP/1.1 từ trước mà vẫn lỗi → thử lại một lần
                h2_tried = True
                attempt -= 1
                continue
        if attempt < retries:
            await asyncio.sleep(2 ** attempt + random.random())
    return None, last


def open_session(a):
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
               "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}
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
            limit=a.concurrency, limit_per_host=max(a.per_host_big, a.per_host), ttl_dns_cache=600, ssl=False),
            headers=headers, cookie_jar=aiohttp.CookieJar(unsafe=True))
    return engine, sess_cm


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
        if a.retry_failed and (s.startswith(("err_", "blocked_")) or s in retry_status or s == "empty"):
            return False
        return True
    items = [(i, u) for i, u in items if not finished(i)]
    skip = {h.strip().lower() for h in a.skip_hosts.split(",") if h.strip()}
    if skip:
        n0 = len(items)
        def skipped(u):
            h = urlparse(u).netloc.lower()
            return any(h == x or h.endswith("." + x) or h == "www." + x for x in skip)
        items = [(i, u) for i, u in items if not skipped(u)]
        print(f"Bỏ qua {n0 - len(items):,} URL thuộc: {sorted(skip)}")
    if a.limit:
        random.Random(0).shuffle(items)
        items = items[:a.limit]

    by_host = collections.defaultdict(list)          # nhóm tốc độ -> [(id, url, host)]
    rule = {}
    hosts = set()
    for i, u in items:
        h = urlparse(u).netloc.lower()
        g, r = rate_group(h)
        by_host[g].append((i, u, h)); rule[g] = r; hosts.add(h)
    for g in by_host:
        if rule[g] is not None:                      # nhóm có quy tắc riêng, gồm nhiều tên miền con: trộn đều
            random.Random(0).shuffle(by_host[g])
        by_host[g] = collections.deque(by_host[g])
    total = len(items)
    print(f"Shard {a.shard}: {total:,} URL cần crawl, {len(hosts)} tên miền, đã xong trước đó: {len(done):,}")
    for g, q in by_host.items():
        r = rule[g]
        if r is not None and q:
            print(f"  Quy tắc riêng: {g}: {len(q):,} URL, {r['workers']} kết nối, nghỉ {r['delay']}s "
                  f"→ ít nhất ~{len(q) * r['delay'] / r['workers'] / 3600:.1f} giờ")
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

    async def host_worker(grp, q, delay, timeout, retries):
        while q and not stop.is_set():
            if state[grp]["dead"]:
                return
            i, u, host = q.popleft()
            async with glob:
                body, st = await fetch(session, engine, u, timeout, retries)
            if body is None:
                writer.write(None, (i, u, st)); stats[st] += 1; hstats[host][st] += 1
                if st not in BENIGN:
                    # gặp CAPTCHA thì dừng gần như ngay: gọi tiếp chỉ làm trang giữ lệnh chặn lâu hơn
                    state[grp]["consec"] += a.block_threshold // 3 + 1 if st == "blocked_captcha" else 1
                    if state[grp]["consec"] >= a.block_threshold and not state[grp]["dead"]:
                        state[grp]["dead"] = True
                        print(f"[DỪNG TÊN MIỀN] {grp}: lỗi liên tiếp ({st}) → dừng nhóm này trong lần chạy này. "
                              f"Còn {len(q):,} URL chưa thử (không ghi vào progress, lần sau chạy tiếp).", flush=True)
                else:
                    state[grp]["consec"] = 0
            else:
                state[grp]["consec"] = 0
                if html_saved[host] < a.save_html:
                    html_saved[host] += 1
                    os.makedirs(os.path.join(a.out, "html_samples"), exist_ok=True)
                    safe = re.sub(r"[^A-Za-z0-9.\-]", "_", host)
                    with gzip.open(os.path.join(a.out, "html_samples", f"{safe}__{i}.html.gz"), "wb") as hf:
                        hf.write(body)
                try:
                    title, text, lang = await loop.run_in_executor(pool, extract, body, host)
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
            await asyncio.sleep(delay * (0.5 + random.random()))

    async def reporter():
        while True:
            await asyncio.sleep(30)
            writer.flush()
            n = sum(stats.values()); el = time.time() - t0
            eta = (total - n) / max(n / el, 1e-9) / 3600
            print(f"[{el/60:6.1f} phút] {n:,}/{total:,} ({100*n/total:.1f}%) "
                  f"{n/el:.1f} trang/s | ok={stats['ok']:,} lỗi={n-stats['ok']:,} | còn ~{eta:.1f} giờ", flush=True)

    engine, sess_cm = open_session(a)
    async with sess_cm as session:
        tasks = []
        for grp, q in by_host.items():
            r = rule[grp]
            if r is None:
                k, d, to, re_ = (a.per_host_big if len(q) > 50_000 else a.per_host), a.delay, a.timeout, a.retries
            else:
                k, d = r["workers"], r["delay"]
                to = r["timeout"] or a.timeout
                re_ = a.retries if r["retries"] is None else r["retries"]
            tasks += [asyncio.create_task(host_worker(grp, q, d, to, re_)) for _ in range(k)]
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



# ====================== CHẾ ĐỘ THĂM DÒ (--probe) ======================
# Bước 1: mỗi tên miền gọi NHẸ vài trang (rải đều trong danh sách URL) → bắt lỗi luôn xảy ra:
#         bị chặn, trang xác minh, sai bảng mã, trích chữ rỗng.
# Bước 2: mỗi tên miền gọi một loạt ngắn ở ĐÚNG nhịp sẽ dùng khi chạy thật → bắt lỗi chỉ lộ ra khi gọi dày
#         (CAPTCHA theo tốc độ, quá thời gian) và đo tốc độ thật để ước lượng thời gian crawl.
def _spread(lst, n):
    """n phần tử rải đều trong lst (để chạm nhiều kiểu URL khác nhau của cùng một trang)."""
    if len(lst) <= n:
        return list(lst)
    return [lst[int((j + 0.5) * len(lst) / n)] for j in range(n)]


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


async def probe_async(a):
    os.makedirs(os.path.join(a.out, "html_samples"), exist_ok=True)
    tbl = pq.read_table(a.input).to_pydict()
    by = collections.defaultdict(list)
    for i, u in zip(tbl["id"], tbl["url"]):
        by[urlparse(u).netloc.lower()].append((i, u))
    skip = {h.strip().lower() for h in a.skip_hosts.split(",") if h.strip()}
    only = {h.strip().lower() for h in a.probe_hosts.split(",") if h.strip()}
    def match(h, names):
        return any(h == x or h.endswith("." + x) or h == "www." + x for x in names)
    hosts = [h for h in by if not match(h, skip) and (not only or match(h, only))]
    print(f"Thăm dò {len(hosts)} tên miền: bước 1 = {a.probe} trang/tên miền (gọi nhẹ), "
          f"bước 2 = {a.probe_load} trang/tên miền (nhịp chạy thật)")

    pool = ProcessPoolExecutor(max_workers=a.cpu)
    loop = asyncio.get_running_loop()
    glob = asyncio.Semaphore(a.concurrency)
    pages = collections.defaultdict(list)                 # host -> [dict]
    rng = random.Random(1)

    async def one(session, engine, host, i, u, phase, save, timeout, retries):
        async with glob:
            t = time.time()
            body, st = await fetch(session, engine, u, timeout, retries)
            fetch_sec = time.time() - t                    # chỉ tính thời gian tải, không tính lúc chờ trích chữ
        row = {"host": host, "id": i, "url": u, "phase": phase, "status": st, "fetch_sec": round(fetch_sec, 2),
               "html_bytes": len(body) if body else 0, "text_chars": 0, "title": "", "lang": "", "text": ""}
        if body is not None:
            if save:
                safe = re.sub(r"[^A-Za-z0-9.\-]", "_", host)
                with gzip.open(os.path.join(a.out, "html_samples", f"{safe}__{i}.html.gz"), "wb") as hf:
                    hf.write(body)
            try:
                title, text, lang = await loop.run_in_executor(pool, extract, body, host)
                row.update(title=title, lang=lang, text=text, text_chars=len(text))
                if len(text) < a.min_chars:
                    row["status"] = "empty"
            except Exception as e:
                row["status"] = "err_extract:" + type(e).__name__
        pages[host].append(row)
        return row["status"]

    groups = collections.defaultdict(list)
    rule = {}
    for h in hosts:
        g, r = rate_group(h)
        groups[g].append(h); rule[g] = r

    def settings(g, n_urls):
        r = rule[g]
        if r is None:
            return (a.per_host_big if n_urls > 50_000 else a.per_host), a.delay, a.timeout, a.retries
        return r["workers"], r["delay"], r["timeout"] or a.timeout, a.retries if r["retries"] is None else r["retries"]

    engine, sess_cm = open_session(a)
    t_all = time.time()
    plan = {}                                               # nhóm -> (số kết nối, nghỉ)
    async with sess_cm as session:
        # ---------- bước 1: gọi nhẹ ----------
        async def phase1(g):
            k, d, to, re_ = settings(g, sum(len(by[h]) for h in groups[g]))
            for h in groups[g]:
                for i, u in _spread(by[h], a.probe):
                    await one(session, engine, h, i, u, 1, True, to, min(re_, 1))
                    await asyncio.sleep(max(1.0, d))
        await asyncio.gather(*[phase1(g) for g in groups])
        ok1 = {h: sum(1 for r in pages[h] if r["status"] == "ok") for h in hosts}
        print(f"Bước 1 xong sau {time.time() - t_all:.0f}s: {sum(1 for h in hosts if ok1[h])}/{len(hosts)} tên miền lấy được nội dung")

        # ---------- bước 2: gọi ở nhịp chạy thật ----------
        async def phase2(g):
            live = [h for h in groups[g] if ok1[h] or all(r["status"] in BENIGN for r in pages[h])]
            if not live or a.probe_load <= 0:
                return
            k, d, to, re_ = settings(g, sum(len(by[h]) for h in groups[g]))
            plan[g] = (k, d)
            budget = a.probe_load if rule[g] is None else min(a.probe_load * 3, a.probe_load * len(live))
            per = max(1, budget // len(live))
            q = collections.deque()
            for h in live:
                used = {r["id"] for r in pages[h]}
                rest = [x for x in by[h] if x[0] not in used]
                for i, u in rng.sample(rest, min(per, len(rest))):
                    q.append((h, i, u))
            rng.shuffle(q)
            st = {"consec": 0, "dead": False}

            async def w():
                while q and not st["dead"]:
                    h, i, u = q.popleft()
                    s_ = await one(session, engine, h, i, u, 2, False, to, min(re_, 1))
                    if s_ in ("ok", "empty") or s_ in BENIGN:
                        st["consec"] = 0
                    else:
                        st["consec"] += 3 if s_ == "blocked_captcha" else 1
                        if st["consec"] >= 6:               # đừng gọi tiếp khi trang đã từ chối
                            st["dead"] = True
                    await asyncio.sleep(d * (0.5 + random.random()))
            await asyncio.gather(*[w() for _ in range(k)])
        await asyncio.gather(*[phase2(g) for g in groups])
    pool.shutdown(wait=False, cancel_futures=True)

    # ---------- báo cáo ----------
    import hashlib
    rows = []
    for h in hosts:
        g = rate_group(h)[0]
        p1 = [r for r in pages[h] if r["phase"] == 1]
        p2 = [r for r in pages[h] if r["phase"] == 2]
        allp = pages[h]
        okp = [r for r in allp if r["status"] == "ok"]
        s1 = collections.Counter(r["status"] for r in p1)
        s2 = collections.Counter(r["status"] for r in p2)
        gone = sum(1 for r in allp if r["status"] in BENIGN)
        med_txt = _median([r["text_chars"] for r in okp])
        med_html = _median([r["html_bytes"] for r in okp])
        bad1 = {k: v for k, v in s1.items() if k != "ok" and k not in BENIGN}
        bad2 = {k: v for k, v in s2.items() if k != "ok" and k not in BENIGN}
        dup = max(collections.Counter(hashlib.md5(r["text"].encode()).hexdigest() for r in okp).values(), default=0)
        note = ""
        if not okp:
            if all(k in BLOCKED or k.startswith("blocked_") for k in s1):
                verdict = "BỊ CHẶN"
            elif all(r["status"] in BENIGN for r in allp):
                verdict = "KHÔNG CÒN TRANG"
            elif all(k == "empty" for k in s1):
                verdict = "TRÍCH CHỮ RỖNG"
            else:
                verdict = "LỖI MÁY CHỦ / KẾT NỐI"
            note = json.dumps(dict(s1), ensure_ascii=False)
        elif p2 and sum(bad2.values()) > 0.2 * len(p2):
            verdict, note = "LỖI KHI GỌI DÀY", f"{sum(bad2.values())}/{len(p2)} trang lỗi ở bước 2 {json.dumps(bad2, ensure_ascii=False)}"
        elif bad1 or sum(bad2.values()) > 0.05 * max(1, len(p2)):
            verdict, note = "CHẬP CHỜN", json.dumps({**bad1, **bad2}, ensure_ascii=False)
        elif len(okp) >= 4 and dup >= 0.3 * len(okp):
            verdict, note = "CẦN XEM", f"{dup}/{len(okp)} trang có nội dung giống hệt nhau"
        elif med_txt < 150:
            verdict, note = "CẦN XEM", "chữ trích ra ít"
        else:
            verdict = "OK"
        if gone >= 0.3 * len(allp) and verdict in ("OK", "CHẬP CHỜN", "CẦN XEM"):
            note = (note + " | " if note else "") + f"{gone}/{len(allp)} trang đã gỡ (404 / chuyển về trang chủ)"
        hours = ""
        fs = [r["fetch_sec"] for r in p2 if r["status"] == "ok"]
        if g in plan and fs and verdict in ("OK", "CHẬP CHỜN", "CẦN XEM"):
            k, d = plan[g]
            n_grp = sum(len(by[x]) for x in groups[g])
            hours = f"{n_grp * (sum(fs) / len(fs) + d) / k / 3600:.1f}"   # giờ để crawl hết nhóm trên MỘT máy
        rows.append({"host": h, "url_trong_kho": len(by[h]), "ket_luan": verdict, "ghi_chu": note,
                     "buoc1": f"{s1.get('ok', 0)}/{len(p1)}", "buoc2": f"{s2.get('ok', 0)}/{len(p2)}" if p2 else "",
                     "da_go": gone, "html_trung_vi": med_html, "chu_trung_vi": med_txt, "gio_1_may": hours,
                     "nhom": g if g != h else "", "tieu_de_mau": (okp[0]["title"][:60] if okp else "")})
    rows.sort(key=lambda r: -r["url_trong_kho"])
    cols = list(rows[0].keys())
    with open(os.path.join(a.out, "probe_report.tsv"), "w", encoding="utf-8-sig") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    with gzip.open(os.path.join(a.out, "probe_pages.jsonl.gz"), "wt", encoding="utf-8") as f:
        for h in hosts:
            for r in pages[h]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(r["url_trong_kho"] for r in rows)
    print(f"\nXong sau {(time.time() - t_all) / 60:.1f} phút. Kết quả: {a.out}/probe_report.tsv (mở bằng Excel), "
          f"probe_pages.jsonl.gz (văn bản từng trang), html_samples/ (HTML gốc)\n")
    print(f"{'tên miền':38s} {'URL':>8s}  {'bước 1':>6s} {'bước 2':>7s} {'chữ':>6s}  kết luận")
    for r in rows:
        if r["ket_luan"] != "OK" or r["ghi_chu"]:
            print(f"{r['host'][:38]:38s} {r['url_trong_kho']:8,d}  {r['buoc1']:>6s} {r['buoc2']:>7s} {r['chu_trung_vi']:6d}  "
                  f"{r['ket_luan']}  {r['ghi_chu']}")
    agg = collections.Counter()
    for r in rows:
        agg[r["ket_luan"]] += r["url_trong_kho"]
    print("\nTỷ lệ kho theo kết luận:")
    for k, v in agg.most_common():
        print(f"  {k:28s} {v:10,d} URL  ({100 * v / total:.1f}%)")
    seen, slow = set(), []
    for r in sorted([r for r in rows if r["gio_1_may"]], key=lambda r: -float(r["gio_1_may"])):
        g = rate_group(r["host"])[0]
        if g not in seen:
            seen.add(g); slow.append((g, r["gio_1_may"], plan[g]))
    if slow:
        print("\nƯớc lượng thời gian crawl cho MỘT máy (chia 3 máy thì khoảng 1/3), 5 nhóm lâu nhất:")
        for g, hrs, (k, d) in slow[:5]:
            print(f"  {g:38s} ~{hrs:>6s} giờ   ({k} kết nối, nghỉ {d}s)")


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
    p.add_argument("--probe", type=int, default=0,
                   help="chế độ thăm dò: gọi nhẹ N trang mỗi tên miền rồi in bảng lỗi (không ghi vào progress)")
    p.add_argument("--probe-hosts", default="", help="chỉ thăm dò các tên miền này (cách nhau dấu phẩy)")
    p.add_argument("--slow-scale", type=float, default=1.0,
                   help="nhân nhịp nghỉ của các quy tắc riêng (0.5 = nhanh gấp đôi; chỉ giảm sau khi đã thử bằng --probe)")
    p.add_argument("--probe-load", type=int, default=40,
                   help="thăm dò bước 2: số trang mỗi tên miền gọi ở nhịp chạy thật (0 = bỏ bước này)")
    p.add_argument("--engine", choices=["auto", "curl", "aiohttp"], default="auto",
                   help="curl = curl_cffi (giống trình duyệt, ít bị 403 hơn)")
    p.add_argument("--save-html", type=int, default=3, help="lưu HTML gốc của N trang đầu mỗi tên miền (để kiểm tra bộ trích chữ)")
    p.add_argument("--block-threshold", type=int, default=25,
                   help="dừng một tên miền nếu lỗi từng này URL liên tiếp (bị chặn, máy chủ sập, quá thời gian)")
    p.add_argument("--skip-hosts", default="", help="các tên miền bỏ qua, cách nhau dấu phẩy, vd: zysjonline.com,laodong.vn")
    a = p.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for r in SLOW_RULES:
        r["delay"] *= a.slow_scale
    if a.probe and a.out == "out":
        a.out = "out_probe"
    try:
        asyncio.run(probe_async(a) if a.probe else main_async(a))
    except KeyboardInterrupt:
        print("\nĐã dừng. Chạy lại cùng lệnh để tiếp tục.")


if __name__ == "__main__":
    main()
