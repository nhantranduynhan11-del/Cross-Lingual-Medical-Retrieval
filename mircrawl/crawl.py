"""Vòng crawl chính (asyncio).

- Trang web: chia theo nhóm tên miền (fetch.rate_group), mỗi nhóm có số kết nối / nhịp nghỉ riêng và điều tốc thích ứng.
- PubMed: lấy theo lô qua E-utilities (pubmed.py).
- Trích nội dung chạy ở các process con (ProcessPoolExecutor) để tận dụng mọi lõi CPU.
- Kết quả ghi thành file parquet theo phần (store.py); dừng êm trước khi hết --time-budget (phiên Kaggle tối đa 12 giờ).
- Chạy lại = chạy tiếp: bỏ qua id đã xong ở out/ và ở các thư mục --resume (output của phiên trước).
"""
import asyncio
import collections
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from urllib.parse import urlparse

from . import XV, __version__
from .extract import extract_job
from .fetch import Fetched, HostCtl, fetch, open_session, rate_group
from .pubmed import RateLimiter, fetch_batch, parse_efetch, pmid_of
from .store import PAGE_SCHEMA, RAW_SCHEMA, PartWriter, carry_over, find_parts, now_iso, read_state

PUBMED_HOST = "pubmed.ncbi.nlm.nih.gov"
# Trang "thông báo" thực chất là trang chặn / xác minh → báo cho bộ điều tốc lùi lại.
_BLOCKISH = re.compile(r"验证|captcha|just a moment|checking your browser|verify you are|access denied|网络不给力",
                       re.I)


def log(*args):
    print(*args, flush=True)


def host_of(url):
    return urlparse(url).netloc.lower()


def _match(h, names):
    return any(h == x or h.endswith("." + x) for x in names)


def load_links(path):
    """[(id, url)] từ file parquet (cột id, url) hoặc JSONL ({"id":…, "url":…} mỗi dòng)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=["id", "url"])
        return list(zip(t.column("id").to_pylist(), t.column("url").to_pylist()))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out.append((int(r["id"]), r["url"]))
    return out


def parse_shard(s):
    si, sn = (int(x) for x in s.split("/"))
    if not 0 <= si < sn:
        raise ValueError(f"--shard {s}: cần 0 <= i < N")
    return si, sn


def _spread(lst, n):
    if len(lst) <= n:
        return list(lst)
    return [lst[int((j + 0.5) * len(lst) / n)] for j in range(n)]


def select_items(a, links):
    si, sn = parse_shard(a.shard)
    items = [(i, u) for i, u in links if i % sn == si]
    only = {h.strip().lower() for h in (a.hosts or "").split(",") if h.strip()}
    skip = {h.strip().lower() for h in (a.skip_hosts or "").split(",") if h.strip()}
    if only:
        items = [(i, u) for i, u in items if _match(host_of(u), only)]
    if skip:
        items = [(i, u) for i, u in items if not _match(host_of(u), skip)]
    if a.probe:
        by = collections.defaultdict(list)
        for i, u in sorted(items):
            by[host_of(u)].append((i, u))
        items = []
        for h, lst in by.items():
            items += _spread(lst, a.probe * 20 if h == PUBMED_HOST else a.probe)
    return items, f"s{si}of{sn}"


class Pool:
    """ProcessPoolExecutor tự tạo lại nếu một process con chết (vd thiếu bộ nhớ)."""

    def __init__(self, n):
        self.n, self.ex = n, None
        self.reset()

    def reset(self):
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)
        import multiprocessing as mp
        try:   # thay process con định kỳ để tránh phình bộ nhớ khi chạy hàng trăm nghìn trang
            self.ex = ProcessPoolExecutor(self.n, mp_context=mp.get_context("spawn"), max_tasks_per_child=2000)
        except TypeError:                                   # Python < 3.11
            self.ex = ProcessPoolExecutor(self.n)

    async def run(self, fn, *args):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.ex, fn, *args)
        except BrokenProcessPool:
            log("[CẢNH BÁO] process trích nội dung bị hỏng → tạo lại pool")
            self.reset()
            return await loop.run_in_executor(self.ex, fn, *args)

    def shutdown(self):
        self.ex.shutdown(wait=False, cancel_futures=True)


async def crawl_async(a):
    t0 = time.monotonic()
    deadline = t0 + a.time_budget * 3600 if a.time_budget else None
    links = load_links(a.links)
    items, tag = select_items(a, links)
    session_id = f"{tag}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    os.makedirs(a.out, exist_ok=True)
    resume = [d for d in (a.resume or []) if d]
    if resume and not a.no_carry:
        n = carry_over(resume, a.out, tag)
        if n:
            log(f"Chép {n} file của các phiên trước (shard {tag}) từ {resume} → {a.out}")
    files = {}
    for f in find_parts([a.out] + resume, "pages"):         # cùng tên file (đã chép sang out/) chỉ đọc một lần
        files.setdefault(os.path.basename(f), f)
    ids = {i for i, _ in items}
    state = read_state(sorted(files.values()), ids)
    n_done = sum(1 for i in ids if state.get(i, (False, 0))[0])
    gave_up = sum(1 for i in ids if not state.get(i, (False, 0))[0] and state.get(i, (False, 0))[1] >= a.max_attempts)
    todo = [(i, u) for i, u in items
            if not state.get(i, (False, 0))[0] and state.get(i, (False, 0))[1] < a.max_attempts]
    if a.limit:
        random.Random(0).shuffle(todo)
        todo = todo[:a.limit]
    log(f"mircrawl {__version__} (xv={XV}) | phiên {session_id} | shard {a.shard}: {len(items):,} URL, "
        f"đã xong {n_done:,}, bỏ cuộc (đủ {a.max_attempts} lần lỗi) {gave_up:,}, cần làm {len(todo):,}")
    if not todo:
        return session_id
    web = [(i, u, host_of(u)) for i, u in todo if host_of(u) != PUBMED_HOST or not pmid_of(u)]
    pm = [(i, u, pmid_of(u)) for i, u in todo if host_of(u) == PUBMED_HOST and pmid_of(u)]

    pages_w = PartWriter(os.path.join(a.out, "pages"), session_id, PAGE_SCHEMA, a.part_rows, a.flush_secs)
    raw_w = PartWriter(os.path.join(a.out, "raw"), session_id, RAW_SCHEMA, a.part_rows, a.flush_secs) \
        if a.save_raw else None
    writers = [w for w in (pages_w, raw_w) if w]
    stats = collections.Counter()
    hstats = collections.defaultdict(collections.Counter)
    pool = Pool(a.workers)
    loop = asyncio.get_running_loop()
    stop = {"flag": False}

    def stopped():
        return stop["flag"] or (deadline is not None and time.monotonic() > deadline - a.grace)

    def emit(i, u, host, rec, raw=None):
        row = dict.fromkeys(PAGE_SCHEMA.names)
        row.update(rec)
        row.update(id=i, url=u, host=host, processed_at=now_iso(), session=session_id)
        row["xv"] = row["xv"] or XV
        pages_w.add(row)
        stats[row["status"]] += 1
        hstats[host][row["status"]] += 1
        if raw is not None and raw_w is not None:
            raw_w.add(raw)

    # ---------------------------------------------------------------- trang web
    groups, rules = collections.defaultdict(list), {}
    for i, u, h in web:
        g, r = rate_group(h)
        groups[g].append((i, u, h))
        rules[g] = r
    queues, ctls = {}, {}
    for g, lst in groups.items():
        random.Random(0).shuffle(lst)
        r = rules[g]
        if r:
            k, d = r["workers"], r["delay"] * a.slow_scale
            to, rt = r.get("timeout") or a.timeout, r.get("retries", a.retries)
        else:
            k, d, to, rt = (a.per_host_big if len(lst) >= a.big_host else a.per_host), a.delay, a.timeout, a.retries
        ctls[g] = HostCtl(g, k, d, to, rt, a.fail_threshold, a.cooldown,
                          backoff_on_timeout=(r or {}).get("backoff_on_timeout", True))
        queues[g] = collections.deque(lst)
    for g in sorted(queues, key=lambda g: -len(queues[g]))[:8]:
        log(f"  nhóm {g}: {len(queues[g]):,} URL, tối đa {ctls[g].max} kết nối, nghỉ {ctls[g].base_delay}s")
    glob_sem = asyncio.Semaphore(a.concurrency)

    async def web_worker(session, g):
        ctl, q = ctls[g], queues[g]
        while q and not stopped():
            if not await ctl.acquire(stopped):
                return
            if not q:
                ctl.inflight -= 1
                return
            i, u, h = q.popleft()
            try:
                async with glob_sem:
                    f = await fetch(session, u, ctl.timeout, ctl.retries)
            except Exception as e:                          # không để mất "chỗ" kết nối của nhóm
                f = Fetched(None, "err_" + type(e).__name__, u)
            ctl.release(f.status, log)
            if f.retry_after:
                ctl.ban(f.retry_after, deadline - time.monotonic() if deadline else None, log)
            fetched_at = now_iso()
            if f.body is None:
                emit(i, u, h, {"status": f.status, "final_url": f.final_url, "fetched_at": fetched_at})
            else:
                try:
                    rec, raw = await pool.run(extract_job, f.body, h, u, f.charset, a.save_raw, a.min_chars,
                                              a.min_chars_zh)
                except Exception as e:
                    rec, raw = {"status": "err_extract", "extractor": f"{type(e).__name__}: {str(e)[:200]}"}, None
                if rec["status"] == "notice" and _BLOCKISH.search((rec.get("title") or "") + (rec.get("text") or "")):
                    rec["status"] = "blocked_notice"               # trang xác minh: không coi là xong, phiên sau thử lại
                    ctl.feedback("blocked_notice", log)
                rec.update(final_url=f.final_url, fetched_at=fetched_at)
                emit(i, u, h, rec, {"id": i, "url": u, "final_url": f.final_url, "host": h, "charset": f.charset,
                                    "fetched_at": fetched_at, "html": raw} if raw is not None else None)
            await asyncio.sleep(ctl.delay * (0.5 + random.random()))

    # ---------------------------------------------------------------- PubMed
    pm_q = collections.deque(pm[k:k + a.pubmed_batch] for k in range(0, len(pm), a.pubmed_batch))
    api_key = os.environ.get("NCBI_API_KEY", "")
    limiter = RateLimiter(a.pubmed_rps or (9 if api_key else 2.8))
    if pm:
        log(f"  PubMed: {len(pm):,} PMID → {len(pm_q):,} lô × {a.pubmed_batch}, "
            f"{'có' if api_key else 'không có'} NCBI_API_KEY")

    async def pubmed_worker(session):
        while pm_q and not stopped():
            batch = pm_q.popleft()
            xml, st = await fetch_batch(session, [p for _, _, p in batch], limiter, api_key, a.ncbi_email)
            fetched_at = now_iso()
            recs = {}
            if xml is not None:
                try:
                    recs = await pool.run(parse_efetch, xml)
                except Exception as e:
                    xml, st = None, "err_parse_" + type(e).__name__
            for i, u, p in batch:
                if xml is None:
                    rec = {"status": st}
                else:
                    rec = dict(recs.get(p) or {"status": "pm_missing"})
                rec.update(final_url=u, fetched_at=fetched_at)
                emit(i, u, PUBMED_HOST, rec)

    # ---------------------------------------------------------------- báo cáo + ghi định kỳ
    total = len(todo)

    async def flush(force=False):
        for w in writers:
            if force or w.due():
                await loop.run_in_executor(None, w.write, w.take())

    async def reporter():
        while True:
            await asyncio.sleep(a.report_secs)
            await flush()
            n = sum(stats.values())
            el = time.monotonic() - t0
            rate = n / max(el, 1e-9)
            left = sum(len(q) for q in queues.values()) + sum(len(b) for b in pm_q)
            good = stats["ok"] + stats["empty"] + stats["notice"]
            log(f"[{el / 60:6.1f} phút] {n:,}/{total:,} ({100 * n / max(total, 1):.1f}%) {rate:.1f} trang/s | "
                f"ok={stats['ok']:,} rỗng={stats['empty']:,} thông báo={stats['notice']:,} lỗi={n - good:,} | "
                f"còn {left:,} (~{left / max(rate, 1e-9) / 3600:.1f} giờ theo tốc độ hiện tại)")
            busy = sorted((g for g in queues if queues[g]), key=lambda g: -len(queues[g]))[:5]
            if busy:
                log("    còn nhiều nhất: " + " | ".join(
                    f"{g} {len(queues[g]):,} [{ctls[g].limit}/{ctls[g].max} kết nối, nghỉ {ctls[g].delay:.1f}s"
                    f"{', TẠM DỪNG' if time.monotonic() < ctls[g].pause_until else ''}"
                    f"{', ĐÃ DỪNG' if ctls[g].dead else ''}]" for g in busy))

    async def guard(coro, name):
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:                              # một worker hỏng không được làm dừng cả phiên
            log(f"[LỖI] worker {name}: {type(e).__name__}: {e}")

    session = open_session(a.concurrency + 8)
    rep = asyncio.create_task(reporter())
    try:
        async with session:
            tasks = [asyncio.create_task(guard(web_worker(session, g), g)) for g in queues for _ in range(ctls[g].max)]
            tasks += [asyncio.create_task(guard(pubmed_worker(session), "pubmed")) for _ in range(min(3, len(pm_q)))]
            await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        stop["flag"] = True
        log("Đang dừng…")
    finally:
        rep.cancel()
        await flush(force=True)
        pool.shutdown()
        left = sum(len(q) for q in queues.values()) + sum(len(b) for b in pm_q)
        summary = {"session": session_id, "version": __version__, "xv": XV, "shard": a.shard,
                   "minutes": round((time.monotonic() - t0) / 60, 1), "attempted": sum(stats.values()),
                   "left_for_next_session": left, "status": dict(stats),
                   "hosts": {h: dict(c) for h, c in sorted(hstats.items())},
                   "dead_groups": [g for g, c in ctls.items() if c.dead], "args": vars(a)}
        os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
        with open(os.path.join(a.out, "runs", f"{session_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
        log(f"Xong phiên {session_id} sau {summary['minutes']} phút: {dict(stats)}")
        if left:
            log(f"Còn {left:,} URL chưa thử (hết giờ hoặc nhóm bị dừng) → chạy phiên tiếp với --resume trỏ tới output này.")
        bad = sorted(((h, c) for h, c in hstats.items() if sum(c.values()) - c["ok"] > 0.3 * sum(c.values())),
                     key=lambda x: -sum(x[1].values()))[:15]
        if bad:
            log("Tên miền nhiều lỗi (>30% không ok):")
            for h, c in bad:
                log(f"  {h:38s} {c['ok']:,}/{sum(c.values()):,} ok  {dict(c.most_common(4))}")
    return session_id


def run(a):
    if sys.platform == "win32":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        return asyncio.run(crawl_async(a))
    except KeyboardInterrupt:
        log("Đã dừng. Chạy lại cùng lệnh để tiếp tục.")
