"""Tải trang: phiên HTTP giống trình duyệt (curl_cffi), xử lý lỗi, và điều tốc theo tên miền.

Điều tốc thích ứng (HostCtl): mỗi nhóm tên miền bắt đầu với một nửa số kết nối tối đa;
  - đủ 25 trang thành công liên tiếp → tăng 1 kết nối (hoặc giảm nhịp nghỉ nếu trước đó đã phải tăng);
  - 429 / 5xx / quá thời gian / lỗi kết nối → giảm một nửa số kết nối; đã còn 1 thì tăng nhịp nghỉ (tối đa 15 s);
  - bị chặn (403, CAPTCHA, Cloudflare) tính nặng hơn; lỗi liên tiếp quá ngưỡng → tạm dừng nhóm 5, 10, 20 phút;
    quá 3 lần → dừng nhóm trong phiên này. URL chưa thử KHÔNG bị ghi gì nên phiên sau tự chạy tiếp.
Script không vượt CAPTCHA / thử thách Cloudflare: gặp là lùi lại.
"""
import asyncio
import random
import re
import time
from urllib.parse import urlparse

UA_LANG = "vi,zh-CN;q=0.9,zh;q=0.8,en;q=0.7"
FINAL_HTTP = {400, 401, 403, 404, 410, 451}          # không thử lại trong phiên
# Trang xác minh người dùng (CAPTCHA). Script KHÔNG giải CAPTCHA: gặp là lùi lại.
CAPTCHA = re.compile(rb"AJ-Captcha|slideVerify|\xe6\xbb\x91\xe5\x8a\xa8\xe6\x8b\xbc\xe5\x9b\xbe\xe9\xaa\x8c\xe8\xaf\x81|"
                     rb"cf-turnstile|g-recaptcha|h-captcha|wappass\.baidu\.com|\xe7\x99\xbe\xe5\xba\xa6\xe5\xae\x89"
                     rb"\xe5\x85\xa8\xe9\xaa\x8c\xe8\xaf\x81", re.I)    # 滑动拼图验证 / 百度安全验证
JS_COOKIE = re.compile(rb'''document\.cookie\s*=\s*["']([A-Za-z0-9_\-]+)=([^"';]+)["']''')
LOGIN_PATH = re.compile(r"login|signin|sign-in|dang-nhap|passport", re.I)
_CT_CHARSET = re.compile(r"charset=([\w-]+)", re.I)

# Kết quả theo góc nhìn điều tốc
GOOD = {"ok", "empty", "notice"}
BENIGN = {"gone", "not_html", "http_400", "http_404", "http_410", "err_too_big", "err_extract"}
BLOCK_PREFIX = ("blocked_", "http_401", "http_403", "http_451")

# Quy tắc tốc độ riêng: gộp mọi tên miền con vào MỘT nhóm với số kết nối / nhịp nghỉ / thời gian chờ riêng.
RATE_RULES = [
    # Các kênh tin 39.net: gọi dày là bật CAPTCHA kéo ghép hình → 1 kết nối, nghỉ lâu (đã thử 120 lần không bị chặn).
    {"suffix": "39.net", "exclude": {"ask.39.net", "jbk.39.net"}, "group": "39.net (kênh tin)",
     "workers": 1, "delay": 2.0},
    {"suffix": "ask.39.net", "group": "ask.39.net", "workers": 3, "delay": 0.8},
    # youlai.cn: khoảng một nửa số trang treo tới hết thời gian chờ, không phụ thuộc tốc độ gọi → chờ ngắn, không thử
    # lại ngay (để phiên sau), và không giảm tốc vì quá thời gian (vẫn lùi lại nếu bị chặn).
    {"suffix": "youlai.cn", "group": "youlai.cn", "workers": 4, "delay": 0.3, "timeout": 8, "retries": 0,
     "backoff_on_timeout": False},
    {"suffix": "qihuangzhishu.com", "group": "qihuangzhishu.com", "workers": 1, "delay": 1.0, "retries": 1},
    # nhathuoclongchau: Cloudflare cấm IP ~13 giờ (429 + Retry-After ~47.000 s) sau khoảng 500 trang gọi ở ~0,6 trang/s
    # (cả IP Kaggle lẫn IP Việt Nam, 12/10/2026) → 1 kết nối, nghỉ ~8 s; điều tốc không được tự tăng.
    {"suffix": "nhathuoclongchau.com.vn", "group": "nhathuoclongchau.com.vn", "workers": 1, "delay": 8.0},
    # dantri: 429 rồi từ chối kết nối khi gọi dày (cả Kaggle lẫn máy nhà).
    {"suffix": "dantri.com.vn", "group": "dantri.com.vn", "workers": 1, "delay": 4.0},
    {"suffix": "baidu.com", "group": "baidu.com", "workers": 1, "delay": 3.0},
]


def rate_group(host):
    """(tên nhóm, quy tắc hoặc None)."""
    for r in RATE_RULES:
        suf = r["suffix"]
        if host not in r.get("exclude", ()) and (host == suf or host.endswith("." + suf)):
            return r["group"], r
    return host, None


def outcome(status):
    if status in GOOD:
        return "good"
    if status in BENIGN:
        return "benign"
    if status.startswith(BLOCK_PREFIX):
        return "block"
    return "throttle"                                       # 429, 5xx, quá thời gian, lỗi kết nối


class HostCtl:
    def __init__(self, name, max_conns, delay, timeout, retries, fail_threshold=12, cooldown=300, max_strikes=3,
                 backoff_on_timeout=True):
        self.name, self.max = name, max(1, max_conns)
        self.backoff_on_timeout = backoff_on_timeout
        self.limit = max(1, (self.max + 1) // 2)
        self.base_delay = self.delay = delay
        self.timeout, self.retries = timeout, retries
        self.fail_threshold, self.cooldown, self.max_strikes = fail_threshold, cooldown, max_strikes
        self.inflight = self.ok_streak = self.fail_streak = self.strikes = 0
        self.pause_until = 0.0
        self.dead = False
        self.last_bad = ""

    async def acquire(self, stopped):
        while True:
            if self.dead or stopped():
                return False
            now = time.monotonic()
            if now < self.pause_until:
                await asyncio.sleep(min(5.0, self.pause_until - now))
                continue
            if self.inflight < self.limit:
                self.inflight += 1
                return True
            await asyncio.sleep(0.2 + random.random() * 0.2)

    def ban(self, seconds, budget_left, log=print):
        """Trang báo cấm trong `seconds` giây (429 + Retry-After): nghỉ đúng chừng đó, quá thời gian còn lại → bỏ nhóm."""
        if self.dead or time.monotonic() < self.pause_until:
            return
        if budget_left is not None and seconds >= budget_left:
            self.dead = True
            log(f"[DỪNG] {self.name}: trang báo cấm {seconds / 3600:.1f} giờ (Retry-After) — quá thời gian còn lại "
                f"của phiên; URL chưa thử để phiên sau.")
        else:
            self.pause_until = time.monotonic() + seconds
            self.limit = 1
            log(f"[TẠM DỪNG] {self.name}: trang báo cấm {seconds / 60:.0f} phút (Retry-After) → nghỉ đúng chừng đó.")

    def release(self, status, log=print):
        self.inflight -= 1
        self.feedback(status, log)

    def feedback(self, status, log=print):
        """Điều chỉnh tốc độ theo kết quả một trang (gọi thêm được sau khi trích, vd trang xác minh lọt qua)."""
        o = outcome(status)
        if o in ("good", "benign"):
            self.fail_streak = 0
            self.ok_streak += 1
            if self.ok_streak >= 25:
                self.ok_streak = 0
                if self.delay > self.base_delay:
                    self.delay = max(self.base_delay, self.delay / 1.5)
                elif self.limit < self.max:
                    self.limit += 1
            return
        if o == "throttle" and status == "err_timeout" and not self.backoff_on_timeout:
            return
        self.ok_streak = 0
        self.last_bad = status
        self.fail_streak += 4 if (o == "block" or status == "blocked_captcha") else 1
        if self.limit > 1:
            self.limit = max(1, self.limit // 2)
        else:
            self.delay = min(15.0, max(self.delay * 2, 1.0))
        if self.fail_streak >= self.fail_threshold:
            self.fail_streak = 0
            self.strikes += 1
            if self.strikes > self.max_strikes:
                self.dead = True
                log(f"[DỪNG] {self.name}: lỗi liên tiếp ({status}) sau {self.max_strikes} lần tạm dừng → bỏ nhóm này "
                    f"trong phiên; URL chưa thử để phiên sau.")
            else:
                wait = self.cooldown * 2 ** (self.strikes - 1)
                self.pause_until = time.monotonic() + wait
                self.limit = 1
                log(f"[TẠM DỪNG] {self.name}: lỗi liên tiếp ({status}) → nghỉ {wait // 60:.0f} phút (lần {self.strikes}).")


# ---------------------------------------------------------------- phiên HTTP
def open_session(concurrency):
    """curl_cffi giả lập Chrome (vân tay TLS/HTTP2 giống trình duyệt → ít bị 403 hơn hẳn aiohttp)."""
    from curl_cffi.requests import AsyncSession
    return AsyncSession(impersonate="chrome", max_clients=concurrency,
                        headers={"Accept-Language": UA_LANG})


HTTP1_HOSTS = set()      # tên miền từng báo lỗi khung HTTP/2 (curl 16/92) → từ đó gọi bằng HTTP/1.1
INSECURE_HOSTS = set()   # tên miền có chứng chỉ TLS lỗi (curl 60) → từ đó bỏ kiểm tra chứng chỉ


async def _get(session, url, timeout):
    host = urlparse(url).netloc.lower()
    kw = {}
    if host in HTTP1_HOSTS:
        from curl_cffi.const import CurlHttpVersion
        kw["http_version"] = CurlHttpVersion.V1_1
    if host in INSECURE_HOSTS:
        kw["verify"] = False
    r = await session.get(url, timeout=timeout, allow_redirects=True, max_redirects=6, **kw)
    return r.status_code, r.headers, r.content, str(r.url)


def set_cookie(session, url, name, value):
    session.cookies.set(name, value, domain=urlparse(url).hostname, path="/")


class Fetched:
    __slots__ = ("body", "status", "final_url", "charset", "secs", "retry_after")

    def __init__(self, body, status, final_url="", charset=None, secs=0.0, retry_after=0):
        self.body, self.status, self.final_url, self.charset, self.secs = body, status, final_url, charset, secs
        self.retry_after = retry_after          # giây, khi trang trả 429 kèm Retry-After


async def fetch(session, url, timeout=25, retries=2, max_bytes=8_000_000):
    """Tải một URL → Fetched(body hoặc None, trạng thái, URL cuối, charset ở header, thời gian)."""
    t0 = time.monotonic()
    last, final = "err_unknown", url
    cookie_tried = h2_tried = tls_tried = False
    attempt = -1
    while attempt < retries:
        attempt += 1
        try:
            code, hdr, body, final = await _get(session, url, timeout)
            if hdr.get("cf-mitigated"):                     # Cloudflare đang đòi thử thách (challenge)
                return Fetched(None, "blocked_cf", final, None, time.monotonic() - t0)
            if code == 200:
                if len(body) > max_bytes:
                    return Fetched(None, "err_too_big", final, None, time.monotonic() - t0)
                if len(body) < 600:
                    # Trang nhỏ chỉ để đặt cookie rồi tải lại (vd laodong.vn):
                    #   <script>document.cookie="D1N=…";window.location.reload(true);</script>
                    m = JS_COOKIE.search(body)
                    if m and not cookie_tried:
                        cookie_tried = True
                        set_cookie(session, url, m.group(1).decode(), m.group(2).decode())
                        attempt -= 1                         # lần gọi lại này không tính là một lần thử lỗi
                        continue
                    return Fetched(None, "blocked_tiny", final, None, time.monotonic() - t0)
                if len(body) < 20000 and CAPTCHA.search(body):
                    return Fetched(None, "blocked_captcha", final, None, time.monotonic() - t0)
                ctype = str(hdr.get("content-type", "")).lower()
                if (ctype and not any(x in ctype for x in ("html", "xml", "text/plain"))) \
                        or body[:3] == b"\xff\xd8\xff" or body[:4] in (b"%PDF", b"\x89PNG", b"GIF8"):
                    return Fetched(None, "not_html", final, None, time.monotonic() - t0)
                o, f = urlparse(url), urlparse(final)
                if o.path.strip("/") and f.path != o.path and (not f.path.strip("/") or LOGIN_PATH.search(f.path)):
                    return Fetched(None, "gone", final, None, time.monotonic() - t0)   # bài đã gỡ → trang chủ/đăng nhập
                m = _CT_CHARSET.search(ctype)
                return Fetched(body, "ok", final, m.group(1) if m else None, time.monotonic() - t0)
            if code in FINAL_HTTP:
                return Fetched(None, f"http_{code}", final, None, time.monotonic() - t0)
            last = f"http_{code}"
            if code == 429:
                ra = str(hdr.get("Retry-After", "")).strip()
                if ra.isdigit() and int(ra) > 120:      # bị cấm dài (Cloudflare longchau: ~13 giờ) → không thử lại
                    return Fetched(None, "http_429", final, None, time.monotonic() - t0, int(ra))
                await asyncio.sleep(int(ra) if ra.isdigit() else 10 * (attempt + 1))
        except asyncio.TimeoutError:
            last = "err_timeout"
        except Exception as e:
            code = getattr(e, "code", None)                  # curl_cffi: mã lỗi libcurl (28 quá giờ, 56 đứt kết nối…)
            code = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else None
            last = "err_timeout" if code == 28 else f"err_{type(e).__name__}" + (f"_curl{code}" if code else "")
            host = urlparse(url).netloc.lower()
            if code in (16, 92) and not h2_tried:
                # máy chủ trả khung HTTP/2 hỏng: chuyển tên miền sang HTTP/1.1 và gọi lại ngay, không tính là lần thử
                HTTP1_HOSTS.add(host)
                h2_tried = True
                attempt -= 1
                continue
            if code == 60 and not tls_tried:                 # chứng chỉ TLS lỗi (hay gặp ở trang .gov.vn cũ)
                INSECURE_HOSTS.add(host)
                tls_tried = True
                attempt -= 1
                continue
        if attempt < retries:
            await asyncio.sleep(2 ** attempt + random.random())
    return Fetched(None, last, final, None, time.monotonic() - t0)
