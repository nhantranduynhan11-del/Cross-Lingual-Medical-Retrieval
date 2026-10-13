"""Tài liệu tiếng Anh: URL https://pubmed.ncbi.nlm.nih.gov/<PMID>/ → tiêu đề + tóm tắt qua NCBI E-utilities.

Không tải trang HTML của PubMed (chậm, dễ bị chặn): gọi efetch theo lô (mặc định 200 PMID / yêu cầu, POST),
tôn trọng giới hạn của NCBI: 3 yêu cầu/giây, 10 yêu cầu/giây nếu có API key (biến môi trường NCBI_API_KEY).
240k PMID ≈ 1.200 yêu cầu ≈ 10–20 phút.
"""
import asyncio
import json
import random
import re
import time

import lxml.etree

from . import XV
from .extract import text_hash
from .htmlx import clean_lines, to_plain

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMID_RX = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_MON = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                         "dec"])}


def pmid_of(url):
    m = PMID_RX.search(url or "")
    return m.group(1) if m else None


class RateLimiter:
    """Không quá `rps` yêu cầu mỗi giây (tính theo thời điểm bắt đầu yêu cầu)."""

    def __init__(self, rps):
        self.gap, self.next, self.lock = 1.0 / rps, 0.0, asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            if now < self.next:
                await asyncio.sleep(self.next - now)
            self.next = max(now, self.next) + self.gap


async def fetch_batch(session, pmids, limiter, api_key="", email="", timeout=120, retries=4):
    """→ (bytes XML hoặc None, trạng thái)."""
    data = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": "mircrawl"}
    if api_key:
        data["api_key"] = api_key
    if email:
        data["email"] = email
    last = "err_unknown"
    for attempt in range(retries + 1):
        await limiter.wait()
        try:
            r = await session.post(EFETCH, data=data, timeout=timeout)
            if r.status_code == 200 and b"<PubmedArticleSet" in r.content[:2000]:
                return r.content, "ok"
            last = f"http_{r.status_code}" if r.status_code != 200 else "err_bad_xml"
        except Exception as e:
            last = "err_" + type(e).__name__
        await asyncio.sleep(min(60, 2 ** attempt * 3) + random.random())
    return None, last


# ---------------------------------------------------------------- phân tích XML
def _txt(el):
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _date(el):
    """<PubDate>/<ArticleDate>/<DateRevised> → 'YYYY-MM-DD' | 'YYYY-MM' | 'YYYY' | ''."""
    if el is None:
        return ""
    y = (el.findtext("Year") or "").strip()
    if not y:
        m = re.search(r"((?:19|20)\d{2})(?:\s+([A-Za-z]{3}))?", el.findtext("MedlineDate") or "")
        if not m:
            return ""
        y, mon = m.group(1), _MON.get((m.group(2) or "").lower()[:3])
        return f"{y}-{mon:02d}" if mon else y
    mo = (el.findtext("Month") or "").strip()
    mo = int(mo) if mo.isdigit() else _MON.get(mo.lower()[:3])
    d = (el.findtext("Day") or "").strip()
    if not mo:
        return y
    if not d.isdigit():
        return f"{y}-{mo:02d}"
    return f"{y}-{mo:02d}-{int(d):02d}"


def _authors(parent, limit=50):
    names = []
    for a in parent.findall("AuthorList/Author") if parent is not None else []:
        if a.get("ValidYN") == "N":
            continue
        n = _txt(a.find("CollectiveName")) or " ".join(
            x for x in ((a.findtext("ForeName") or a.findtext("Initials") or "").strip(),
                        (a.findtext("LastName") or "").strip()) if x)
        if n:
            names.append(n)
    s = "; ".join(names[:limit])
    return s + ("; et al." if len(names) > limit else "")


def _abstract_lines(abs_el):
    """Tóm tắt có cấu trúc → '## BACKGROUND' + đoạn, '## METHODS' + đoạn …; không cấu trúc → các đoạn."""
    lines = []
    for at in abs_el.findall("AbstractText") if abs_el is not None else []:
        label = (at.get("Label") or "").strip()
        body = _txt(at)
        if label and label.upper() not in ("UNLABELLED", "UNASSIGNED"):
            lines.append("## " + label)
        if body:
            lines.append(body)
    return clean_lines("\n".join(lines))


def parse_efetch(xml_bytes):
    """XML của efetch → {pmid: bản ghi (chưa có id/url/thời gian)}."""
    parser = lxml.etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)
    root = lxml.etree.fromstring(xml_bytes, parser)
    out = {}
    if root is None:
        return out
    for art in root:
        if art.tag == "PubmedArticle":
            mc = art.find("MedlineCitation")
            a = mc.find("Article") if mc is not None else None
            if a is None:
                continue
            pmid = (mc.findtext("PMID") or "").strip()
            title = _txt(a.find("ArticleTitle")) or _txt(a.find("VernacularTitle"))
            lines = _abstract_lines(a.find("Abstract"))
            journal = _txt(a.find("Journal/Title"))
            iso = _txt(a.find("Journal/ISOAbbreviation"))
            pub = _date(a.find("Journal/JournalIssue/PubDate"))
            epub = _date(a.find("ArticleDate"))
            published = pub if len(pub) == 10 else (epub or pub)
            modified = _date(mc.find("DateRevised"))
            authors = _authors(a)
            langs = [x.text for x in a.findall("Language") if x.text]
            pubtypes = [_txt(x) for x in a.findall("PublicationTypeList/PublicationType")]
            mesh = [_txt(x) for x in mc.findall("MeshHeadingList/MeshHeading/DescriptorName")]
            kws = [_txt(x) for x in mc.findall("KeywordList/Keyword")]
            ids = {x.get("IdType"): (x.text or "").strip() for x in art.findall("PubmedData/ArticleIdList/ArticleId")}
        elif art.tag == "PubmedBookArticle":
            bd = art.find("BookDocument")
            if bd is None:
                continue
            pmid = (bd.findtext("PMID") or "").strip()
            title = _txt(bd.find("ArticleTitle")) or _txt(bd.find("Book/BookTitle"))
            lines = _abstract_lines(bd.find("Abstract"))
            journal = _txt(bd.find("Book/BookTitle"))
            iso = ""
            published = _date(bd.find("Book/PubDate"))
            modified = _date(bd.find("DateRevised"))
            authors = _authors(bd) or _authors(bd.find("Book"))
            langs = [x.text for x in bd.findall("Language") if x.text]
            pubtypes = [_txt(x) for x in bd.findall("PublicationType")]
            mesh, kws = [], [_txt(x) for x in bd.findall("KeywordList/Keyword")]
            ids = {x.get("IdType"): (x.text or "").strip() for x in bd.findall("ArticleIdList/ArticleId")}
        else:
            continue
        if not pmid:
            continue
        text = "\n".join(lines)
        plain = to_plain(text)
        extra = {"pmid": pmid, "journal": journal, "journal_iso": iso, "doi": ids.get("doi", ""),
                 "pmcid": ids.get("pmc", ""), "pub_types": pubtypes, "mesh": mesh,
                 "language": ",".join(langs)}
        out[pmid] = {
            "status": "ok" if plain else "empty",           # chỉ có tiêu đề, không có tóm tắt → "empty" (vẫn giữ tiêu đề)
            "lang": "en", "doc_type": "paper", "title": title, "text": text, "description": "",
            "author": authors, "published": published, "modified": modified, "section": journal,
            "keywords": "; ".join(dict.fromkeys(k for k in kws if k)), "site_name": "PubMed",
            "extra": json.dumps({k: v for k, v in extra.items() if v}, ensure_ascii=False),
            "extractor": "pubmed-efetch", "xv": XV, "n_chars": len(plain),
            "text_hash": text_hash(plain) if plain else "",
        }
    return out
