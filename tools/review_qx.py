"""Rà soát tự động đầu ra mirret/query_expand.py (records*.jsonl) — đánh dấu lượt nghi lỗi để đọc tay.

  python tools/review_qx.py --records "results/mir2-qx-full/qexp/rw2-qwen25_14b/records*.jsonl" [--out results/mir2-qx-full]
  → <out>/review.md (bảng tỉ lệ + ví dụ), <out>/flagged.csv (mỗi dòng một lỗi: qid, lang, kiểm tra, chi tiết, base,
    rewrite, terms)

A. cấu trúc: đủ 1.200 câu × 3 ngôn ngữ, lỗi theo why, đầu ra thô bị cắt (chạm max_new_tokens)
B. rewrite: số lạ (không có trong câu gốc), thêm tuổi / giới / thai kỳ, mất ý (ít từ nội dung của câu gốc), độ dài
C. terms: từ mới hay chép câu, từ chung chung, câu hỏi chẩn đoán ("bệnh gì") → term mới để đọc tay, term gần giống một
   cụm trong câu gốc (lỗi chính tả, vd "hẹn họng"), sai chữ viết, độ phủ bảng thuật ngữ
"""
import argparse
import csv
import difflib
import glob
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import bm25, config, translate  # noqa: E402

LANGS = ("vi", "en", "zh")
NUM = re.compile(r"\d+(?:[.,]\d+)?")
DEMO = {
    "vi": [r"\d+\s*tuổi", r"\bnam giới\b", r"\bnữ giới\b", r"\bphụ nữ\b", r"\bđàn ông\b", r"\bbé trai\b",
           r"\bbé gái\b", r"\bmang thai\b", r"\bcó thai\b", r"\bthai phụ\b", r"\bngười cao tuổi\b", r"\bsơ sinh\b"],
    "en": [r"year-old", r"years? old", r"\bmale\b", r"\bfemale\b", r"\bman\b", r"\bwoman\b", r"\bboy\b", r"\bgirl\b",
           r"pregnan", r"\belderly\b", r"\bnewborn\b", r"\binfant\b"],
    "zh": [r"\d+岁", r"男性", r"女性", r"男孩", r"女孩", r"孕", r"老年", r"新生儿", r"婴儿"],
}
GENERIC = {"thuốc", "bệnh", "triệu chứng", "nguyên nhân", "điều trị", "chữa trị", "cách chữa", "bác sĩ", "sức khỏe",
           "medication", "medicine", "drug", "drugs", "disease", "symptom", "symptoms", "cause", "causes", "treatment",
           "doctor", "health", "药物", "药", "疾病", "病", "症状", "原因", "治疗", "医生", "健康"}
DIAG_Q = {
    "vi": re.compile(r"bệnh gì|bị gì|là bệnh|bị bệnh|có phải (là )?(bị|bệnh)|bị làm sao|là sao|dấu hiệu (của )?gì|"
                     r"có nguy hiểm|là bị", re.I),
    "en": re.compile(r"what (illness|disease|condition|is wrong|could (this|it) be|might)|is (this|it) (a sign|"
                     r"a symptom|serious)|could (this|it) be|do i have", re.I),
    "zh": re.compile(r"什么病|是不是|怎么回事|是否为|会不会是|哪种病|什么原因"),
}


def norm(s):
    return unicodedata.normalize("NFC", str(s or "")).casefold()


def load(pattern):
    recs = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                k = (int(r["id"]), r["lang"])
                if not (k in recs and recs[k].get("ok") and not r.get("ok")):
                    recs[k] = r
    return recs


def script_bad(term, lang):
    han = bool(translate._HAN.search(term))
    if lang == "zh":
        return not han
    if lang == "en":
        return han or bool(translate._VI.search(term))
    return han


def near_copy(term, base):
    """Term không có trong câu gốc nhưng rất giống một đoạn của câu gốc (sai chính tả / biến dạng) → đoạn đó."""
    t, b = norm(term), norm(base)
    if len(t) < 3 or t in b:
        return None
    best, best_s = 0.0, None
    for w in {len(t) - 1, len(t), len(t) + 1}:
        for i in range(0, max(1, len(b) - w + 1)):
            s = b[i:i + w]
            r = difflib.SequenceMatcher(None, t, s).ratio()
            if r > best:
                best, best_s = r, s
    return best_s if best >= 0.85 else None


def stopwords(recs, lang, ratio=0.15):
    """Từ xuất hiện trong > ratio số câu hỏi của ngôn ngữ → coi là từ chức năng khi đo độ phủ."""
    df = Counter()
    n = 0
    for (q, l), r in recs.items():
        if l == lang:
            n += 1
            df.update(set(bm25.tokenize(r["base"], lang)))
    return {t for t, c in df.items() if c > ratio * n}


def glossary_expect(vi_query, lang, gl):
    out = []
    for txt, g in translate.glossary_hits(vi_query, gl):
        name = {"vi": g[4], "en": g[1], "zh": g[2]}[lang]
        name = re.split(r"[（(；;]", name)[0].strip()
        if name:
            out.append(name)
    return out


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    recs = load(a.records)
    out = Path(a.out or Path(sorted(glob.glob(a.records))[0]).parents[2])
    gl = translate.load_glossary("configs/med_glossary_vi.tsv")
    vi_base = {q: r["base"] for (q, l), r in recs.items() if l == "vi"}
    flags = []                                             # (qid, lang, kiểm tra, chi tiết)
    stats = defaultdict(Counter)

    def flag(r, check, detail=""):
        flags.append((r["id"], r["lang"], check, detail))
        stats[r["lang"]][check] += 1

    # A. cấu trúc
    qids = {q for q, _ in recs}
    missing = {lg: sorted(qids - {q for q, l in recs if l == lg}) for lg in LANGS}
    for lg in LANGS:
        stats[lg]["lượt"] = sum(1 for _, l in recs if l == lg)
    sw = {lg: stopwords(recs, lg) for lg in LANGS}
    for (q, lg), r in sorted(recs.items()):
        raw = re.sub(r"```(?:json)?", "", r.get("raw") or "").strip()
        if not r["ok"]:
            flag(r, f"lỗi: {r['why']}", raw[:200])
            continue
        if not raw.endswith("}"):
            flag(r, "đầu ra bị cắt / chữ thừa sau JSON", raw[-120:])
        base, rw, terms = r["base"], r["rewrite"], r["terms"]
        # B. rewrite
        extra = sorted(set(NUM.findall(rw)) - set(NUM.findall(base)))
        if extra:
            flag(r, "rewrite: số không có trong câu gốc", ", ".join(extra))
        for p in DEMO[lg]:
            m = re.search(p, rw, re.I)
            if m and not re.search(p, base, re.I):
                flag(r, "rewrite: thêm tuổi/giới/thai kỳ", m.group(0))
                break
        bt = [t for t in dict.fromkeys(bm25.tokenize(base, lg)) if t not in sw[lg] and not t.isdigit()]
        rt = set(bm25.tokenize(rw, lg))
        if len(bt) >= 6:
            cov = sum(t in rt for t in bt) / len(bt)
            stats[lg]["_cov_sum"] += cov
            stats[lg]["_cov_n"] += 1
            if cov < 0.35:
                flag(r, "rewrite: giữ ít từ nội dung của câu gốc", f"{cov:.0%}; mất: {[t for t in bt if t not in rt][:10]}")
        if r["n_words"] > (60 if lg == "zh" else 40):
            flag(r, "rewrite: quá dài", str(r["n_words"]))
        if norm(rw) == norm(base):
            stats[lg]["rewrite = câu gốc"] += 1
        # C. terms
        stats[lg]["_terms"] += len(terms)
        stats[lg]["_cands"] += bool(r.get("candidates"))
        if not terms:
            flag(r, "terms: rỗng")
        new = [t for t in terms if norm(t) not in norm(base)]
        stats[lg]["_terms_new"] += len(new)
        gen = [t for t in terms if norm(t) in GENERIC]
        if gen:
            flag(r, "terms: từ chung chung", ", ".join(gen))
        bad = [t for t in terms if script_bad(t, lg)]
        if bad:
            flag(r, "terms: sai chữ viết", ", ".join(bad))
        for t in new:
            s = near_copy(t, base)
            if s:
                flag(r, "terms: gần giống cụm trong câu (chính tả?)", f"{t!r} ~ {s!r}")
        if DIAG_Q[lg].search(base) and new:
            stats[lg]["_diag_q"] += 1
            flag(r, "terms: câu hỏi chẩn đoán → term mới (đọc tay)", ", ".join(new))
        exp = glossary_expect(vi_base.get(q, ""), lg, gl)
        if exp:
            stats[lg]["_gl_q"] += 1
            miss = [e for e in exp if norm(e) not in norm(" ".join(terms))]
            stats[lg]["_gl_terms"] += len(exp)
            stats[lg]["_gl_miss"] += len(miss)

    # báo cáo
    checks = sorted({c for _, _, c, _ in flags}, key=lambda c: (not c.startswith("lỗi"), c))
    lines = ["# Rà soát rewrite + thuật ngữ\n", f"Nguồn: `{a.records}` — {len(recs)} lượt, {len(qids)} câu.\n"]
    lines.append("| | " + " | ".join(LANGS) + " |\n|---|" + "---|" * len(LANGS))
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(f(lg) for lg in LANGS) + " |")
    row("lượt", lambda lg: str(stats[lg]["lượt"]))
    row("thiếu câu", lambda lg: str(len(missing[lg])))
    row("TB terms / lượt", lambda lg: f"{stats[lg]['_terms'] / max(1, stats[lg]['lượt']):.1f}")
    row("lượt có candidates (chẩn đoán có khả năng)", lambda lg: str(stats[lg]["_cands"]))
    row("terms mới (không có trong câu)", lambda lg: f"{stats[lg]['_terms_new'] / max(1, stats[lg]['_terms']):.0%}")
    row("độ phủ từ nội dung của rewrite (TB)", lambda lg: f"{stats[lg]['_cov_sum'] / max(1, stats[lg]['_cov_n']):.0%}")
    row("rewrite = câu gốc", lambda lg: str(stats[lg]["rewrite = câu gốc"]))
    row("câu khớp bảng thuật ngữ / tên thiếu trong terms",
        lambda lg: f"{stats[lg]['_gl_q']} / {stats[lg]['_gl_miss']} trên {stats[lg]['_gl_terms']}")
    for c in checks:
        row(c, lambda lg: str(stats[lg][c]))
    byc = defaultdict(list)
    for f in flags:
        byc[f[2]].append(f)
    for c in checks:
        lines.append(f"\n## {c} ({len(byc[c])})\n")
        for q, lg, _, d in byc[c][:a.examples]:
            r = recs[(q, lg)]
            lines.append(f"- **{q} [{lg}]** {d}\n  - base: {r['base'][:220]}\n  - rewrite: {r['rewrite']}\n"
                         f"  - terms: {r['terms']}")
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "flagged.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "lang", "kiểm tra", "chi tiết", "base", "rewrite", "terms", "candidates"])
        for q, lg, c, d in flags:
            r = recs[(q, lg)]
            w.writerow([q, lg, c, d, r["base"], r["rewrite"], " | ".join(r["terms"]), " | ".join(r.get("candidates") or [])])
    print("\n".join(lines[:4 + 8 + len(checks) + 2]))
    print(f"\n→ {out / 'review.md'}, {out / 'flagged.csv'} ({len(flags)} dòng, "
          f"{len({(q, lg) for q, lg, _, _ in flags})} lượt bị đánh dấu)")


if __name__ == "__main__":
    main()
