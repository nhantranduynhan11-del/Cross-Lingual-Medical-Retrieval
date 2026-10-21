"""Tập đánh giá thủ công có kiểm soát (thay cho nhãn BTC không có): chọn câu hỏi → lập pool → phiếu gán nhãn ẩn
nguồn → nhập nhãn, đo độ đồng thuận → qrels.parquet. Quy trình đầy đủ: docs/EVAL_PLAN.md.

  python -m mirret.judge select [--n 60] [--pilot 6]
        chọn câu hỏi phân tầng theo (độ dài câu hỏi × mức bất đồng giữa các nhánh) → eval/judge/queries.csv
  python -m mirret.judge select --add 40
        thêm 40 câu MỚI (cùng cách phân tầng, không trùng câu đã có) vào queries.csv, nhóm "add1" ("add2"…)
  python -m mirret.judge pool --sets add1 [--branch-depth 5] [--fused …]
        pool + phiếu chỉ cho các câu của nhóm đó → sheet_add1_<người>.csv; key.parquet được nối thêm
  python -m mirret.judge pool [--branch-depth 10] [--fused rrf9:30,rrf9_colbert:30] [--labelers A,B,C] [--overlap 0.2]
        pool = hợp các đoạn cha trong top của 9 nhánh + run gộp/xếp hạng lại → phiếu (xáo thứ tự, KHÔNG ghi nguồn):
        sheet_pilot_<người>.csv (câu pilot, MỌI người chấm), sheet_pool_<người>.csv (câu chính, chia nhau, --overlap
        phần câu chấm chéo) + key.parquet (nguồn của từng mục, chỉ dùng khi tính điểm)
  python -m mirret.judge extend --runs <run mới>:30
        thêm vào pool các mục CHƯA chấm của run mới (vd reranker mới) → phiếu bổ sung, không chấm lại mục cũ
  python -m mirret.judge ingest [--adjudicated file.csv]
        đọc mọi sheet đã điền → độ đồng thuận (kappa) + danh sách bất đồng + eval/judge/qrels.parquet
Nhãn: 2 = trả lời trực tiếp / chứa thông tin cần tìm; 1 = liên quan một phần (một ý của câu nhiều ý, thông tin then
chốt nhưng chưa đủ); 0 = không liên quan, kể cả CÙNG CHỦ ĐỀ nhưng không trả lời câu hỏi. Bỏ trống = chưa chấm.
"""
import argparse
import csv
import hashlib
import random
import re

import numpy as np
import pandas as pd

from . import config, runs
from .diag import BRANCHES
from .rerank import unit_rows

GRADE = "grade (0/1/2)"


def jdir(cfg):
    d = cfg.path("eval") / "judge"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _queries(cfg):
    qs = {}
    for lg in ("vi", "en", "zh"):
        p = cfg.path("qtrans") / f"queries_{lg}.parquet"
        if p.exists():
            t = pd.read_parquet(p)
            qs[lg] = dict(zip(t["id"].astype(int), t["query"]))
    if "vi" not in qs:
        t = pd.read_parquet(cfg.path("queries"))
        qs["vi"] = dict(zip(t["id"].astype(int), t["query"]))
    return qs


# ---------------------------------------------------------------- 1. chọn câu hỏi
def select(cfg, n=60, pilot=6, k=20, add=0, log=print):
    rdir = cfg.path("runs")
    names = [b for b in BRANCHES if (rdir / f"{b}.parquet").exists()]
    if len(names) < 2:
        raise SystemExit("Cần ít nhất 2 run nhánh trong runs/ để đo mức bất đồng.")
    lists = {b: runs.topk_lists(runs.read(rdir / f"{b}.parquet", depth=k * 3), k, "doc") for b in names}
    qs = _queries(cfg)["vi"]
    rows = []
    for q, text in qs.items():
        J = []
        for a, b in [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]:
            A, B = set(lists[a].get(q, [])), set(lists[b].get(q, []))
            if A and B:
                J.append(len(A & B) / len(A | B))
        rows.append({"qid": q, "query": text, "len": len(text), "disagree": 1 - (np.mean(J) if J else 0)})
    df = pd.DataFrame(rows)
    df["len_bin"] = pd.qcut(df["len"].rank(method="first"), 3, labels=["ngắn", "vừa", "dài"])
    df["dis_bin"] = pd.qcut(df["disagree"].rank(method="first"), 3, labels=["đồng thuận", "trung bình", "bất đồng"])
    out = jdir(cfg) / "queries.csv"
    cols = ["qid", "set", "len_bin", "dis_bin", "len", "disagree", "query"]
    old = pd.read_csv(out, encoding="utf-8-sig") if add else None
    if add:                                               # thêm câu mới, giữ nguyên các câu (và nhãn) đã có
        tag = f"add{sum(str(x).startswith('add') for x in old['set'].unique()) + 1}"
        df = df[~df["qid"].isin(set(old["qid"].astype(int)))]
        n, pilot = add, 0
        rng = random.Random(f"{cfg['seed']}-{tag}")
    else:
        rng = random.Random(cfg["seed"])
    picked = []
    strata = list(df.groupby(["len_bin", "dis_bin"], observed=True))
    per = max(1, n // len(strata))
    for _, g in strata:                                   # đều mỗi tầng (9 tầng), rồi bù ngẫu nhiên cho đủ n
        ids = g["qid"].tolist()
        picked += rng.sample(ids, min(per, len(ids)))
    rest = [q for q in df["qid"] if q not in set(picked)]
    picked += rng.sample(rest, max(0, n - len(picked)))
    rng.shuffle(picked)
    sel = df.set_index("qid").loc[picked].reset_index()
    sel["set"] = [tag] * len(sel) if add else ["pilot" if i < pilot else "main" for i in range(len(sel))]
    if add:
        sel = pd.concat([old[cols], sel[cols]], ignore_index=True)
    sel[cols].to_csv(out, index=False, encoding="utf-8-sig")
    log(f"→ {out}: {len(sel)} câu" + (f" (thêm {n} câu nhóm {tag})" if add else f" ({pilot} pilot)")
        + f" — phân tầng {len(strata)} tầng theo độ dài × mức bất đồng")
    log(sel.groupby(["set", "len_bin", "dis_bin"], observed=True).size().unstack("set", fill_value=0).to_string())


# ---------------------------------------------------------------- 2. pool + phiếu
def _code(qid, unit_id, salt):
    return hashlib.md5(f"{salt}:{qid}:{unit_id}".encode()).hexdigest()[:8]


def collect(cfg, qids, specs):
    """specs: [(tên run, độ sâu)] → DataFrame (qid, unit_id, doc_id, provenance)."""
    rdir = cfg.path("runs")
    rows = []
    for name, depth in specs:
        base, _, agg = name.partition("~")                # "run~sum2": đoạn tốt nhất của top-K TÀI LIỆU theo cách gộp đó
        p = rdir / f"{base}.parquet"
        if not p.exists():
            print(f"  [bỏ qua] thiếu run {base}")
            continue
        if agg:
            df = runs.doc_ranking(runs.read(p, depth=depth * 30, qids=qids), agg)
            df = df[df["drank"] <= depth].assign(rank=lambda x: x["drank"])
        else:
            df = runs.read(p, depth=depth, qids=qids)
        for q, u, d, r in zip(df["qid"], df["unit_id"], df["doc_id"], df["rank"]):
            rows.append((int(q), u, int(d), f"{name}@{int(r)}"))
    df = pd.DataFrame(rows, columns=["qid", "unit_id", "doc_id", "src"])
    return df.groupby(["qid", "unit_id", "doc_id"], as_index=False)["src"].agg(";".join)


def write_sheets(cfg, items, labelers, overlap, tag, log=print):
    """items: DataFrame (qid, unit_id, doc_id, src). Câu hỏi chia đều cho người gán; `overlap` phần câu hỏi được
    gán thêm cho người thứ hai (để đo đồng thuận); overlap >= 1 → MỌI người chấm mọi câu (vòng pilot)."""
    d = jdir(cfg)
    qs = _queries(cfg)
    rows = unit_rows(cfg, items["unit_id"].unique().tolist())
    salt = str(cfg["seed"])
    items = items[items["unit_id"].isin(rows)].copy()
    items["item"] = [_code(q, u, salt) for q, u in zip(items["qid"], items["unit_id"])]
    items["lang"] = items["unit_id"].map(lambda u: rows[u][0])
    rng = random.Random(f"{cfg['seed']}-{tag}")
    qids = sorted(items["qid"].unique())
    rng.shuffle(qids)
    owner = {q: [labelers[i % len(labelers)]] for i, q in enumerate(qids)}
    n_dup = int(round(min(overlap, 1.0) * len(qids)))
    for q in qids[:n_dup]:
        owner[q] = list(labelers) if overlap >= 1 else             [owner[q][0], labelers[(labelers.index(owner[q][0]) + 1) % len(labelers)]]
    key = items.assign(assigned=items["qid"].map(lambda q: ",".join(owner[q])))
    kp = d / "key.parquet"
    if kp.exists():
        key = pd.concat([pd.read_parquet(kp), key], ignore_index=True).drop_duplicates(["qid", "unit_id"], keep="first")
    key.to_parquet(kp, index=False)
    for who in labelers:
        mine = items[items["qid"].map(lambda q: who in owner[q])].copy()
        mine["_r"] = [rng.random() for _ in range(len(mine))]
        mine = mine.sort_values(["qid", "_r"])              # theo câu hỏi; trong câu hỏi xáo ngẫu nhiên (ẩn thứ hạng)
        f = d / f"sheet_{tag}_{who}.csv"
        with f.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["item", "qid", "query_vi", "query_en", "query_zh", "doc_lang", "context", "chunk_text",
                        GRADE, "note"])
            for _, r in mine.iterrows():
                lg, ctx, text = rows[r["unit_id"]]
                w.writerow([r["item"], r["qid"], qs["vi"].get(r["qid"], ""), qs.get("en", {}).get(r["qid"], ""),
                            qs.get("zh", {}).get(r["qid"], ""), lg, ctx, text, "", ""])
        log(f"  → {f}: {mine['qid'].nunique()} câu, {len(mine):,} mục")
    log(f"[{tag}] {len(items):,} mục / {len(qids)} câu ({len(items) / max(1, len(qids)):.0f} mục/câu); "
        + (f"mọi câu được cả {len(labelers)} người chấm" if overlap >= 1 else f"{n_dup} câu được chấm chéo (2 người)"))


def _specs(branch_depth, fused):
    return [(b, branch_depth) for b in BRANCHES] + [(x.split(":")[0], int(x.split(":")[1]))
                                                   for x in fused.split(",") if x]


def pool(cfg, branch_depth=10, fused="rrf9:30", labelers=("A", "B", "C"), overlap=0.2, sets=None, log=print):
    sel = pd.read_csv(jdir(cfg) / "queries.csv", encoding="utf-8-sig")
    if sets:                                               # chỉ nhóm câu mới thêm (select --add) → sheet_<nhóm>_<người>
        sel = sel[sel["set"].isin(sets.split(","))]
        if sel.empty:
            raise SystemExit(f"queries.csv không có câu nhóm {sets}")
        items = collect(cfg, sel["qid"].astype(int).tolist(), _specs(branch_depth, fused))
        write_sheets(cfg, items, list(labelers), overlap, sets.replace(",", ""), log)
        return
    qids = sel["qid"].astype(int).tolist()
    items = collect(cfg, qids, _specs(branch_depth, fused))
    pilot = set(sel.loc[sel["set"] == "pilot", "qid"].astype(int))
    if pilot:                                              # vòng pilot: mọi người chấm cùng các câu → kappa, sửa hướng dẫn
        write_sheets(cfg, items[items["qid"].isin(pilot)], list(labelers), 1.0, "pilot", log)
    write_sheets(cfg, items[~items["qid"].isin(pilot)], list(labelers), overlap, "pool", log)


def extend(cfg, specs_str, labelers=("A", "B", "C"), log=print):
    """Thêm các mục chưa có trong key của run mới → phiếu bổ sung (để tránh thiên lệch pool khi so run mới)."""
    d = jdir(cfg)
    key = pd.read_parquet(d / "key.parquet")
    qids = sorted(key["qid"].unique())
    specs = [(x.split(":")[0], int(x.split(":")[1])) for x in specs_str.split(",")]
    items = collect(cfg, qids, specs)
    have = set(zip(key["qid"], key["unit_id"]))
    new = items[[(q, u) not in have for q, u in zip(items["qid"], items["unit_id"])]]
    if new.empty:
        log("Không có mục mới (run mới nằm trọn trong pool).")
        return
    tag = "ext" + str(len(list(d.glob("sheet_ext*_*.csv"))) // max(1, len(labelers)) + 1)
    write_sheets(cfg, new, list(labelers), 0.0, tag, log)


# ---------------------------------------------------------------- 3. nhập nhãn
def kappa(a, b, labels):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def ingest(cfg, adjudicated=None, log=print):
    d = jdir(cfg)
    key = pd.read_parquet(d / "key.parquet")
    lab = []
    for f in sorted(d.glob("sheet_*.csv")):
        m = re.match(r"sheet_([a-z]+\d*)_([A-Za-z0-9]+)", f.stem)  # Google Sheets đổi tên: "sheet_pool_A - sheet_pool_A"
        tag, who = (m.group(1), m.group(2)) if m else ("?", f.stem)
        try:
            t = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
        except UnicodeDecodeError:                         # Excel lưu lại bằng bảng mã Windows: item/nhãn vẫn là ASCII
            log(f"  [CẢNH BÁO] {f.name} không phải UTF-8 (lưu bằng Excel?) → chỉ đọc cột item/nhãn")
            t = pd.read_csv(f, encoding="latin-1", dtype=str).fillna("")
        t = t[t[GRADE].str.strip() != ""]
        bad = t[~t[GRADE].str.strip().isin(["0", "1", "2"])]
        if len(bad):
            log(f"  [LỖI] {f.name}: {len(bad)} nhãn không hợp lệ (chỉ 0/1/2), vd mục {bad['item'].head(3).tolist()}")
        t = t[t[GRADE].str.strip().isin(["0", "1", "2"])]
        lab.append(pd.DataFrame({"item": t["item"], "labeler": who, "tag": tag,
                                 "grade": t[GRADE].str.strip().astype(int), "note": t["note"]}))
    if not lab:
        raise SystemExit("Chưa có nhãn nào trong eval/judge/sheet_*.csv")
    lab = pd.concat(lab, ignore_index=True).drop_duplicates(["item", "labeler"], keep="last")
    key["item"] = [_code(q, u, str(cfg["seed"])) for q, u in zip(key["qid"], key["unit_id"])]
    # đồng thuận trên mục được chấm bởi >= 2 người (mọi cặp), tách theo vòng (pilot / pool / ext…)
    piv = lab.pivot_table(index="item", columns="labeler", values="grade", aggfunc="first")
    tag_of = dict(zip(lab["item"], lab["tag"]))
    pairs, multi = [], []
    for i in piv.index:
        v = piv.loc[i].dropna().astype(int).tolist()
        pairs += [(i, v[x], v[y], tag_of[i]) for x in range(len(v)) for y in range(x + 1, len(v))]
        if len(v) >= 2 and min(v) != max(v):
            multi.append((i, min(v), max(v), ",".join(f"{w}={g}" for w, g in piv.loc[i].dropna().astype(int).items())))
    L = ["# Nhập nhãn", "", f"- {lab['item'].nunique():,} mục đã chấm bởi {lab['labeler'].nunique()} người"]
    if pairs:
        for name, ps in [("tất cả", pairs)] + [(t, [p for p in pairs if p[3] == t]) for t in sorted({p[3] for p in pairs})]:
            a, b = [p[1] for p in ps], [p[2] for p in ps]
            L.append(f"- Chấm chéo ({name}): {len(ps)} cặp nhãn; trùng khớp {np.mean(np.array(a) == np.array(b)):.1%}; "
                     f"kappa 3 mức {kappa(a, b, [0, 1, 2]):.2f}; kappa nhị phân (>=1) "
                     f"{kappa([x >= 1 for x in a], [x >= 1 for x in b], [False, True]):.2f}")
        L.append("  (kappa nhị phân < 0,6 → hướng dẫn chưa đủ rõ: họp thống nhất, sửa hướng dẫn, chấm lại phần pilot)")
        dis = multi
        pd.DataFrame(dis, columns=["item", "g_min", "g_max", "grades"]).merge(
            key[["item", "qid", "unit_id"]], on="item").to_csv(d / "disagreements.csv", index=False, encoding="utf-8-sig")
        L.append(f"- {len(dis)} mục bất đồng → {d / 'disagreements.csv'} (phân xử, ghi vào file --adjudicated: item,grade)")
    final = lab.groupby("item")["grade"].agg(["min", "max", "count"]).reset_index()
    final["grade"] = final["min"]                          # chưa phân xử: lấy mức THẤP hơn (thận trọng)
    if adjudicated:
        adj = pd.read_csv(adjudicated, encoding="utf-8-sig")
        final = final.merge(adj[["item", "grade"]].rename(columns={"grade": "adj"}), on="item", how="left")
        final["grade"] = final["adj"].fillna(final["grade"]).astype(int)
    q = final.merge(key, on="item", how="inner")
    rows = unit_rows(cfg, q["unit_id"].unique().tolist())
    q["lang"] = q["unit_id"].map(lambda u: rows.get(u, ("", "", ""))[0])
    q["text"] = q["unit_id"].map(lambda u: rows.get(u, ("", "", ""))[2])
    q = q[["qid", "unit_id", "doc_id", "lang", "grade", "count", "src", "text"]].rename(columns={"count": "n_labelers"})
    q.to_parquet(d / "qrels.parquet", index=False)
    per_q = q.groupby("qid")["grade"].agg(n="size", rel=lambda s: int((s >= 1).sum()), rel2=lambda s: int((s >= 2).sum()))
    L += [f"- qrels: {len(q):,} mục / {q['qid'].nunique()} câu; liên quan (>=1) TB {per_q['rel'].mean():.1f}/câu, "
          f"(>=2) {per_q['rel2'].mean():.1f}/câu; câu không có mục liên quan nào: {(per_q['rel'] == 0).sum()}",
          "- Theo ngôn ngữ (mục >=1): " + ", ".join(f"{lg} {n}" for lg, n in q[q["grade"] >= 1]["lang"].value_counts().items()),
          f"→ {d / 'qrels.parquet'}"]
    md = "\n".join(L) + "\n"
    (d / "ingest.md").write_text(md, encoding="utf-8")
    log(md)


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "pool", "extend", "ingest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--pilot", type=int, default=6)
    ap.add_argument("--branch-depth", type=int, default=10)
    ap.add_argument("--fused", default="rrf9:30")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--labelers", default="A,B,C")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--adjudicated", default=None)
    ap.add_argument("--add", type=int, default=0, help="select: thêm N câu mới vào queries.csv đã có")
    ap.add_argument("--sets", default=None, help="pool: chỉ các câu thuộc nhóm này (vd add1)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    labelers = a.labelers.split(",")
    if a.cmd == "select":
        select(cfg, a.n, a.pilot, add=a.add)
    elif a.cmd == "pool":
        pool(cfg, a.branch_depth, a.fused, labelers, a.overlap, a.sets)
    elif a.cmd == "extend":
        extend(cfg, a.runs, labelers)
    else:
        ingest(cfg, a.adjudicated)


if __name__ == "__main__":
    main()
