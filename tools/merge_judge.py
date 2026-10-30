"""Ghép nhãn giám khảo LLM (mirret.llmjudge, k10) vào baseline lin07 → danh sách tài liệu + run cho span_chunks. Chạy trên
máy, không GPU. Cách ghép cố định trước khi nộp (không dò trên đáp án):

  - tập tài liệu mỗi câu: tài liệu có đoạn trong top-400 của run ens hoặc run Qwen (như baseline); đã chấm = hạng ens 1..hi;
    chưa chấm: P_judge / P(2) ước bằng trung bình của 50 hạng chấm cuối của câu;
  - HAI thứ tự, mỗi thứ tự thay một nửa tín hiệu baseline bằng giám khảo (z chuẩn hoá theo câu trên các tài liệu đã chấm):
      danh sách tài liệu: sd = (1 − w)·z(logit P(yes) Qwen) + w·z(logit P_judge)   — baseline xếp tài liệu theo P(yes) Qwen;
      thứ tự khúc (--order của span_chunks): s = (1 − w)·z(ens) + w·z(logit P_judge) — baseline xếp khúc theo ens;
    (30/10: bản đầu dùng ens cho cả danh sách tài liệu → 92% trùng baseline nhưng gần hết chênh lệch là do ens ≠ Qwen);
  - số tài liệu mỗi câu (--k-mode): base = đúng số của --base-docs (bản giữ PubMed) → chỉ đổi thứ tự (J1); auto = autok
    k ∈ [--k-min, --k-max] tối đa 5·Σp / (4·Σp_all + k) trên p của --prob (J2); bản _np bỏ PubMed sau khi cắt;
  - --strat-lang: giữ ngôn ngữ của từng VỊ TRÍ như thứ tự baseline (Qwen cho tài liệu, ens cho khúc), chỉ xếp lại trong từng
    ngôn ngữ → mọi đoạn đầu danh sách có đúng tỉ lệ vi / zh / en như baseline (30/10: giám khảo xếp bài vi thấp hơn Qwen —
    suckhoedoisong, medlatec, thanhnien, youmed — và nhỉnh hơn với 120ask, PubMed: thang điểm lệch giữa ngôn ngữ);
  - p cho autok + phân bổ đoạn (--prob): qwen = P(yes) như baseline (mặc định); judge = P_judge = P(2) + ½P(1); p2 = P(2);
    mean / mean2 = trung bình P(yes) với P_judge / P(2). Cả 1.200 câu: Σ P_judge trung vị 172 / 250 tài liệu (rộng tay),
    Σ P(2) 112 (p10 28, p90 218);
  - câu chưa chấm (chạy thử một phần câu): giữ nguyên danh sách tài liệu của --base-docs và run gốc.
Đầu ra (--out): <name>_docs_pm.json, <name>_docs_np.json, runs/<name>_order.parquet (run thứ tự cho --order),
runs/<name>_probs.parquet (P mỗi tài liệu cho --probs) → tools/span_chunks.py → (chạy thử) tools/splice_sub.py.

  J1:  python tools/merge_judge.py --labels results/judge_0 results/judge_1 results/judge_2 results/judge_3 --name j1 \
           --base-docs results/mix_submit/final1200_vi2300_pm.json
  J1s: … --name j1s --strat-lang
  J2:  … --name j2 --w 1 --k-mode auto --prob mean2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirret import config, runs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_submission import auto_k  # noqa: E402


def logit(p):
    p = np.asarray(p, dtype=np.float64).clip(1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def slot_fill(pattern, order, lang):
    """Giữ ngôn ngữ của từng vị trí theo `pattern` (tài liệu theo thứ tự baseline), điền tài liệu theo `order` trong từng
    ngôn ngữ; tài liệu còn lại nối sau theo `order`."""
    by = {}
    for x in order:
        by.setdefault(lang.get(x), []).append(x)
    pos, out = {k: 0 for k in by}, []
    for x in pattern:
        lg = lang.get(x)
        if pos.get(lg, 0) < len(by.get(lg, [])):
            out.append(by[lg][pos[lg]])
            pos[lg] += 1
    seen = set(out)
    return out + [x for x in order if x not in seen]


def load_labels(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("labels-*.parquet")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"không thấy labels-*.parquet trong {paths}")
    lab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lab["qid"], lab["doc_id"] = lab["qid"].astype(int), lab["doc_id"].astype(int)
    dup = lab.duplicated(["qid", "doc_id"]).sum()
    if dup:
        raise SystemExit(f"{dup} cặp (câu, tài liệu) bị chấm hai lần — trùng phần giữa các tài khoản?")
    print(f"nhãn: {len(files)} file, {len(lab)} cặp, {lab['qid'].nunique()} câu, model {lab['model'].unique().tolist()}; "
          f"không đọc được {int(lab['prel'].isna().sum())}")
    return lab


def main():
    config.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", nargs="+", required=True, help="thư mục output k10 (tìm labels-*.parquet) hoặc file")
    ap.add_argument("--ens", default="results/full1200/runs/lin07_ensi_1200.parquet")
    ap.add_argument("--qwen", default="results/full1200/runs/lin07_qwen3i_1200.parquet")
    ap.add_argument("--base-docs", required=True,
                    help="bài nộp baseline GIỮ PubMed (.json): số tài liệu mỗi câu (--k-mode base) + câu chưa chấm")
    ap.add_argument("--w", type=float, default=0.5, help="trọng số giám khảo (1 = chỉ giám khảo)")
    ap.add_argument("--prob", default="qwen", choices=["qwen", "judge", "p2", "mean", "mean2"])
    ap.add_argument("--k-mode", default="base", choices=["base", "auto"])
    ap.add_argument("--strat-lang", action="store_true", help="giữ ngôn ngữ từng vị trí như thứ tự baseline (xem trên)")
    ap.add_argument("--k-min", type=int, default=100)
    ap.add_argument("--k-max", type=int, default=400)
    ap.add_argument("--top-n", type=int, default=400)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/judge")
    a = ap.parse_args()
    cfg = config.load(a.config)
    pm = runs.pubmed_ids(cfg)
    lab = load_labels(a.labels)
    jq = set(lab["qid"])
    hi = int(lab["drank"].max())
    base = {int(s["id"]): s["relevant_docs"] for s in json.loads(Path(a.base_docs).read_text("utf-8"))}
    if not any(x in pm for v in base.values() for x in v):
        raise SystemExit("--base-docs không có PubMed: cần bản giữ PubMed (vd final1200_vi2300_pm.json)")
    ens = runs.read(a.ens, depth=1000)
    qw = runs.read(a.qwen, depth=a.top_n)
    de = runs.doc_ranking(ens[ens["qid"].isin(jq)], "max", a.top_n)
    de = de[de["rank"] <= a.top_n][["qid", "doc_id", "unit_id", "drank", "score"]].rename(columns={"score": "ens"})
    dq = runs.doc_ranking(qw[qw["qid"].isin(jq) & (qw["score"] >= 0) & (qw["score"] <= 1)], "max", a.top_n)
    dq = dq[["qid", "doc_id", "unit_id", "drank", "score"]].rename(
        columns={"unit_id": "unit_q", "drank": "qrank", "score": "pqwen"})
    d = de.merge(dq, on=["qid", "doc_id"], how="outer")
    d["unit_id"] = d["unit_id"].fillna(d["unit_q"])
    d = d.merge(lab[["qid", "doc_id", "prel", "p2", "lang"]], on=["qid", "doc_id"], how="left")
    d["judged"] = d["prel"].notna()
    miss = int(((d["drank"] <= hi) & ~d["judged"]).sum())
    if miss:
        print(f"  cảnh báo: {miss} tài liệu hạng ens ≤ {hi} chưa có nhãn (coi như chưa chấm)")
    tp = d[d["judged"] & (d["drank"] > hi - 50)].groupby("qid")[["prel", "p2"]].mean()
    for c in ("prel", "p2"):
        d[c] = d[c].fillna(d["qid"].map(tp[c]))
    d["pqwen"] = d["pqwen"].fillna(0.0)
    d["ens"] = d["ens"].fillna(d.groupby("qid")["ens"].transform("min"))       # tài liệu chỉ có trong top-400 Qwen
    d["drank"], d["qrank"] = d["drank"].fillna(10 ** 6), d["qrank"].fillna(10 ** 6)
    d["doc_id"] = d["doc_id"].astype(int)
    nq = len(jq)
    print(f"tập tài liệu: TB {len(d) / nq:.0f}/câu, đã chấm {d['judged'].sum() / nq:.0f}, chưa chấm "
          f"{(~d['judged']).sum() / nq:.0f} (ước P_judge {tp['prel'].mean():.3f}, P(2) {tp['p2'].mean():.3f})")
    need = sorted(set(d.loc[~d["judged"], "doc_id"]) | {x for v in base.values() for x in v})
    t = ds.dataset(str(cfg.path("corpus"))).to_table(columns=["id", "lang"], filter=ds.field("id").isin(need)).to_pydict()
    lang = dict(zip([int(i) for i in t["id"]], t["lang"]))
    lang.update(zip(d.loc[d["judged"], "doc_id"], d.loc[d["judged"], "lang"]))
    d["jl"], d["ql"] = logit(d["prel"]), logit(d["pqwen"])
    j = d[d["judged"]].groupby("qid")
    for c in ("ens", "jl", "ql"):
        mu, sd = j[c].mean(), j[c].std().replace(0, np.nan)
        d["z_" + c] = ((d[c] - d["qid"].map(mu)) / d["qid"].map(sd)).fillna(0.0)
    d["s"] = (1 - a.w) * d["z_ens"] + a.w * d["z_jl"]                       # thứ tự khúc
    d["sd"] = (1 - a.w) * d["z_ql"] + a.w * d["z_jl"]                       # danh sách tài liệu
    d["p"] = {"qwen": d["pqwen"], "judge": d["prel"], "p2": d["p2"], "mean": (d["prel"] + d["pqwen"]) / 2,
              "mean2": (d["p2"] + d["pqwen"]) / 2}[a.prob]
    docs, chunk_order = {}, {}
    for q, g in d.groupby("qid"):
        od = g.sort_values(["sd", "qrank"], ascending=[False, True])["doc_id"].tolist()
        oc = g.sort_values(["s", "drank"], ascending=[False, True])["doc_id"].tolist()
        if a.strat_lang:
            od = slot_fill(g.sort_values("qrank")["doc_id"].tolist(), od, lang)
            oc = slot_fill(g.sort_values("drank")["doc_id"].tolist(), oc, lang)
        docs[int(q)], chunk_order[int(q)] = od, oc
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    # run P: một dòng / tài liệu (đơn vị tốt nhất), xếp theo danh sách tài liệu; câu chưa chấm giữ run Qwen gốc
    rk = pd.DataFrame([(q, x, i + 1) for q, v in docs.items() for i, x in enumerate(v)], columns=["qid", "doc_id", "rank"])
    pr = rk.merge(d[["qid", "doc_id", "unit_id", "p"]], on=["qid", "doc_id"]).rename(columns={"p": "score"})
    pr = pd.concat([pr[["qid", "unit_id", "doc_id", "rank", "score"]],
                    qw[~qw["qid"].isin(jq)][["qid", "unit_id", "doc_id", "rank", "score"]]], ignore_index=True)
    runs.write(pr.sort_values(["qid", "rank"]).reset_index(drop=True), out / "runs" / f"{a.name}_probs.parquet",
               {"merge_judge": "probs", "prob": a.prob, "labels": a.labels})
    # run thứ tự khúc: đơn vị ens theo thứ tự khúc của tài liệu (trong tài liệu giữ thứ tự ens), rồi phần còn lại theo ens
    ck = pd.DataFrame([(q, x, i + 1) for q, v in chunk_order.items() for i, x in enumerate(v)],
                      columns=["qid", "doc_id", "crank"])
    e = ens[ens["qid"].isin(jq)].merge(ck, on=["qid", "doc_id"], how="left")
    e["crank"] = e["crank"].fillna(10 ** 6)
    e = e.sort_values(["qid", "crank", "rank"])
    e["rank"] = e.groupby("qid").cumcount() + 1
    e["score"] = -e["rank"].astype(float)
    od = pd.concat([e[["qid", "unit_id", "doc_id", "rank", "score"]],
                    ens[~ens["qid"].isin(jq)][["qid", "unit_id", "doc_id", "rank", "score"]]], ignore_index=True)
    runs.write(od.sort_values(["qid", "rank"]).reset_index(drop=True), out / "runs" / f"{a.name}_order.parquet",
               {"merge_judge": "order", "w": a.w, "strat_lang": a.strat_lang, "ens": a.ens, "labels": a.labels})
    if a.k_mode == "base":
        docs = {q: v[:len(base.get(q, v))] for q, v in docs.items()}
    else:
        docs = auto_k(docs, out / "runs" / f"{a.name}_probs.parquet", a.k_min, a.k_max, top_n=10 ** 9)
    full = {q: docs.get(q, base.get(q, [])) for q in sorted(set(base) | set(docs))}
    sets = {f"{a.name}_docs_pm": full,
            f"{a.name}_docs_np": {q: ([x for x in v if x not in pm] if q in jq else v) for q, v in full.items()}}
    for nm, dd in sets.items():
        (out / f"{nm}.json").write_text(json.dumps([{"id": q, "relevant_docs": v} for q, v in sorted(dd.items())]),
                                        encoding="utf-8")
        kj = [len(v) for q, v in dd.items() if q in jq]
        print(f"→ {out / (nm + '.json')}: câu đã chấm TB {np.mean(kj):.1f} tài liệu (trung vị {np.median(kj):.0f})")
    ov = [len(set(docs[q]) & set(base.get(q, []))) / max(len(docs[q]), 1) for q in docs]
    add = [x for q in docs for x in set(docs[q]) - set(base.get(q, []))]
    rem = [x for q in docs for x in set(base.get(q, [])) - set(docs[q])]

    def share(xs):
        return {k: round(v, 3) for k, v in pd.Series([lang.get(x) for x in xs]).value_counts(normalize=True).items()}
    print(f"so với --base-docs: trùng {np.mean(ov):.1%}, base TB {np.mean([len(base.get(q, [])) for q in docs]):.1f} "
          f"tài liệu; thêm {len(add) / len(docs):.1f}/câu {share(add)}, bớt {len(rem) / len(docs):.1f}/câu {share(rem)}")
    (out / f"{a.name}_meta.json").write_text(json.dumps(vars(a), indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
