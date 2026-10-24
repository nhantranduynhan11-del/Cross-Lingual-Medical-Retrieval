"""Sinh các notebook Kaggle của pipeline truy hồi (notebooks/k1…k8). Chạy: python tools/make_notebooks.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = "15/10/2026"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def setup_cell(version, extra_vars, inputs, overrides="{}"):
    return code(f'''
# (1) Chuẩn bị. Chỉ sửa các biến viết HOA ngay dưới đây.
NB_VERSION = '{version} ({DATE})'
{extra_vars}
import glob, os, shutil, sys, zipfile
KAGGLE = os.environ.get('KAGGLE_ROOT', '/kaggle')            # chạy thử ngoài Kaggle: đặt biến môi trường KAGGLE_ROOT
CODE = f'{{KAGGLE}}/working/code'
shutil.rmtree(CODE, ignore_errors=True)
def _ver(h):                                                  # output notebook cũ (vd mir2-chunks/code) cũng có mirret/
    v = os.path.join(os.path.dirname(os.path.dirname(h)), 'VERSION.txt')
    return open(v, encoding='utf-8').read().strip() if os.path.exists(v) else ''
hits = sorted(glob.glob(f'{{KAGGLE}}/input/**/mirret/__init__.py', recursive=True),
              key=lambda h: (_ver(h), -len(h)), reverse=True)   # bản mã mới nhất theo VERSION.txt (ngày giờ)
if hits:
    shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE)
else:
    zips = sorted(glob.glob(f'{{KAGGLE}}/input/**/mir2_code.zip', recursive=True))
    assert zips, 'Chưa thêm dataset mir2-code (Add Input → Your Datasets)'
    zipfile.ZipFile(zips[0]).extractall(CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
from mirret.kaggle_nb import setup, sh, par, locate, locate_all, report
K = setup(NB_VERSION, {overrides}, inputs={inputs})
''')


def save(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    p = ROOT / "notebooks" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in cells:                                           # kiểm tra cú pháp
        if c["cell_type"] == "code":
            compile("".join(l for l in c["source"] if not l.lstrip().startswith("!")), name, "exec")
    print("→", p)


# ---------------------------------------------------------------- k1: chia đoạn + BM25 (CPU)
save("k1_chunk_bm25.ipynb", [
    md("""
# k1 — Chia đoạn (A+B+C) + chỉ mục BM25 (CPU)

**Cài đặt:** Accelerator = **None (CPU)**, Internet = On. **Add Input:** `mir2-code`, `mir2-data`
(`data/query.parquet`, `data/links_corpus_vi_en_zh.parquet`, `corpus_out/corpus/`).

**Chạy:** *Save Version → Save & Run All*. Ước tính ~1 giờ (chia đoạn ~15 phút, tách từ pyvi/jieba phần lớn thời gian).
**Đầu ra:** `chunks/{parents,children}/`, `index/bm25/parents_{vi,en,zh}/`, `results/chunk_report.md` →
tab *Output* → *New Dataset* tên **`mir2-chunks`** (dùng cho k3, k4, k5).
Khi kho đổi (vd crawl xong longchau): tạo `mir2-data` bản mới rồi chạy lại notebook này.
"""),
    setup_cell("k1-v1", "", "()"),
    code('''
# (2) Chia đoạn: đoạn cha (nộp) + đoạn con (tìm) + ngữ cảnh; báo cáo phân bố độ dài, kiểm tra bất biến
sh(f'python -m mirret.chunk run --config {K.cfg} --workers {os.cpu_count()}', 'chunk')
sh(f'python -m mirret.chunk report --config {K.cfg}', 'chunk_report')
'''),
    code('''
# (3) BM25 trên đoạn cha, mỗi ngôn ngữ tài liệu một chỉ mục (câu hỏi vi ↔ tài liệu vi, en ↔ en, zh ↔ zh)
K.set({'bm25.workers': os.cpu_count()})
sh(f'python -m mirret.index bm25 --config {K.cfg} --level parents', 'bm25')
report('k1')
'''),
])

# ---------------------------------------------------------------- k2: dịch câu hỏi (GPU)
save("k2_translate.ipynb", [
    md("""
# k2 — Dịch 1.200 câu hỏi vi → en → zh, so sánh model / phiên bản prompt (GPU T4 ×2)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On. **Add Input:** `mir2-code`, `mir2-data`
(+ `mir2-qtrans` bản trước: để so với bản cũ, để phiếu chấm mới không lặp câu đã chấm, và để chạy tiếp khi bị ngắt).

Mỗi model chạy 2 tiến trình song song (mỗi GPU một bản 4-bit, mỗi bản một nửa số câu). Kết quả ở
`qtrans/<model>-<prompt>/` (prompt `t1` cũ giữ tên `qtrans/<model>/`).
**Prompt t2** (mặc định): sửa theo lỗi đã gặp khi so Qwen2.5-14B / Qwen3-8B với t1 — quy tắc thuật ngữ, triệu chứng không
nâng thành chẩn đoán, đủ ý / không thêm, đúng người, chữ số, chữ viết tắt, biệt dược; gợi ý thuật ngữ đã kiểm chứng theo
từng câu (`configs/med_glossary_vi.tsv`). Đo thực tế với t1: 14B ~2 giờ, 8B ~1,2 giờ cho 1.200 câu (lô 8).
Ô (1): `LIMIT = 20` chạy thử để đọc mẫu, rồi `LIMIT = 0`. `COMPARE` = các cặp cần so (tên thư mục trong qtrans/).

**Đầu ra:** `qtrans/<khoá>/records-*.jsonl`, `qtrans/compare_*/{auto_metrics.md, sheet.csv, key.json}` → dataset
**`mir2-qtrans`** (bản mới). Phiếu `sheet.csv` chấm mù (không mở `key.json`), rồi
`python -m mirret.translate score --sheet <phiếu đã chấm>` (docs/EVAL_PLAN.md mục 0).
"""),
    setup_cell("k2-v3", "MODELS = ['qwen25_14b', 'qwen3_14b']  # tên trong translate.models hoặc id Hugging Face\n"
                        "PROMPT = 't2'\nLIMIT = 0\n"
                        "BATCH = {'qwen25_14b': 8, 'qwen3_14b': 8, 'qwen3_8b': 16}   # câu/lô mỗi GPU; hết bộ nhớ → tự giảm\n"
                        "QUANT = '4bit'                       # '4bit' (nhanh nhất trên T4) | '8bit' (chậm hơn)\n"
                        "COMPARE = [('qwen25_14b-t2', 'qwen3_14b-t2'),   # so 2 model cùng prompt\n"
                        "           ('qwen25_14b', 'qwen25_14b-t2')]     # so prompt t1 → t2 (cần mir2-qtrans bản cũ)",
               "()", "{'translate.prompt': PROMPT}"),
    code('''
# (2) Chép bản ghi + phiếu so sánh của các phiên trước (nếu có); cài bitsandbytes (4-bit)
for d in locate_all('qtrans'):
    for f in glob.glob(f'{d}/qtrans/*/records*.jsonl') + glob.glob(f'{d}/qtrans/compare_*/*'):
        dst = f'{K.work}/qtrans/' + os.path.relpath(f, f'{d}/qtrans')
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst); print('chép', f)
sh('python -m pip install -q -U bitsandbytes accelerate')
'''),
    code('''
# (3) Sinh bản dịch cho từng model, 2 GPU song song
for m in MODELS:
    lim = f' --limit {LIMIT}' if LIMIT else ''
    bs = f' --batch {BATCH[m]}' if m in BATCH else ''
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.translate generate --config {K.cfg} --model {m} --shard {g}/2'
         f'{lim}{bs} --quant {QUANT} --prompt {PROMPT}' for g in range(2)], f'translate_{m}')
'''),
    code('''
# (4) So sánh: chỉ số tự động + phiếu chấm mù MỚI (bỏ các câu đã chấm ở lần trước)
for a, b in COMPARE:
    if all(os.path.isdir(f'{K.work}/qtrans/{x}') for x in (a, b)):
        sh(f'python -m mirret.translate compare --config {K.cfg} --a {a} --b {b} --n 100')
    else:
        print('bỏ qua so sánh', a, b, '(thiếu thư mục bản dịch)')
report('k2')
'''),
])

# ---------------------------------------------------------------- k3: mã hoá (GPU, mỗi người một phần)
save("k3_encode.ipynb", [
    md("""
# k3 — Mã hoá BGE-M3 đoạn cha + đoạn con (GPU T4 ×2, chia cho nhiều người)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On. **Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`
(+ output phiên trước của CHÍNH người này nếu bị ngắt).

**Chia việc:** `MEMBERS` người × 2 GPU. Người `m` đặt `MEMBER = m`; GPU g chạy phần `2m+g / 2·MEMBERS`. Mọi người phải
dùng **cùng bản `mir2-chunks`**: dòng "dấu vân tay" in ra phải giống nhau.
**Mã hoá bổ sung** (kho thêm bài, vd longchau): Add Input các dataset emb CŨ ĐẦY ĐỦ (vd `mir2-emb-m0..m2` bản trước) và
ghi tên chúng vào `SKIP_FROM` → chỉ mã hoá đoạn mới / đổi nội dung (theo cặp unit_id + hash). Mọi người phải dùng CÙNG
`SKIP_FROM`. Output dở dang của chính mình (để chạy tiếp) KHÔNG ghi vào `SKIP_FROM` — nó tự được dùng để chép khối đã xong.
Ước tính (chưa đo trên kho thật): đoạn cha ~2,6 triệu, đoạn con ~6,5 triệu; tổng ~8–10 giờ GPU ×2 → ~3 giờ/người khi 3 người.
**Đầu ra:** `emb/{parents,children}/<khối>/` → dataset **`mir2-emb-m{MEMBER}`**.
"""),
    setup_cell("k3-v1", "MEMBER = None        # ← 0, 1, 2 …\nMEMBERS = 3\nLEVELS = ['parents', 'children']\n"
                        "SKIP_FROM = []       # tên dataset emb cũ đầy đủ (vd ['mir2-emb-m0-v1']) — giống nhau ở mọi người\n"
                        "RUN_BENCH = False", "('chunks',)"),
    code('''
# (2) Thư mục emb của phiên trước / bản cũ trong /kaggle/input
assert MEMBER is not None and 0 <= MEMBER < MEMBERS, 'Đặt MEMBER ở ô (1)'
prev = [str(d / 'emb') for d in locate_all('emb')]
print('emb trong input:', prev)
old = [p for p in prev if any(f'/{x}/' in p + '/' for x in SKIP_FROM)]
mine_prev = [p for p in prev if p not in old]
skip = f' --skip {",".join(old)}' if old else ''
resume = f' --resume {",".join(mine_prev)}' if mine_prev else ''
print('bỏ qua (bản cũ):', old, '| chạy tiếp từ:', mine_prev)
if RUN_BENCH:
    sh(f'python -m mirret.encode bench --config {K.cfg} --level parents --n 2000 --device cuda:0')
for lv in LEVELS:
    sh(f'python -m mirret.encode plan --config {K.cfg} --level {lv}{skip}')
'''),
    code('''
# (3) Mã hoá phần của MEMBER trên 2 GPU, lần lượt từng mức
n = 2 * MEMBERS
for lv in LEVELS:
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.encode run --config {K.cfg} --level {lv} '
         f'--shard {2 * MEMBER + g}/{n} --device cuda:0{skip}{resume}' for g in range(2)], f'encode_{lv}')
report(f'k3_m{MEMBER}')
'''),
])

# ---------------------------------------------------------------- k4: chỉ mục + 9 nhánh + RRF + chẩn đoán
save("k4_index_retrieve.ipynb", [
    md("""
# k4 — Chỉ mục dense, 9 nhánh (vi/en/zh × Dense/LSR/BM25), RRF, chẩn đoán không cần nhãn

**Cài đặt:** Accelerator = **GPU T4** (chỉ để mã hoá câu hỏi), Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`, `mir2-qtrans`, mọi `mir2-emb-m*`.
`USE` = bản dịch dùng cho nhánh en/zh (tên thư mục trong qtrans/, chọn sau khi chấm bản dịch).
RAM: IVF-SQ8 ~1 KB/vector (đoạn con ~6,5 triệu → ~6,5 GB) — vừa 30 GB. `SAVE_INDEX = False` để không lưu file chỉ mục
(~9 GB) vào output nếu không cần.
**Đầu ra:** `runs/` (9 nhánh, các mức @parents/@children, rrf9 …), `qtrans/queries_*.parquet`, `qemb/`,
`eval/diag/diag.md` → dataset **`mir2-runs`**.
"""),
    setup_cell("k4-v1", "USE = 'qwen25_14b'\nSAVE_INDEX = False", "('chunks',)"),
    code('''
# (2) Câu hỏi 3 ngôn ngữ: chép bản dịch từ mir2-qtrans rồi dựng qtrans/queries_{vi,en,zh}.parquet
for d in locate_all('qtrans'):
    for f in glob.glob(f'{d}/qtrans/*/records*.jsonl'):
        dst = f'{K.work}/qtrans/' + os.path.relpath(f, f'{d}/qtrans')
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst)
sh(f'python -m mirret.translate build --config {K.cfg} --use {USE}')
bm = locate('index/bm25', required=False)
if bm is not None:
    K.set({'bm25.index_dir': str(bm / 'index' / 'bm25')})
EMB = ','.join(sorted({str(d / 'emb') for d in locate_all('emb')}))
print('emb:', EMB)
'''),
    code('''
# (3) Chỉ mục dense từng mức + mã hoá câu hỏi (GPU)
for lv in ('parents', 'children'):
    sh(f'python -m mirret.index dense --config {K.cfg} --level {lv} --emb {EMB}', f'dense_{lv}')
sh(f'python -m mirret.retrieve qencode --config {K.cfg} --device cuda:0', 'qencode')
'''),
    code('''
# (4) 9 nhánh + gộp RRF + chẩn đoán không cần nhãn (trên cả 1.200 câu)
sh(f'python -m mirret.retrieve run --config {K.cfg} --emb {EMB}', 'retrieve')
sh(f'python -m mirret.fuse --config {K.cfg}', 'fuse')
sh(f'python -m mirret.diag --config {K.cfg}', 'diag')
if not SAVE_INDEX:
    for f in glob.glob(f'{K.work}/index/dense_*'):
        os.remove(f)
report('k4')
'''),
])

# ---------------------------------------------------------------- k5: rerank + bài nộp
save("k5_rerank_submit.ipynb", [
    md("""
# k5 — Stage 2: xếp hạng lại top-N (ColBERT BGE-M3 / cross-encoder) + bài nộp (GPU T4 ×2)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`, `mir2-runs` (có `qtrans/queries_*.parquet`).
**Baseline: `RUN = 'rrf9_w'`, `METHODS = ['cross']`** (`BAAI/bge-reranker-v2-m3`), chọn theo tập có nhãn 40 câu
(19/10): cross-encoder hơn rrf9 có ý nghĩa, ColBERT thì không; rrf9_w (en_lsr, en_bm25 trọng số 0,5) hơn rrf9. `TOP_N` phải lớn hơn
k_chunk (F2 tính theo TẬP: xếp hạng lại trong đúng tập nộp không đổi điểm). Dự án cũ: ColBERT top-400 ~100 phút / 2 GPU.
**Đầu ra:** `runs/<run>_<method>.parquet`, `submissions/*.zip` (doc_id số nguyên, tự giảm k_chunk nếu > 99 MB).
**Thử trên bộ nhãn:** thêm file danh sách câu (vd `queries.csv`) làm Input, đặt `QIDS = 'queries.csv'` → chỉ rerank các câu
đó, ra `runs/<run>_<method>_sub.parquet`, không tạo bài nộp. bge top-400 ~6 s/câu/GPU (đo 19/10).
"""),
    setup_cell("k5-v4", "RUN = 'rrf9_w'                       # baseline: rrf9_w + cross (đánh giá 19/10)\n"
                        "METHODS = ['cross']                  # thêm 'colbert' nếu muốn so sánh; [] = chỉ nộp RUN có sẵn\n"
                        "TOP_N = 400\nK_DOC, K_CHUNK = 200, 185\n"
                        "QIDS = ''                            # thử trên bộ nhãn: 'queries.csv' (tìm trong /kaggle/input) "
                        "hoặc '1,2,3'; '' = cả 1.200 câu\n"
                        "DOC_AGG = 'max'                      # thứ hạng tài liệu khi nộp: max (đoạn tốt nhất) | sum2 | sum3\n"
                        "SUBMIT_RUN = True                    # False: không nộp RUN chưa rerank (chỉ nộp run đã rerank)",
               "('chunks', 'qtrans')"),
    code('''
# (2) Xếp hạng lại trên 2 GPU (chia theo câu hỏi), ghi vào /kaggle/working/runs
hits = locate_all(f'runs/{RUN}.parquet') or locate_all('runs/vi_dense.parquet')   # bỏ qua runs/ của output k5 cũ
assert hits, (f'Không thấy runs/{RUN}.parquet hay 9 nhánh (runs/vi_dense.parquet…) trong /kaggle/input: '
              'Add Input mir2-runs (output k4)')
RUNS_IN = hits[0] / 'runs'
OUT = f'{K.work}/runs'
os.makedirs(OUT, exist_ok=True)
if QIDS and not os.path.exists(QIDS) and not QIDS.replace(',', '').isdigit():
    hits = sorted(glob.glob(f'{KAGGLE}/input/**/{QIDS}', recursive=True), key=len)
    assert hits, f'Không thấy {QIDS} trong /kaggle/input'
    QIDS = hits[0]
QARG = f' --qids "{QIDS}"' if QIDS else ''        # run thử trên bộ nhãn tên <run>_<method>_sub
print('câu hỏi:', QIDS or 'cả 1.200 câu')
if not os.path.exists(f'{RUNS_IN}/{RUN}.parquet'):        # vd mir2-runs cũ chưa có rrf9_w → gộp lại từ 9 nhánh (CPU, ~1 phút)
    for f in glob.glob(f'{RUNS_IN}/*.parquet'):
        if '@' not in os.path.basename(f) and not os.path.exists(f'{OUT}/' + os.path.basename(f)):
            shutil.copy(f, OUT)
    K.set({'retrieve.out_dir': OUT})
    sh(f'python -m mirret.fuse --config {K.cfg} --only {RUN}', 'fuse')
    assert os.path.exists(f'{OUT}/{RUN}.parquet'), f'fuse không tạo được {RUN} (thiếu nhánh, xem dòng "thiếu run" ở trên)'
    RUNS_IN = OUT
print('run đầu vào:', f'{RUNS_IN}/{RUN}.parquet')
for m in METHODS:
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.rerank --config {K.cfg} --run {RUNS_IN}/{RUN}.parquet '
         f'--method {m} --top-n {TOP_N} --shard {g}/2 --device cuda:0 --out {OUT}{QARG}' for g in range(2)],
        f'rerank_{m}')
    sh(f'python -m mirret.rerank --config {K.cfg} --run {RUNS_IN}/{RUN}.parquet --method {m} --top-n {TOP_N} '
       f'--shard 0/2 --out {OUT}{QARG}')              # gộp 2 phần
'''),
    code('''
# (3) Bài nộp: run gộp (không rerank) + từng run đã rerank. Run thử trên bộ nhãn (QIDS) thì không nộp.
K.set({'retrieve.out_dir': OUT})
if os.path.abspath(str(RUNS_IN)) != os.path.abspath(OUT):
    shutil.copy(f'{RUNS_IN}/{RUN}.parquet', f'{OUT}/{RUN}.parquet')
for name in ([] if QIDS else [RUN] * SUBMIT_RUN + [f'{RUN}_{m}' for m in METHODS]):
    sh(f'python -m mirret.submit --config {K.cfg} --run {OUT}/{name}.parquet --k-doc {K_DOC} --k-chunk {K_CHUNK} '
       f'--doc-agg {DOC_AGG}')
report('k5')
'''),
])

# ---------------------------------------------------------------- k6: đánh giá (CPU)
save("k6_eval.ipynb", [
    md("""
# k6 — Đánh giá 9 nhánh: pool + phiếu chấm mù → nhập nhãn → bảng điểm (CPU)

**Cài đặt:** Accelerator = **None (CPU)**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`, `mir2-runs` (output k4), output k5 (các run đã rerank, nếu có);
với `MODE = 'score'` / `'extend'` / `'add'` thêm dataset **`mir2-labels`** (thư mục `judge/` đã điền nhãn). Quy trình: docs/EVAL_PLAN.md.

- `MODE = 'pool'`: chẩn đoán không nhãn (kèm thay đổi do rerank) → chọn `N_QUERIES` câu phân tầng → pool + phiếu ẩn
  nguồn: `sheet_pilot_<người>.csv` (`PILOT` câu, MỌI người cùng chấm) và `sheet_pool_<người>.csv` (phần còn lại, chia
  nhau, 20% câu chấm chéo). Tải cả thư mục `eval/judge/` về, điền cột `grade (0/1/2)` (Google Sheets → *Download CSV*;
  Excel dễ hỏng dấu tiếng Việt), tạo dataset `mir2-labels` từ cả thư mục đó (giữ `key.parquet`, `queries.csv`).
  Chấm pilot trước → `MODE = 'score'` xem kappa pilot → họp, sửa hướng dẫn → mới chấm phiếu pool.
- `MODE = 'add'`: thêm `ADD` câu MỚI (cùng phân tầng, không trùng câu cũ) vào bộ nhãn đã có → `queries.csv` nhóm
  `add1` + phiếu `sheet_add1_<người>.csv` (pool như câu cũ: nhánh top-`BRANCH_DEPTH` + `FUSED`); `EXTEND` (vd
  `'rrf9_w_cross:20'`) đồng thời bổ sung run mới cho các câu cũ → `sheet_ext<k>_<người>.csv`. Cần `mir2-labels` mới nhất
  và output k5 có `rrf9_w_cross`. Tải `eval/judge/` về, chấm các phiếu mới, tạo bản `mir2-labels` mới → `MODE = 'score'`.
- `MODE = 'extend'`: thêm vào pool các mục CHƯA chấm của run mới (`EXTEND`, vd `'rrf9_w:30'`; run gộp có trọng số
  tạo ngay trong notebook qua `FUSE`) → phiếu bổ sung. Giữ nguyên `FUSE` ở lần `MODE = 'score'` sau đó.
- `MODE = 'score'`: nhập nhãn (kappa, bất đồng) → `eval/score.md`, `eval/marginal.md`, `eval/table_k*.md`.
"""),
    setup_cell("k6-v2", "MODE = 'pool'                      # 'pool' | 'add' | 'extend' | 'score'\n"
                        "N_QUERIES, PILOT = 40, 6           # đo trên run thật (19/10): 40 câu, nhánh top-5 + gộp top-20\n"
                        "LABELERS = 'A,B,C'                  # tên không dấu, không cách (vd 'Nhan,Phong,Thanh')\n"
                        "BRANCH_DEPTH = 5                    # top-? của MỖI nhánh vào pool (10 → ~108 mục/câu, 5 → ~66)\n"
                        "FUSED = 'rrf9:20,rrf9_w:20,rrf9_cross:20,rrf9_w_cross:20,rrf9_colbert:20'   # chỉ dùng run có trong input\n"
                        "ADD = 40                           # MODE='add': số câu mới thêm vào bộ nhãn\n"
                        "EXTEND = ''                        # MODE='extend'/'add': 'run:độ sâu,…' (vd 'rrf9_w_cross:20')\n"
                        "ADJUDICATED = ''                   # file phân xử (item,grade) trong mir2-labels, nếu có\n"
                        "FUSE = {}   # run gộp thử thêm, vd {'rrf9_w': 'vi_dense,vi_lsr,vi_bm25,en_dense,en_lsr,en_bm25,"
                        "zh_dense,zh_lsr,zh_bm25*0.5'}\n"
                        "KS = '5,10,20,50,100,185,200'   # K = BRANCH_DEPTH: so 9 nhánh công bằng (đã chấm đủ)", "('chunks', 'qtrans')"),
    code('''
# (2) Gom mọi run (k4 + k5) vào /kaggle/working/runs; chép thư mục nhãn (nếu có) vào eval/judge
OUT = f'{K.work}/runs'
os.makedirs(OUT, exist_ok=True)
for d in locate_all('runs'):
    for f in glob.glob(f'{d}/runs/*.parquet') + glob.glob(f'{d}/runs/*.json'):
        if not os.path.exists(f'{OUT}/' + os.path.basename(f)):
            shutil.copy(f, OUT)
K.set({'retrieve.out_dir': OUT})
print('run:', sorted(os.path.basename(f)[:-8] for f in glob.glob(f'{OUT}/*.parquet')))
JD = f'{K.work}/eval/judge'
lab = locate_all('judge/key.parquet')
if MODE != 'pool':
    assert lab, 'Chưa Add Input dataset nhãn (thư mục judge/ có key.parquet)'
    if len(lab) > 1:
        print('[CẢNH BÁO] nhiều thư mục judge/ trong input, dùng', lab[0], '— chỉ nên thêm bản mir2-labels mới nhất')
    shutil.copytree(lab[0] / 'judge', JD, dirs_exist_ok=True)
    print('nhãn từ', lab[0] / 'judge', ':', sorted(os.listdir(JD)))
'''),
    code('''
# (3) Chạy theo MODE (trước đó tạo các run gộp thử trong FUSE — rẻ, chạy lại mỗi lần)
for name, spec in FUSE.items():
    sh(f'python -m mirret.fuse --config {K.cfg} --runs "{spec}" --name {name}', 'fuse')
have = {os.path.basename(f)[:-8] for f in glob.glob(f'{OUT}/*.parquet')}
if MODE == 'pool':
    rr = ','.join(n for n in ('rrf9_colbert', 'rrf9_cross') if n in have)
    sh(f'python -m mirret.diag --config {K.cfg}' + (f' --reranked {rr}' if rr else ''), 'diag')
    sh(f'python -m mirret.judge select --config {K.cfg} --n {N_QUERIES} --pilot {PILOT}', 'select')
    fused = ','.join(x for x in FUSED.split(',') if x.split(':')[0] in have)
    sh(f'python -m mirret.judge pool --config {K.cfg} --branch-depth {BRANCH_DEPTH} --fused {fused} '
       f'--labelers {LABELERS}', 'pool')
elif MODE == 'add':
    import pandas as pd, yaml
    cf = yaml.safe_load(open(K.cfg, encoding='utf-8'))['fuse']['configs']
    for n in [x.split(':')[0] for x in FUSED.split(',') if x]:
        if n not in have and n in cf:                    # vd rrf9_w: gộp lại từ 9 nhánh (CPU, ~1 phút)
            sh(f'python -m mirret.fuse --config {K.cfg} --only {n}', 'fuse')
    have = {os.path.basename(f)[:-8] for f in glob.glob(f'{OUT}/*.parquet')}
    miss = [x for x in FUSED.split(',') if x and x.split(':')[0] not in have]
    if miss:
        print('[CẢNH BÁO] thiếu run, pool câu mới sẽ không có:', miss)
    assert 'rrf9_w_cross' in have, 'Add Input output k5 có runs/rrf9_w_cross.parquet (baseline)'
    sh(f'python -m mirret.judge select --config {K.cfg} --add {ADD}', 'select')
    new = sorted(x for x in pd.read_csv(f'{JD}/queries.csv', encoding='utf-8-sig')['set'].unique()
                 if str(x).startswith('add'))[-1]
    fused = ','.join(x for x in FUSED.split(',') if x and x.split(':')[0] in have)
    sh(f'python -m mirret.judge pool --config {K.cfg} --sets {new} --branch-depth {BRANCH_DEPTH} --fused {fused} '
       f'--labelers {LABELERS}', 'pool')
    if EXTEND:
        sh(f'python -m mirret.judge extend --config {K.cfg} --runs {EXTEND} --labelers {LABELERS}', 'extend')
elif MODE == 'extend':
    sh(f'python -m mirret.judge extend --config {K.cfg} --runs {EXTEND} --labelers {LABELERS}', 'extend')
else:
    adj = f' --adjudicated {lab[0] / "judge" / ADJUDICATED}' if ADJUDICATED else ''
    sh(f'python -m mirret.judge ingest --config {K.cfg}{adj}', 'ingest')
    allruns = ','.join(sorted(have))                  # gồm cả run từng mức (@parents/@children) và run rerank
    sh(f'python -m mirret.evaluate score --config {K.cfg} --ks {KS} --runs {allruns}', 'score')
    sh(f'python -m mirret.evaluate marginal --config {K.cfg} --k 50', 'marginal')
    for k in (BRANCH_DEPTH, 20, 50, 200):
        sh(f'python -m mirret.evaluate table --config {K.cfg} --ks {k}', f'table_{k}')
shutil.rmtree(OUT)                                     # run đã có trong dataset input, không chép lại vào output
report('k6')
'''),
])

# ---------------------------------------------------------------- k7: tài liệu giả (Query2Doc / HyDE) + truy xuất + rerank
save("k7_qexp.ipynb", [
    md("""
# k7 — Mở rộng câu hỏi bằng tài liệu giả: Query2Doc vs HyDE (task B1 + B2)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`, `mir2-qtrans` (concepts của bản dịch t2, cho `bm25c`),
`mir2-runs` (output k4: 9 nhánh baseline + `qtrans/queries_*`), mọi `mir2-emb-m*`, dataset chứa `queries.csv`
(danh sách câu có nhãn); thêm output k7 phiên trước nếu chạy tiếp (chỉ sinh phần đoạn giả còn thiếu).

**Thiết kế (R2MED 2025, phụ lục E):** mỗi câu hỏi × ngôn ngữ một đoạn giả ~100–150 từ do Qwen2.5-14B (4-bit) viết thẳng
bằng ngôn ngữ của nhánh, sinh tất định.
- `q2d` (Query2Doc): prompt 2 ví dụ; truy hồi bằng câu hỏi + đoạn giả (BM25: câu hỏi lặp 3 lần + đoạn giả).
- `hyde` (HyDE): prompt không ví dụ; truy hồi bằng riêng đoạn giả.
- `bm25c`: câu hỏi + concepts của bản dịch, chỉ nhánh BM25 (không cần LLM).
Reranker (bge-reranker-v2-m3) luôn dùng câu hỏi gốc / bản dịch.

**Run gửi chấm** (so với `rrf9_w_cross`): `q2d9_cross_sub`, `hyde9_cross_sub`, `q2dlex_cross_sub` (đoạn giả chỉ ở 6
nhánh LSR + BM25), `bm25c9_cross_sub`. Mẫu đoạn giả: `qexp/<biến thể>/samples.md`; độ trùng top-100 với nhánh baseline
in ở ô (6).
**Thời gian:** ô (3) in s/lượt — 40 câu × 3 ngôn ngữ × 2 chế độ ước ~20–40 phút; chỉ mục dense như k4; rerank 4 run × 40 câu
~10 phút. Muốn chạy 1.200 câu: nhân thời gian sinh đo được lên 30 trước khi quyết định.
**Chạy đủ 1.200 câu (1 phiên):** `QIDS = ''`, `MODES = ['q2d']`, `FUSED = ['q2d9']`, `BATCH = 16` → thêm
`runs/q2d9_cross.parquet` + `submissions/q2d9_cross.zip`. Ước tính 6,5–10 giờ (sinh ~3 giờ, chỉ mục dense 1–4,5 giờ).
**Đầu ra:** `qexp/`, `runs/` (nhánh biến thể, run gộp, `*_cross_sub`) → dataset **`mir2-qexp`**.
"""),
    setup_cell("k7-v1", "MODES = ['q2d', 'hyde']\nMODEL = 'qwen25_14b'\n"
                        "QIDS = 'queries.csv'                 # '' = cả 1.200 câu\n"
                        "LIMIT = 0                            # >0: chỉ N câu đầu (đọc thử đoạn giả)\n"
                        "BATCH, QUANT = 8, '4bit'\n"
                        "FUSED = ['q2d9', 'hyde9', 'q2dlex', 'bm25c9']   # fuse.configs trong retrieval.yaml\n"
                        "TOP_N = 400\nSAVE_INDEX = False", "('chunks',)"),
    code('''
# (2) Đầu vào: danh sách câu; câu hỏi 3 ngôn ngữ GIỐNG baseline; bản ghi dịch (concepts); đoạn giả phiên trước; 9 nhánh
if QIDS and not os.path.exists(QIDS) and not QIDS.replace(',', '').isdigit():
    nrow = lambda h: sum(1 for _ in open(h, encoding='utf-8-sig')) - 1
    hits = sorted(glob.glob(f'{KAGGLE}/input/**/{QIDS}', recursive=True), key=lambda h: (-nrow(h), len(h)))
    assert hits, f'Không thấy {QIDS} trong /kaggle/input'
    print('các file', QIDS, 'trong input:', [(h, nrow(h)) for h in hits])
    QIDS = hits[0]                                    # bộ nhãn chỉ tăng → lấy bản nhiều câu nhất (vd 80, không phải 40)
QARG = f' --qids "{QIDS}"' if QIDS else ''
print('câu hỏi:', QIDS or 'cả 1.200 câu')
base = (locate_all('runs/vi_dense.parquet') or [None])[0]
assert base is not None, 'Add Input mir2-runs (output k4)'
QT, RUNS = f'{K.work}/qtrans', f'{K.work}/runs'
os.makedirs(QT, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
for f in glob.glob(f'{base}/qtrans/queries_*'):
    shutil.copy(f, QT)
for d in locate_all('qtrans'):
    for f in glob.glob(f'{d}/qtrans/*/records*.jsonl'):
        dst = f'{QT}/' + os.path.relpath(f, f'{d}/qtrans')
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst)
def take(src, dst):                                   # trùng tên giữa các phiên → giữ bản lớn hơn (nhiều câu hơn)
    if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst); print('chép', src)
for d in locate_all('qexp'):                           # output k7 phiên trước: đoạn giả, chỉ mục dense, run biến thể
    for f in glob.glob(f'{d}/qexp/*/records*.jsonl'):
        take(f, f'{K.work}/qexp/' + os.path.relpath(f, f'{d}/qexp'))
    for f in glob.glob(f'{d}/index/dense_*'):
        take(f, f'{K.work}/index/' + os.path.basename(f))
    for f in glob.glob(f'{d}/runs/**/*.parquet', recursive=True) + glob.glob(f'{d}/runs/*.json'):
        take(f, f'{K.work}/runs/' + os.path.relpath(f, f'{d}/runs'))
copied = []
for f in glob.glob(f'{base}/runs/*.parquet'):
    if '@' not in os.path.basename(f):
        shutil.copy(f, RUNS); copied.append(f'{RUNS}/' + os.path.basename(f))
K.set({'translate.out_dir': QT, 'retrieve.out_dir': RUNS})
bm = locate('index/bm25', required=False)
if bm is not None:
    K.set({'bm25.index_dir': str(bm / 'index' / 'bm25')})
EMB = ','.join(sorted({str(d / 'emb') for d in locate_all('emb')}))
print('emb:', EMB)
sh('python -m pip install -q -U bitsandbytes accelerate')
'''),
    code('''
# (3) Sinh đoạn giả: mỗi chế độ 2 GPU song song (mỗi GPU một bản 4-bit, mỗi bản một nửa số câu)
lim = f' --limit {LIMIT}' if LIMIT else ''
for m in MODES:
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.expand generate --config {K.cfg} --mode {m} --model {MODEL} '
         f'--shard {g}/2 --batch {BATCH} --quant {QUANT}{QARG}{lim}' for g in range(2)], f'expand_{m}')
'''),
    code('''
# (4) Dựng câu hỏi cho từng biến thể + đọc mẫu đoạn giả
for m in MODES + ['bm25c']:
    sh(f'python -m mirret.expand build --config {K.cfg} --mode {m} --model {MODEL}')
for m in MODES:
    print(open(f'{K.work}/qexp/{m}/samples.md', encoding='utf-8').read()[:3000])
'''),
    code('''
# (5) Chỉ mục dense (như k4) + mã hoá câu hỏi các biến thể (GPU)
for lv in ('parents', 'children'):
    sh(f'python -m mirret.index dense --config {K.cfg} --level {lv} --emb {EMB}', f'dense_{lv}')
for m in MODES:
    sh(f'python -m mirret.retrieve qencode --config {K.cfg} --device cuda:0 --variant {m}', f'qencode_{m}')
'''),
    code('''
# (6) 9 nhánh cho mỗi biến thể (bm25c: chỉ BM25) → gộp; độ trùng top-100 với nhánh baseline (không cần nhãn)
import pandas as pd
def done(m, methods):                                  # đủ nhánh của phiên trước, đủ số câu của biến thể → bỏ qua
    n = pd.read_parquet(glob.glob(f'{K.work}/qexp/{m}/queries_vi*.parquet')[0]).shape[0]
    fs = [f'{RUNS}/{m}_{lg}_{me}.parquet' for lg in ('vi', 'en', 'zh') for me in methods]
    return all(os.path.exists(f) and pd.read_parquet(f, columns=['qid'])['qid'].nunique() >= n - 5 for f in fs)
for m in MODES:
    if done(m, ('dense', 'lsr', 'bm25')):
        print(m, ': 9 nhánh đã có từ phiên trước → bỏ qua truy xuất'); continue
    sh(f'python -m mirret.retrieve run --config {K.cfg} --emb {EMB} --variant {m}', f'retrieve_{m}')
if not done('bm25c', ('bm25',)):
    sh(f'python -m mirret.retrieve run --config {K.cfg} --methods bm25 --variant bm25c', 'retrieve_bm25c')
sh(f'python -m mirret.fuse --config {K.cfg} --only {",".join(FUSED)}', 'fuse')
import pandas as pd
from mirret import runs as R
rows = []
for m in MODES + ['bm25c']:
    for b in [f'{lg}_{me}' for lg in ('vi', 'en', 'zh') for me in ('dense', 'lsr', 'bm25')]:
        p, q = f'{RUNS}/{m}_{b}.parquet', f'{RUNS}/{b}.parquet'
        if os.path.exists(p) and os.path.exists(q):
            a = R.topk_sets(R.read(p, 100), 100, 'unit')
            c = R.topk_sets(R.read(q, 100, qids=set(a)), 100, 'unit')
            rows.append({'biến thể': m, 'nhánh': b,
                         'trùng@100': round(sum(len(a[x] & c.get(x, set())) for x in a) / (100 * len(a)), 3)})
print('Tỷ lệ đoạn top-100 trùng với nhánh baseline (thấp = biến thể đưa nhiều tài liệu mới vào):')
print(pd.DataFrame(rows).pivot(index='nhánh', columns='biến thể', values='trùng@100'))
'''),
    code('''
# (7) Rerank bge-reranker-v2-m3 top-400 (câu hỏi gốc) → run gửi chấm; bỏ bản sao 9 nhánh baseline khỏi output
for f in FUSED:
    if not os.path.exists(f'{RUNS}/{f}.parquet'):
        print('bỏ qua', f, '(chưa gộp được)'); continue
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.rerank --config {K.cfg} --run {RUNS}/{f}.parquet --method cross '
         f'--top-n {TOP_N} --shard {g}/2 --device cuda:0{QARG}' for g in range(2)], f'rerank_{f}')
    sh(f'python -m mirret.rerank --config {K.cfg} --run {RUNS}/{f}.parquet --method cross --top-n {TOP_N} '
       f'--shard 0/2{QARG}')
if not QIDS:                                          # chạy đủ 1.200 câu → bài nộp cho từng run đã rerank
    K.set({'retrieve.out_dir': RUNS})
    for f in FUSED:
        if os.path.exists(f'{RUNS}/{f}_cross.parquet'):
            sh(f'python -m mirret.submit --config {K.cfg} --run {RUNS}/{f}_cross.parquet', f'submit_{f}')
for f in copied:
    os.remove(f)
if not SAVE_INDEX:
    for f in glob.glob(f'{K.work}/index/dense_*'):
        os.remove(f)
print('run gửi chấm:', sorted(os.path.basename(f) for f in glob.glob(f'{RUNS}/*_cross*.parquet')))
report('k7')
'''),
])

# ---------------------------------------------------------------- k8: viết lại câu hỏi + thuật ngữ (query_expand)
save("k8_qx.ipynb", [
    md("""
# k8 — Viết lại câu hỏi + thuật ngữ mở rộng (mirret/query_expand.py)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code` (bản mới có `mirret/query_expand.py`), `mir2-data`, `mir2-chunks` (chỉ mục BM25 — cho ô (5)),
dataset có `qtrans/queries_{vi,en,zh}.parquet` (output k4 `mir2-runs` hoặc `mir2-qtrans`), dataset có file câu hỏi
`80queries.csv`; thêm output k8 phiên trước (`mir2-qx`) nếu chạy tiếp — lượt đã có trong cache không sinh lại.

Qwen2.5-14B (4-bit) sinh cho mỗi câu × {vi, en, zh} một JSON `{"rewrite", "terms"}` (greedy). vi = câu gốc, en/zh =
bản dịch t2 (cùng bản baseline). Gợi ý thuật ngữ từ `configs/med_glossary_vi.tsv`.
**Thử 5 câu** (mặc định): `LIMIT = 5` → 15 lượt, 1 GPU, ~10 phút kể cả nạp mô hình.
**Đủ 80 câu / 1.200 câu:** `LIMIT = 0` (+ `QIDS = ''` cho 1.200 câu) → 2 GPU song song; ô (3) in s/lượt.
**Chia việc với máy khác** (vd Colab, notebook `c8_qx_colab.ipynb`): `SHARDS = ['0/3', '1/3']` (mỗi GPU một phần), máy kia
chạy `2/3`; mỗi phần ghi `records-<i>of<N>.jsonl` riêng, `build` gộp mọi file.
**Đầu ra:** `qexp/rw3-qwen25_14b/records*.jsonl` (cache theo phiên bản prompt, có đầu ra thô), `qexp/rw/`, `qexp/rwt/` → dataset **`mir2-qx`**.
"""),
    setup_cell("k8-v4", "MODEL = 'qwen25_14b'\n"
                        "QIDS = '80queries.csv'               # '' = cả 1.200 câu\n"
                        "LIMIT = 5                            # >0: chỉ N câu đầu (thử); 0 = mọi câu trong QIDS\n"
                        "SHARDS = []                          # vd ['0/3', '1/3']: mỗi GPU một phần (phần 2/3 chạy ở máy khác)\n"
                        "BATCH, QUANT = 8, '4bit'\n"
                        "QID_EXPLAIN = None                   # câu in token + trọng số BM25 (None = câu nhỏ nhất)",
               "('qtrans',)"),
    code('''
# (2) Đầu vào: file câu hỏi, chỉ mục BM25, cache phiên trước; bản dịch đang dùng
if QIDS and not os.path.exists(QIDS) and not QIDS.replace(',', '').isdigit():
    nrow = lambda h: sum(1 for _ in open(h, encoding='utf-8-sig')) - 1
    hits = sorted(glob.glob(f'{KAGGLE}/input/**/{QIDS}', recursive=True), key=lambda h: (-nrow(h), len(h)))
    assert hits, f'Không thấy {QIDS} trong /kaggle/input (Add Input dataset chứa file này)'
    print('các file', QIDS, 'trong input:', [(h, nrow(h)) for h in hits])
    QIDS = hits[0]
QARG = f' --qids "{QIDS}"' if QIDS else ''
qt = locate('qtrans/queries_vi.parquet') / 'qtrans'
print('câu hỏi:', QIDS or 'cả 1.200 câu', '| bản dịch:', qt,
      open(qt / 'queries_meta.json', encoding='utf-8').read() if (qt / 'queries_meta.json').exists() else '')
bm = locate('index/bm25', required=False)
if bm is not None:
    K.set({'bm25.index_dir': str(bm / 'index' / 'bm25')})
else:
    print('[chú ý] không thấy index/bm25 (mir2-chunks) → ô (5) sẽ báo thiếu chỉ mục')
for d in locate_all('qexp'):                          # cache phiên trước → chỉ sinh phần còn thiếu
    for f in glob.glob(f'{d}/qexp/rw*-*/records*.jsonl'):
        dst = f'{K.work}/qexp/' + os.path.relpath(f, f'{d}/qexp')
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(f, dst); print('chép', f)
sh('python -m pip install -q -U bitsandbytes accelerate')
'''),
    code('''
# (3) Sinh rewrite + thuật ngữ (Qwen2.5-14B 4-bit). SHARDS: mỗi GPU một phần; thử ít câu: 1 GPU; còn lại: mỗi GPU một nửa
lim = f' --limit {LIMIT}' if LIMIT else ''
gen = f'python -m mirret.query_expand generate --config {K.cfg} --model {MODEL} --batch {BATCH} --quant {QUANT}{QARG}{lim}'
if SHARDS:
    par([f'CUDA_VISIBLE_DEVICES={g} {gen} --shard {s}' for g, s in enumerate(SHARDS)], 'qx_generate')
elif LIMIT and LIMIT <= 40:
    sh(f'CUDA_VISIBLE_DEVICES=0 {gen}', 'qx_generate')
else:
    par([f'CUDA_VISIBLE_DEVICES={g} {gen} --shard {g}/2' for g in range(2)], 'qx_generate')
'''),
    code('''
# (4) Dựng câu hỏi biến thể (qexp/rw, qexp/rwt) + in đầu ra 3 ngôn ngữ, số lượt lỗi JSON
sh(f'python -m mirret.query_expand build --config {K.cfg} --model {MODEL}')
sh(f'python -m mirret.query_expand show --config {K.cfg} --model {MODEL} --limit {LIMIT or 5}')
'''),
    code('''
# (5) Một câu: token + trọng số BM25 trước / sau mở rộng (w_base, w_rewrite, w_terms, max_df_ratio trong qexpand)
qa = f' --qid {QID_EXPLAIN}' if QID_EXPLAIN is not None else ''
sh(f'python -m mirret.query_expand explain --config {K.cfg}{qa}', check=False)
report('k8')
print('→ Save Version, tạo dataset mir2-qx từ output (qexp/)')
'''),
])

# ---------------------------------------------------------------- c8: phần câu hỏi của k8 chạy trên Google Colab
save("c8_qx_colab.ipynb", [
    md("""
# c8 — Viết lại câu hỏi + thuật ngữ trên Google Colab (một phần của k8)

**Runtime → Change runtime type → T4 GPU.** Mỗi phiên Colab 1 GPU → chạy một phần (`SHARD`), các phần khác ở Kaggle
(k8, `SHARDS = ['0/3', '1/3']`).
**Chuẩn bị trên Google Drive** thư mục `MyDrive/mir2/`:
- `mir2_code.zip` (CÙNG bản với Kaggle — ô (1) in VERSION để đối chiếu);
- `qtrans/queries_vi.parquet`, `queries_en.parquet`, `queries_zh.parquet` (tải từ dataset `mir2-runs`, thư mục `qtrans/`).

Kết quả ghi thẳng vào Drive: `MyDrive/mir2/work/qexp/rw3-qwen25_14b/records.jsonl`, nối sau MỖI lô → mất kết
nối thì chạy lại mọi ô, lượt đã có được bỏ qua. Xong: tải file đó về.
"""),
    code('''
# (1) Chuẩn bị. Chỉ sửa các biến viết HOA.
SHARD = '0/1'                                  # cả 1.200 câu; chia với Kaggle: '2/3' (k8 chạy 0/3, 1/3)
MODEL, BATCH, QUANT = 'qwen25_14b', 32, 'fp16'  # GPU 96 GB (G4); T4: 8, '4bit'
DRIVE = '/content/drive/MyDrive/mir2'
import os, shutil, sys, zipfile
from google.colab import drive
drive.mount('/content/drive')
CODE, CFG = '/content/code', '/content/config.yaml'
shutil.rmtree(CODE, ignore_errors=True)
zipfile.ZipFile(f'{DRIVE}/mir2_code.zip').extractall(CODE)
print('mir2-code', open(f'{CODE}/VERSION.txt', encoding='utf-8').read().strip())
miss = [lg for lg in ('vi', 'en', 'zh') if not os.path.exists(f'{DRIVE}/qtrans/queries_{lg}.parquet')]
assert not miss, f'Thiếu {DRIVE}/qtrans/queries_{miss}.parquet'
os.chdir(CODE); sys.path.insert(0, CODE)
from mirret.kaggle_nb import write_config
write_config(f'{CODE}/configs/retrieval.yaml', CFG, {'translate.out_dir': f'{DRIVE}/qtrans', 'paths.work_dir': f'{DRIVE}/work'})
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U bitsandbytes accelerate
'''),
    code('''
# (2) Sinh: G4 fp16 ước ~20–50 phút cho 1.200 câu × 3 ngôn ngữ (T4 4-bit: ~4 s/lượt), chưa kể ~10 phút tải + nạp mô hình
!python -m mirret.query_expand generate --config {CFG} --model {MODEL} --shard {SHARD} --batch {BATCH} --quant {QUANT}
'''),
    code('''
# (3) Kiểm tra nhanh: số lượt, lỗi JSON, 3 câu mẫu → tải records-*.jsonl trong Drive về
!python -m mirret.query_expand show --config {CFG} --model {MODEL} --limit 3
!ls -la {DRIVE}/work/qexp/rw3-qwen25_14b/
'''),
])
