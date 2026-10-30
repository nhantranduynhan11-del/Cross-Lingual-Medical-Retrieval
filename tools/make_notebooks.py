"""Sinh các notebook Kaggle của pipeline truy hồi (notebooks/k1…k6). Chạy: python tools/make_notebooks.py"""
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
    setup_cell("k5-v15","RUN = 'rrf9_w'                     # baseline: rrf9_w + cross (đánh giá 19/10)\n"
                        "METHODS = ['cross']                  # thêm 'colbert' nếu muốn so sánh; [] = chỉ nộp RUN có sẵn\n"
                        "TOP_N = 400\nK_DOC, K_CHUNK = 200, 185\n"
                        "QIDS = ''                            # thử trên bộ nhãn: 'queries.csv' (tìm trong /kaggle/input) "
                        "hoặc '1,2,3'; '' = cả 1.200 câu\n"
                        "DOC_AGG = 'max'                      # thứ hạng tài liệu khi nộp: max (đoạn tốt nhất) | sum2 | sum3\n"
                        "MAX_PER_DOC = [0]                    # tối đa n đoạn mỗi tài liệu trong 185 đoạn; [0] = không giới hạn; "
                        "[1, 2] = 2 bài nộp _mpd1, _mpd2\n"
                        "CHUNK_PROBS = ''                     # phân bổ lại ngân sách đoạn theo P(yes): tên run Qwen3 trong "
                        "Input (vd 'lin07_qwen3i') → thêm bài nộp _cbud; '' = tắt\n"
                        "CHUNK_K_MIN = 100                    # số đoạn tối thiểu mỗi câu khi phân bổ (k tối đa 400)\n"
                        "CHUNK_CALIB = ''                     # hiệu chỉnh P(yes) theo 'precision,recall' đoạn trên LB của bài "
                        "185 đoạn (vd '0.0525,0.1555') → bài _cbudcal\n"
                        "CALIB_QIDS = 'public600_qids.txt'   # câu LB đã chấm (Input) để khớp hiệu chỉnh\n"
                        "REUSE = {}                           # dùng lại điểm đã chấm: {phương pháp: run hoặc [các run] trong "
                        "Input}, vd {'cross': ['lin07_cross', 'lin12_07_cross'], 'qwen3i': 'lin07_qwen3i_1200'} → chỉ chấm "
                        "đoạn mới\n"
                        "ENSEMBLE = False                     # True + METHODS có 'cross' và 'qwen3i' → thêm run <RUN>_ensi\n"
                        "SUBMIT_RUN = True                    # False: không nộp RUN chưa rerank (chỉ nộp run đã rerank)\n"
                        "CROSS_MODEL = ''                     # model bge đã fine-tune (thư mục có finetune_meta.json trong Input, "
                        "vd 'bge') → run <RUN>_crossft\n"
                        "QWEN_LORA = ''                       # adapter LoRA Qwen3 đã fine-tune (vd 'qwen') → run <RUN>_qwen3ift\n"
                        "PUBMED_ONLY = False                  # True: chỉ chấm đoạn bài PubMed (model ft dùng riêng cho tiếng "
                        "Anh) → run <RUN>_<method>ftpm, không ensemble / bài nộp\n"
                        "FT_ALPHA = 1.0                       # WiSE-FT: < 1 kéo model ft về gần model gốc (0,5 = nửa đường) "
                        "→ tên run ft50",
               "('chunks', 'qtrans')"),
    code('''
# (2) Xếp hạng lại trên 2 GPU (chia theo câu hỏi), ghi vào /kaggle/working/runs
NAMES = [x.strip() for x in RUN.split(',') if x.strip()]  # METHODS = [] → nộp được nhiều run có sẵn: 'a,b,c'
assert len(NAMES) == 1 or not METHODS, 'Nhiều RUN chỉ dùng với METHODS = [] (chỉ tạo bài nộp)'
RUN = NAMES[0]
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
        if not os.path.exists(f'{OUT}/' + os.path.basename(f)):   # cả @parents/@children: gộp theo mức như lin07
            shutil.copy(f, OUT)
    K.set({'retrieve.out_dir': OUT})
    sh(f'python -m mirret.fuse --config {K.cfg} --only {RUN}', 'fuse')
    assert os.path.exists(f'{OUT}/{RUN}.parquet'), f'fuse không tạo được {RUN} (thiếu nhánh, xem dòng "thiếu run" ở trên)'
    RUNS_IN = OUT
print('run đầu vào:', f'{RUNS_IN}/{RUN}.parquet')
def run_path(n):                                     # run trong phiên này (OUT) hoặc trong Input
    if os.path.exists(f'{OUT}/{n}.parquet'):
        return f'{OUT}/{n}.parquet'
    h = locate_all(f'runs/{n}.parquet')
    assert h, f'Không thấy runs/{n}.parquet trong /kaggle/input'
    return str(h[0] / 'runs' / f'{n}.parquet')
DEPTH = '' if TOP_N == 400 else str(TOP_N)           # TOP_N khác 400 → tên run riêng, vd lin07_cross800
sfx = '_sub' if QIDS else ''
def locate_model(name):                              # thư mục output k8c trong Input (có finetune_meta.json)
    h = sorted(glob.glob(f'{KAGGLE}/input/**/{name}/finetune_meta.json', recursive=True), key=len)
    assert h, f'Không thấy {name}/finetune_meta.json trong /kaggle/input (Add Input output k8c)'
    return os.path.dirname(h[0])
FT = {'cross': bool(CROSS_MODEL), 'qwen3i': bool(QWEN_LORA), 'qwen3': bool(QWEN_LORA)}
if CROSS_MODEL:
    K.set({'rerank.cross_model': locate_model(CROSS_MODEL)})
if QWEN_LORA:
    K.set({'rerank.qwen3_lora': locate_model(QWEN_LORA)})
    sh('python -m pip install -q peft'); sh('python -m pip uninstall -y -q torchao')   # peft mới đòi torchao > 0.16, Kaggle có 0.10, không dùng tới
assert not any(FT.get(m) for m in REUSE), 'model đã fine-tune: không dùng lại điểm cũ của phương pháp đó (bỏ khỏi REUSE)'
K.set({'rerank.ft_alpha': FT_ALPHA})
FTN = ('ft2' if any(x.endswith('synth2') for x in (CROSS_MODEL, QWEN_LORA)) else 'ft') + \
      (str(round(FT_ALPHA * 100)) if FT_ALPHA < 1 else '')    # vd ft50 (v1, FT_ALPHA 0,5), ft2 (model v2)
TAG = lambda m: f'{RUN}_{m}{FTN * FT.get(m, False)}{"pm" * PUBMED_ONLY}{DEPTH}{sfx}'
PM = ' --pubmed-only' if PUBMED_ONLY else ''
assert not (PUBMED_ONLY and ENSEMBLE), 'PUBMED_ONLY: tắt ENSEMBLE (ghép trên máy bằng tools/en_reorder.py)'
for m in METHODS:
    RU = REUSE.get(m) or []
    RA = ' --reuse ' + ' '.join(run_path(x) for x in ([RU] if isinstance(RU, str) else RU)) if RU else ''
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.rerank --config {K.cfg} --run {RUNS_IN}/{RUN}.parquet '
         f'--method {m} --top-n {TOP_N} --shard {g}/2 --device cuda:0 --out {OUT} --name {TAG(m)}{QARG}{RA}{PM}'
         for g in range(2)], f'rerank_{m}')
    sh(f'python -m mirret.rerank --config {K.cfg} --run {RUNS_IN}/{RUN}.parquet --method {m} --top-n {TOP_N} '
       f'--shard 0/2 --out {OUT} --name {TAG(m)}{QARG}{RA}{PM}')      # gộp 2 phần
ENS = []
if ENSEMBLE:
    assert 'cross' in METHODS and 'qwen3i' in METHODS, 'ENSEMBLE cần METHODS có cả cross và qwen3i'
    sh(f'python -m mirret.ensemble --base {RUNS_IN}/{RUN}.parquet --bge {OUT}/{TAG("cross")}.parquet '
       f'--qwen {OUT}/{TAG("qwen3i")}.parquet --top-n {TOP_N} --out {OUT} --suffix i{FTN * any(FT.values())}{DEPTH}{sfx}', 'ensemble')
    for f in glob.glob(f'{OUT}/{RUN}_ip0[68]i*'):                # ensemble.py ghi thêm hai run nội suy — không dùng
        os.remove(f)
    ENS = [f'{RUN}_ensi{FTN * any(FT.values())}{DEPTH}']
'''),
    code('''
# (3) Bài nộp: run gộp (không rerank) + từng run đã rerank. Run thử trên bộ nhãn (QIDS) thì không nộp.
K.set({'retrieve.out_dir': OUT})
if os.path.abspath(str(RUNS_IN)) != os.path.abspath(OUT):
    shutil.copy(f'{RUNS_IN}/{RUN}.parquet', f'{OUT}/{RUN}.parquet')
for n in NAMES[1:]:
    h = locate_all(f'runs/{n}.parquet')
    assert h, f'Không thấy runs/{n}.parquet trong /kaggle/input'
    shutil.copy(h[0] / 'runs' / f'{n}.parquet', f'{OUT}/{n}.parquet')
CPARG = ''
if CHUNK_PROBS:
    CPARG = f" --chunk-probs {run_path(CHUNK_PROBS)} --chunk-k-min {CHUNK_K_MIN}"   # run của phiên này hoặc Input
    if CHUNK_CALIB:
        hq = sorted(glob.glob(f'{KAGGLE}/input/**/{CALIB_QIDS}', recursive=True), key=len)
        assert hq, f'Không thấy {CALIB_QIDS} trong /kaggle/input'
        CPARG += f' --chunk-calib {CHUNK_CALIB} --calib-qids {hq[0]}'
for name in ([] if QIDS or PUBMED_ONLY else [RUN] * SUBMIT_RUN + NAMES[1:] + [TAG(m) for m in METHODS] + ENS):
    for mpd in MAX_PER_DOC:
        for cp in sorted({'', CPARG}):                    # CHUNK_PROBS → cả bài thường (185 đoạn) lẫn bài _cbud
            sh(f'python -m mirret.submit --config {K.cfg} --run {OUT}/{name}.parquet --k-doc {K_DOC} '
               f'--k-chunk {K_CHUNK} --doc-agg {DOC_AGG} --max-per-doc {mpd}{cp}')
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

**Biến thể không cần sinh (25/10):** `MODES = ['prf']` (phản hồi giả định: câu hỏi + 3 đoạn cùng ngôn ngữ đầu `FEEDBACK`,
vd `lin07_cross` — thêm dataset có `runs/lin07_cross.parquet`), `FUSED = ['lin07prf']`; `MODES = ['rw']` (câu viết lại
của nhóm, thêm dataset có `qexp/rw/queries_*.parquet`), `FUSED = ['lin07rw']`. Thêm output k7 cũ có `index/dense_*` để
khỏi dựng lại chỉ mục dense. Nhánh baseline chép cả mức `@parents/@children` (lin07 gộp theo mức như bản đã nộp).

**Fine-tune (27/10):** câu hỏi tổng hợp của k8a → `MODES = ['synth']`, `FUSED = ['synth_lin07']`, `QIDS = ''`,
`RERANK = False`; thêm Input `mir2-synth` (output k8a, có `qexp/synth/queries_*.parquet`) → `runs/synth_lin07.parquet`; v2 (30/10): `MODES = ['synth2']`, `FUSED = ['synth2_lin07']`, Input `mir2-synth2`
cho k8b.
"""),
    setup_cell("k7-v3", "MODES = ['q2d', 'hyde']             # 'q2d' | 'hyde' (sinh bằng LLM) | 'prf' | 'rw' | 'synth' (không sinh)\n"
                        "MODEL = 'qwen25_14b'\n"
                        "QIDS = 'queries.csv'                 # '' = cả 1.200 câu\n"
                        "LIMIT = 0                            # >0: chỉ N câu đầu (đọc thử đoạn giả)\n"
                        "BATCH, QUANT = 8, '4bit'\n"
                        "FUSED = ['q2d9', 'hyde9', 'q2dlex', 'bm25c9']   # fuse.configs trong retrieval.yaml\n"
                        "FEEDBACK = 'lin07_cross'             # 'prf': run đã rerank làm phản hồi (runs/<tên>.parquet trong Input)\n"
                        "TOP_N = 400\nSAVE_INDEX = False\n"
                        "RERANK = True                        # False: chỉ truy hồi + gộp (MODES=['synth'] cho fine-tune)",
               "('chunks',)"),
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
    for f in glob.glob(f'{d}/qexp/*/records*.jsonl') + glob.glob(f'{d}/qexp/*/queries_*.parquet'):
        take(f, f'{K.work}/qexp/' + os.path.relpath(f, f'{d}/qexp'))   # queries_*: biến thể dựng sẵn (vd rw)
    for f in glob.glob(f'{d}/index/dense_*'):
        take(f, f'{K.work}/index/' + os.path.basename(f))
    for f in glob.glob(f'{d}/runs/**/*.parquet', recursive=True) + glob.glob(f'{d}/runs/*.json'):
        take(f, f'{K.work}/runs/' + os.path.relpath(f, f'{d}/runs'))
copied = []
for f in glob.glob(f'{base}/runs/*.parquet'):               # cả mức @parents/@children: lin07 gộp theo mức
    shutil.copy(f, RUNS); copied.append(f'{RUNS}/' + os.path.basename(f))
FB = None
if 'prf' in MODES:
    h = locate_all(f'runs/{FEEDBACK}.parquet')
    assert h, f'MODES có prf: Add Input dataset chứa runs/{FEEDBACK}.parquet'
    FB = h[0] / 'runs' / f'{FEEDBACK}.parquet'
    print('phản hồi PRF:', FB)
GEN = [m for m in MODES if m in ('q2d', 'hyde')]           # chỉ các chế độ này cần LLM
BM25C = any('bm25c' in f for f in FUSED)
K.set({'translate.out_dir': QT, 'retrieve.out_dir': RUNS})
bm = locate('index/bm25', required=False)
if bm is not None:
    K.set({'bm25.index_dir': str(bm / 'index' / 'bm25')})
EMB = ','.join(sorted({str(d / 'emb') for d in locate_all('emb')}))
print('emb:', EMB)
if GEN:
    sh('python -m pip install -q -U bitsandbytes accelerate')
'''),
    code('''
# (3) Sinh đoạn giả: mỗi chế độ 2 GPU song song (mỗi GPU một bản 4-bit, mỗi bản một nửa số câu)
lim = f' --limit {LIMIT}' if LIMIT else ''
for m in GEN:
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.expand generate --config {K.cfg} --mode {m} --model {MODEL} '
         f'--shard {g}/2 --batch {BATCH} --quant {QUANT}{QARG}{lim}' for g in range(2)], f'expand_{m}')
'''),
    code('''
# (4) Dựng câu hỏi cho từng biến thể + đọc mẫu đoạn giả
for m in GEN + ['bm25c'] * BM25C:
    sh(f'python -m mirret.expand build --config {K.cfg} --mode {m} --model {MODEL}')
if 'prf' in MODES:
    sh(f'python -m mirret.expand build --config {K.cfg} --mode prf --feedback {FB}', 'build_prf')
for m in MODES:
    assert glob.glob(f'{K.work}/qexp/{m}/queries_vi*.parquet'), f'thiếu qexp/{m}/queries_*.parquet (rw: Add Input dataset của nhóm)'
    if os.path.exists(f'{K.work}/qexp/{m}/samples.md'):
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
if BM25C and not done('bm25c', ('bm25',)):
    sh(f'python -m mirret.retrieve run --config {K.cfg} --methods bm25 --variant bm25c', 'retrieve_bm25c')
sh(f'python -m mirret.fuse --config {K.cfg} --only {",".join(FUSED)}', 'fuse')
import pandas as pd
from mirret import runs as R
rows = []
for m in MODES + ['bm25c'] * BM25C:
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
for f in (FUSED if RERANK else []):                   # câu tổng hợp (synth) không có trong qtrans → không rerank
    if not os.path.exists(f'{RUNS}/{f}.parquet'):
        print('bỏ qua', f, '(chưa gộp được)'); continue
    par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.rerank --config {K.cfg} --run {RUNS}/{f}.parquet --method cross '
         f'--top-n {TOP_N} --shard {g}/2 --device cuda:0{QARG}' for g in range(2)], f'rerank_{f}')
    sh(f'python -m mirret.rerank --config {K.cfg} --run {RUNS}/{f}.parquet --method cross --top-n {TOP_N} '
       f'--shard 0/2{QARG}')
if not QIDS and RERANK:                               # chạy đủ 1.200 câu → bài nộp cho từng run đã rerank
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

# ---------------------------------------------------------------- k9: nhánh MedCPT cho PubMed (GPU)
save("k9_medcpt.ipynb", [
    md("""
# k9 — Nhánh MedCPT cho tài liệu tiếng Anh / PubMed (GPU T4 ×2)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks` (đoạn cha), `mir2-runs` (9 nhánh + `qtrans/queries_en.parquet`).
MedCPT (NCBI 2023, BERT-base, mở): `ncbi/MedCPT-Article-Encoder` mã hoá ~240 nghìn bài PubMed dạng [tiêu đề, tóm tắt],
`ncbi/MedCPT-Query-Encoder` mã hoá câu hỏi tiếng Anh (bản dịch) → top-1.000 → `runs/medcpt_en_dense.parquet`, rồi gộp
`FUSE` (fuse.configs: lin07m = lin07 + MedCPT; lin07um = lin07u + MedCPT). Ước tính ~20–30 phút.
**Đầu ra:** tab *Output* → *New Dataset* **`mir2-medcpt`** → k5 với `RUN = 'lin07m'` (rerank, `REUSE` điểm cũ).
Đáp án public không có PubMed → kiểm ở vòng private (1–4/11).
"""),
    setup_cell("k9-v1", "FUSE = ['lin07m', 'lin07um']             # fuse.configs có nhánh medcpt_en_dense",
               "('chunks', 'qtrans')"),
    code('''
# (2) Mã hoá đoạn cha tiếng Anh: mỗi GPU một nửa
par([f'CUDA_VISIBLE_DEVICES={g} python -m mirret.medcpt encode --config {K.cfg} --shard {g}/2 --device cuda:0'
     for g in range(2)], 'medcpt_encode')
'''),
    code('''
# (3) Tìm top-1.000 bằng câu hỏi tiếng Anh → runs/medcpt_en_dense; chép 9 nhánh (@parents/@children) rồi gộp FUSE
OUT = f'{K.work}/runs'
os.makedirs(OUT, exist_ok=True)
RUNS_IN = locate('runs/vi_dense.parquet') / 'runs'
copied = []
for f in glob.glob(f'{RUNS_IN}/*.parquet'):
    dst = f'{OUT}/' + os.path.basename(f)
    if not os.path.exists(dst):
        shutil.copy(f, dst); copied.append(dst)
K.set({'retrieve.out_dir': OUT})
sh(f'python -m mirret.medcpt search --config {K.cfg} --device cuda:0', 'medcpt_search')
sh(f'python -m mirret.fuse --config {K.cfg} --only {",".join(FUSE)}', 'fuse')
keep = {f'{OUT}/{n}.parquet' for n in ['vi_dense', 'vi_lsr', 'vi_bm25', 'en_dense', 'en_lsr', 'en_bm25', 'zh_dense',
                                        'zh_lsr', 'zh_bm25']}
for f in copied:                                      # giữ 9 nhánh gộp mức (k5 cần runs/vi_dense.parquet để định vị)
    if f not in keep:
        os.remove(f)
from mirret import config as C, runs as R
pm = R.pubmed_ids(C.load(K.cfg))
for n in ['medcpt_en_dense', 'en_dense'] + FUSE:
    r = R.read(f'{OUT}/{n}.parquet', depth=400)
    print(f'{n:18s} top-400: PubMed {r["doc_id"].isin(pm).mean():.1%}')
report('k9')
'''),
])

# ---------------------------------------------------------------- k8: dữ liệu fine-tune reranker (GPU, vLLM)
VLLM_NOTE = """**vLLM trên T4:** `VLLM_SPEC` ghim bản dùng PyTorch CUDA 12.x như Kaggle (bản vLLM mới nhất kéo PyTorch CUDA
13 → lệch torchaudio của Kaggle và cần driver ≥ 580). Ô cài đặt gỡ torchaudio (không dùng) và dừng nếu torch mới không
thấy GPU. Lỗi khi nạp model → thử `VLLM_SPEC = 'vllm==0.11.0'`, hoặc `MODEL = 'Qwen/Qwen3-14B-AWQ'` (nhanh ~2 lần, chất
lượng thấp hơn)."""
VLLM_INSTALL = """
sh('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader')
sh(f'python -m pip install -q "{VLLM_SPEC}"', 'pip_vllm')
sh('python -m pip uninstall -y -q torchaudio')       # vLLM thay torch của Kaggle; torchaudio cũ lệch CUDA, không dùng tới
sh('python -c "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available())"')
try:
    sh('python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"')
except Exception:
    raise SystemExit('torch mới không thấy GPU (driver Kaggle không hỗ trợ CUDA của bản torch này) → đổi VLLM_SPEC')
"""
save("k8a_synth.ipynb", [
    md(f"""
# k8a — Câu hỏi tổng hợp cho fine-tune reranker (GPU T4 ×2, vLLM)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On. **Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`.
Lấy ngẫu nhiên đoạn nguồn trong kho (`N`: vi/zh/en — có PubMed vì đáp án private có), teacher `MODEL` (Qwen3-32B-AWQ,
chỉ dùng lúc huấn luyện) viết mỗi đoạn một câu hỏi tiếng Việt kiểu người dùng + bản dịch en/zh. KHÔNG dùng câu thi.
**Chạy thử trước:** `LIMIT = 100` → đọc `samples.md` + tốc độ (token/s) ở log, rồi `LIMIT = 0`.
**v2 (30/10, mặc định):** `VARIANT = 'synth2'`, `STYLE = 'mix'` (câu ngắn / vừa / dài theo phân bố độ dài câu thi),
`QA_CAP = 0.3` (trang hỏi-đáp tối đa 30% mỗi ngôn ngữ). Lý do: Qwen-ft v1 lệch nhiều nhất ở câu dài và đẩy trang
hỏi-đáp tiếng Trung lên. v1: `VARIANT = 'synth'`, `STYLE = 'none'`, `QA_CAP = 1.0`.
**Đầu ra:** `<VARIANT>/source.parquet`, `<VARIANT>/gen.jsonl`, `qexp/<VARIANT>/queries_{{vi,en,zh}}.parquet` → *New
Dataset* **`mir2-synth2`** → k7 (`MODES = ['synth2']`, `FUSED = ['synth2_lin07']`, `RERANK = False`) → k8b.
**Đợt bổ sung:** Add Input output k8a cũ cùng VARIANT, `TAG = 'pm'`, `N = 'en:2000'` → chỉ sinh phần mới, `qexp/<VARIANT>`
dựng lại từ MỌI đợt.
{VLLM_NOTE}
"""),
    setup_cell("k8a-v5", "VARIANT = 'synth2'                   # 'synth' = v1 (27/10); 'synth2' = v2\n"
                         "N = 'vi:2400,zh:1400,en:2200'        # dư vì bị bỏ qua (v1 hợp lệ: vi ~51%, zh ~74%; en xem log)\n"
                         "STYLE = 'mix'                        # 'mix' = ngắn 25% / vừa 50% / dài 25%; 'none' = như v1\n"
                         "QA_CAP = 0.3                         # tỉ lệ tối đa trang hỏi-đáp mỗi ngôn ngữ; 1.0 = như v1\n"
                         "TAG = ''                             # '' = đợt đầu; 'pm' = đợt bổ sung (Add Input output k8a cũ)\n"
                         "MODEL = 'Qwen/Qwen3-32B-AWQ'\n"
                         "LIMIT = 100                          # >0: chỉ N đoạn đầu (chạy thử); 0 = tất cả\n"
                         "VLLM_SPEC = 'vllm==0.9.2'            # PyTorch 2.7 (CUDA 12.x), T4 chạy engine V0", "('chunks',)"),
    code('''
# (2) vLLM + đoạn nguồn (CPU); đợt bổ sung: chép các đợt trước từ Input''' + VLLM_INSTALL + '''for d in (locate_all(f'{VARIANT}/gen.jsonl') if TAG else []):
    shutil.copytree(f'{d}/{VARIANT}', f'{K.work}/{VARIANT}', dirs_exist_ok=True)
    print('đợt trước:', d, sorted(os.listdir(f'{d}/{VARIANT}')))
assert not TAG or os.path.exists(f'{K.work}/{VARIANT}/gen.jsonl'), 'TAG đợt bổ sung: Add Input output k8a đợt đầu'
VA = f' --variant {VARIANT}'
TG = f' --tag {TAG}' if TAG else ''
sh(f'python -m mirret.synth sample --config {K.cfg} --n {N} --qa-cap {QA_CAP}{TG}{VA}', 'sample')
'''),
    code('''
# (3) Sinh câu hỏi (2 GPU, tensor parallel) → câu hỏi cho k7 (gộp mọi đợt)
lim = f' --limit {LIMIT}' if LIMIT else ''
sh(f'python -m mirret.synth generate --config {K.cfg} --model {MODEL} --tp 2 --style {STYLE}{lim}{TG}{VA}', 'generate')
sh(f'python -m mirret.synth build --config {K.cfg}{VA}', 'build')
print(open(f'{K.work}/qexp/{VARIANT}/samples.md', encoding='utf-8').read())
report('k8a')
'''),
])

save("k8b_label.ipynb", [
    md(f"""
# k8b — Ứng viên khó + nhãn teacher cho câu hỏi tổng hợp (GPU T4 ×2, vLLM)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks`, output k8a (`mir2-synth2`), output k7 có `runs/<VARIANT>_lin07.parquet`.
Mỗi câu tổng hợp: đoạn nguồn + đoạn tốt nhất của top-`TOP` tài liệu (âm khó) + `FAR` tài liệu ngẫu nhiên hạng 50–200
(âm dễ); bỏ bản gần trùng đoạn nguồn. Teacher chấm 0/1/2 (tắt chế độ suy nghĩ), lấy xác suất từng mức từ logprob → nhãn mềm.
**v2 (mặc định `LEVEL = 'doc'`):** teacher xem tiêu đề + phần đầu bài + đoạn liên quan nhất và chấm CẢ TÀI LIỆU, tiêu chí
chặt hơn (trang hỏi-đáp về ca của người khác tối đa 1). v1: `LEVEL = 'unit'` (chấm từng đoạn).
**Nhiều tài khoản:** cùng Input, `SHARD = '0/4'` … `'3/4'` (ứng viên tính lại giống hệt ở mọi phiên).
**Chạy thử:** `LIMIT = 30` (30 câu) → xem cặp/s ở log để chia phần.
**Đầu ra:** `<VARIANT>/pairs.parquet`, `<VARIANT>/labels-<phần>.parquet` → *New Dataset* → k8c.
{VLLM_NOTE}
"""),
    setup_cell("k8b-v3", "VARIANT = 'synth2'                   # 'synth' = v1; 'synth2' = v2\n"
                         "LEVEL = 'doc'                        # 'doc' = chấm cả tài liệu (v2); 'unit' = chấm từng đoạn (v1)\n"
                         "MODEL = 'Qwen/Qwen3-32B-AWQ'\n"
                         "TOP, FAR = 12, 4\n"
                         "SHARD = '0/1'\n"
                         "LIMIT = 30                           # >0: chỉ N câu đầu của phần (đo tốc độ); 0 = cả phần\n"
                         "VLLM_SPEC = 'vllm==0.9.2'            # PyTorch 2.7 (CUDA 12.x), T4 chạy engine V0", "('chunks',)"),
    code('''
# (2) Dữ liệu k8a + run gộp của k7 → ứng viên (CPU; v2 đọc thêm phần đầu bài trong kho)
for d in locate_all(f'{VARIANT}/source.parquet'):
    shutil.copytree(f'{d}/{VARIANT}', f'{K.work}/{VARIANT}', dirs_exist_ok=True)
for d in locate_all(f'qexp/{VARIANT}/queries_vi.parquet'):
    shutil.copytree(f'{d}/qexp/{VARIANT}', f'{K.work}/qexp/{VARIANT}', dirs_exist_ok=True)
RUN = locate(f'runs/{VARIANT}_lin07.parquet') / 'runs' / f'{VARIANT}_lin07.parquet'
print('run:', RUN)
VA = f' --variant {VARIANT}'
sh(f'python -m mirret.synth candidates --config {K.cfg} --run {RUN} --top {TOP} --far {FAR}{VA}', 'candidates')
'''),
    code('''
# (3) Teacher chấm (2 GPU, tensor parallel)''' + VLLM_INSTALL + '''lim = f' --limit {LIMIT}' if LIMIT else ''
sh(f'python -m mirret.synth label --config {K.cfg} --model {MODEL} --tp 2 --shard {SHARD} --level {LEVEL}{lim}{VA}', 'label')
report('k8b')
'''),
])

save("k8c_train.ipynb", [
    md("""
# k8c — Fine-tune reranker (GPU T4 ×2): bge (GPU 0) và Qwen3-Reranker-4B + LoRA (GPU 1) song song

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, mọi output k8b cùng `VARIANT` (`<VARIANT>/labels-*.parquet`, `pairs.parquet`,
`qexp/<VARIANT>`).
Loss: chưng cất theo danh sách (KL tới softmax(3 × điểm teacher 0–2) trên ứng viên của chính câu + 2 đoạn câu khác) +
BCE kéo điểm về nhãn mềm của teacher. Mỗi câu tổng hợp một nhóm (bge 16 đoạn, qwen 8 đoạn).
5% câu tổng hợp để kiểm tra: in AUC / độ chính xác cặp TRƯỚC và SAU khi train (sau phải cao hơn).
**Đầu ra:** `ft/bge_<VARIANT>/` (model đầy đủ), `ft/qwen_<VARIANT>/` (adapter LoRA) → *New Dataset* → k5 với
`CROSS_MODEL = 'bge_synth2'` / `QWEN_LORA = 'qwen_synth2'` (+ `FT_ALPHA` để nội suy về model gốc; không dùng REUSE cho
phương pháp đã fine-tune).
"""),
    setup_cell("k8c-v4", "VARIANT = 'synth2'                   # 'synth' = v1; 'synth2' = v2\n"
                         "TRAIN = ['bge', 'qwen']              # bỏ bớt nếu chỉ train một model\n"
                         "MAX_GROUPS = {'bge': 0, 'qwen': 3000}   # 0 = mọi nhóm; qwen ~8 s/nhóm 8 đoạn trên T4\n"
                         "EPOCHS = 1", "()"),
    code('''
# (2) Gom nhãn của mọi phần k8b
os.makedirs(f'{K.work}/{VARIANT}', exist_ok=True)
for d in locate_all(f'{VARIANT}/pairs.parquet'):
    for f in glob.glob(f'{d}/{VARIANT}/labels*.parquet') + [f'{d}/{VARIANT}/pairs.parquet']:
        shutil.copy(f, f'{K.work}/{VARIANT}/')
for d in locate_all(f'qexp/{VARIANT}/queries_vi.parquet'):
    shutil.copytree(f'{d}/qexp/{VARIANT}', f'{K.work}/qexp/{VARIANT}', dirs_exist_ok=True)
print(sorted(os.listdir(f'{K.work}/{VARIANT}')))
sh('python -m pip install -q peft')
sh('python -m pip uninstall -y -q torchao')           # peft mới đòi torchao > 0.16 (Kaggle có 0.10) — không dùng tới
'''),
    code('''
# (3) Train song song: mỗi model một GPU
cmds = []
for g, kind in enumerate(TRAIN):
    mg = f' --max-groups {MAX_GROUPS[kind]}' if MAX_GROUPS.get(kind) else ''
    cmds.append(f'CUDA_VISIBLE_DEVICES={g} python -m mirret.finetune {kind} --config {K.cfg} '
                f'--out {K.work}/ft/{kind}_{VARIANT} --device cuda:0 --epochs {EPOCHS} --variant {VARIANT}{mg}')
par(cmds, 'finetune')
for kind in TRAIN:
    print(kind, open(f'{K.work}/ft/{kind}_{VARIANT}/finetune_meta.json', encoding='utf-8').read())
report('k8c')
'''),
])

# ---------------------------------------------------------------- k10: giám khảo LLM lúc suy luận (GPU, vLLM)
save("k10_judge.ipynb", [
    md(f"""
# k10 — Giám khảo LLM chấm tài liệu của câu thi (GPU T4 ×2, vLLM, Qwen3-14B-AWQ)

**Cài đặt:** Accelerator = **GPU T4 ×2**, Internet = On.
**Add Input:** `mir2-code`, `mir2-data`, `mir2-chunks` (đoạn cha), `mir2-runs` (`qtrans/queries_*.parquet`),
`mir2-private-prep` (bản mới có `runs/lin07_ensi_1200.parquet`).
Mỗi câu thi: top-`HI` tài liệu của lin07_ensi (đoạn tốt nhất + tiêu đề + phần đầu bài) → Qwen3-14B (mở, ≤ 14B, 04/2025:
hợp luật suy luận) chấm 0/1/2 cả tài liệu (prompt như teacher v2, tắt suy nghĩ), xác suất từng mức từ logprob. Không
huấn luyện gì; model > 14B không dùng ở đây. Ghép trên máy: `tools/merge_judge.py` → `span_chunks.py`.
**Chia việc:** tài khoản i (0…n−1) đặt `SHARD = 'i/n'` — mọi tài khoản cùng `HI`, `QIDS`. Trong tài khoản: 2 tiến trình,
mỗi GPU một bản model (`TP = 1`). Hết RAM GPU khi nạp → `TP = 2` (một tiến trình, chậm hơn ~30%).
**Tốc độ đo 29/10 (`LIMIT = 4`):** 1,19 cặp/s mỗi GPU (~800 token/cặp) → cả phần ≈ 1200 × HI / (2 × n × 1,19) giây =
**~5,8 giờ** với HI 250, n 6 (cộng ~20 phút cài đặt + nạp model). Đo lại: `LIMIT = 4` (~15 phút).
**Chạy:** `LIMIT = 0`, *Save Version → Save & Run All*; không cần thêm output của lượt đo làm Input. Mỗi lô 20 câu ghi một file ngay → phiên bị dừng vẫn giữ phần đã
chấm: chạy lại với output cũ làm Input, các lô đã có được bỏ qua.
**Đầu ra:** `judge/labels-<phần>-<GPU>-b<lô>.parquet` (vài MB) → tải về `results/judge_<i>/`.
{VLLM_NOTE}
"""),
    setup_cell("k10-v2","SHARD = '0/6'                        # tài khoản i trong n: '0/6' … '5/6'\n"
                         "HI = 250                             # số tài liệu đầu mỗi câu được chấm\n"
                         "QIDS = ''                            # '' = cả 1.200 câu; 'public600_a.txt' = chỉ các câu đó\n"
                         "MODEL = 'Qwen/Qwen3-14B-AWQ'\n"
                         "TP = 1                               # 1 = mỗi GPU một bản model; 2 = một bản trên 2 GPU\n"
                         "LIMIT = 0                            # >0: N câu đầu mỗi GPU (đo tốc độ); 0 = cả phần\n"
                         "VLLM_SPEC = 'vllm==0.9.2'            # PyTorch 2.7 (CUDA 12.x), T4 chạy engine V0", "('chunks', 'qtrans')"),
    code('''
# (2) Cặp (câu, tài liệu) của phần này (CPU: đọc đoạn cha + phần đầu bài trong kho); nhãn cũ trong Input → bỏ qua lô đã có
RUN = locate('runs/lin07_ensi_1200.parquet') / 'runs' / 'lin07_ensi_1200.parquet'
print('run:', RUN)
if QIDS and not os.path.exists(QIDS) and not QIDS.replace(',', '').isdigit():
    hits = sorted(glob.glob(f'{KAGGLE}/input/**/{QIDS}', recursive=True), key=len)
    assert hits, f'Không thấy {QIDS} trong /kaggle/input'
    QIDS = hits[0]
QARG = f' --qids "{QIDS}"' if QIDS else ''
TAG = SHARD.replace('/', 'of')
os.makedirs(f'{K.work}/judge', exist_ok=True)
for f in glob.glob(f'{KAGGLE}/input/**/judge/labels-{TAG}-*.parquet', recursive=True):
    shutil.copy(f, f'{K.work}/judge/')
print('nhãn cũ:', len(glob.glob(f'{K.work}/judge/labels-*.parquet')))
sh(f'python -m mirret.llmjudge pairs --config {K.cfg} --run {RUN} --hi {HI} --shard {SHARD}{QARG}', 'pairs')
'''),
    code('''
# (3) Chấm: mỗi GPU một tiến trình (TP = 1) hoặc một tiến trình trên 2 GPU (TP = 2)''' + VLLM_INSTALL + '''lim = f' --limit {LIMIT}' if LIMIT else ''
base = f'python -m mirret.llmjudge label --config {K.cfg} --model {MODEL} --shard {SHARD} --tp {TP}{lim}'
if TP == 1:
    par([f'CUDA_VISIBLE_DEVICES={g} {base} --part {g}/2' for g in range(2)], 'label')
else:
    sh(f'{base} --part 0/1', 'label')
for f in glob.glob(f'{K.work}/judge/pairs-*.parquet'):
    os.remove(f)                                      # chỉ giữ nhãn (nhỏ) trong output
print(sorted(os.listdir(f'{K.work}/judge')))
report('k10')
'''),
])
