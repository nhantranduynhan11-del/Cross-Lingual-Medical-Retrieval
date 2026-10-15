# Kế hoạch đánh giá Stage 1 (9 nhánh) khi không có nhãn của BTC

**Mục tiêu:** quyết định **GIỮ / BỎ / ĐIỀU CHỈNH** từng nhánh trong 9 nhánh truy hồi
(câu hỏi vi / en / zh × Dense BGE-M3 / LSR BGE-M3 / BM25) **trước** khi đầu tư thêm vào Query Expansion và reranking.
Ràng buộc: Kaggle (CPU 4 nhân, GPU T4 ×2, 12 giờ/phiên, ~30 giờ GPU/tuần/tài khoản), nhóm 3 người, không có nhãn của BTC.

Mọi bước đều có lệnh sẵn trong `mirret` và notebook `k4` (chẩn đoán), `k6` (pool, phiếu chấm, tính điểm).
Ở cuối tài liệu có danh sách lệnh (Phụ lục A) và checklist trước khi báo cáo (Phụ lục C).

---

## Ba loại kết quả — luôn ghi nhãn loại trong báo cáo

| Nhãn | Nghĩa | Ví dụ | Được phép kết luận |
|---|---|---|---|
| **[ĐO]** | Đo trực tiếp trên cả 1.200 câu, không cần nhãn | độ trùng, số tài liệu mới, Kendall τ trước/sau rerank | độ **đa dạng / ảnh hưởng** giữa các run. KHÔNG nói gì về đúng/sai |
| **[NHÃN-n]** | Đo trên tập chấm tay *n* câu, kèm KTC 95% | P@10, Recall trên pool R@50, F2 | chất lượng **tương đối** giữa các run, trong giới hạn của pool (mục 2.5) |
| **[GT]** | Giả thuyết, suy luận | "zh_bm25 kém có thể do bản dịch thuật ngữ" | chỉ là hướng kiểm tra tiếp |
| **[LB]** | Điểm leaderboard | F2 của một bài nộp | điểm thật nhưng ít lượt, không tách được đóng góp từng nhánh |

---

## Lộ trình tổng thể

| Bước | Việc | Ở đâu | Thời gian (ước tính, chưa đo) | Đầu ra | Dùng để |
|---|---|---|---|---|---|
| 0 | Dịch câu hỏi bằng 2 model, chấm mù bản dịch | k2 (GPU ×2) + người | 2–4 giờ GPU; ~2 giờ người | `qtrans/`, `compare_*/` | chọn bản dịch cho nhánh en/zh (mục 0) |
| 1 | 9 nhánh + RRF + chẩn đoán không nhãn | k4 | ~1 giờ GPU (mã hoá câu hỏi) + CPU | `runs/`, `eval/diag/diag.md` | giả thuyết; đo bất đồng để chọn câu hỏi (mục 1, 5) |
| 2 | Rerank ColBERT / cross-encoder trên `rrf9` | k5 (GPU ×2) | dự án cũ: ColBERT top-400 ≈ 100 phút | `rrf9_colbert`, `rrf9_cross` | đưa vào pool ngay từ đầu (công bằng hơn `extend` về sau) |
| 3 | Chọn câu hỏi, lập pool, phiếu chấm ẩn nguồn | k6 `MODE='pool'` (CPU) | < 30 phút | `eval/judge/` | mục 2.1–2.2 |
| 4 | Chấm **pilot** (6 câu, cả 3 người cùng chấm) | người | ~2 giờ/người | kappa pilot | sửa hướng dẫn chấm (mục 2.3–2.4) |
| 5 | Chấm vòng chính (54 câu, 20% chấm chéo) | người | ~6–10 giờ/người (mục 2.2) | `qrels.parquet` | — |
| 6 | Tính điểm, giá trị biên, bảng tổng hợp | k6 `MODE='score'` (CPU) | < 1 giờ | `score.md`, `marginal.md`, `table_k*.md` | quyết định giữ/bỏ/chỉnh (mục 8) |
| 7 | Xác nhận bằng 1–2 lượt nộp | k5 | — | [LB] | kiểm tra chéo quyết định quan trọng nhất |

Nếu bước 2 chưa xong khi lập pool: lập pool không có rerank, sau đó `MODE='extend'` với `EXTEND='rrf9_colbert:30'`
(chỉ chấm thêm các mục chưa có trong pool).

---

## 0. Chọn bản dịch câu hỏi (Qwen2.5-14B so với Qwen3)

- **Qwen3 không có bản 7B** (các bản dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B). Dùng **Qwen3-8B** (gần 7B nhất);
  muốn so cùng cỡ thì thêm `Qwen/Qwen3-14B` vào `MODELS` của k2. Qwen3 chạy với `enable_thinking=False` (không sinh phần
  suy nghĩ), 4-bit trên T4.
- Đã có bản dịch cũ (Qwen2.5-14B, prompt v2): 1.195/1.200 câu, thiếu 85, 219, 510, 683, 697. Nạp bằng
  `python -m mirret.translate import-old --qexp <thư mục qexp> --name old_qwen25_14b`. Câu thiếu bản dịch thì nhánh en/zh
  **bỏ trống** câu đó (không thay bằng câu tiếng Việt để khỏi nhân đôi trọng số tiếng Việt trong RRF).
- Bản dịch chỉ ảnh hưởng 6 nhánh en/zh. BM25 nhạy nhất với việc chọn thuật ngữ (khớp từ đúng chữ); Dense ít nhạy hơn.

**Bước 0.1 — chỉ số tự động [ĐO]** (`translate compare`, có trong k2): tỷ lệ JSON hợp lệ, đúng chữ viết (en không có
chữ Hán/dấu Việt, zh có chữ Hán), giữ đủ con số (liều, tuổi, chỉ số xét nghiệm), tỷ lệ độ dài, câu quá ngắn, thời gian.
Các chỉ số này chỉ bắt **lỗi hình thức**, không đo bản dịch đúng hay sai. Ví dụ đã đo trên bản cũ: chỉ 84,2% (en) /
79,1% (zh) câu giữ đủ các con số → đáng kiểm tra bằng mắt.

**Bước 0.2 — chấm mù 100 câu** (`compare_*/sheet.csv`): một nửa chọn ngẫu nhiên phân tầng theo độ dài, một nửa là các
câu hai model dịch khác nhau nhiều nhất (nơi có thông tin để phân định). X/Y được xáo ngẫu nhiên mỗi dòng, khoá ở
`key.json` (người chấm không mở). Mỗi dòng: chọn bản tốt hơn cho en và cho zh (`X`/`Y`/`=`), đánh dấu **lỗi nặng**
(`severe_X`, `severe_Y` = 1) nếu bản dịch làm sai thông tin cần tìm: sai thuật ngữ bệnh/thuốc/xét nghiệm, đổi người
mắc bệnh (mẹ ↔ con), sai/mất con số, mất một ý hỏi, thêm triệu chứng không có, hoặc trả lời câu hỏi thay vì dịch.
Hai người chấm (mỗi người cả 100 câu nếu kịp, ~1 giờ), rồi `python -m mirret.translate score --sheet a.csv --sheet b.csv`.

**Quy tắc quyết định (đã cài trong `translate score`):** chọn model có tỷ lệ lỗi nặng thấp hơn; nếu tương đương, chọn
model thắng nhiều hơn ở tiếng Trung (nhánh zh là đường chính tới ~240 nghìn tài liệu tiếng Trung). Chênh lệch có KTC
chứa 0 → coi là hoà; khi đó chọn bản rẻ hơn (bản cũ đã có sẵn), hoặc nếu còn thời gian thì đánh giá **ngoại tại**:
chạy nhánh en/zh với cả hai bản dịch (k4 hai lần, `USE` khác nhau), đưa top-10 của các nhánh mới vào pool
(`judge extend`) và so P@10 / R@50 của nhánh en/zh trên tập có nhãn. Đây mới là mục tiêu thật (bản dịch nào tìm được
tài liệu đúng hơn), nhưng tốn thêm công chấm.

---

## 1. Phân tích độ trùng giữa các nhánh (không cần nhãn) — [ĐO]

Chạy tự động ở cuối k4 (`python -m mirret.diag`) trên cả 1.200 câu, kết quả ở `eval/diag/diag.md` và
`pairwise_doc_k*.csv`. Đơn vị: **tài liệu** (thứ hạng tài liệu = đoạn tốt nhất của nó), vì nhiều đoạn cùng bài làm
độ trùng ở mức đoạn khó đọc. Khoảng trong ngoặc là KTC 95% bootstrap theo câu hỏi.

| Chỉ số | Định nghĩa (A, B = top-K của hai nhánh, theo từng câu, rồi lấy trung bình) | Đọc thế nào |
|---|---|---|
| Jaccard@K | \|A ∩ B\| / \|A ∪ B\| | trùng tập, đối xứng |
| Overlap coefficient@K | \|A ∩ B\| / min(\|A\|, \|B\|) | dùng khi hai danh sách khác độ dài (vd BM25 zh có ít kết quả) |
| RBO (p = 0,9) | trùng có trọng số theo thứ hạng, đầu danh sách nặng hơn | cùng tập nhưng khác thứ tự → RBO thấp hơn Jaccard |
| Tài liệu mới khi thêm nhánh | \|top-K nhánh mới \ union các nhánh trước\|, theo thứ tự cố định vi → en → zh và thứ tự tham lam | nhánh nào mở rộng tập ứng viên |
| Tài liệu riêng | tài liệu chỉ một nhánh có trong top-K | nhánh nào "khác biệt" nhất |
| Hồ sơ ngôn ngữ | tỷ lệ tài liệu vi / zh / en trong top-K mỗi nhánh | nhánh zh có thật sự mang về tài liệu tiếng Trung không |

K = 10 (khớp độ sâu pool của nhánh), 50, 100 (gần quy mô bài nộp k_doc = 200).
Các mục của `diag.md`: §1 độ phủ và ngôn ngữ; §2 ma trận overlap cặp ở K = 10/50/100; §3 union (cố định + tham lam) và
tài liệu riêng; §4 thành phần của `rrf9`; §5 bỏ từng nhánh khỏi RRF, độ nhạy theo k của RRF, thay đổi do rerank.

### Vì sao đa dạng KHÔNG đồng nghĩa với Recall

Recall = |tìm được ∩ liên quan| / |liên quan|. Mọi chỉ số ở trên chỉ nhìn vào **phần "tìm được"**; không chỉ số nào
biết tài liệu nào **liên quan**. Bốn tình huống đều cho cùng một con số độ trùng:

| Tình huống | Độ trùng | Ảnh hưởng thật tới Recall |
|---|---|---|
| Hai nhánh trùng nhiều, cùng đúng | cao | nhánh thứ hai thừa về tập, nhưng có thể có ích nhờ "bỏ phiếu" trong RRF |
| Hai nhánh trùng nhiều, cùng sai (cùng mắc bẫy từ khoá) | cao | không có ích |
| Nhánh có nhiều tài liệu riêng, phần lớn đúng | thấp | tăng Recall thật |
| Nhánh có nhiều tài liệu riêng, phần lớn sai (vd BM25 zh với thuật ngữ dịch sai) | thấp | chỉ thêm nhiễu, có thể đẩy tài liệu đúng ra khỏi top-K của RRF |

Một nhánh trả về tài liệu **ngẫu nhiên** sẽ có độ trùng thấp nhất và "tài liệu mới" nhiều nhất — tức là điểm "đa dạng"
tốt nhất. Vì vậy đa dạng chỉ là **điều kiện cần** để một nhánh đóng góp Recall, không phải bằng chứng.

**Độ trùng dùng được vào việc gì:**
1. Phát hiện **thừa**: overlap@50 > 0,8 giữa hai nhánh → gần như cùng tập; nhánh sau thêm ít tài liệu mới dù đúng.
2. Phát hiện **lỗi**: một nhánh gần như không trùng nhánh nào, hoặc hồ sơ ngôn ngữ lạ (vd zh_bm25 trả về tài liệu
   tiếng Việt) → kiểm tra bản dịch, bộ tách từ, chỉ mục trước khi đánh giá tiếp.
3. **Chọn câu hỏi để chấm**: câu có các nhánh bất đồng nhiều là nơi chấm tay cho nhiều thông tin nhất (mục 2.1).
4. **Dự đoán ảnh hưởng** của nhánh lên RRF (§4, §5 của `diag.md`): ảnh hưởng lớn ≠ tốt, chỉ là nơi cần kiểm tra bằng nhãn.

---

## 2. Tập đánh giá thủ công có kiểm soát

### 2.1 Chọn câu hỏi — `judge select`

- **60 câu** (6 pilot + 54 chính), mặc định trong k6. Phân tầng **3 × 3**: độ dài câu hỏi (ngắn / vừa / dài, theo tam
  phân vị) × mức bất đồng giữa các nhánh (1 − Jaccard@20 trung bình giữa mọi cặp nhánh; tam phân vị). Mỗi tầng chọn đều,
  bù ngẫu nhiên cho đủ; hạt giống ngẫu nhiên cố định (`seed` trong cấu hình) → lặp lại được.
  - Câu **bất đồng** cao là nơi các nhánh khác nhau → quyết định giữ/bỏ phụ thuộc chủ yếu vào chúng.
  - Câu **đồng thuận** giữ cho tập không thiên về các câu "khó", tránh đánh giá quá cao giá trị của đa dạng.
  - **Độ dài** đại diện cho kiểu câu: ngắn ~ tra cứu từ khoá (BM25 thường tốt), dài ~ câu hỏi tư vấn kiểu Q&A (Dense
    thường tốt).
- Mở `eval/judge/queries.csv` xem nhanh **độ phủ chủ đề** (thuốc, bệnh, triệu chứng, dinh dưỡng, mẹ & bé, y học cổ
  truyền…). Nếu thiếu hẳn một loại, đổi vài câu bằng tay **trước khi** xem bất kỳ kết quả nào, ghi lại lý do. Không bao
  giờ thêm/bớt câu sau khi đã thấy điểm.
- **Độ nhạy thống kê:** với ~50–60 câu, KTC 95% của chênh lệch F2 giữa hai run thường rộng vài điểm phần trăm (đo thật ở
  `score.md`). Khác biệt nhỏ hơn độ rộng đó → coi là **hoà**, không dùng để quyết định.

### 2.2 Lập pool — `judge pool`

- **Đơn vị chấm = đoạn cha** (đúng đơn vị nộp `chunk_text`, chữ nguyên văn), kèm ngữ cảnh "tiêu đề > mục" để người chấm
  biết đoạn nằm ở đâu. Mức liên quan của **tài liệu** = nhãn cao nhất trong các đoạn đã chấm của nó.
- **Nguồn pool:** top-10 của mỗi nhánh (9 × 10) ∪ top-30 của `rrf9`, `rrf9_colbert`, `rrf9_cross`.
  - top-10 nhánh → so 9 nhánh công bằng ở K = 10 (mọi mục đều đã chấm).
  - top-30 run gộp/rerank → các ứng viên bài nộp được chấm sâu hơn; rerank đưa vào top-30 những đoạn nằm ở hạng
    31–400 của `rrf9`.
- **Phiếu:** `sheet_pilot_<người>.csv` (câu pilot, cả 3 người cùng chấm) và `sheet_pool_<người>.csv` (câu chính, chia
  đều, 20% câu được người thứ hai chấm lại). Trong mỗi câu, thứ tự đoạn **xáo ngẫu nhiên**; mã mục là chuỗi băm;
  **không có nhánh, hạng, điểm**. Nguồn của từng mục chỉ nằm trong `key.parquet` (không mở trước khi chấm xong).
- **Khối lượng:** dòng log `[pool] … mục/câu` cho số thật. Ước tính khi lập kế hoạch:
  `giờ ≈ mục/câu × (pilot × số người + chính × 1,2) × giây/mục / 3600`.
  Ví dụ 65 mục/câu, 25 giây/mục: 65 × (6 × 3 + 54 × 1,2) × 25 / 3600 ≈ 37 giờ ≈ **12 giờ/người**.
  Phương án nhẹ: 40 câu (6 pilot), `--branch-depth 5`, run gộp top-20 → ~40 mục/câu → ~16 giờ ≈ **5–6 giờ/người**
  (đổi lại: chỉ so nhánh công bằng ở K = 5). Đo giây/mục thật ở vòng pilot rồi chọn.
- **Run mới về sau** (trọng số RRF mới, QE, reranker khác): `MODE='extend'`, `EXTEND='ten_run:30'` → chỉ phát phiếu
  cho các mục **chưa chấm** của run đó (`sheet_ext<n>_<người>.csv`). Không chấm lại mục cũ.

### 2.3 Hướng dẫn gán nhãn (bản 1 — chỉnh sau pilot, ghi số phiên bản)

Câu hỏi người chấm tự hỏi: *"Nếu người hỏi đọc riêng đoạn này, họ có nhận được thông tin trả lời câu hỏi không?"*

| Nhãn | Khi nào |
|---|---|
| **2** — trả lời trực tiếp | Đoạn chứa câu trả lời / thông tin chính cần tìm, đủ để trả lời ý chính của câu hỏi. |
| **1** — liên quan một phần | Trả lời **một ý** của câu hỏi nhiều ý; hoặc chứa thông tin then chốt nhưng chưa đủ; hoặc thông tin chung áp dụng được cho trường hợp được hỏi. |
| **0** — không liên quan | Khác chủ đề; **cùng chủ đề nhưng không trả lời câu hỏi**; menu, quảng cáo, trang thông báo / chọn ngôn ngữ; đoạn chỉ chép lại câu hỏi (bài hỏi đáp chưa có trả lời); danh sách tiêu đề bài khác. |

Quy tắc:
- **R1. Chấm đoạn, không chấm cả bài.** Chỉ dựa vào `chunk_text` (+ `context`). Không mở bài gốc trừ khi đoạn mơ hồ
  (ghi `note`).
- **R2. Ngôn ngữ không ảnh hưởng nhãn.** Đoạn tiếng Anh/Trung: dùng cột `query_en` / `query_zh` để đối chiếu thuật ngữ.
  Không đọc được thì dùng **cùng một** công cụ dịch máy cho mọi đoạn và ghi `note` = `MT`. Không chấm theo "có trùng từ
  khoá" — trùng từ khoá mà không trả lời vẫn là 0.
- **R3. Không chấm độ đúng y khoa.** Đoạn trả lời câu hỏi là liên quan, kể cả khi khuyến cáo khác nguồn khác. Quảng
  cáo bán thuốc không có thông tin → 0.
- **R4. Đối tượng / điều kiện.** Câu hỏi nêu rõ đối tượng (bà bầu, trẻ 2 tuổi, người suy thận) mà đoạn nói về đối tượng
  khác và thông tin **không áp dụng được** → 0; đoạn nói chung và áp dụng được → 1; đúng đối tượng và trả lời → 2.
- **R5. Câu hỏi nhiều ý.** Trả lời ý chính (ý người hỏi cần nhất, thường là câu hỏi cuối) → 2; chỉ một ý phụ → 1.
- **R6. Nội dung trùng** (một bài đăng lại ở nhiều trang): chấm như nhau.
- **R7. Không chắc → chọn mức thấp hơn** và ghi `note` = `?` để phân xử.
- **R8. Nhịp độ** 20–30 giây/mục; đã rõ là 0 thì không cần đọc hết. Mỗi phiên ≤ 60 phút.

Ví dụ (tự soạn để minh hoạ):

*"Bà bầu 3 tháng đầu bị sốt 38,5 độ có uống được thuốc hạ sốt không, uống loại nào?"*
- 2: đoạn nói paracetamol là thuốc hạ sốt được dùng cho phụ nữ mang thai, liều dùng, tránh aspirin/ibuprofen.
- 1: đoạn khuyên phụ nữ mang thai bị sốt nên đi khám, không tự ý dùng thuốc (thông tin then chốt nhưng không nêu thuốc).
- 0 (cùng chủ đề): đoạn giải thích các nguyên nhân gây sốt khi mang thai, không nói cách xử lý; đoạn về liều hạ sốt
  cho **trẻ em** theo cân nặng (R4).
- 0: menu "Mẹ & bé | Dinh dưỡng | Tin tức"; trang chọn ngôn ngữ.

*"Uống metformin lâu năm có bị thiếu vitamin B12 không, cần làm gì?"*
- 2: đoạn nói dùng metformin kéo dài làm giảm hấp thu B12 và khuyến cáo xét nghiệm định kỳ / bổ sung.
- 1: đoạn (tiếng Anh) liệt kê tác dụng phụ của metformin, có dòng "may reduce vitamin B12 absorption" (một ý).
- 0 (cùng chủ đề): đoạn về cơ chế hạ đường huyết của metformin.
- **Cần thống nhất ở pilot:** đoạn về thực phẩm giàu B12 không nhắc tới metformin (1 hay 0?). Ghi quyết định vào
  hướng dẫn bản 2.

### 2.4 Giảm thiên lệch

- **Ẩn nguồn và thứ hạng** (đã cài trong phiếu). Ngôn ngữ của đoạn vẫn gợi ý loại nhánh (đoạn tiếng Trung thường đến từ
  nhánh zh) — không tránh được; giảm bằng R2 và kiểm tra sau: `ingest.md` in số mục liên quan theo ngôn ngữ đoạn. Nếu
  đoạn zh bị chấm thấp hệ thống vì khó đọc (nhiều `MT`, nhiều `?`), phân xử lại phần đó.
- **Không xem kết quả trước khi chấm xong:** người chấm không mở `diag.md`, `score.md`, `key.parquet`. Người xây nhánh
  nào cũng chấm như mọi người (phiếu không cho biết nhánh).
- **Pilot trước, đóng băng hướng dẫn sau pilot:** cả 3 người chấm cùng 6 câu pilot → `MODE='score'` xem kappa pilot →
  họp, thống nhất các mục bất đồng (`disagreements.csv`), sửa hướng dẫn thành bản 2 → mới chấm vòng chính. Không sửa
  hướng dẫn giữa vòng chính; nếu buộc phải sửa, chấm lại các mục bị ảnh hưởng và ghi lại.
- **Đo đồng thuận:** `ingest` tính kappa 3 mức và kappa nhị phân (≥ 1) trên mọi cặp người chấm, tách theo vòng
  (pilot / pool / ext). Ngưỡng làm việc: kappa nhị phân ≥ 0,6. Thấp hơn → hướng dẫn chưa đủ rõ.
- **Nhãn cuối:** chưa phân xử → lấy **mức thấp hơn** (thận trọng, không thổi phồng P/R). Phân xử: một người **không biết
  nguồn** đọc `disagreements.csv`, ghi `adjudicated.csv` (cột `item,grade`) vào `mir2-labels`, đặt `ADJUDICATED` trong k6.
- **Hai ngưỡng liên quan:** mặc định "≥ 1" (nới). Báo cáo thêm ngưỡng chặt "≥ 2" (đặt `eval.rel_threshold: 2`) cho các
  kết luận quan trọng; kết luận nào đảo chiều giữa hai ngưỡng thì ghi là chưa chắc.

### 2.5 Giới hạn của Recall trên pool — phải ghi trong báo cáo

1. **R_pool chỉ đếm tài liệu liên quan nằm trong pool.** Tài liệu đúng mà không run nào đưa vào pool thì không ai biết →
   R_pool **cao hơn** Recall thật, và các run chỉ được so với nhau trên phần pool tìm được.
2. **Thiên lệch về các run đã góp vào pool.** Run không có trong pool (QE, reranker mới) có nhiều mục chưa chấm, bị tính
   là sai → bị đánh giá thấp. Luôn xem cột **judged@K**; judged@K < 0,8 → chạy `extend` trước khi kết luận.
3. **Độ sâu pool có hạn:** nhánh chỉ được chấm tới hạng 10, run gộp tới 30. Ở K = 185/200 phần lớn mục chưa chấm → P@200
   chỉ là **cận dưới**; R@200 vẫn dùng được như "tỷ lệ tài liệu đúng trong pool lọt vào top-200" nhưng lạc quan (điểm 1).
4. **Tài liệu được suy từ đoạn:** tài liệu có phần đúng nằm ở một đoạn chưa chấm mà các đoạn đã chấm đều 0 → bị tính là
   không liên quan.
5. **Không so được với leaderboard về giá trị tuyệt đối:** định nghĩa liên quan và cách gom trùng của BTC (một id cho mỗi
   nội dung trùng) khác ta. Chỉ so **tương đối** giữa các run trên cùng pool.
6. **n nhỏ:** KTC rộng; xem mục 2.1.

---

## 3. Chỉ số trên tập có nhãn — `evaluate score` / `evaluate table`

Trung bình **macro** trên các câu có nhãn. R_q = các mục liên quan (nhãn ≥ ngưỡng) **trong pool** của câu q.

**Cấp tài liệu** (thứ hạng tài liệu = đoạn tốt nhất):
- P@K = |top-K ∩ R_q| / K — mục chưa chấm tính là sai → **cận dưới**.
- Pj@K = |top-K ∩ R_q| / |top-K ∩ đã chấm| — chỉ trên mục đã chấm → cận trên thô ("condensed precision").
- R@K = |top-K ∩ R_q| / |R_q| — **Recall trên pool**.
- F2@K = 5·P·R / (4·P + R).
- judged@K = tỷ lệ mục của top-K đã được chấm.

**Cấp đoạn** (giống máy chấm BTC): một đoạn trả về **khớp** một đoạn liên quan đã chấm nếu cùng câu hỏi, cùng `doc_id`
và LCS theo token / |đoạn đúng| ≥ 0,4 sau chuẩn hoá NFKC + chữ thường (tiếng Trung tách theo ký tự).
unit P@K = tỷ lệ đoạn trong top-K khớp ít nhất một đoạn liên quan; unit R@K = tỷ lệ đoạn liên quan được ít nhất một
đoạn trong top-K khớp. Đoạn chưa chấm nhưng trùng ≥ 0,4 với đoạn liên quan đã chấm vẫn được tính (hữu ích khi so hai
cách chia đoạn khác nhau).

**nDCG@10** (nhãn 0/1/2, cấp đoạn): chất lượng thứ tự ở đầu danh sách — thước đo cho reranker, nhưng **không** quyết
định F2 dạng tập.

**Độ bất định:** KTC 95% bootstrap theo câu hỏi; so cặp với `rrf9`: chênh lệch trung bình theo câu, KTC bootstrap và
p của kiểm định hoán vị dấu (hai phía). `score.md` in bảng ở **mọi K** và so cặp ở K nhỏ nhất và lớn nhất.

**Chọn K:**
- **K = 10** — so 9 nhánh (đã chấm đủ, công bằng).
- **K = 20, 50** — so các cấu hình gộp / rerank (judged ≈ 1 với run đã vào pool tới 30).
- **K = 185 (đoạn) / 200 (tài liệu)** — giống bài nộp. Vì F2 của BTC tính trên **tập**:
  P@K = R@K · |R_q| / K, nên với K cố định cả P và R đều tỉ lệ với **số mục đúng lọt vào top-K** → thứ tự các run theo
  F2@K gần như trùng thứ tự theo R@K. Hệ quả: ở K của bài nộp hãy nhìn **R@K**; rerank chỉ có tác dụng khi nó đổi
  **tập** top-K (top_n > k_chunk; với top_n = K điểm không đổi — đã kiểm chứng trong môi trường thử).

**So sánh gì với gì:** 9 nhánh; `union9` cùng ngân sách (mục 6); `rrf9` và các biến thể (`rrf_vi`, `rrf_dense3`, bỏ
một nhánh, có trọng số); reranker trên cùng đầu vào.

**Đoạn con (đề xuất B):** mỗi nhánh Dense/LSR có run từng mức `…@parents`, `…@children` và run gộp mức (tên nhánh).
Pool chỉ chứa run gộp mức, nên muốn so mức thì chấm bổ sung:
`EXTEND='vi_dense@parents:10,vi_dense@children:10,…'`. Giữ tìm-bằng-đoạn-con nếu `@children` mang thêm tài liệu đúng
(giá trị biên, mục 4) hoặc nhánh gộp mức tốt hơn `@parents` có KTC không chứa 0.

---

## 4. Giá trị biên của từng nhánh — `evaluate marginal`

Tính trên tập có nhãn, top-50 tài liệu mỗi nhánh (`marginal.md`):

| Cột | Ý nghĩa | Loại |
|---|---|---|
| mới TB | tài liệu chưa có trong union các nhánh trước | [ĐO] trên câu có nhãn |
| mới đã chấm TB | phần "mới" đã có nhãn | — (độ phủ của nhãn) |
| **ĐÚNG mới TB** | mới **và** được chấm liên quan | [NHÃN-n] — chỉ cột này nói về Recall |
| Recall pool của union | R_pool của union các nhánh tới bước này | [NHÃN-n] |
| Tài liệu ĐÚNG chỉ một nhánh tìm ra | đúng và không nhánh nào khác có trong top-50 | [NHÃN-n] |
| Bỏ nhánh khỏi RRF: Δ doc R@50, Δ doc F2@50, Δ unit F2@50 | RRF 8 nhánh − RRF 9 nhánh, KTC + p | [NHÃN-n] — giá trị thật trong hệ thống gộp |

Hai thứ tự: **cố định** (vi → en → zh, dùng để báo cáo) và **tham lam** theo số tài liệu ĐÚNG mới (lạc quan vì chọn trên
chính dữ liệu đánh giá — chỉ để trực giác).

Cách đọc:
- **Nhiều mới, ít đúng mới** → nhánh thêm nhiễu → ứng viên bỏ hoặc giảm trọng số (xác nhận bằng Δ khi bỏ khỏi RRF ≥ 0).
- **Ít mới nhưng bỏ đi làm giảm điểm** (Δ < 0, KTC không chứa 0) → nhánh có giá trị nhờ **củng cố thứ hạng** (bỏ phiếu
  trong RRF), không nhờ tài liệu mới.
- **Có tài liệu đúng riêng nhưng Δ ≈ 0** → tài liệu riêng của nhánh bị xếp quá thấp trong RRF để lọt top-K → thử tăng
  trọng số, hoặc dựa vào rerank với top_n sâu hơn.
- **Bỏ đi làm tăng điểm** (Δ > 0, KTC không chứa 0) → nhánh đang làm hại → bỏ hoặc trọng số 0,5.

---

## 5. Chẩn đoán không nhãn trên cả 1.200 câu

| Chỉ số | Đo gì | Ở đâu | Cần nhãn? |
|---|---|---|---|
| Độ phủ, số kết quả, hồ sơ ngôn ngữ | nhánh có chạy đúng không, mang về ngôn ngữ nào | `diag.md` §1 | Không |
| Jaccard / overlap / RBO từng cặp ở K = 10/50/100 | thừa / bổ sung về tập và thứ tự | §2 + `pairwise_doc_k*.csv` | Không |
| Tài liệu mới khi thêm nhánh, tài liệu riêng | mở rộng tập ứng viên | §3 | Không |
| Thành phần RRF: tỷ lệ top-K của RRF có trong nhánh, tỷ lệ đoạn CHỈ nhánh đó đưa vào, số nhánh cùng ủng hộ một đoạn | thay đổi do gộp, nhánh nào chi phối | §4 | Không |
| Bỏ từng nhánh: Jaccard / RBO giữa RRF 8 nhánh và RRF 9 nhánh | mức ảnh hưởng của nhánh lên danh sách cuối | §5 | Không |
| Độ ổn định theo k của RRF (k = 10, 30, 100 so với 60) | danh sách cuối nhạy với tham số gộp tới đâu | §5 | Không |
| Thay đổi do rerank: Jaccard top-K, Kendall τ trên đoạn chung | rerank đổi tập hay chỉ đổi thứ tự | §5 (`--reranked`) | Không |
| Mức bất đồng theo câu | chọn câu để chấm | `judge select` | Không |
| P, R_pool, F2, nDCG, judged@K | chất lượng | `evaluate score/table` | **Có** |
| Tài liệu ĐÚNG mới, ĐÚNG riêng, Δ khi bỏ nhánh | giá trị biên thật | `evaluate marginal` | **Có** |

Ngưỡng gợi ý để **đặt câu hỏi** (không phải để quyết định) [GT]: overlap@50 > 0,8 → hai nhánh có thể thừa nhau;
nhánh zh có < 50% tài liệu tiếng Trung trong top-50 → kiểm tra bản dịch / bộ tách từ jieba; tỷ lệ đoạn CHỈ nhánh đó
đưa vào RRF < 2% → nhánh ít ảnh hưởng lên kết quả cuối (giữ hay bỏ đều ít khác biệt); RRF giữ nguyên < 70% top-50 khi
đổi k → nên đánh giá trọng số / k trên tập có nhãn.

"Đoạn được ≥ 3 nhánh ủng hộ" có thể dùng làm **gợi ý** (giả nhãn) khi chọn câu hay soi lỗi, nhưng không bao giờ được gọi
là Recall hay Precision.

---

## 6. Thiết kế thí nghiệm và bảng báo cáo

**Nguyên tắc so sánh công bằng:** cùng tập câu hỏi có nhãn; cùng K; cùng **ngân sách ứng viên**; cùng pool (cùng bộ
phiếu, đã `extend` cho run mới); cùng đơn vị (đoạn cha); cùng bản dịch; cấu hình mỗi run lưu trong file `.json` cạnh run.

**Cùng ngân sách ứng viên:**
- Nhánh với nhánh: top-K mỗi nhánh.
- Union với RRF: `union9` = lấy xen kẽ hạng 1 của mọi nhánh, rồi hạng 2… tới khi đủ **đúng K** đoạn (`evaluate table`)
  → so với RRF top-K. Union "thô" (gộp hết top-K của 9 nhánh = tới 9K mục) luôn có Recall cao hơn nhưng không nộp được.
- Reranker với reranker: cùng đầu vào (`rrf9` top-N), cùng N, cùng đầu ra K; phần sau hạng N giữ thứ tự của `rrf9`.
  So N ∈ {200, 400} chỉ khi còn thời gian (thời gian rerank tỉ lệ với N). Ở K = 185, F2 chỉ đổi khi N > 185.
- QE về sau: so có/không QE với cùng số nhánh và độ sâu.

**Chọn Top-N / K:** bài nộp cố định k_doc = 200, k_chunk = 185 (file nén ≤ 100 MB; `submit` tự giảm k_chunk nếu
vượt). Rerank top_n = 400 (dự án cũ: ColBERT top-400 tốt nhất, 0,1674 [LB]).

**Các bảng nên có** (mỗi bảng ghi rõ: n câu, K, ngưỡng nhãn, phiên bản pool/phiếu, bản dịch):

*Bảng 1 — Chẩn đoán [ĐO] (1.200 câu):* run · số câu có kết quả · %vi/%zh/%en top-50 · overlap@50 lớn nhất với nhánh
khác (và với nhánh nào) · tài liệu riêng@50 · tỷ lệ đoạn chỉ nhánh này đưa vào `rrf9` · Jaccard(RRF bỏ nhánh, RRF đủ).

*Bảng 2 — Từng nhánh ở K = 10 [NHÃN-n]:* run · judged@10 · doc P@10 [KTC] · doc R_pool@10 · unit P@10 · unit F2@10 ·
nDCG@10 · Δ so với nhánh tốt nhất [KTC]. (lấy từ `score.md`, mục K = 10)

*Bảng 3 — Giá trị biên [NHÃN-n]:* nhánh · mới@50 · ĐÚNG mới@50 · ĐÚNG riêng · Δ doc R@50 khi bỏ [KTC] · Δ doc F2@50 khi
bỏ [KTC] · **quyết định** (mục 8). (lấy từ `marginal.md`)

*Bảng 4 — Hệ thống ở K = 20 / 50 / 200 [NHÃN-n]:* loại (nhánh / union / gộp / rerank) · run · judged@K · doc P, R, F2 ·
unit P, R, F2 · nDCG@10 · Δ so với `rrf9` [KTC], p. (lấy từ `table_k20.md`, `table_k50.md`, `table_k200.md`)

*Bảng 5 — Leaderboard [LB]:* bài nộp · cấu hình (run, k_doc, k_chunk, top_n) · F2 tài liệu / đoạn · ngày.

---

## 7. Báo cáo trung thực

- **Không gọi độ trùng / số tài liệu mới / tài liệu riêng là Recall.**
  - Sai: "zh_bm25 tăng Recall 12%."
  - Đúng: "[ĐO] zh_bm25 thêm trung bình 12 tài liệu chưa có ở các nhánh khác trong top-50 (1.200 câu). [NHÃN-54] Trong
    đó trung bình 0,8 tài liệu/câu được chấm liên quan; bỏ zh_bm25 khỏi RRF làm R_pool@50 thay đổi −0,02 [−0,04; 0,00]
    — chưa đủ bằng chứng."
- **Luôn viết "Recall trên pool (R_pool@K)"**, kèm n câu và K.
- **Luôn kèm KTC**; chênh lệch có KTC chứa 0 → "chưa phân biệt được", không viết "tốt hơn".
- **Nhiều phép so sánh:** ~30 run so cùng lúc → vài p < 0,05 có thể là ngẫu nhiên. Coi p trong khoảng 0,01–0,05 là bằng
  chứng yếu, nhất là khi chỉ xuất hiện ở một K.
- **Luôn kèm judged@K** cạnh P và F2.
- **Tách bạch** bốn loại kết quả [ĐO] / [NHÃN-n] / [GT] / [LB] trong văn bản và trong bảng.
- **Không vừa chỉnh vừa chấm trên cùng tập:** nếu chọn trọng số RRF / k / top_n bằng tập có nhãn, chia đôi câu hỏi
  (một nửa để chỉnh, một nửa để báo cáo) hoặc xác nhận bằng leaderboard; ghi rõ đã làm gì.
- **Ghi cả kết quả âm và thay đổi kế hoạch:** sửa hướng dẫn chấm, đổi câu hỏi, nhánh bị lỗi rồi chạy lại.

---

## 8. Quy tắc quyết định giữ / bỏ / điều chỉnh từng nhánh

Dùng Δ khi **bỏ nhánh khỏi RRF** ở K = 50 (và kiểm tra lại ở K = 200), cùng tài liệu ĐÚNG riêng:

| Quyết định | Điều kiện |
|---|---|
| **GIỮ** | Bỏ nhánh làm giảm doc R@50 hoặc doc F2@50 với KTC không chứa 0; **hoặc** nhánh có tài liệu ĐÚNG riêng đáng kể (vd ≥ 0,3 tài liệu/câu — ngưỡng tự đặt, ghi rõ). |
| **BỎ** | Bỏ nhánh làm **tăng** doc F2@50 với KTC không chứa 0 và gần như không có tài liệu ĐÚNG riêng; hoặc nhánh lỗi (chẩn đoán mục 1, 5). |
| **GIẢM TRỌNG SỐ** | Nhiều tài liệu mới nhưng ít ĐÚNG mới, Δ ≈ 0: thử `python -m mirret.fuse --runs vi_dense,…,zh_bm25*0.5 --name rrf9_w`, `extend` rồi so trên tập có nhãn (chia đôi câu hỏi như mục 7). |
| **CHƯA ĐỦ BẰNG CHỨNG** | Còn lại → **giữ nguyên** (RRF chịu được nhánh yếu; giữ chỉ tốn tính toán), ghi là giả thuyết; xác nhận bằng leaderboard nếu còn lượt. |

Chi phí giữ một nhánh thấp: BM25 rẻ; LSR dùng lại vector đã mã hoá; Dense cần chỉ mục FAISS. Lý do chính để bỏ là
**chất lượng**, không phải tính toán.

**Xác nhận leaderboard** (1–2 lượt): `rrf9` so với cấu hình sau khi bỏ/giảm trọng số. Chênh lệch nhỏ hơn mức dao động
giữa hai bài nộp gần như giống nhau → coi là hoà.

**Sau khi quyết định:** đầu tư QE cho các nhánh được giữ (ưu tiên nhánh có tài liệu đúng riêng — nơi QE có thể mở rộng
thêm), rồi mới tối ưu rerank (top_n, ColBERT hay cross-encoder) — so trên tập có nhãn bằng `extend`.

---

## 9. Tài nguyên và thời gian (ước tính, chưa đo trên kho thật)

| Việc | Máy | Ước tính |
|---|---|---|
| k2 dịch 2 model | GPU ×2 | 14B ~1,5–2,5 giờ; 8B ~1 giờ |
| k3 mã hoá đoạn cha + con | GPU ×2 × 3 người | ~8–10 giờ GPU ×2 tổng → ~3 giờ/người (đo bằng `RUN_BENCH`) |
| k4 chỉ mục + 9 nhánh + chẩn đoán | GPU (mã hoá câu hỏi), CPU | ~1–2 giờ |
| k5 rerank top-400 | GPU ×2 | ColBERT ~100 phút (dự án cũ); cross-encoder chậm hơn |
| k6 pool / tính điểm | CPU | < 30 phút / < 1 giờ |
| Chấm tay | 3 người | pilot ~2 giờ/người; vòng chính 5–10 giờ/người (mục 2.2) |

Hạn mức GPU Kaggle ~30 giờ/tuần/tài khoản: chia k3 cho 3 tài khoản; k2, k4, k5 mỗi việc một tài khoản.

**Mở rộng (chưa cài):** dùng LLM (vd Qwen2.5-14B) chấm sơ bộ để lọc các mục rõ ràng là 0 (menu, trang thông báo).
Chỉ dùng sau khi đo đồng thuận LLM–người trên phiếu pilot (kappa như giữa người với người), và báo cáo riêng; LLM dễ
thiên về trùng từ khoá.

---

## Phụ lục A — Lệnh

Trên Kaggle dùng notebook (`notebooks/k*.ipynb`); trên máy (cần `chunks/` và `runs/`):

```bash
# chẩn đoán không nhãn (k4 tự chạy; thêm rerank khi có)
python -m mirret.diag --reranked rrf9_colbert,rrf9_cross
# tập có nhãn (k6 MODE='pool')
python -m mirret.judge select --n 60 --pilot 6
python -m mirret.judge pool --branch-depth 10 --fused rrf9:30,rrf9_colbert:30,rrf9_cross:30 --labelers A,B,C
# run mới (k6 MODE='extend')
python -m mirret.judge extend --runs rrf9_w:30
# nhập nhãn + điểm (k6 MODE='score')
python -m mirret.judge ingest [--adjudicated eval/judge/adjudicated.csv]
python -m mirret.evaluate score --ks 10,20,50,100,185,200 --runs <danh sách run>
python -m mirret.evaluate marginal --k 50
python -m mirret.evaluate table --ks 20      # → eval/table_k20.md
# bản dịch
python -m mirret.translate compare --a qwen25_14b --b qwen3_8b --n 100
python -m mirret.translate score --sheet a.csv --sheet b.csv
```

## Phụ lục B — Thư mục `eval/`

```
eval/
  diag/diag.md, pairwise_doc_k{10,50,100}.csv      [ĐO] chẩn đoán không nhãn
  judge/queries.csv                                câu hỏi đã chọn (cột set = pilot/main, tầng)
  judge/key.parquet                                nguồn của từng mục (KHÔNG mở khi chấm)
  judge/sheet_{pilot,pool,ext<n>}_<người>.csv      phiếu chấm
  judge/ingest.md, disagreements.csv, qrels.parquet
  score.md, perq_<run>.csv                         [NHÃN-n] điểm theo K, so cặp với rrf9
  marginal.md                                      [NHÃN-n] giá trị biên, bỏ từng nhánh
  table_k{20,50,200}.md                            [NHÃN-n] bảng tổng hợp cùng ngân sách
```

## Phụ lục C — Checklist trước khi báo cáo

- [ ] Kappa nhị phân pilot và vòng chính ≥ 0,6 (hoặc đã ghi rõ lý do và cách xử lý).
- [ ] Bất đồng đã phân xử, hoặc ghi rõ đang dùng "mức thấp hơn".
- [ ] Mọi run được so có judged@K ≥ 0,8 ở K dùng để kết luận (hoặc đã `extend`).
- [ ] Mỗi con số có nhãn loại [ĐO] / [NHÃN-n] / [GT] / [LB], có n câu và K.
- [ ] Không có câu nào gọi độ trùng / tài liệu mới là Recall.
- [ ] Mọi chênh lệch đều kèm KTC; chênh lệch có KTC chứa 0 ghi là "chưa phân biệt được".
- [ ] Nếu đã chỉnh tham số bằng tập có nhãn: đã chia đôi câu hỏi hoặc xác nhận bằng leaderboard.
- [ ] Ghi phiên bản hướng dẫn chấm, bản dịch dùng cho nhánh en/zh, phiên bản `mir2-chunks` / `mir2-runs`.
