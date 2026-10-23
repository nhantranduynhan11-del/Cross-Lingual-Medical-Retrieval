# BÁO CÁO TIẾN ĐỘ & KẾT QUẢ THỰC HIỆN - ROLE C (TASK C1)
**Dự án:** ViBioMIR - Cross-Lingual Medical Information Retrieval (Vi - En - Zh)  
**Nhánh Git:** `c-e5`  
**Thành viên phụ trách:** Role C (Kaggle: `huyphong09`)  
**Mục tiêu chính:** Triển khai Dense Retrieval với mô hình `intfloat/multilingual-e5-large` trên GPU độc lập T4 ×2, hoàn thành mốc đánh giá Go/No-Go trước 21/10/2026 và hoàn thành mã hoá trước 24/10/2026.
Kế hoạch: https://claude.ai/artifact/KSmsvjrsipMogjb3B5SHBN
---

## 📌 TỔNG QUAN KẾT QUẢ MỐC GO / NO-GO (20/10/2026)

* **Hạn chót đánh giá:** 21/10/2026.
* **Thời điểm hoàn thành:** **20/10/2026** *(vượt trước tiến độ 1 ngày)*.
* **Quyết định chính thức:** **`>>> GO <<<`**
* **Ý nghĩa quyết định:** Tài nguyên GPU Tesla T4 ×2 độc lập của Role C hoàn toàn đủ khả năng mã hoá **100% kho văn bản (cả khối Parents và khối Children)**, đảm bảo nằm trọn vẹn trong hạn mức GPU Quota khả dụng (30 giờ GPU) và kịp tiến độ toàn đội trước 24/10/2026.

---

## 🛠️ CHI TIẾT CÁC CÔNG VIỆC CHÍNH ĐÃ HOÀN THÀNH

### 1. Nâng cấp Core Module `mirret/encode.py` chuẩn mE5-large
* **Kiến trúc Encoder:** Xây dựng lớp `E5Encoder` và bộ nạp `load_encoder()` hỗ trợ song song cả `BAAI/bge-m3` và `intfloat/multilingual-e5-large`.
* **Quy chuẩn định dạng chuỗi (Text Prefixing):**
  * Tự động đệm tiền tố `"passage: "` cho tài liệu corpus (cả đoạn cha và đoạn con).
  * Tự động đệm tiền tố `"query: "` cho câu hỏi truy vấn.
  * Tích hợp cơ chế kiểm tra chống lặp đệm (`startswith("passage: ")` / `startswith("query: ")`).
* **Cấu hình mô hình:**
  * Giới hạn độ dài token: `max_length = 512`.
  * Chế độ suy luận: Bán chính xác `fp16 = True`, hỗ trợ SDPA (`Scaled Dot-Product Attention`) tăng tốc tối đa trên kiến trúc Turing (Tesla T4).
  * Cơ chế Pooling: **Mean Pooling** trên ma trận đặc trưng ẩn `last_hidden_state` có xét mặt nạ `attention_mask`.
  * Chuẩn hoá vector: **L2 Normalization** (`torch.nn.functional.normalize(p=2, dim=1)`), đảm bảo tích vô hướng (Dot Product) tương đương Cosine Similarity phục vụ tìm kiếm Faiss Inner Product (`IndexFlatIP`).

---

### 2. Xây dựng công cụ Speed Profiling & Công thức tính ETA tự động
* Bổ sung lệnh CLI `profile` vào module `mirret.encode`:
  ```bash
  python -m mirret.encode profile --config configs/retrieval_e5.yaml --device cuda:0 --n-children 2000 --n-parents 1000 --quota 30.0
  ```
* **Lấy mẫu thực nghiệm:** Đo thời gian mã hoá thực tế trên $2.000$ đoạn con ($T_{children\_sample}$) và $1.000$ đoạn cha ($T_{parents\_sample}$) trực tiếp từ dữ liệu chunking.
* **Mô hình toán học dự báo ETA:**
  $$\text{ETA} = \left(\frac{N_{children\_total}}{N_{children\_sample}} \times T_{children\_sample}\right) + \left(\frac{N_{parents\_total}}{N_{parents\_sample}} \times T_{parents\_sample}\right)$$
* **Bộ điều hướng quyết định:**
  * Nếu $\text{ETA} \le 30.0\text{h}$ GPU $\rightarrow$ Kết luận **GO**: Triển khai mã hoá toàn bộ kho (Parents + Children).
  * Nếu $\text{ETA} > 30.0\text{h}$ GPU $\rightarrow$ Kích hoạt **Fallback**:
    * *Option A:* Chỉ mã hoá Parents (bỏ qua khối Children).
    * *Option B:* Hỗ trợ Shard cho Role A ở Task A5.

---

### 3. Thiết lập hệ thống cấu hình & Notebook Kaggle
* **File cấu hình riêng:** Tạo [configs/retrieval_e5.yaml](file:///d:/BKAi/ViBioMIR/Cross-Lingual-Medical-Retrieval/configs/retrieval_e5.yaml) định nghĩa đường dẫn `emb_e5`, chỉ mục `index_e5`, tham số phân khối `block: 100000`, batch size và index IVF/Flat.
* **Notebook tối ưu 2x GPU T4:** Tạo [notebooks/k3_encode_e5.ipynb](file:///d:/BKAi/ViBioMIR/Cross-Lingual-Medical-Retrieval/notebooks/k3_encode_e5.ipynb) với phân việc Role C:
  * 1 thành viên chạy độc lập (`MEMBERS = 1, MEMBER = 0`).
  * Chia việc song song: **GPU 0 chạy Shard 0/2**, **GPU 1 chạy Shard 1/2** $\rightarrow$ Thời gian thực tế (wall-clock) giảm một nửa so với ETA 1 GPU.

---

### 4. Giải quyết triệt để các rào cản triển khai trên Kaggle
* **Loại bỏ sự phụ thuộc vào file/dataset code thủ công (`mir2-code`):**
  * *Vấn đề:* Ban đầu Kaggle nạp nhầm dataset `mir2-code` cũ của teammate (từ 07/10) thiếu lệnh `profile`.
  * *Giải pháp:* Thiết kế Ô (1) tự động `git clone --depth 1 -b c-e5` trực tiếp từ GitHub repo về `/kaggle/working/code` chỉ mất ~2 giây, luôn đồng bộ mã nguồn mới nhất mà người dùng không cần quan tâm đến file dataset code.
* **Xử lý lỗi môi trường Linux (`sh: 0: getcwd() failed`):**
  * *Vấn đề:* Khi chạy lại Ô (1), lệnh `shutil.rmtree(CODE)` xoá thư mục khi tiến trình Python đang đứng bên trong, khiến các tiến trình con `subprocess` bị mất thư mục hiện hành và không tìm thấy module `mirret`.
  * *Giải pháp:* Cập nhật [mirret/kaggle_nb.py](file:///d:/BKAi/ViBioMIR/Cross-Lingual-Medical-Retrieval/mirret/kaggle_nb.py) bổ sung `os.chdir(KAGGLE)` trước khi xoá, truyền `cwd` tường minh cho `subprocess.Popen` và tiêm `PYTHONPATH` vào biến môi trường hệ thống.
* **Khắc phục cảnh báo Hugging Face Hub (`HF_TOKEN`):**
  * Cung cấp tài liệu hướng dẫn về bản chất cảnh báo (chỉ là thông báo ưu tiên rate limit, model là public nên tải bình thường).
* **Vượt rào cản giới hạn ổ đĩa 20 GB trên Kaggle (Task C1 Resume Pipeline):**
  * *Vấn đề:* Tổng dung lượng embedding mE5 (Parents + Children) đạt 23.22 GB, vượt mức trần 20.0 GB của Kaggle Container, gây tràn ổ đĩa khi chạy đến khối 62/73.
  * *Giải pháp:* Nâng cấp `mirret/encode.py` hỗ trợ cờ `--blocks`, phân bổ 20 khối còn thiếu cho 2 GPU song song và thu hồi output gọn nhẹ (~3.82 GB).
  * *Kết quả:* Đã hợp nhất thành công 100% toàn bộ 121 khối (34 khối Parents + 87 khối Children, đủ 12.008.828 đơn vị văn bản).

---

## 🏆 KẾT QUẢ HOÀN THÀNH TASK C1 (23/10/2026)

* **Tiến độ:** **100% HOÀN TẤT** *(vượt trước hạn chót 24/10/2026 1 ngày)*.
* **Tổng số đơn vị mã hoá:** **12,008,828 đơn vị** (100% kho ViBioMIR).
  * **Khối Parents:** 34 / 34 khối (3,332,632 đoạn cha) — 6.45 GB.
  * **Khối Children:** 87 / 87 khối (8,676,196 đoạn con) — 16.77 GB.
* **Tổng dung lượng bộ vector:** **23.22 GB**.
* **Trạng thái xác thực:** Tất cả 121 khối đều có đủ `dense.npy`, `sparse.npz`, `ids.parquet`, `_DONE.json`.

---

## 🚀 CÁC BƯỚC TIẾP THEO (TASK C2: LẬP CHỈ MỤC & TRUY HỒI)

1. **Đóng gói & Đăng tải Dataset `huyphong09/mir2-emb-e5` lên Kaggle:**
   * Sử dụng Kaggle API hoặc giao diện Web để tạo dataset từ thư mục `emb_e5` (hạn mức upload dataset 100 GB).
2. **Lập chỉ mục Dense FAISS IVF-SQ8 (`index_e5`):**
   * Dựng chỉ mục `dense_parents.faiss` và `dense_children.faiss` theo cấu hình `configs/retrieval_e5.yaml`.
3. **Mã hoá câu hỏi & Chạy truy hồi (Dense Retrieval):**
   * Mã hoá câu hỏi `query.parquet` (1.200 câu) bằng `mE5-large`.
   * Chạy truy hồi trên FAISS index để sinh run `dense_e5`.
4. **Hợp nhất đa mô hình (Multi-model Fusion):**
   * Đóng góp run `dense_e5` vào hệ thống RRF cùng với BM25 (Role A) và BGE-M3 (Role B) để tối ưu điểm F2.

---
*Báo cáo được cập nhật tự động vào ngày 23/10/2026 - Nhánh `c-e5` - Dự án ViBioMIR 2026.*
---------------------
Cập nhật cho ae baseline điểm cao nhất hiện tại của mình nha @All 
Bước 1: Chunking
Đoạn cha: 150–400 token BGE-M3 (đích 300), chia theo cấu trúc bài, không chồng lấn, là đơn vị nộp + dùng để rerank
Đoạn con: ≤ 128 token, nằm trong đoạn cha; Chỉ để tìm mịn hơn. Điểm đoạn con được dồn về đoạn cha.
Ngữ cảnh: Dòng “tiêu đề > mục”, ≤ 220 ký tự; ghép vào trước đoạn khi lập chỉ mục và khi rerank. Không nộp

Bước 2: Dịch câu query: Câu query tiếng Việt được dịch sang tiếng anh và tiếng trung bằng Qwen2.5-14B (prompt t2, có bảng thuật ngữ y khoa)

Bước 3: Truy hồi 9 nhánh: 
Mỗi ngôn ngữ chạy 3 kiểu tìm, tổng 9 nhánh:
Dense (BGE-M3, FAISS IVF-SQ8, nprobe 64): tìm ở cả đoạn cha và đoạn con. Tìm theo nghĩa, bắt được cách diễn đạt khác.
LSR (trọng số từ thưa của BGE-M3): tìm ở cả đoạn cha và đoạn con. Giống tìm từ khoá nhưng có trọng số học được.
BM25: chỉ trên đoạn cha. Khớp từ khoá chính xác, mạnh với tên thuốc, tên bệnh, viết tắt.

Bước 4: Gộp tuyến tính thay cho RRF (linear với alpha = 0.7) gọi là lin07
1. Với mỗi nhánh và mỗi câu hỏi, chuẩn hoá điểm min-max về [0, 1]. Đoạn cha lấy điểm cao nhất giữa mức cha và mức con.
2. Điểm cuối = 0,7 × trung bình 3 nhánh dense + 0,3 × trung bình có trọng số 6 nhánh từ khoá. Hai nhánh en_lsr và en_bm25 có trọng số 0,5, giống rrf9_w.
3. Kết quả là một danh sách đoạn cha xếp theo điểm.

Bước 5: Rerank top-400 bằng 2 mô hình (bge vs qwen)
Cả hai mô hình chấm cùng 400 đoạn cha đứng đầu lin07. Mỗi đoạn được chấm bằng câu hỏi cùng ngôn ngữ với nó (match_lang): đoạn zh chấm bằng câu hỏi zh, đoạn en bằng câu hỏi en.
sau đó ens (ensemble) gộp lại:
1. Với từng câu, trên đúng 400 đoạn đó, chuẩn hoá z-score điểm bge.
2. Đổi P(yes) của Qwen3 sang logit trước rồi mới chuẩn hoá z-score. Lý do: xác suất dồn sát 1, chuẩn hoá trực tiếp sẽ mất khả năng phân biệt.
3. Điểm ens = trung bình hai z-score.
4. Các đoạn sau hạng 400 giữ nguyên thứ tự của lin07.

Bước 6: Hai danh sách nộp, chọn riêng
FINAL = (doc F2 + chunk F2) / 2. Hai danh sách được chấm độc lập, nên mỗi danh sách lấy từ nguồn tốt nhất cho nó.

6a. Danh sách tài liệu: Qwen3i + cắt theo từng câu (autok400)
dùng xác suất của mô hình Qwen3 để tự động quyết định: "Câu hỏi này nên nộp bao nhiêu tài liệu là lời nhất?"
1. Gán mỗi tài liệu một xác suất đúng p = P(yes) cao nhất của các đoạn của nó trong top-400. Tài liệu ngoài top-400 có p = 0.
2. Xếp các tài liệu từ cao xuống thấp theo xác suất p này
3. Tìm điểm cắt k tốt nhất (autok): Máy tính sẽ thử chạy k từ 100 đến 400. Tại giá trị k nào mà con số EF_2 đạt đỉnh cao nhất thì dừng lại và lấy đúng k tài liệu đó để nộp trong đó EF_2 là kỳ vọng điểm F2

6b. Danh sách đoạn: 185 đoạn cha theo thứ tự ens
1. Lấy 185 đoạn cha đứng đầu theo điểm ens.
2. Trong cùng một tài liệu, nếu có các đoạn văn nội dung gần giống hệt nhau thì bỏ bớt đi để không lãng phí lượt nộp