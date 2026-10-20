# BÁO CÁO TIẾN ĐỘ & KẾT QUẢ THỰC HIỆN - ROLE C (TASK C1)
**Dự án:** ViBioMIR - Cross-Lingual Medical Information Retrieval (Vi - En - Zh)  
**Nhánh Git:** `c-e5`  
**Thành viên phụ trách:** Role C (Kaggle: `huyphong09`)  
**Mục tiêu chính:** Triển khai Dense Retrieval với mô hình `intfloat/multilingual-e5-large` trên GPU độc lập T4 ×2, hoàn thành mốc đánh giá Go/No-Go trước 21/10/2026 và hoàn thành mã hoá trước 24/10/2026.

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

---

## 🚀 KẾ HOẠCH BƯỚC TIẾP THEO (GIAI ĐOẠN 20/10 - 24/10)

1. **Mã hoá toàn bộ kho Corpus (Task C1 Execution):**
   * Đổi `RUN_PROFILE = False` ở Ô (1) của notebook.
   * Kích hoạt chạy Ô (3) trên phiên GPU T4 ×2:
     * Shard 0/2 trên GPU 0 (CUDA_VISIBLE_DEVICES=0).
     * Shard 1/2 trên GPU 1 (CUDA_VISIBLE_DEVICES=1).
   * Thực hiện qua cơ chế **Save Version $\rightarrow$ Save & Run All (Commit)** để tiến trình chạy ngầm trên đám mây của Kaggle qua đêm an toàn.
2. **Đóng gói Dataset đầu ra `mir2-emb-e5`:**
   * Sau khi chạy xong, kết quả lưu tại thư mục `/kaggle/working/emb_e5/`:
     * `emb_e5/parents/<khối>/{dense.npy, ids.parquet, _DONE.json}`
     * `emb_e5/children/<khối>/{dense.npy, ids.parquet, _DONE.json}`
   * Tạo dataset Kaggle **`huyphong09/mir2-emb-e5`** làm đầu vào cho pha Lập chỉ mục Dense (`index_e5`) và Pha Reranking.
3. **Phối hợp tích hợp:**
   * Cung cấp các vector `mE5-large` để nhóm phối hợp hợp nhất (Fusion / RRF) với nhánh BM25 (Task A) và BGE-M3 (Task B).

---
*Báo cáo được khởi tạo tự động và cập nhật trên nhánh `c-e5` - Dự án ViBioMIR 2026.*
