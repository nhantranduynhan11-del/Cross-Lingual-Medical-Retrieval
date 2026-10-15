"""mirret — pipeline truy hồi cho ViBioMIR: chia đoạn (cha/con + ngữ cảnh), dịch câu hỏi, 9 nhánh truy hồi
(Dense / LSR / BM25 × câu hỏi vi / en / zh), RRF, xếp hạng lại, bài nộp, và bộ công cụ đánh giá khi KHÔNG có nhãn BTC.

Phần lõi đã kiểm chứng trên Kaggle được dùng lại từ codebase cũ (mir/): mã hoá BGE-M3, BM25 thống kê toàn cục chia
shard, chỉ mục đảo CSC, RRF, ColBERT MaxSim, prompt dịch v2.
"""
__version__ = "0.1.0"
