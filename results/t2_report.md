# T2 — Chia đoạn: báo cáo

Đầu vào: `corpus_clean` (9,184 bài). Độ dài = token BGE-M3. T3 mã hoá `tiêu đề + \n + đoạn` với max_length 512.
Cấu hình: max_tokens=400, min_tokens=40, overlap=0, parent=480, child=128.

## structure

- Số đoạn: **22,849** (2.49 đoạn/bài); bất biến char_start/char_end sai: **0**
- Token mỗi đoạn: p5=69, p25=172, p50=292, p75=361, p95=395, p99=399, max=401
  - vi (7,043 đoạn): p5=67, p25=216, p50=323, p75=370, p95=395, p99=399, max=401
  - zh (15,806 đoạn): p5=69, p25=158, p50=275, p75=355, p95=394, p99=399, max=401
- `tiêu đề + đoạn` vượt 512 token (bị cắt ở T3): **0 (0.00%)**
- Ước lượng toàn kho (~3,410,000 bài × 2.49): **~8,483,785 đoạn**

