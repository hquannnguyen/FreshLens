# Kiểm tra lần tiếp theo ngày 02/11/2026

Đối chiếu source, model và báo cáo sau khi bổ sung hash nội dung batch, đổi quy tắc đánh giá SVM và thêm benchmark. Đây là trạng thái mới hơn hai báo cáo kiểm tra trước trong cùng ngày.

## Phần đã xác nhận

- `python -m compileall -q app.py src scripts`: exit 0.
- `python -m unittest discover -s tests -v`: 4/4 đạt.
- `python scripts/check_batch_content_cache.py`: nội dung ảnh khác, cùng tên tệp, kết quả có ID mới; `result_reused_after_content_change: false`, không exception.
- `python scripts/check_ui_release.py`: lô mẫu đủ 6 ảnh, chuyển sang model ID CNN, bấm chạy lại và nạp lại lô đều đạt; ảnh hỏng không làm trang crash.
- Model SVM thực tế có `svc__C=0.5`. Đánh giá lại trên feature cache val/test bằng argmax xác suất khớp `baseline_metrics.json`; so sánh với `predict_from_features()` có 0 nhãn lệch trên cả hai split.

| Split SVM | Accuracy | Macro-F1 |
|---|---:|---:|
| Val (180) | 0.8722222222 | 0.8712809239 |
| Test (180) | 0.8500000000 | 0.8486280616 |

Phép kiểm tra này dùng feature cache sẵn có, không chạy lại phân vùng toàn bộ dataset và không huấn luyện lại. Hash SHA-256 SVM hiện tại là `c6b05978b4b01c302425c6912bc74f1f0a73b9bae1537db701ef8bc7d7f21420`.

CNN checkpoint có cùng SHA-256 với lần kiểm tra trước (`3c9ee1b5628ff4361bda127e333bb39f1569340884ecd385bad89070d1b2a890`). Source service vẫn truyền `work` và giữ bước từ chối mask trước CNN. Kết quả đã đo trước là 179 ảnh được chấp nhận/180 ảnh, không đổi nhãn trên 179 ảnh đó; lần này không lặp lại phép đo toàn bộ CNN vì phần đó không có thay đổi cần kiểm tra lại.

## Các điểm chưa hoàn tất

### 1. Benchmark chưa tái lập được từ script được bàn giao

Lệnh `python scripts/benchmark_inference.py` dừng ở dòng 16 với `ModuleNotFoundError: No module named 'src'`. Script tính `ROOT` nhưng không thêm root vào đường dẫn import như các script huấn luyện.

Ngay cả khi chạy bằng module để giải quyết import, mã hiện tại chỉ xuất các khóa `mean_latency_ms`, `median_latency_ms`, `p95_latency_ms`. Nó không có phép đo/tách các trường `pure_inference_ms`, `feature_extraction_ms`, `pure_forward_pass_ms` đang có trong `outputs/inference_benchmark.json`. Vì vậy tệp công bố chưa tái lập được bằng script này.

### 2. Số độ trễ không phải toàn trình của ứng dụng

JSON ghi GrabCut trung bình 6407,6 ms, nhưng tổng SVM 8,47 ms chỉ cộng feature 6,67 ms và model 1,795 ms. Tổng CNN 87,79 ms chỉ cộng transform và forward. Cả hai đường `AnalysisService` hiện đều chạy phân vùng, đánh giá chất lượng và tạo ảnh minh họa trước khi trả kết quả. Các số 8,47/87,79 ms không đại diện độ trễ service.

Script benchmark còn tính segmentation vào thời gian SVM (dòng 57–64), nhưng bỏ segmentation khỏi thời gian CNN (dòng 70–78). Cần đo hai phạm vi rõ ràng: classifier/transform riêng và `AnalysisService.analyze()` cho cả hai mô hình, kèm cấu hình threads, môi trường, hash model, warmup và mẫu thời gian thực tế. Tệp benchmark hiện không chứa các thông tin nhận diện lần đo này.

### 3. Kích thước model và giao diện vẫn sai

SVM hiện tại là **148.693 byte = 145,21 KiB**, trong khi JSON/README/HANDOFF còn ghi 136.485 byte/133,3 KB của artifact cũ. CNN hiện là 9.145.099 byte = 8,72 MiB. Nên ghi đúng đơn vị nhị phân khi chia cho 1024.

`app.py:468` trở đi vẫn ghi cứng SVM 86,7%/86,6%, kích thước ~28 KB, CNN ~25 ms và confusion matrix cũ. Tab đối chuẩn chưa đọc các số mới, nên tuyên bố đã cập nhật giao diện chưa khớp source.

### 4. `--eval-only` chưa chuyển sang `weights=None`

`scripts/train_cnn.py:133` vẫn gọi `models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)` trước nhánh `if args.eval_only` ở dòng 140. Máy thiếu cache ImageNet vẫn có thể cần tải trọng số. Nhánh eval còn ghi đè metrics/manifest. Khả năng giữ metadata đã có không xác nhận nguồn gốc lịch sử 10 epoch được thêm vào checkpoint; giới hạn bằng chứng trong báo cáo trước vẫn còn.

### 5. Tài liệu còn mâu thuẫn

ARCHITECTURE đã cập nhật vị trí các module, nhưng vẫn mô tả Quality Assessment đo cả ROI trong khi service gọi `assess_image(work)`, và nêu Model Registry kiểm tra phiên bản/cấu hình trong khi source chủ yếu nạp theo mtime. HANDOFF vừa ghi C=1,0/F1=86,6% ở các phần cũ, vừa ghi C=0,5/F1=84,9% ở bảng mới. Chưa thể đánh dấu tài liệu hoàn toàn đồng bộ.

## Bằng chứng runtime

`outputs/check_2026-11-02/batch_content_cache_check.json` và `ui_check.json` đã được tạo lại bằng các lệnh kiểm tra trong lần này. Model, metrics và benchmark gốc không bị ghi đè trong quá trình kiểm tra. Việc chạy benchmark trực tiếp thất bại trước khi ghi kết quả.
