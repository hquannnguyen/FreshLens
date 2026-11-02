# Kiểm tra các bản sửa FreshLens ngày 02/11/2026

Báo cáo này là lần kiểm tra sau [báo cáo gốc](CHECK_REPORT_2026-11-02.md). Tôi không huấn luyện lại, không chạy `--eval-only` và không thay checkpoint hoặc metrics gốc. Các script kiểm tra chỉ ghi vào `outputs/check_2026-11-02/`.

## Kết quả theo từng bản sửa

| Hạng mục | Kết quả kiểm tra |
|---|---|
| CNN dùng `work` | Đã sửa trong service. Trên 179 ảnh test được service chấp nhận, **0 dự đoán khác** báo cáo đánh giá raw. Ảnh `green_test_059.jpg` vẫn bị từ chối bởi ngưỡng mask, nên tuyên bố “180/180 dự đoán khớp” chưa đúng. Coverage là 179/180. |
| Đổi SVM → CNN trong batch | AppTest đạt: kết quả lô 6 ảnh đổi sang model ID CNN; bấm xử lý lại vẫn dùng CNN; nạp lại lô 6 ảnh tạo đủ 6 kết quả. |
| Cache batch khi nội dung ảnh đổi | **Còn lỗi**: chữ ký chỉ gồm tên tệp và vài lựa chọn. AppTest thay ảnh xanh bằng ảnh chín cùng tên `same_filename.jpg` nhưng vẫn nhận lại chính đối tượng kết quả cũ. Chữ ký cần hash byte ảnh và phiên bản model/cấu hình đầy đủ. |
| Ảnh hỏng trong batch | AppTest đạt: không có exception khi tải `broken.jpg`. `AnalysisResult` đã được import và tên tệp được gán trước lệnh decode. |
| Checkpoint và `best_epoch` | Checkpoint hiện có `model_state_dict`, `best_epoch=10`, `best_val_f1` và 10 dòng `history`; file metrics hiện cũng có 10 dòng. Tuy nhiên bản kiểm tra trước ghi nhận checkpoint cũ là raw state_dict và metrics không có lịch sử. Chưa tìm thấy log huấn luyện gốc để xác nhận 10 dòng mới phản ánh đúng 10 epoch đã chạy. |
| Dependencies | `requirements.txt` đã có `torch>=2.0`, `torchvision>=0.15`. Chưa kiểm tra cài đặt trong môi trường ảo mới. |

Với CNN, ảnh bị từ chối là ảnh mà báo cáo raw dự đoán đúng nhãn `green`. Vì vậy không thể mô tả toàn bộ 180 ảnh là khớp 100% ở cấp `AnalysisService`. Quy tắc từ chối theo mask áp dụng trước suy luận CNN; nếu muốn công bố điểm cấp ứng dụng, phải báo cả coverage và cách tính ảnh bị từ chối.

Đánh giá lại trọng số checkpoint hiện tại trên cả 180 ảnh test theo `scripts/train_cnn.py` đạt Accuracy 0,8333333 và Macro-F1 0,8303360; cả 180 nhãn trùng `cnn_metrics.json`. Điều này xác nhận điểm test của trọng số hiện tại, không xác nhận quá trình tạo lịch sử epoch.

## Giới hạn bằng chứng huấn luyện

Checksum checkpoint hiện tại khác checksum đã ghi trong lần kiểm tra trước (`500b63d5...` → `3c9ee1b5...`). `scripts/train_cnn.py` hiện ghi giá trị loss/F1 trực tiếp từ tính toán với độ chính xác float đầy đủ, trong khi mọi giá trị lịch sử của checkpoint hiện tại đã được làm tròn 4 chữ số thập phân. Điều này phù hợp với việc đóng gói metadata sau khi huấn luyện; nó **không chứng minh** lịch sử được tạo giả, nhưng cũng không xác thực được chuỗi 10 epoch. Chỉ số Val Macro-F1 hiện tại 0,834059957... của trọng số đã lưu vẫn là phép đo có thể tái lập.

Nhánh `--eval-only` giờ đọc được checkpoint đóng gói và giữ `history` hiện có, nhưng vẫn ghi đè `outputs/cnn_metrics.json` và `models/manifest_cnn.json`. Nó còn khởi tạo MobileNetV2 với `weights=DEFAULT` trước khi nạp checkpoint, nên có thể cần cache/tải trọng số ImageNet dù chỉ đánh giá lại. Nên lưu kết quả eval vào tệp riêng và khởi tạo `weights=None` ở nhánh này.

## Các vấn đề khác còn mở

- R6: Script đánh giá SVM dùng `predict()`, còn service chọn nhãn bằng `argmax(predict_proba())`; hai quy tắc khác 3/180 ảnh từ feature cache test hiện tại.
- R7: Tab đối chuẩn và README vẫn ghi cứng điểm số, kích thước SVM `~28 KB` (file thực tế 136.485 byte) và thời gian suy luận chưa có benchmark lưu lại.
- `ARCHITECTURE.md` vẫn mô tả nhiều module đã có là “cần bổ sung”; tài liệu chưa đồng bộ với source.

## Bằng chứng và lệnh chạy

```powershell
python -m compileall -q app.py src scripts/train_cnn.py scripts/check_fixes.py
python -m unittest discover -s tests -v
python scripts/check_ui_release.py
python scripts/check_batch_content_cache.py
python scripts/check_fixes.py
```

Biên dịch đạt, 4/4 unittest đạt. Kết quả runtime được lưu trong `outputs/check_2026-11-02/ui_check.json`, `batch_content_cache_check.json` và `fixes_check.json`. Script CNN ghi nhận `artifacts_unchanged: true` sau khi đối chiếu. Lệnh chạy xong của script audit chỉ có nghĩa kiểm tra hoàn tất; cần đọc nội dung JSON để biết mục nào đạt.
