# Kế hoạch triển khai FreshLens

Đọc [ARCHITECTURE.md](ARCHITECTURE.md) để triển khai theo cùng thiết kế. Trạng thái thực tế, lỗi đã biết và các đầu việc ưu tiên nằm trong [HANDOFF.md](HANDOFF.md); các giai đoạn dưới đây là kế hoạch, không phải xác nhận đã hoàn thành.

## Mục tiêu

Hoàn thành một ứng dụng hỗ trợ phân loại cà chua thành ba mức độ chín từ ảnh. Sản phẩm phải cho thấy pipeline xử lý ảnh, có baseline học máy chạy được, hỗ trợ xử lý lô, báo cáo kết quả và nêu giới hạn rõ ràng.

## Giai đoạn và tiêu chí hoàn thành

| Giai đoạn | Công việc | Kết quả bàn giao | Điều kiện hoàn thành |
|---|---|---|---|
| 0. Chốt thiết kế | Quy tắc ba nhãn, nền chụp, định dạng dữ liệu, vai trò nhóm | Quy ước gán nhãn và lịch thu thập | Hai người có thể gán cùng ảnh nhất quán; giảng viên duyệt phạm vi |
| 1. Dữ liệu | Thu thập, ghi fruit/session ID, rà lỗi và nhãn | Tập train/val/test và CSV kê ảnh | Có ảnh thật ở cả ba lớp; split theo quả/phiên; không trùng ảnh gần giống giữa các split |
| 2. Xử lý ảnh | Kiểm tra chất lượng, phân vùng, morphology, crop/mask | Pipeline và bộ ảnh minh họa trung gian | Mask được lưu để kiểm tra; lỗi nền/nhòe được ghi nhận |
| 3. Baseline | HSV histogram + SVM, validation và test | Mô hình lưu tại `models/`, JSON metrics | Có confusion matrix và precision/recall/F1 từng lớp; dùng test một lần sau khi khóa mô hình |
| 4. Ứng dụng | Ảnh đơn, ảnh lô, review flag, CSV, ZIP | Streamlit app | Người mới chạy theo README và hoàn tất luồng từ upload tới tải kết quả |
| 5. CNN & nghiên cứu | Huấn luyện CNN nhỏ hoặc transfer learning; so sánh công bằng | Bảng so sánh baseline/CNN, nếu đủ dữ liệu | Cùng split, cùng quy trình đánh giá; có phân tích lỗi chứ không chỉ báo accuracy |
| 6. Hoàn thiện | Đóng gói, demo, báo cáo và slide | Bộ nộp cuối kỳ | Demo chạy được; mô tả chính xác dữ liệu, điều kiện chụp và giới hạn |

## Lộ trình 6 tuần

| Tuần | Trọng tâm | Cột mốc |
|---|---|---|
| 1 | Nhãn, thiết kế chụp và dựng app | Chụp thử 10–20 quả; chốt hướng dẫn chụp |
| 2 | Thu dữ liệu và pipeline phân vùng | Bản xem mask + báo cáo ảnh lỗi |
| 3 | Hoàn thiện dataset, khóa split | CSV nhãn và thống kê cân bằng lớp |
| 4 | Baseline SVM và đánh giá | Metrics validation/test và lỗi điển hình |
| 5 | Batch UI và tích hợp | App xử lý nhiều ảnh, CSV, ZIP |
| 6 | CNN nếu đủ dữ liệu, kiểm định cuối, tài liệu | Bản demo và báo cáo hoàn chỉnh |

## Phân công nhóm

- **Dữ liệu:** thống nhất nhãn, tổ chức chụp, kiểm tra metadata, tách split.
- **Xử lý ảnh:** chất lượng ảnh, segmentation, morphology và minh họa.
- **Mô hình:** baseline, CNN (nếu đủ), đánh giá và phân tích lỗi.
- **Sản phẩm:** Streamlit, xử lý lô, xuất CSV/ZIP và hướng dẫn chạy.
- **Tích hợp/báo cáo:** kết nối pipeline, kiểm tra lại kết quả, viết báo cáo và diễn tập.

Nhóm ít người có thể gộp vai trò; mỗi thành viên cần chịu trách nhiệm rõ cho ít nhất một đầu ra.

## Quyết định kỹ thuật hiện tại

- **UI:** Streamlit, để demo nhanh trên máy cá nhân.
- **Xử lý ảnh:** OpenCV, mask HSV cho phông xanh; GrabCut là phương án thay thế.
- **Baseline:** histogram HSV trên ROI + StandardScaler + SVM.
- **Phân loại:** `green`, `turning`, `ripe`.
- **Độ tin cậy:** xác suất từ SVM chỉ dùng làm tín hiệu rà soát; ngưỡng 0.65 là mặc định ban đầu và phải hiệu chỉnh trên validation.
- **CNN:** một nhánh so sánh tùy tài nguyên và số mẫu; không chặn MVP.

## Rủi ro và cách giảm

| Rủi ro | Tác động | Cách giảm |
|---|---|---|
| Khó gán nhãn ranh giới lớp | Nhãn nhiễu, mô hình nhầm | Ảnh ví dụ chuẩn, hai người gán độc lập, họp xử lý bất đồng |
| Ít quả khác nhau | Rò rỉ giữa train/test và quá khớp | Chia theo fruit ID/session ID; ưu tiên nhiều quả thật hơn nhiều góc cùng một quả |
| Ánh sáng và giống cà chua khác nhau | Mô hình học điều kiện chụp thay vì độ chín | Dùng ánh sáng tán xạ, ghi metadata, có test ngoài điều kiện chuẩn |
| Segmentation sai | Đặc trưng màu bị lẫn nền | Nền tương phản, hiện mask, bật GrabCut, đưa mask lỗi vào phân tích |
| CNN tốn thời gian hoặc kém baseline | Trễ tiến độ | MVP chốt baseline trước; CNN là nhánh sau, có thể bỏ nếu không có lợi |
| Điểm số cao do ảnh tương tự | Báo cáo sai thực tế | Chia theo từng quả/phiên trước augment; lưu file split và seed |

## Cổng quyết định phạm vi

1. **Sau tuần 1:** nếu chưa có quy tắc nhãn đồng thuận, chưa chụp quy mô lớn.
2. **Sau tuần 3:** nếu số ảnh/đa dạng còn thấp, ưu tiên hoàn tất baseline và giới hạn kết luận; không tuyên bố mô hình tổng quát.
3. **Sau tuần 4:** chỉ tiếp tục CNN nếu baseline và dữ liệu đã ổn định.
4. **Trước nộp:** không thay đổi test set để chọn mô hình; ghi rõ mọi điều chỉnh đã xem test nếu có.

## Bộ bàn giao

- Mã nguồn và lệnh cài đặt/chạy.
- Dữ liệu được phép chia sẻ cùng `annotations.csv` hoặc hướng dẫn tái tạo nếu quyền dữ liệu hạn chế.
- Mô hình và `baseline_metrics.json`.
- Báo cáo thuật toán, dữ liệu, thí nghiệm, kết quả, lỗi và giới hạn.
- Slide và video demo có ít nhất một ca đúng, một ca sai, một ca cần xem lại.
