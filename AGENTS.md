# Hướng dẫn tiếp tục phát triển FreshLens

Áp dụng cho các file trong thư mục FreshLens. Người dùng có thể điều chỉnh phạm vi hoặc quyết định kiến trúc trong từng yêu cầu.

## Đọc trước khi sửa

- `README.md`: chạy project.
- `ARCHITECTURE.md`: thiết kế đích, hợp đồng và phần chưa triển khai.
- `HANDOFF.md`: trạng thái, lỗi đã biết, task có thể nhận tiếp.
- `PLAN.md`: mốc triển khai và đầu ra BTL.

Luôn đối chiếu với source thực tế. Trong tài liệu, “đã viết” không đồng nghĩa “đã kiểm tra runtime” hoặc “đã huấn luyện”.

## Phạm vi sản phẩm

- Một quả cà chua mỗi ảnh; batch gồm nhiều ảnh độc lập.
- Nhãn ổn định: `green`, `turning`, `ripe`.
- Preview theo màu phải được ghi rõ; không tạo confidence ML hoặc metrics giả.
- Baseline chính: HSV histogram 50 chiều + StandardScaler + SVM. CNN là phần mở rộng sau khi baseline/dữ liệu ổn định.

## Quy tắc module

- UI gọi service; module thuật toán không import Streamlit hoặc `app.py`.
- BGR uint8 là định dạng nội bộ; mask uint8 0/255 phải khớp ảnh làm việc.
- Chuẩn hóa/phân vùng một lần và dùng cùng mask cho hiển thị, đặc trưng và suy luận.
- Giữ pipeline train/inference thống nhất; thay đặc trưng hoặc segmentation phải cập nhật cấu hình/phiên bản model.
- Phân biệt lỗi đọc ảnh, ảnh cần rà soát và mask không hợp lệ; không biến lỗi thành nhãn mặc định.
- Model chỉ được nạp từ artifact nhóm kiểm soát; không thêm chức năng upload joblib từ người dùng.

## Dữ liệu và mô hình

- Không tự tạo dữ liệu huấn luyện hoặc metrics để lấp phần còn thiếu.
- Không chia ảnh cùng quả sang nhiều split. Chính sách session split và ảnh trùng phải được ghi nhận.
- Chọn tham số bằng validation; chỉ chạy test cuối theo bước đánh giá đã định.
- Không ghi đè dữ liệu gốc hoặc model đang dùng khi chưa tạo artifact ứng viên và ghi rõ kết quả.

## Khi bàn giao

- Kiểm tra đúng phần đã đổi; ghi chính xác lệnh và kết quả, phân biệt kiểm tra kỹ thuật với đánh giá model.
- Khi đổi lệnh chạy/cấu trúc, cập nhật README. Khi đổi hợp đồng, cập nhật ARCHITECTURE.
- Cập nhật HANDOFF: task hoàn thành, trạng thái runtime/dataset/model, lỗi còn lại và việc tiếp theo.
- Với thay đổi chỉ là tài liệu, kiểm tra đường dẫn và tính nhất quán; không cần huấn luyện hoặc chạy test model.
