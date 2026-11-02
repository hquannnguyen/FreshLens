# Cấu trúc dữ liệu

Đặt ảnh theo lớp và theo split. Mỗi lớp là một trong `green`, `turning`, `ripe`.

```text
data/images/
  train/green/
  train/turning/
  train/ripe/
  val/green/
  val/turning/
  val/ripe/
  test/green/
  test/turning/
  test/ripe/
```

Chia tập theo quả vật lý hoặc phiên chụp trước khi augment. Tất cả ảnh của cùng một quả phải nằm trong một split. Dùng ảnh gốc trong thư mục này; không chép ảnh augment vào validation/test.

## Gợi ý bảng kê nhãn

`annotations.csv` nên có các cột: `filename,class,fruit_id,session_id,device,lighting,annotator,note`. Lưu bảng kê riêng ở thư mục `data/`.

## Quy tắc gán nhãn

- `green`: quả xanh, chưa có vùng đỏ đáng kể.
- `turning`: quả có cả vùng xanh/vàng và vùng đỏ/cam rõ rệt.
- `ripe`: phần lớn bề mặt có màu đỏ/cam của giống được thu thập.

Đây là quy ước nhãn của dự án học tập. Thống nhất ảnh ví dụ với người gán nhãn trước khi chụp số lượng lớn; ghi riêng giống không phù hợp với ba lớp này.
