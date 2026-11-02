# Đề xuất bài tập lớn Xử lý ảnh

## FreshLens — Trợ lý phân loại độ chín cà chua bằng ảnh

**Phiên bản đề xuất để nhóm chốt phạm vi và triển khai.** Sản phẩm hướng tới hộ trồng, điểm thu mua hoặc cửa hàng nông sản nhỏ cần phân loại nhanh cà chua theo độ chín. Người dùng chụp một quả trên nền chụp đơn giản hoặc tải nhiều ảnh lên; ứng dụng tách vùng quả, dự đoán mức độ chín, cho biết độ tin cậy và tổng hợp kết quả theo lô.

## 1. Vì sao chọn đề tài này

Mẫu tham khảo là một ứng dụng chỉnh sửa ảnh có giao diện trực quan và có giải thích cách cài đặt các thao tác như làm mờ, làm nét, lọc trung vị, cân bằng histogram, phân vùng K-Means và Haar Cascade. Đây là nền tảng tốt để học cách biến thuật toán thành phần mềm.

FreshLens nâng chất lượng đề tài ở **tính thống nhất và khả năng đánh giá**: mọi bước xử lý đều phục vụ một mục tiêu duy nhất — phân loại độ chín. Nhóm sẽ xây dựng bộ dữ liệu có nhãn, so sánh mô hình nền với CNN, đánh giá trên tập kiểm tra tách biệt và đưa kết quả vào một quy trình người dùng hoàn chỉnh. Thay vì trình diễn nhiều phép biến đổi không liên quan trong cùng ứng dụng, sản phẩm giải quyết một quyết định cụ thể.

Đề tài cũng vừa sức hơn một hệ thống nông nghiệp ngoài đồng: phiên bản chính xử lý **một quả mỗi ảnh** trong điều kiện chụp có kiểm soát; xử lý chùm quả trong video và đếm sản lượng được để dành làm hướng mở rộng.

## 2. Bài toán và người dùng

### Bài toán

Đánh giá độ chín bằng mắt thường có thể không nhất quán giữa người phân loại. FreshLens hỗ trợ sàng lọc dựa trên biểu hiện màu sắc nhìn thấy được trong ảnh.

### Người dùng mục tiêu

- Hộ trồng hoặc điểm thu mua quy mô nhỏ.
- Nhân viên phân loại, đóng gói nông sản.
- Sinh viên cần một sản phẩm trình diễn được kiến thức xử lý ảnh và học sâu.

### Đầu ra cho mỗi ảnh

- Nhãn: **xanh/chưa chín**, **đang chuyển màu**, hoặc **đỏ/chín**.
- Mức tin cậy của dự đoán.
- Ảnh vùng quả đã tách và phần màu đóng góp vào quyết định.
- Gợi ý kiểm tra thủ công nếu ảnh không đạt chuẩn hoặc độ tin cậy thấp.

### Đầu ra cho một lô ảnh

- Bảng kết quả có tên ảnh, lớp dự đoán, độ tin cậy và trạng thái kiểm tra.
- Biểu đồ số lượng và tỷ lệ theo ba lớp.
- Tải CSV kết quả và ZIP ảnh minh họa đã xử lý.

FreshLens là công cụ hỗ trợ phân loại bằng hình ảnh nhìn thấy được. Sản phẩm không đo độ ngọt, độ cứng, độ an toàn thực phẩm hay chất lượng bên trong quả.

## 3. Cách dùng dự kiến

1. Người dùng chọn chế độ **một ảnh** hoặc **một lô ảnh**.
2. Ứng dụng kiểm tra tệp, kích thước, độ sáng và độ nhòe; nếu ảnh khó đánh giá, ứng dụng giải thích vấn đề và gợi ý chụp lại.
3. Hệ thống tách vùng cà chua khỏi nền, làm sạch mặt nạ và chuẩn hóa kích thước ảnh.
4. Bộ phân loại dự đoán một trong ba mức chín.
5. Giao diện hiển thị ảnh gốc, mặt nạ phân vùng, nhãn, độ tin cậy và thông tin màu liên quan.
6. Người dùng xác nhận/chỉnh nhãn nếu cần; kết quả được tổng hợp và tải xuống.

### Điều kiện chụp cho phiên bản học kỳ

Để kết quả đo được có ý nghĩa, nhóm công bố rõ điều kiện sử dụng: một quả nằm trọn trong ảnh, nền xanh lam hoặc xanh lục lam mờ để tương phản với quả, ánh sáng tán xạ, tránh phản sáng mạnh và không che khuất phần lớn bề mặt. Giao diện cung cấp khung hướng dẫn chụp. Một tập ảnh chụp ngoài điều kiện chuẩn được giữ riêng để kiểm tra độ bền, không dùng để làm đẹp điểm số.

## 4. Kiến thức môn học được áp dụng

| Nội dung | Ứng dụng trong FreshLens | Vai trò |
|---|---|---|
| Biểu diễn ảnh, kênh màu, lấy mẫu | Kiểm tra ảnh đầu vào, đổi RGB/HSV/LAB | Chuẩn hóa dữ liệu và phân tích màu |
| Tăng cường ảnh, gamma, histogram | Hiệu chỉnh sáng mức nhẹ; CLAHE khi cần | Giảm ảnh hưởng của ảnh tối/thiếu tương phản |
| Lọc ảnh | Gaussian/median để giảm nhiễu trước phân vùng | Ổn định mặt nạ và đặc trưng |
| Phát hiện biên | Canny/Sobel để trực quan hóa đường bao | Hỗ trợ kiểm tra vùng quả, không làm đầu ra phân loại chính |
| Phân vùng ảnh | Ngưỡng màu trong HSV hoặc GrabCut khởi tạo từ vùng trung tâm | Tạo mặt nạ quả để loại nền khỏi đặc trưng |
| Hình thái học | Opening bỏ đốm nhỏ, closing lấp khe nhỏ, chọn thành phần liên thông lớn nhất | Làm sạch mặt nạ phân vùng |
| Trích chọn đặc trưng | Histogram HSV, thống kê màu, HOG/đặc trưng kết cấu đơn giản | Tạo mô hình đối chứng cổ điển |
| PCA (mở rộng có kiểm soát) | Giảm chiều và trực quan hóa đặc trưng | Phân tích dữ liệu, không bắt buộc trong đường chạy sản phẩm |
| Mạng nơ-ron/CNN | Phân loại ảnh vùng quả thành ba mức chín | Mô hình chính để so sánh với mô hình cổ điển |
| Đánh giá mô hình | Confusion matrix, precision, recall, macro-F1 | Xác định lớp nào dễ bị nhầm và mức hữu ích thực tế |

Chương phát hiện khuôn mặt không đưa vào sản phẩm vì không phục vụ bài toán phân loại nông sản. Việc nêu rõ nội dung không sử dụng giúp đề tài tập trung và cho thấy nhóm chọn kỹ thuật theo yêu cầu thực tế.

## 5. Thiết kế kỹ thuật đề xuất

```text
Ảnh đầu vào
   ↓
Kiểm tra chất lượng ảnh (độ phân giải, sáng, nhòe)
   ↓
Hiệu chỉnh nhẹ / lọc nhiễu
   ↓
Tách quả: ngưỡng màu nền hoặc GrabCut
   ↓
Opening + closing + chọn vùng liên thông
   ↓
Chuẩn hóa vùng quả và trích màu HSV
   ├── Baseline: đặc trưng màu + SVM / Random Forest
   └── Mô hình chính: CNN nhỏ hoặc transfer learning
   ↓
Nhãn + độ tin cậy + ảnh giải thích + báo cáo lô
```

### Hai nhánh mô hình để bài làm có chiều sâu

1. **Mô hình đối chứng cổ điển:** histogram màu HSV hoặc các thống kê màu trên vùng đã tách; huấn luyện SVM hoặc Random Forest. Nhánh này giúp giải thích tường minh vai trò của xử lý ảnh và đặc trưng.
2. **Mô hình CNN:** học đặc trưng không gian từ ảnh quả đã chuẩn hóa. Ưu tiên một CNN nhỏ hoặc mô hình học chuyển giao phù hợp với tài nguyên của nhóm; khóa quy trình huấn luyện và lưu phiên bản mô hình.

Không giả định CNN chắc chắn tốt hơn. So sánh trên cùng tập test; nếu dữ liệu nhỏ hoặc CNN quá khớp, mô hình cổ điển có thể là lựa chọn triển khai hợp lý hơn.

### Phân vùng có thể giải thích được

Vì phiên bản học kỳ yêu cầu nền chụp tương phản, có thể thử ngưỡng HSV để tách nền; nếu ảnh có bóng hoặc ngưỡng màu không ổn định, so sánh với GrabCut. Sau đó dùng opening/closing và thành phần liên thông để sửa mặt nạ. Lưu ảnh mặt nạ và chỉ số IoU trên một tập con được gán mask thủ công để chứng minh bước phân vùng hoạt động, thay vì chỉ chụp một ảnh kết quả đẹp.

## 6. Dữ liệu và gán nhãn

### Kế hoạch khả thi

- Thu thập khoảng **600–900 ảnh gốc**, mục tiêu cân bằng khoảng 200–300 ảnh mỗi lớp.
- Chụp ít nhất 100–150 quả khác nhau; mỗi quả có thể chụp nhiều góc nhưng tất cả ảnh của cùng một quả phải nằm trong cùng một split.
- Ghi lại mã quả, ngày chụp, thiết bị, điều kiện sáng và nhãn người gán.
- Hai thành viên gán nhãn độc lập theo quy tắc màu bề mặt; bất đồng được rà soát, không âm thầm chọn nhãn thuận tiện.
- Tách train/validation/test theo **quả vật lý hoặc phiên chụp**, ví dụ 70/15/15, trước khi augment. Không chia ngẫu nhiên từng ảnh gần trùng nhau giữa các tập.
- Chỉ augment trong tập train; không tính ảnh augment là ảnh độc lập mới.

### Quy tắc nhãn ban đầu

Ba mức **xanh/chưa chín – đang chuyển màu – đỏ/chín** được định nghĩa bằng quy ước màu bề mặt và hướng dẫn bằng ảnh ví dụ. Có thể khởi đầu bằng tỷ lệ vùng đỏ trong mặt nạ quả, nhưng cần hiệu chỉnh cùng người có kinh nghiệm trồng/thu mua cà chua; đây là quy ước của bài tập, không tự nhận là tiêu chuẩn nông nghiệp chính thức. Ghi chú riêng các trường hợp màu không đại diện, ví dụ giống cà chua màu vàng.

### Nghiên cứu liên quan

Các nghiên cứu gần đây về phân loại độ chín cà chua tổ chức nhãn theo các mức xanh, bán chín và chín, đồng thời xem xét ánh sáng thay đổi, che khuất và các điều kiện ảnh khác nhau. Điều này củng cố lựa chọn bài toán nhưng cũng cho thấy mô hình cần được đánh giá trong điều kiện thu nhận cụ thể, không chỉ trên ảnh đã chuẩn hóa đẹp. [Frontiers in Plant Science, nghiên cứu nhận diện độ chín cà chua trong nhà kính](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1853690/full).

## 7. Kế hoạch đánh giá

### Chỉ số bắt buộc

- **Macro-F1** làm chỉ số chính để không che khuất lớp ít mẫu.
- Accuracy, precision, recall theo từng lớp và confusion matrix.
- Thời gian suy luận trung vị trên máy demo.
- Tỷ lệ ảnh bị từ chối vì không đạt chất lượng đầu vào.
- IoU của mặt nạ phân vùng trên một tập con có mask tham chiếu.
- Đánh giá người dùng: thời gian thao tác/lô, mức dễ hiểu của giải thích, tỷ lệ đồng ý với gợi ý.

### Các phép thử cần có

1. So sánh đặc trưng màu + SVM/Random Forest với CNN.
2. So sánh ảnh gốc và ảnh đã hiệu chỉnh nhẹ.
3. Báo cáo kết quả tổng thể **và theo lớp**; phân tích các nhầm lẫn giữa xanh và đang chuyển màu.
4. Kiểm tra riêng ảnh từ phiên chụp hoặc thiết bị chưa xuất hiện ở train.
5. Hiển thị các lỗi điển hình: bóng đổ, phản sáng, nền không đúng, quả bị che, giống màu vàng.

Không đưa ra mục tiêu accuracy cao tùy tiện trước khi thu thập dữ liệu. Nhóm đặt ngưỡng chấp nhận sau khi xem baseline và trao đổi với giảng viên; kết quả trung thực kèm phân tích lỗi có giá trị hơn một con số đẹp do rò rỉ dữ liệu.

## 8. Thiết kế giao diện

### Màn hình 1 — Phân loại

Khung tải ảnh/chụp ảnh; trạng thái chất lượng; ảnh gốc và mặt nạ; nhãn, độ tin cậy; nút xác nhận hoặc sửa nhãn.

### Màn hình 2 — Xử lý lô

Bảng thumbnail, trạng thái, lớp, độ tin cậy, bộ lọc ảnh cần kiểm tra; biểu đồ thống kê và tải CSV/ZIP.

### Màn hình 3 — Thử nghiệm mô hình

Chọn mô hình đối chứng/CNN; nhập tập test; hiển thị confusion matrix, macro-F1, thời gian suy luận và ảnh lỗi. Đây là chế độ minh bạch cho báo cáo, không cần xuất hiện với người dùng vận hành thông thường.

### Hướng giao diện kỹ thuật

Streamlit là lựa chọn triển khai nhanh cho bản trình diễn: hỗ trợ tải ảnh, webcam, bảng và biểu đồ. Tách riêng `preprocessing`, `segmentation`, `features`, `inference` và `ui` để dễ kiểm tra từng thành phần. Nếu muốn giao diện desktop, có thể thay lớp UI mà giữ nguyên pipeline.

## 9. Phạm vi MVP và phần mở rộng

### MVP — bắt buộc hoàn thành

- Quy trình hướng dẫn chụp và nhập một ảnh.
- Kiểm tra cơ bản chất lượng ảnh.
- Phân vùng, hậu xử lý hình thái và hiển thị mặt nạ.
- Baseline đặc trưng màu + bộ phân loại cổ điển.
- CNN và so sánh định lượng nếu dữ liệu đủ.
- Nhập nhiều ảnh, bảng kết quả, biểu đồ và xuất CSV.
- Báo cáo dữ liệu, đánh giá, lỗi và giới hạn.

### Chỉ làm khi MVP ổn định

- Watershed/YOLO để tách nhiều quả chồng lấp trong một ảnh.
- Phân loại video hoặc theo dõi quả qua băng chuyền.
- Phát hiện đốm bệnh/hư hỏng như một nhãn phụ.
- Ứng dụng di động, API, cơ sở dữ liệu lịch sử.

Không đưa các phần mở rộng vào cam kết ban đầu; chúng làm tăng đáng kể gánh nặng dữ liệu và kiểm thử.

## 10. Phân công nhóm gợi ý

| Vai trò | Trách nhiệm |
|---|---|
| Dữ liệu và quy trình chụp | Quy tắc nhãn, chụp ảnh, kiểm tra cân bằng và chia tập |
| Xử lý ảnh | Hiệu chỉnh, phân vùng, morphology, chỉ số chất lượng ảnh |
| Mô hình | Baseline, CNN, huấn luyện và đánh giá |
| Sản phẩm và tích hợp | Giao diện, xử lý lô, xuất báo cáo, đóng gói demo |
| Báo cáo và kiểm định | Nhật ký thí nghiệm, phân tích lỗi, hình minh họa và kiểm tra kết quả |

Với nhóm nhỏ, có thể gộp vai trò; mỗi thành viên vẫn cần ghi lại phần chịu trách nhiệm và kết quả cụ thể.

## 11. Lộ trình 6 tuần

| Tuần | Kết quả cần có |
|---|---|
| 1 | Chốt quy tắc nhãn, thiết kế nền chụp, làm giao diện phác thảo |
| 2 | Thu thập một phần dữ liệu, viết công cụ xem/kiểm tra ảnh và phân vùng |
| 3 | Hoàn tất dữ liệu, rà nhãn, khóa train/validation/test theo quả/phiên chụp |
| 4 | Huấn luyện baseline và CNN; lập bảng so sánh đầu tiên |
| 5 | Tích hợp UI, nhập lô, độ tin cậy, xuất CSV và báo cáo lỗi |
| 6 | Chạy đánh giá cuối, sửa trải nghiệm, hoàn thiện tài liệu và diễn tập demo |

## 12. Bộ bàn giao cuối kỳ

1. Ứng dụng FreshLens chạy được từ máy nhóm.
2. Mã nguồn có cấu trúc, hướng dẫn cài đặt/chạy và phiên bản mô hình.
3. Bộ dữ liệu do nhóm thu thập hoặc nguồn dữ liệu được phép sử dụng, kèm quy tắc nhãn và cách chia tập.
4. Báo cáo kỹ thuật: bài toán, lý thuyết, thuật toán, dữ liệu, thí nghiệm, kết quả, phân tích lỗi và giới hạn.
5. Video demo ngắn với cả trường hợp dự đoán đúng, dự đoán sai và ảnh đầu vào kém.
6. Slide bảo vệ: vấn đề → sản phẩm → pipeline → dữ liệu → so sánh → demo → kết luận.

## 13. Điểm nhấn khi bảo vệ

- **Một sản phẩm, một quyết định rõ:** phân loại mức độ chín từ ảnh.
- **Thể hiện kiến thức xử lý ảnh thực chất:** segmentation và morphology tạo ROI trước khi phân loại.
- **Có đối chứng khoa học:** đặc trưng màu + classifier cổ điển so với CNN.
- **Có quản lý rủi ro:** ảnh kém chất lượng hoặc dự đoán thiếu tin cậy được chuyển cho người xem lại.
- **Có đánh giá không rò rỉ:** ảnh cùng quả/phiên không bị chia sang cả train và test.
- **Có giới hạn trung thực:** mô hình chỉ được khuyến nghị trong điều kiện dữ liệu và chụp đã đánh giá.

## 14. Tài liệu nền tảng

- Slide Xử lý ảnh số, Chương 3: tăng cường ảnh.
- Slide Xử lý ảnh số, Chương 4: phát hiện đường và biên ảnh.
- Slide Xử lý ảnh số, Chương 5: phân vùng ảnh.
- Slide Xử lý ảnh số, Chương 6: xử lý hình thái.
- Slide Xử lý ảnh số, Chương 7: trích chọn đặc trưng và PCA.
- Slide Xử lý ảnh số, Chương 8: mạng nơ-ron/CNN.
- OpenCV, tài liệu hàm GrabCut: https://docs.opencv.org/doc/doxygen/html/d3/d47/group__imgproc__segmentation.html
- Nghiên cứu liên quan về phân loại độ chín cà chua trong môi trường nhà kính: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1853690/full
