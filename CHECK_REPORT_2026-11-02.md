# Kiểm tra độc lập FreshLens ngày 02/11/2026

**Lưu ý:** Đây là trạng thái trước các bản sửa tiếp theo trong cùng ngày. Xem [kiểm tra sau bản sửa](CHECK_FOLLOWUP_2026-11-02.md) để biết hiện trạng.

## Kết luận kiểm tra

Checkpoint MobileNetV2 có thể nạp và tái lập đúng kết quả đã công bố trên ảnh đầu vào của script đánh giá. Tuy nhiên, phần tích hợp và tái lập thí nghiệm còn các lỗi dưới đây. Không nên coi T08/T06 đã hoàn tất kiểm định phát hành chỉ dựa vào bảng điểm offline.

Lần kiểm tra này không huấn luyện lại, không chọn lại checkpoint và không ghi đè các model/báo cáo thực nghiệm hiện có. Chỉ thêm các chương trình kiểm tra, bằng chứng trong `outputs/check_2026-11-02/` và tài liệu bàn giao.

## 1. Kết quả đã xác minh

| Kiểm tra | Kết quả |
|---|---|
| Cấu trúc dataset | 540 train, 180 val, 180 test; mỗi split cân bằng ba lớp; validator hiện tại báo hợp lệ |
| Checkpoint CNN | `models/freshlens_cnn.pt` có thể nạp; kích thước thực tế 9.144.203 byte |
| CNN trên 180 ảnh val, theo pipeline của script train | Accuracy 83,3333%; Macro-F1 83,4060%; 180 nhãn dự đoán trùng báo cáo |
| CNN trên 180 ảnh test, theo pipeline của script train | Accuracy 83,3333%; Macro-F1 83,0336%; 180 nhãn dự đoán trùng báo cáo |
| CNN qua service của ứng dụng, 180 ảnh test | 179 ảnh có nhãn, 1 ảnh rejected; 151/180 đúng khi tính rejected là không phân loại đúng (83,8889%) |
| CNN trên riêng 179 ảnh service chấp nhận | Accuracy 84,3575%; Macro-F1 84,2545%; đây là chỉ số có điều kiện, không so sánh trực tiếp với điểm trên toàn bộ 180 ảnh |
| Khác biệt đầu vào raw và service | 12/179 ảnh có nhãn đổi dự đoán; `green_test_059.jpg` bị service từ chối |
| Cờ rà soát confidence thấp | Không phát hiện trường hợp confidence < 0,65 mà không mang trạng thái needs_review |
| SVM trên feature cache val/test hiện có | Tái lập đúng điểm từ `SVC.predict()` trong báo cáo baseline |
| Trang Streamlit khởi động | AppTest không ghi nhận exception ở lần khởi động mặc định |
| Hợp đồng/phân vùng/xuất kết quả | 4 kiểm tra unittest đạt |
| Chọn checkpoint theo Validation Macro-F1 | Logic chọn tốt nhất có trong source; chưa đủ bằng chứng để xác nhận lịch sử epoch 10 của artifact hiện tại |

Thứ tự hàng/cột của confusion matrix là `green`, `ripe`, `turning`. CNN test tái lập được:

```text
[[58,  0,  2],
 [ 2, 50,  8],
 [ 5, 13, 42]]
```

Lớp `turning`: 42/60 đúng, 13 bị dự đoán thành `ripe`, 5 thành `green`; Recall 70%. Các giá trị này đúng cho pipeline ảnh gốc của script đánh giá.

Các checksum của checkpoint CNN, manifest CNN, metrics CNN và checkpoint SVM trước/sau kiểm tra không thay đổi. Kết quả service sử dụng RNG OpenCV seed 42; việc điểm service lần này cao hơn nhẹ không làm hai pipeline trở thành tương đương và không phải cơ sở chọn lại mô hình trên test.

## 2. Các lỗi cần xử lý

### R1 — P1: Tiền xử lý CNN của ứng dụng khác tiền xử lý train/evaluation

**Vị trí:** `src/services/analysis_service.py:182`, đối chiếu `scripts/train_cnn.py:51` và `scripts/train_cnn.py:70`.

Script huấn luyện và đánh giá đọc trực tiếp ảnh JPEG từ dataset, resize 224×224 rồi normalize. Trong khi đó service chạy GrabCut, có thể loại ảnh theo tỷ lệ mask, rồi đưa `crop_to_mask()` vào CNN. Như vậy cùng một ảnh test có thể bị cắt thành một đầu vào khác với đầu vào đã tạo ra điểm 83,03%. Mask lớn nhất còn có thể chọn vùng khác khi ảnh có nhiều vật thể.

Đây là lỗi tương thích giữa model và đường chạy sản phẩm. Cần thống nhất một hợp đồng tiền xử lý: giữ nguyên ảnh đầu vào cho model v1 hiện có, hoặc phát triển phiên bản model mới được train/validation trên cùng phép crop của service. Đánh giá pipeline đầy đủ phải tính cả ảnh bị từ chối và công bố độ bao phủ; không chỉ báo accuracy trên phần còn lại.

**Bằng chứng:** `outputs/check_2026-11-02/checkpoint_check.json` chứa điểm raw, điểm qua service, nhãn từng ảnh, số ảnh bị từ chối và số dự đoán thay đổi. Điểm qua service là kiểm tra triển khai, không được dùng để chọn lại checkpoint trên test.

### R2 — P1: Cache batch chỉ dùng số ảnh, giữ kết quả model cũ

**Vị trí:** `app.py:324–359`, trọng tâm `batch_files_len` tại dòng 325.

Đã tái hiện bằng AppTest: xử lý lô 6 ảnh bằng SVM, chuyển sang CNN, bấm nút xử lý lại nhưng kết quả vẫn có `model_id=freshlens_svm_baseline_v1`. Điều kiện cache chỉ so sánh số ảnh nên không nhận biết nội dung ảnh, loại model, phiên bản checkpoint hoặc segmentation đã đổi. Thay một lô khác có cùng số ảnh cũng có nguy cơ hiển thị kết quả cũ.

Một lỗi liên quan: nút thử nhanh xóa `batch_results` nhưng giữ `batch_files_len`; chọn lại lô 6 ảnh rồi xử lý cho bảng rỗng (0 kết quả).

**Cần sửa:** dùng khóa gồm hash nội dung ảnh, cấu hình đầy đủ và phiên bản artifact; trạng thái xử lý phải gắn với khóa đó. Nút chạy cần tạo kết quả cho yêu cầu mới; nút download chỉ sử dụng kết quả tương ứng đã hoàn thành.

### R3 — P2: Một ảnh hỏng làm batch lỗi vì thiếu import

**Vị trí:** `app.py:349`; phần import `app.py:13` chỉ có `AnalysisRequest, ProcessingConfig`.

Nhánh bắt lỗi tạo `AnalysisResult(...)` nhưng tên này chưa được import. AppTest với một tệp `broken.jpg` không đọc được đã thu được exception `name 'AnalysisResult' is not defined`. Vì vậy một tệp hỏng dừng cả trang thay vì ghi một dòng failed và tiếp tục lô.

**Cần sửa:** import đúng hợp đồng, khởi tạo filename riêng trước khi decode từng ảnh và kiểm tra lô trộn ảnh hợp lệ/ảnh hỏng.

### R4 — P2: `--eval-only` xóa bằng chứng huấn luyện và gán cứng best_epoch

**Vị trí:** `scripts/train_cnn.py:164` và `scripts/train_cnn.py:187`.

Nhánh đánh giá lại gán `best_epoch=10` cho mọi checkpoint và ghi đè `cnn_metrics.json` bằng đối tượng chỉ có `model_id`, `val`, `test`. Nó loại bỏ `history` và `best_epoch` của báo cáo huấn luyện. Artifact hiện tại đúng là chỉ còn ba khóa này; checkpoint là state_dict không chứa lịch sử chọn epoch.

Vì vậy kiểm tra này xác minh được điểm của checkpoint hiện tại nhưng không thể xác minh tuyên bố “epoch 10 là tốt nhất” từ artifact đang có. Không được tái tạo lịch sử bằng số đoán.

**Cần sửa:** lưu training history và metadata đi cùng checkpoint, đọc best_epoch từ artifact; ghi báo cáo đánh giá lại ra tệp riêng. Khởi tạo model với `weights=None` trong eval-only để không cần tải ImageNet trước khi nạp state_dict đã có.

### R5 — P2: Dependencies chưa đủ để người mới chạy CNN

**Vị trí:** `requirements.txt`; đối chiếu import `torch`/`torchvision` trong `scripts/train_cnn.py:11–15`.

Requirements hiện chỉ khai báo các dependency cho SVM/UI, thiếu `torch` và `torchvision`. Máy kiểm tra đã có sẵn chúng nên inference chạy được; điều này không chứng minh lệnh cài đặt trên môi trường ảo mới trong README sẽ chạy được CNN.

**Cần sửa:** khai báo bộ dependency CNN có phiên bản tương thích, kèm hướng dẫn bản CPU/GPU và môi trường đã kiểm chứng. Không cần thêm pretrained download cho đường chạy suy luận.

### R6 — P2: SVM trong bảng đối chuẩn và SVM trong ứng dụng dùng quy tắc trả nhãn khác nhau

**Vị trí:** `scripts/train_baseline.py:154`, đối chiếu `src/modeling.py:68` (hàm `predict_from_features`).

Script đánh giá baseline dùng `model.predict()`, còn adapter UI dùng argmax của `predict_proba()`. Trên cache test hiện có, hai cách trả nhãn khác nhau ở 3/180 ảnh: cách của báo cáo đạt Accuracy 86,6667%, Macro-F1 86,5839%; cách của adapter đạt Accuracy 87,2222%, Macro-F1 87,0523%. Trên val cũng có 2 nhãn khác nhau.

Không phải quy tắc nào tự động đúng hơn, nhưng pipeline được công bố cần là pipeline thực sự được đánh giá. Cần dùng cùng adapter/quy tắc trong script và service, ghi phiên bản quy tắc; confidence phải đúng lớp được trả về. Kết quả này dùng feature cache sẵn có, chưa phải phép đo lại segmentation SVM trên toàn bộ test.

### R7 — P2: Bảng đối chuẩn ghi cứng và chứa số liệu chưa có căn cứ đo

**Vị trí:** `app.py:419–497`, `README.md` và `HANDOFF.md`.

Accuracy, Macro-F1, F1 từng lớp và confusion matrix trong tab 3 được ghi trực tiếp trong source thay vì đọc từ báo cáo. Các giá trị chính trùng snapshot hiện tại nhưng sẽ sai khi model/báo cáo thay đổi. Kích thước SVM được ghi ~28 KB trong khi file thực tế là 136.485 byte (~133,3 KiB). Chưa có kết quả benchmark lưu lại để chứng minh các mốc ~8 ms/~25 ms hoặc so sánh độ trễ ở cùng phạm vi tính toán.

**Cần sửa:** đọc số liệu từ artifact có ID/checksum khớp model, lấy kích thước bằng filesystem; đo thời gian có nêu phần bao gồm preprocessing hay chỉ classifier. Các giải thích về overfitting và khả năng tổng quát tốt hơn của CNN chỉ nên ghi là giả thuyết khi chưa có thí nghiệm đối chứng.

## 3. Giới hạn bằng chứng và tài liệu

- Validator hiện xác nhận cấu trúc và một số rò rỉ từ manifest; các nhóm gọi `session_id` ở dataset đang tương ứng ảnh gốc (83 train, 20 val, 28 test). Không coi đây là bằng chứng tuyệt đối rằng quả vật lý không xuất hiện trong những ảnh gốc khác nhau.
- Source có hai giai đoạn huấn luyện 3 + 7 epochs và mở tham số block 14–18. Tuy nhiên `model.train()` vẫn bật cập nhật running statistics của BatchNorm ở các block đã khóa gradient. Nếu mục tiêu là đóng băng toàn bộ backbone, cần kiểm soát cả chế độ BatchNorm và ghi rõ chiến lược.
- `ARCHITECTURE.md` vẫn ghi mốc 31/10/2026 và đánh dấu một số module đã tồn tại là “CẦN BỔ SUNG”. Tài liệu chưa phản ánh đầy đủ trạng thái CNN như mô tả bàn giao.
- Chưa có mask ground truth để xác nhận IoU/Dice; không đánh dấu toàn bộ T07 segmentation là hoàn tất.

## 4. Lệnh và bằng chứng tái lập

Chạy từ thư mục `FreshLens`:

```powershell
python scripts/validate_dataset.py
python -m unittest discover -s tests -v
python scripts/check_ui_release.py
python scripts/check_release.py
```

- `tests/test_release_contracts.py`: 4/4 đạt, dùng fixture kỹ thuật; không sinh metric phân loại từ ảnh giả.
- `outputs/check_2026-11-02/ui_check.json`: thao tác chuyển model, lô cùng số ảnh và ảnh hỏng.
- `outputs/check_2026-11-02/checkpoint_check.json`: tái lập CNN trên val/test, kiểm tra đường service và checksum artifact.
- `outputs/check_2026-11-02/supporting_check.json`: thống kê dataset, kích thước file và đối chiếu quy tắc nhãn SVM từ cache.

Các script `check_*` tạo báo cáo kiểm tra, không huấn luyện hoặc sửa artifact. Cần đọc các cờ kết quả trong JSON; chạy xong script không đồng nghĩa mọi kiểm tra đều đạt.

Môi trường kiểm tra: Python trên máy Windows hiện tại, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Streamlit 1.66.1, scikit-learn 1.9.1. Kiểm tra CNN offline dùng 4 CPU threads; kiểm tra UI dùng 2 CPU threads. Thời gian trong báo cáo là phép đo tại máy này, không phải cam kết triển khai.

## 5. Thứ tự xử lý tiếp theo

1. R1: chốt hợp đồng đầu vào của CNN và đánh giá đúng pipeline.
2. R2–R3: sửa cache batch và đường xử lý tệp lỗi; chạy lại AppTest.
3. R4–R5: bảo toàn provenance/checkpoint và hoàn thiện cài đặt môi trường.
4. R6–R7: đồng bộ benchmark với adapter thực tế, bỏ số liệu ghi cứng và cập nhật tài liệu.

Các thay đổi nghiệp vụ ở trên chưa được sửa trong đợt kiểm tra này; báo cáo giúp người/agent tiếp theo tiếp nhận rõ vấn đề và bằng chứng.
