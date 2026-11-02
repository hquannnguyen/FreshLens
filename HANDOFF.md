# Bàn giao phát triển FreshLens

Cập nhật: 02/11/2026. Dành cho người hoặc agent mới tiếp tục BTL. Đây là ảnh chụp trạng thái tại thời điểm bàn giao; cập nhật khi hoàn thành hoặc thay đổi một đầu việc.

**Kiểm tra độc lập mới nhất 02/11/2026:** Đã khắc phục và xác minh toàn bộ 5 điểm từ báo cáo [CHECK_RELEASE_2026-11-02.md](CHECK_RELEASE_2026-11-02.md): benchmark đã tái lập thành công trên đủ 180 ảnh test, độ trễ toàn trình ghi nhận trung thực kèm GrabCut (~1.819 ms SVM / ~1.851 ms CNN), kích thước SVM khớp chính xác artifact hiện hành (148.693 byte ~ 145.21 KB), giao diện đọc số liệu tự động qua JSON metrics không còn ghi số cứng, và `--eval-only` khởi tạo với `weights=None`. Mọi kiểm tra kỹ thuật đạt 100% không có regression.

## 1. Đọc theo thứ tự

1. `README.md`: mục đích, cài đặt và lệnh chạy.
2. `ARCHITECTURE.md`: thành phần, luồng dữ liệu, hợp đồng và kiến trúc đích.
3. Tài liệu này: trạng thái thật, kết quả thực nghiệm và công việc tiếp theo.
4. `PLAN.md`: mốc thời gian, vai trò và bộ bàn giao.
5. `DE_XUAT_BAI_TAP_LON.md`: bối cảnh đề tài và tiêu chí nghiên cứu.
6. Mã nguồn theo đường đi `app.py → src/services/analysis_service.py → src/image_pipeline.py / src/quality.py / src/features.py → src/modeling.py`.

## 2. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| UI ảnh đơn, nhiều ảnh, tab thực nghiệm | Hoàn thành; chuyển đổi linh hoạt giữa Baseline SVM và Deep Learning CNN |
| HSV nền xanh, GrabCut tăng tốc, morphology, chọn vùng lớn nhất | Hoàn thành trong `src/image_pipeline.py`; GrabCut downscale/upscale tăng tốc ~10x |
| Kiểm tra sáng/nét/kích thước và xoay EXIF | Hoàn thành trong `src/quality.py` và `app.py` |
| HSV histogram 50 chiều | Hoàn thành trong `src/features.py` (18 H + 16 S + 16 V) |
| Hợp đồng dữ liệu có kiểu (`src/contracts.py`) | Hoàn thành; đầy đủ dataclass cho request, config, quality, segmentation, prediction, analysis |
| Analysis Service & Exporter Service | Hoàn thành trong `src/services/`; chuẩn hóa 1 lần, phân vùng 1 lần, dùng cùng mặt nạ |
| Xử lý ảnh bị từ chối (`rejected`) | Hoàn thành; mask rỗng/ngoài 5-95% bị từ chối, nhãn rỗng, không sinh turning giả |
| Đồng bộ nhãn và độ tin cậy | Hoàn thành; nhãn và confidence lấy từ cùng vị trí max trong `predict_proba` / softmax |
| Dataset chuẩn Group-Split 900 ảnh thật ($\ge 640\text{px}$) | Hoàn thành; 300 xanh, 300 chuyển màu, 300 chín; 0 rò rỉ quả/phiên |
| Huấn luyện mô hình Baseline SVM | Hoàn thành; $C=0.5$ trên val, lưu model và `manifest.json`. Test Macro-F1: **84.9%** (Accuracy 85.0%) |
| Huấn luyện Deep Learning (MobileNetV2 CNN) | Hoàn thành; lưu trọng số `models/freshlens_cnn.pt` và `manifest_cnn.json`. Test Macro-F1: **83.0%** |
| Bảng so sánh đối chuẩn SVM vs CNN | Hoàn thành trong `outputs/cnn_metrics.json` và Tab 3 giao diện Streamlit |
| Batch không bị rerun khi tương tác Streamlit | Hoàn thành; lưu kết quả batch trong `session_state` |
| Webcam trực tiếp | Đã tích hợp widget `st.camera_input` trong Tab ảnh đơn |
| Video thời gian thực, nhiều quả/ảnh | Chưa triển khai (phạm vi mở rộng tương lai) |

## 3. Khởi động

Từ PowerShell tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Ứng dụng cho phép lựa chọn linh hoạt mô hình suy luận tại thanh Sidebar:
1. **Baseline SVM (HSV Histogram 50 chiều + SVM RBF)**
2. **Deep Learning CNN (MobileNetV2 Transfer Learning)**
3. **Quy tắc màu xem trước (Color Preview)**

Các script huấn luyện và đánh giá:
```powershell
python scripts\validate_dataset.py
python scripts\train_baseline.py --segmentation grabcut
python scripts\train_cnn.py --eval-only
```

## 4. Báo cáo các Task đã hoàn thành

### Task T01 & T02: Hợp đồng dữ liệu & Analysis Service (Hoàn thành)
- Thêm `src/contracts.py`: Dataclass chuẩn hóa (`ProcessingConfig`, `QualityReport`, `SegmentationResult`, `PredictionResult`, `AnalysisRequest`, `AnalysisResult`).
- Thêm `src/services/analysis_service.py`: Chuẩn hóa ảnh một lần, phân vùng một lần duy nhất, cùng một mặt nạ dùng cho hiển thị quả đã tách, trích xuất đặc trưng và suy luận mô hình.
- Phân biệt rõ ràng trạng thái: `completed` (đạt), `needs_review` (cần rà soát), `rejected` (từ chối do mask hỏng), `failed` (lỗi đọc/định dạng tệp).
- Khắc phục lỗi mask rỗng: trả về `rejected` và nhãn rỗng, không gán nhãn `turning` giả.

### Task T03 & T05: Model Manifest & Phát hành Baseline (Hoàn thành)
- Thêm `models/manifest.json` lưu định danh model `freshlens_svm_baseline_v1`, kiến trúc SVM RBF, phiên bản đặc trưng `hsv_hist_18_16_16_v1`, số chiều 50, phân đoạn `grabcut`, siêu tham số tối ưu $C=0.5$.
- Đồng bộ nhãn trả về và điểm tin cậy: lấy cùng chỉ số có xác suất cao nhất từ `predict_proba`.

### Task T04: Thu thập và kiểm định bộ dữ liệu 900 ảnh chuẩn Group-Split (Hoàn thành)
- Thu thập 900 ảnh thật chất lượng cao ($\ge 640\text{px}$) từ Laboro Tomato benchmark: 300 xanh, 300 chuyển màu, 300 chín.
- Phân chia độc lập theo phiên chụp (Group-Split): Train (540 ảnh từ 83 phiên), Val (180 ảnh từ 20 phiên), Test (180 ảnh từ 28 phiên).
- Tạo `data/annotations.csv` đầy đủ metadata, hash SHA-256. Chạy `scripts/validate_dataset.py` đạt 0 lỗi, 0 rò rỉ quả/phiên.

### Task T06: Dịch vụ xuất kết quả & Quản lý Session State (Hoàn thành)
- Thêm `src/services/export_service.py`: Xuất CSV (UTF-8 with BOM) và gói tệp ZIP chứa ảnh quả tách nền, ảnh crop ROI và ảnh mask với tiền tố ID chống trùng tên.
- `app.py`: Lưu trữ danh sách kết quả batch trong Streamlit `session_state`, việc tải tệp hoặc tương tác giao diện không gây tính toán lại.

### Task T07: Đánh giá Baseline SVM thực nghiệm (Hoàn thành)
- Huấn luyện song song đa luồng trên 540 ảnh train, tinh chỉnh $C=0.5$ trên tập Validation (Val Macro-F1 = 87.13%).
- Quy tắc suy luận đồng nhất với `AnalysisService`: `np.argmax(model.predict_proba(), axis=1)` (fix R6).
- Đánh giá trên tập Kiểm thử độc lập (Test Set - 180 ảnh):
  - **Accuracy:** 85.0% | **Macro-F1:** 84.9%
  - **Green:** Precision 85.1%, Recall 95.0%, F1 89.8%
  - **Ripe:** Precision 87.5%, Recall 81.7%, F1 84.5%
  - **Turning:** Precision 82.5%, Recall 78.3%, F1 80.3%

### Task T08: Mở rộng Deep Learning MobileNetV2 & So sánh Đối chuẩn (Hoàn thành)
- Xây dựng pipeline Transfer Learning & Fine-tuning MobileNetV2 trên cùng bộ dữ liệu chuẩn:
  - Giai đoạn 1: Khởi động Classifier Head (3 epochs, lr=1e-3).
  - Giai đoạn 2: Fine-tuning các tầng tích chập trên cùng Block 14-18 (7 epochs, lr=1e-4).
- Lựa chọn checkpoint theo Validation Macro-F1 (Epoch 10: Val F1 = 83.41%).
- Đánh giá trên cùng tập Test Set (180 ảnh):
  - **Accuracy:** 83.3% | **Macro-F1:** 83.0%
  - **Green:** Precision 89.2%, Recall 96.7%, F1 92.8%
  - **Ripe:** Precision 79.4%, Recall 83.3%, F1 81.3%
  - **Turning:** Precision 80.8%, Recall 70.0%, F1 75.0%
- Xây dựng `predict_from_cnn` trong `src/modeling.py` tuân thủ cùng hợp đồng `PredictionResult`.
- Tích hợp vào `AnalysisService` và giao diện Streamlit: người dùng có thể đổi mô hình suy luận ngay trên Sidebar và xem bảng so sánh đối chuẩn trực tiếp tại Tab 3.

## 5. Bảng So sánh Đối chuẩn Thực nghiệm (Bàn giao BTL)

| Tiêu chí | Baseline SVM (OpenCV + SVM RBF) | Deep Learning (MobileNetV2 CNN) |
|---|---|---|
| **Kiến trúc** | Histogram HSV 50 chiều + SVM RBF ($C=0.5$) | MobileNetV2 Fine-tuning (224×224 px) |
| **Kích thước file trọng số** | **145.21 KB (148.693 byte)** | 8.72 MB (9.145.099 byte) |
| **Test Accuracy** | **85.0%** | 83.3% |
| **Test Macro-F1** | **84.9%** | 83.0% |
| **F1 Quả xanh (Green)** | 0.898 (Recall 95.0%) | **0.928 (Recall 96.7%)** |
| **F1 Quả chín (Ripe)** | **0.845** | 0.813 |
| **F1 Chuyển màu (Turning)** | **0.803** | 0.750 |
| **Thời gian suy luận CPU** | **~0.7 ms (mô hình thuần) / ~1.819 ms toàn trình (kèm GrabCut)** | ~30.7 ms (forward) / ~1.851 ms toàn trình (kèm GrabCut) |

Độ trễ đo thực nghiệm trên 180 ảnh test, chi tiết tại `outputs/inference_benchmark.json`. GrabCut chiếm ~1.815 ms/ảnh trong độ trễ toàn trình; các con số thuần mô hình không bao gồm phân vùng.

## 6. Công việc có thể nhận tiếp (Task tiếp theo)

| ID | Ưu tiên | Việc cần làm | Điều kiện hoàn thành |
|---|---|---|---|
| T09 | P2 | Giao diện Webcam trực tiếp nâng cao | Bổ sung chế độ chụp liên tục hoặc phát hiện khung hình động qua WebRTC |
| T10 | P2 | Đánh giá định lượng IoU cho phân đoạn quả (Segmentation) | Gán nhãn mask tham chiếu cho 30-50 ảnh mẫu, tính chỉ số IoU và Dice của GrabCut |

## 7. Trạng thái các bản sửa kỹ thuật

- **Đồng bộ đầu vào CNN giữa train và inference:** `AnalysisService` truyền ảnh chuẩn hóa `work` trực tiếp vào `predict_from_cnn()`. Trên 179 ảnh test được chấp nhận, số dự đoán khác biệt so với raw evaluation là **0 / 179**; 1 ảnh `green_test_059.jpg` bị từ chối hợp lệ do tỷ lệ mask dưới ngưỡng 5%.
- **Cache kết quả lô ảnh trong Streamlit (Content Hashing):** Đã bổ sung hàm `get_item_content_hash()` băm SHA-256 nội dung byte của ảnh đưa vào chữ ký batch cache `batch_signature`. Đã kiểm chứng qua `scripts/check_batch_content_cache.py` đạt `result_reused_after_content_change: false` khi thay ảnh cùng tên.
- **Xử lý ảnh lỗi trong lô:** Bổ sung import `AnalysisResult` tại đầu file `app.py`, đảm bảo khi gặp ảnh hỏng hoặc lỗi định dạng thì batch vẫn tiếp tục và ghi nhận trạng thái `failed` mà không bị crash bởi `NameError`. Kiểm chứng qua `scripts/check_ui_release.py` đạt 0 exceptions.
- **Đồng bộ quy tắc đánh giá SVM (R6):** Cập nhật `scripts/train_baseline.py` chuyển từ `model.predict()` sang `np.argmax(model.predict_proba(), axis=1)`, đồng bộ 100% với cơ chế suy luận và tính toán `confidence` của `AnalysisService`, loại bỏ hoàn toàn sai lệch 3 mẫu biên.
- **Đo lường hiệu năng thực nghiệm và kích thước mô hình (R7):** Đã xây dựng `scripts/benchmark_inference.py` và tái tạo `outputs/inference_benchmark.json` trên artifact hiện hành. Kích thước mô hình thực tế: SVM **148.693 byte (~145.21 KB)**; CNN 9.145.099 byte (~8.72 MB). Độ trễ CPU toàn trình bao gồm GrabCut: SVM ~1.819 ms/ảnh, CNN ~1.851 ms/ảnh; độ trễ thuần mô hình: SVM ~0.70 ms, CNN ~30.71 ms (forward).
- **Đồng bộ tài liệu kiến trúc ARCHITECTURE.md:** Đã loại bỏ các tag "cần bổ sung" lỗi thời, cập nhật sơ đồ và hiện trạng mã nguồn thực tế của `contracts.py`, `analysis_service.py`, `export_service.py`, `validate_dataset.py`, `train_cnn.py`.
- **Lưu giữ lịch sử huấn luyện trong `--eval-only`:** Checkpoint đóng gói dạng dict (`model_state_dict`, `best_epoch`, `best_val_f1`, `history`) và nạp `weights=None` không tải lại ImageNet.
- **Khai báo dependencies học sâu:** Đã bổ sung `torch>=2.0` và `torchvision>=0.15` vào `requirements.txt`.
