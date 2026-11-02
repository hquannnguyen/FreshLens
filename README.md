# FreshLens

Ứng dụng hỗ trợ phân loại độ chín cà chua từ ảnh bằng xử lý ảnh OpenCV, baseline học máy HSV histogram + SVM, và mở rộng học sâu Deep Learning (MobileNetV2 CNN).

## Bắt đầu cho thành viên hoặc agent mới

Đọc [kiến trúc hệ thống](ARCHITECTURE.md) để hiểu các thành phần và luồng dữ liệu, sau đó đọc [tài liệu bàn giao](HANDOFF.md) để biết hiện trạng, lỗi đã biết và task có thể nhận tiếp. [AGENTS.md](AGENTS.md) quy định cách tiếp tục phát triển; [PLAN.md](PLAN.md) trình bày kế hoạch và mốc bàn giao. Kiến trúc ghi rõ thành phần đã có và thành phần cần bổ sung, không coi thiết kế đích là chức năng đã hoàn thành.

Xem [kiểm tra mới nhất](CHECK_RELEASE_2026-11-02.md) để biết phần đã xác nhận và các vấn đề benchmark/giao diện còn mở. [Báo cáo gốc](CHECK_REPORT_2026-11-02.md) và [lần kiểm tra trước](CHECK_FOLLOWUP_2026-11-02.md) lưu bằng chứng lịch sử trong cùng ngày.

## Chạy giao diện

Windows PowerShell:

```powershell
cd D:\XuLyAnh\FreshLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Hoặc sau khi cài dependencies, chạy `scripts\run_app.bat`.

Giao diện hỗ trợ chuyển đổi linh hoạt giữa 3 mô hình ngay tại thanh Sidebar:
1. **Baseline SVM (HSV Histogram 50 chiều + SVM RBF)**
2. **Deep Learning CNN (MobileNetV2 Transfer Learning)**
3. **Quy tắc màu xem trước (Color Preview)**

## Chuẩn bị dữ liệu

Tổ chức ảnh theo [data/images/README.md](data/images/README.md):

```text
data/images/train/{green,turning,ripe}/
data/images/val/{green,turning,ripe}/
data/images/test/{green,turning,ripe}/
```

Chụp một quả trên phông xanh lam mờ hoặc nền tương phản, ánh sáng tán xạ. Chia theo quả vật lý hoặc phiên chụp (Group-Split) trước khi augment để tránh ảnh cùng quả lọt vào nhiều tập. Ghi thông tin ảnh trong `data/annotations.csv`.

Kiểm tra tính hợp lệ của dữ liệu và phát hiện rò rỉ:

```powershell
python scripts\validate_dataset.py
```

## Huấn luyện Baseline SVM

```powershell
python scripts\train_baseline.py --segmentation grabcut
```

Lệnh này trích xuất đặc trưng histogram HSV (50 chiều) đa luồng, tinh chỉnh siêu tham số $C$ trên tập `val`, đánh giá tập `test` cuối cùng, lưu mô hình vào `models/freshlens_svm.joblib` và xuất báo cáo tại `outputs/baseline_metrics.json`.

## Huấn luyện Deep Learning CNN (MobileNetV2)

```powershell
python scripts\train_cnn.py
```

Lệnh này khởi động Classifier Head (3 epochs) và fine-tuning các tầng tích chập trên cùng của MobileNetV2 (7 epochs), chọn checkpoint theo Validation Macro-F1, đánh giá tập `test` độc lập, lưu trọng số vào `models/freshlens_cnn.pt` và báo cáo tại `outputs/cnn_metrics.json`.

Để chỉ đánh giá lại mô hình đã lưu:
```powershell
python scripts\train_cnn.py --eval-only
```

## Bảng So sánh Đối chuẩn Thực nghiệm (BTL)

Đánh giá độc lập trên cùng **tập Kiểm thử Test Set (180 ảnh thật $\ge 640\text{px}$)**, không rò rỉ phiên chụp:

| Tiêu chí | Baseline (Xử lý ảnh + SVM) | Deep Learning (MobileNetV2 CNN) |
|---|---|---|
| **Kiến trúc mô hình** | StandardScaler + SVM RBF ($C=0.5$) | MobileNetV2 Fine-tuning (224×224 px) |
| **Biểu diễn đặc trưng** | Histogram HSV 50 chiều chuẩn hóa | Biểu diễn tích chập sâu ImageNet |
| **Độ chính xác Test (Accuracy)** | **85.0%** | 83.3% |
| **Test Macro-F1** | **84.9%** | 83.0% |
| **F1 Quả xanh (Green)** | 0.898 (Recall 95.0%) | **0.928 (Recall 96.7%)** |
| **F1 Quả chín (Ripe)** | **0.845** | 0.813 |
| **F1 Chuyển màu (Turning)** | **0.803** | 0.750 |
| **Kích thước file mô hình** | **145.21 KB (148.693 byte)** | 8.72 MB (9.145.099 byte) |
| **Thời gian suy luận CPU** | **~0.7 ms (mô hình thuần) / ~1.819 ms toàn trình (kèm GrabCut)** | ~30.7 ms (forward) / ~1.851 ms toàn trình (kèm GrabCut) |

Độ trễ đo thực nghiệm trên toàn bộ 180 ảnh test (CPU, xử lý tuần tự), chi tiết từng thành phần lưu tại `outputs/inference_benchmark.json`. GrabCut là thành phần chiếm đa số độ trễ toàn trình (~1.815 ms/ảnh); độ trễ thuần mô hình không bao gồm phân vùng.

## Cấu trúc project

```text
app.py                    Giao diện Streamlit: ảnh đơn, lô ảnh, so sánh mô hình
src/contracts.py          Dataclass hợp đồng dữ liệu chuẩn hóa
src/services/             Service điều phối phân tích và xuất dữ liệu
  ├── analysis_service.py Phân tích chuẩn hóa một lần, hỗ trợ SVM & CNN
  └── export_service.py   Tạo báo cáo CSV UTF-8 và gói ảnh ZIP
src/image_pipeline.py     Chuẩn hóa, GrabCut tăng tốc, morphology, mask/crop
src/quality.py            Kiểm tra độ phân giải, độ sáng, độ nét
src/features.py           Histogram HSV 50 chiều và tỷ lệ màu
src/modeling.py           SVM & CNN ClassifierAdapters, nạp model, manifest
scripts/train_baseline.py Huấn luyện và đánh giá baseline SVM
scripts/train_cnn.py      Huấn luyện và fine-tuning MobileNetV2 CNN
scripts/validate_dataset.py Kiểm tra tính toàn vẹn và rò rỉ của bộ dữ liệu
data/                     900 ảnh thật chuẩn Group-Split và metadata annotations.csv
models/                   freshlens_svm.joblib, freshlens_cnn.pt và manifests
outputs/                  baseline_metrics.json, cnn_metrics.json
PLAN.md                   Kế hoạch triển khai theo giai đoạn/tuần
ARCHITECTURE.md           Kiến trúc, hợp đồng dữ liệu và bố trí module đích
HANDOFF.md                Hiện trạng, kết quả thực nghiệm, task và việc tiếp theo
AGENTS.md                 Hướng dẫn cho agent tiếp tục phát triển
DE_XUAT_BAI_TAP_LON.md    Đề xuất bài tập lớn đầy đủ
```
