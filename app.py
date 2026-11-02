from __future__ import annotations

import hashlib
import json
import random
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.contracts import AnalysisRequest, AnalysisResult, ProcessingConfig
from src.modeling import load_cnn_manifest, load_manifest
from src.services import AnalysisService, export_csv_bytes, export_zip_bundle
from src.settings import CLASS_LABELS, DATA_DIR, OUTPUT_DIR


st.set_page_config(page_title="FreshLens - Phân loại độ chín cà chua", page_icon="🍅", layout="wide")


def get_item_content_hash(item) -> str:
    """Tính mã băm SHA256 cho nội dung tệp upload hoặc tệp từ đường dẫn."""
    try:
        if hasattr(item, "getvalue"):
            return hashlib.sha256(item.getvalue()).hexdigest()
        elif hasattr(item, "read"):
            pos = item.tell() if hasattr(item, "tell") else 0
            data = item.read()
            if hasattr(item, "seek"):
                item.seek(pos)
            return hashlib.sha256(data).hexdigest()
        elif isinstance(item, (str, Path)):
            p = Path(item)
            if p.exists():
                return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        pass
    return str(getattr(item, "name", str(item)))


def decode_image_input(image_input) -> np.ndarray:
    """Đọc ảnh từ file uploader, camera bytes, hoặc đường dẫn file với xử lý tự động xoay EXIF."""
    if isinstance(image_input, (str, Path)):
        raw_img = Image.open(str(image_input))
    else:
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        raw_img = Image.open(image_input)
    transposed = ImageOps.exif_transpose(raw_img)
    rgb = np.asarray(transposed.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def as_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def label_vi(label: str) -> str:
    return CLASS_LABELS.get(label, label.replace("_", " ").title() if label else "Không xác định")


def _pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—"


def _f1(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


@st.cache_resource(show_spinner=False)
def get_analysis_service():
    return AnalysisService()


service = get_analysis_service()
manifest_svm = load_manifest()
manifest_cnn = load_cnn_manifest()


@st.cache_data(show_spinner=False)
def load_metrics_summaries() -> dict:
    """Đọc số liệu thực nghiệm trực tiếp từ artifact JSON (nguồn duy nhất của sự thật)."""
    result: dict = {"svm": None, "cnn": None, "benchmark": None}
    for key, fname in (
        ("svm", "baseline_metrics.json"),
        ("cnn", "cnn_metrics.json"),
        ("benchmark", "inference_benchmark.json"),
    ):
        try:
            result[key] = json.loads((OUTPUT_DIR / fname).read_text(encoding="utf-8"))
        except Exception:
            result[key] = None
    return result


metrics_summaries = load_metrics_summaries()

st.title("🍅 FreshLens")
st.caption("Hệ thống trợ lý phân loại độ chín cà chua bằng Xử lý ảnh số & Học máy")

with st.sidebar:
    st.header("⚙️ Thiết lập phân tích")

    # Lựa chọn mô hình (số liệu đọc trực tiếp từ artifact metrics)
    svm_test_metrics = (metrics_summaries.get("svm") or {}).get("test") or {}
    cnn_test_metrics = (metrics_summaries.get("cnn") or {}).get("test") or {}
    model_choices = []
    if service.has_svm_model:
        model_choices.append(f"Baseline SVM (HSV Histogram - Macro-F1: {_pct(svm_test_metrics.get('macro_f1'))})")
    if service.has_cnn_model:
        model_choices.append(f"Deep Learning CNN (MobileNetV2 - Macro-F1: {_pct(cnn_test_metrics.get('macro_f1'))})")
    model_choices.append("Quy tắc màu xem trước (Color Preview)")

    selected_model_str = st.selectbox("Mô hình suy luận:", model_choices, index=0)

    if "Baseline SVM" in selected_model_str:
        chosen_mode = "trained_model"
        chosen_model_type = "svm"
        st.success("✅ Đang dùng: **SVM RBF Baseline**")
    elif "Deep Learning" in selected_model_str:
        chosen_mode = "trained_model"
        chosen_model_type = "cnn"
        st.success("✅ Đang dùng: **MobileNetV2 CNN**")
    else:
        chosen_mode = "color_preview"
        chosen_model_type = "svm"
        st.info("ℹ️ Đang dùng: **Quy tắc tỷ lệ màu xem trước**")

    # Lựa chọn phương pháp phân đoạn
    segmentation_ui = st.selectbox(
        "Thuật toán tách quả (Segmentation):",
        ["GrabCut (Khuyến nghị cho ảnh tự nhiên)", "Nền xanh (Chụp trong buồng lab)"],
        index=0,
    )
    segmentation_method = "grabcut" if "GrabCut" in segmentation_ui else "blue"

    with st.expander("ℹ️ Thông số kỹ thuật mô hình", expanded=False):
        if chosen_model_type == "svm" and manifest_svm:
            st.write(f"**ID:** `{manifest_svm.get('model_id')}`")
            st.write(f"**Kiến trúc:** {manifest_svm.get('model_type')}")
            st.write(f"**Đặc trưng:** {manifest_svm.get('num_features')} chiều HSV")
            st.write(f"**Test Macro-F1:** `{manifest_svm.get('metrics', {}).get('test_macro_f1', 0.866):.1%}`")
        elif chosen_model_type == "cnn" and manifest_cnn:
            st.write(f"**ID:** `{manifest_cnn.get('model_id')}`")
            st.write(f"**Backbone:** {manifest_cnn.get('backbone')}")
            st.write(f"**Độ phân giải đầu vào:** 224×224 px")
            st.write(f"**Test Macro-F1:** `{manifest_cnn.get('metrics', {}).get('test_macro_f1', 0.830):.1%}`")

    st.markdown("---")
    st.markdown(
        "**💡 Khuyến nghị khi chụp ảnh:**\n"
        "- Mỗi ảnh chỉ chứa 1 quả cà chua nằm trọn trong khung hình.\n"
        "- Đặt quả trên nền tương phản (vải xanh hoặc nền đồng màu).\n"
        "- Đủ sáng, ánh sáng tán xạ, hạn chế lóa hoặc bóng gắt."
    )

tab_single, tab_batch, tab_metrics, tab_about = st.tabs([
    "📸 Phân loại ảnh đơn",
    "📦 Phân loại theo lô",
    "📊 So sánh mô hình & Đánh giá",
    "📖 Giới thiệu & Hướng dẫn",
])

# ----------------- TAB 1: ẢNH ĐƠN -----------------
with tab_single:
    st.subheader("Chọn ảnh cà chua cần phân tích")

    source_type = st.radio(
        "Nguồn ảnh:",
        [
            "📂 Chọn ảnh mẫu có sẵn từ bộ dữ liệu (Khuyến nghị để thử ngay)",
            "⬆️ Tải ảnh từ máy tính của bạn",
            "📷 Chụp ảnh trực tiếp từ Camera / Webcam",
        ],
        horizontal=True,
    )

    original_img: np.ndarray | None = None
    input_filename: str = ""

    if source_type.startswith("📂"):
        test_dir = DATA_DIR / "test"
        col_c1, col_c2, col_c3 = st.columns([1.5, 2.5, 1])

        with col_c1:
            stage_option = st.selectbox(
                "Giai đoạn chín:",
                ["🟢 Quả xanh (Green)", "🟡 Đang chuyển màu (Turning)", "🔴 Quả chín đỏ (Ripe)"],
                index=0,
            )
            stage_key = "green" if "Green" in stage_option else "turning" if "Turning" in stage_option else "ripe"

        class_path = test_dir / stage_key
        available_samples = sorted(p.name for p in class_path.glob("*.jpg")) if class_path.exists() else []

        if "random_sample" not in st.session_state:
            st.session_state["random_sample"] = None

        with col_c3:
            st.write("")
            st.write("")
            if st.button("🎲 Chọn ngẫu nhiên", use_container_width=True):
                if available_samples:
                    st.session_state["random_sample"] = random.choice(available_samples)

        with col_c2:
            default_index = 0
            if st.session_state.get("random_sample") in available_samples:
                default_index = available_samples.index(st.session_state["random_sample"])

            selected_sample = st.selectbox(
                "Chọn ảnh quả cà chua mẫu:",
                available_samples,
                index=default_index if available_samples else 0,
            )

        if selected_sample and class_path.exists():
            target_path = class_path / selected_sample
            if target_path.exists():
                original_img = decode_image_input(target_path)
                input_filename = selected_sample

    elif source_type.startswith("⬆️"):
        uploaded = st.file_uploader(
            "Kéo thả hoặc chọn tệp ảnh từ máy tính (JPG, PNG, WEBP, BMP)",
            type=["jpg", "jpeg", "png", "bmp", "webp", "jfif"],
            key="single_uploader",
        )
        if uploaded is not None:
            try:
                original_img = decode_image_input(uploaded)
                input_filename = uploaded.name
            except Exception as exc:
                st.error(f"Lỗi khi đọc tệp ảnh tải lên: {exc}")

    elif source_type.startswith("📷"):
        camera_photo = st.camera_input("Chụp ảnh quả cà chua trước ống kính camera")
        if camera_photo is not None:
            try:
                original_img = decode_image_input(camera_photo)
                input_filename = "webcam_capture.jpg"
            except Exception as exc:
                st.error(f"Lỗi khi xử lý ảnh từ camera: {exc}")

    # Tiến hành phân tích khi đã có ảnh
    if original_img is not None:
        try:
            with st.spinner(f"⏳ Đang xử lý ảnh bằng mô hình {selected_model_str.split(' (')[0]}..."):
                img_id = f"img_{uuid.uuid4().hex[:8]}"
                req = AnalysisRequest(
                    image_id=img_id,
                    filename=input_filename,
                    image_bgr=original_img,
                    mode=chosen_mode,
                    config=ProcessingConfig(
                        segmentation=segmentation_method,
                        model_type=chosen_model_type,
                    ),
                )
                res = service.analyze(req)

            st.markdown("---")
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Ảnh gốc đầu vào")
                st.image(as_rgb(original_img), use_container_width=True)
            with col_right:
                st.subheader("Vùng quả đã phân vùng")
                if res.isolated_image is not None:
                    st.image(as_rgb(res.isolated_image), use_container_width=True)
                else:
                    st.warning("⚠️ Không tách được vùng quả hợp lệ để hiển thị.")

            st.subheader("Kết quả phân loại độ chín")
            k1, k2, k3, k4 = st.columns(4)

            pred = res.prediction
            lbl_vi = label_vi(pred.label) if (pred and pred.label) else "Bị từ chối (Không rõ quả)"
            k1.metric("Mức độ chín", lbl_vi)

            conf_str = (
                f"{pred.confidence:.1%}"
                if (pred and pred.confidence is not None)
                else ("Chế độ xem trước" if chosen_mode == "color_preview" else "N/A")
            )
            k2.metric("Độ tin cậy", conf_str)

            method_str = res.segmentation.method_used if res.segmentation else "Không xác định"
            k3.metric("Phương pháp phân vùng", method_str)

            status_map = {
                "completed": "✅ Đạt chuẩn",
                "needs_review": "⚠️ Cần rà soát",
                "rejected": "❌ Bị từ chối",
                "failed": "❌ Lỗi tệp",
            }
            k4.metric("Trạng thái", status_map.get(res.status, res.status))

            if res.status == "rejected":
                st.error("❌ Ảnh bị từ chối: Không phát hiện quả cà chua hợp lệ. Đề nghị chụp lại sao cho quả nằm trọn trong ảnh.")
            elif res.prediction and res.prediction.mode == "color_preview":
                st.warning("⚠️ Đang chạy chế độ xem trước theo màu sắc (Color Preview). Hãy chọn mô hình Baseline SVM hoặc CNN ở thanh bên.")

            if res.prediction and res.prediction.scores:
                st.write(f"**Phân bố xác suất các lớp ({pred.model_id or 'Mô hình'}):**")
                prob_cols = st.columns(3)
                classes_order = ["green", "turning", "ripe"]
                for i, c in enumerate(classes_order):
                    score = res.prediction.scores.get(c, 0.0)
                    prob_cols[i].progress(score, text=f"{label_vi(c)}: {score:.1%}")

            if res.quality:
                q = res.quality
                st.caption(
                    f"Kích thước: {q.width}×{q.height} px | "
                    f"Độ sáng trung bình: {q.brightness:.1f}/255 | "
                    f"Độ nét (Laplacian): {q.sharpness:.1f} | "
                    f"Thời gian xử lý: {res.duration_ms:.1f} ms"
                )

            if res.issues:
                st.info("🔍 Lưu ý chất lượng: " + " • ".join(res.issues))

            with st.expander("🔍 Chi tiết mặt nạ và vùng cắt quả (ROI)"):
                m_col1, m_col2 = st.columns(2)
                if res.segmentation and res.segmentation.mask is not None:
                    m_col1.image(res.segmentation.mask, caption="Mặt nạ nhị phân (0/255)", clamp=True, use_container_width=True)
                if res.crop_image is not None:
                    m_col2.image(as_rgb(res.crop_image), caption="Vùng cắt quả (ROI Crop)", use_container_width=True)

        except Exception as exc:
            st.error(f"Lỗi khi xử lý ảnh: {exc}")

# ----------------- TAB 2: PHÂN LOẠI THEO LÔ -----------------
with tab_batch:
    st.markdown("Tải lên nhiều ảnh cà chua cùng lúc để xử lý hàng loạt và xuất báo cáo kết quả.")

    col_upload, col_quick = st.columns([2.5, 1.5])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Chọn tệp ảnh từ máy tính (tối đa 30 ảnh mỗi lô)",
            type=["jpg", "jpeg", "png", "bmp", "webp", "jfif"],
            accept_multiple_files=True,
            key="batch_uploader",
        )

    with col_quick:
        st.write("")
        st.write("")
        sample_batch_clicked = st.button("📂 Thử nhanh với Lô 6 ảnh mẫu", use_container_width=True)

    if sample_batch_clicked:
        test_dir = DATA_DIR / "test"
        sample_paths = []
        for c in ["green", "turning", "ripe"]:
            c_dir = test_dir / c
            if c_dir.exists():
                sample_paths.extend(sorted(c_dir.glob("*.jpg"))[:2])

        st.session_state["batch_source_paths"] = sample_paths
        st.session_state.pop("batch_results", None)

    active_files = []
    if uploaded_files:
        active_files = [("upload", f) for f in uploaded_files]
    elif "batch_source_paths" in st.session_state:
        active_files = [("path", p) for p in st.session_state["batch_source_paths"]]

    if active_files:
        st.info(f"Đã chọn {len(active_files)} ảnh để phân tích với mô hình **{selected_model_str.split(' (')[0]}**.")

        batch_signature = (
            tuple(
                (f.name if hasattr(f, "name") else Path(str(f)).name, get_item_content_hash(f))
                for _, f in active_files
            ),
            chosen_model_type,
            chosen_mode,
            segmentation_method,
        )

        btn_run = st.button("🚀 Bắt đầu phân loại lô ảnh", type="primary")
        needs_recompute = btn_run or (st.session_state.get("batch_signature") != batch_signature)

        if btn_run or "batch_results" in st.session_state:
            if needs_recompute:
                batch_results = []
                progress_bar = st.progress(0, text="Đang bắt đầu phân tích lô...")

                for idx, (ftype, file_item) in enumerate(active_files):
                    fname = file_item.name if hasattr(file_item, "name") else Path(str(file_item)).name
                    try:
                        img_bgr = decode_image_input(file_item)

                        req_id = f"batch_{idx+1:03d}_{uuid.uuid4().hex[:6]}"
                        req = AnalysisRequest(
                            image_id=req_id,
                            filename=fname,
                            image_bgr=img_bgr,
                            mode=chosen_mode,
                            config=ProcessingConfig(
                                segmentation=segmentation_method,
                                model_type=chosen_model_type,
                            ),
                        )
                        result = service.analyze(req)
                        batch_results.append(result)
                    except Exception as err:
                        batch_results.append(
                            AnalysisResult(
                                image_id=f"batch_{idx+1:03d}_err",
                                filename=fname,
                                status="failed",
                                issues=[f"Lỗi: {err}"],
                            )
                        )
                    progress_bar.progress((idx + 1) / len(active_files), text=f"Đã xử lý {idx + 1}/{len(active_files)} ảnh...")

                st.session_state["batch_results"] = batch_results
                st.session_state["batch_signature"] = batch_signature

            results_list = st.session_state.get("batch_results", [])

            table_rows = []
            for r in results_list:
                pred = r.prediction
                lbl = label_vi(pred.label) if (pred and pred.label) else ("Từ chối" if r.status == "rejected" else "Lỗi")
                conf = f"{pred.confidence:.1%}" if (pred and pred.confidence is not None) else "-"
                table_rows.append({
                    "Mã": r.image_id,
                    "Tệp": r.filename,
                    "Độ chín": lbl,
                    "Độ tin cậy": conf,
                    "Mô hình": pred.model_id if pred else "-",
                    "Trạng thái": r.status,
                    "Phương pháp": r.segmentation.method_used if r.segmentation else "-",
                    "Vấn đề rà soát": "; ".join(r.issues) if r.issues else "Không",
                    "Thời gian (ms)": f"{r.duration_ms:.1f}",
                })

            df_batch = pd.DataFrame(table_rows)
            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            valid_labels = [row["Độ chín"] for row in table_rows if row["Độ chín"] in [label_vi(c) for c in CLASS_LABELS]]
            if valid_labels:
                st.subheader("📊 Phân bố các mức chín trong lô")
                counts = pd.Series(valid_labels).value_counts().rename_axis("Mức chín").to_frame("Số lượng")
                st.bar_chart(counts)

            st.subheader("📥 Xuất kết quả lô")
            c_csv, c_zip = st.columns(2)

            csv_data = export_csv_bytes(results_list)
            c_csv.download_button(
                "📄 Tải báo cáo CSV (UTF-8)",
                data=csv_data,
                file_name="freshlens_batch_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

            zip_data = export_zip_bundle(results_list)
            c_zip.download_button(
                "🗂️ Tải gói ảnh đã phân vùng (ZIP)",
                data=zip_data,
                file_name="freshlens_batch_bundle.zip",
                mime="application/zip",
                use_container_width=True,
            )

# ----------------- TAB 3: BÁO CÁO SO SÁNH MÔ HÌNH -----------------
with tab_metrics:
    st.subheader("📊 Bảng so sánh Đối chuẩn: Baseline SVM vs Deep Learning CNN")
    st.markdown(
        r"Cả hai mô hình đều được đánh giá độc lập trên cùng **tập Kiểm thử Test Set (180 ảnh thật $\ge 640\text{px}$)**, "
        r"tuân thủ tuyệt đối phân tách độc lập theo phiên chụp (Group-Split) để ngăn chặn rò rỉ dữ liệu."
    )

    # Số liệu đọc trực tiếp từ artifact JSON (baseline_metrics.json / cnn_metrics.json / inference_benchmark.json)
    svm_m = metrics_summaries.get("svm") or {}
    cnn_m = metrics_summaries.get("cnn") or {}
    bench_m = metrics_summaries.get("benchmark") or {}
    svm_test = svm_m.get("test") or {}
    cnn_test = cnn_m.get("test") or {}
    svm_report = svm_test.get("classification_report") or {}
    cnn_report = cnn_test.get("classification_report") or {}
    svm_bench = bench_m.get("svm") or {}
    cnn_bench = bench_m.get("cnn") or {}

    svm_size_txt = f"{svm_bench.get('size_kb', '—')} KB ({svm_bench.get('size_bytes', '—')} byte)" if svm_bench else "—"
    cnn_size_txt = f"{cnn_bench.get('size_mb', '—')} MB ({cnn_bench.get('size_bytes', '—')} byte)" if cnn_bench else "—"

    svm_pure = (svm_bench.get("pure_inference_ms") or {}).get("mean")
    svm_pipe = (svm_bench.get("total_pipeline_ms") or {}).get("mean")
    cnn_pure = (cnn_bench.get("total_model_ms") or {}).get("mean")
    cnn_pipe = (cnn_bench.get("total_pipeline_ms") or {}).get("mean")
    svm_speed_txt = (
        f"{svm_pure} ms thuần mô hình / {svm_pipe} ms toàn trình (kèm GrabCut)" if svm_pure is not None else "—"
    )
    cnn_speed_txt = (
        f"{cnn_pure} ms thuần mô hình / {cnn_pipe} ms toàn trình (kèm GrabCut)" if cnn_pure is not None else "—"
    )

    # Bảng so sánh trực diện
    compare_df = pd.DataFrame({
        "Tiêu chí đánh giá": [
            "Kiến trúc mô hình",
            "Đầu vào / Biểu diễn đặc trưng",
            "Độ chính xác Test (Accuracy)",
            "Test Macro-F1",
            "F1-Score: Quả xanh (Green)",
            "F1-Score: Quả chín (Ripe)",
            "F1-Score: Đang chuyển màu (Turning)",
            "Kích thước tệp mô hình",
            "Tốc độ suy luận thuần (CPU/ảnh)",
            "Tốc độ toàn trình kèm phân vùng (CPU/ảnh)",
        ],
        "Baseline (Xử lý ảnh + SVM)": [
            "StandardScaler + SVM (RBF Kernel)",
            "Histogram màu HSV 50 chiều",
            _pct(svm_test.get("accuracy")),
            _pct(svm_test.get("macro_f1")),
            _f1((svm_report.get("green") or {}).get("f1-score")),
            _f1((svm_report.get("ripe") or {}).get("f1-score")),
            _f1((svm_report.get("turning") or {}).get("f1-score")),
            svm_size_txt,
            f"{svm_pure} ms" if svm_pure is not None else "—",
            f"{svm_pipe} ms" if svm_pipe is not None else "—",
        ],
        "Deep Learning (MobileNetV2 CNN)": [
            "MobileNetV2 Transfer Learning",
            "Biểu diễn tích chập sâu 224×224 px",
            _pct(cnn_test.get("accuracy")),
            _pct(cnn_test.get("macro_f1")),
            _f1((cnn_report.get("green") or {}).get("f1-score")),
            _f1((cnn_report.get("ripe") or {}).get("f1-score")),
            _f1((cnn_report.get("turning") or {}).get("f1-score")),
            cnn_size_txt,
            f"{cnn_pure} ms" if cnn_pure is not None else "—",
            f"{cnn_pipe} ms" if cnn_pipe is not None else "—",
        ],
    })
    st.dataframe(compare_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Phân tích chuyên sâu & Nhận xét Báo cáo BTL")

    svm_f1_txt = _pct(svm_test.get("macro_f1"))
    cnn_f1_txt = _pct(cnn_test.get("macro_f1"))
    col_disc1, col_disc2 = st.columns(2)
    with col_disc1:
        st.markdown(
            f"#### 1. Tại sao Baseline SVM đạt F1 cao hơn CNN ({svm_f1_txt} vs {cnn_f1_txt})?\n"
            "- **Bản chất bài toán độ chín cà chua:** Độ chín biểu hiện trực tiếp qua **tỷ lệ phân bố sắc tố quang phổ** (chlorophyll xanh lục chuyển dần sang carotenoid/lycopene đỏ). "
            "Biểu đồ tần suất HSV (50 bins) là bộ trích xuất đặc trưng thuần túy màu sắc, không phụ thuộc vào vị trí không gian.\n"
            "- **Mạng CNN học đặc trưng không gian (Spatial Features):** MobileNetV2 học cả đường nét, vân bề mặt và hình học. "
            "Trên tập dữ liệu cà chua có bề mặt quả trơn láng, các đặc trưng không gian phức tạp của CNN dễ gặp nhiễu hoặc overfit nhẹ ở giai đoạn chuyển màu (turning) so với histogram màu sắc tập trung của SVM."
        )

    with col_disc2:
        cnn_green = cnn_report.get("green") or {}
        cnn_green_f1 = _f1(cnn_green.get("f1-score"))
        cnn_green_recall = _pct(cnn_green.get("recall"))
        cnn_green_hit = int(round((cnn_green.get("recall") or 0) * 60))
        st.markdown(
            f"#### 2. Thế mạnh riêng của Deep Learning MobileNetV2:\n"
            f"- **Nhận diện quả xanh vượt trội:** MobileNetV2 đạt F1 quả xanh lên tới **{cnn_green_f1}** (Recall {cnn_green_recall}, phát hiện {cnn_green_hit}/60 quả xanh).\n"
            "- **Khả năng khái quát hóa:** MobileNetV2 có thể phân loại trực tiếp trên vùng ảnh crop mà không cần thuật toán phân đoạn màu nền nghiêm ngặt, "
            "phù hợp cho các môi trường có bóng đổ phức tạp hoặc ánh sáng không đồng đều."
        )

    st.markdown("---")
    st.subheader("Ma trận nhầm lẫn chi tiết của 2 mô hình trên tập Test")
    col_cm1, col_cm2 = st.columns(2)

    cm_labels_vi = ["Thực tế: Xanh", "Thực tế: Chín", "Thực tế: Chuyển màu"]
    cm_cols_vi = ["Dự đoán: Xanh", "Dự đoán: Chín", "Dự đoán: Chuyển màu"]

    with col_cm1:
        st.markdown(f"**Baseline SVM RBF (Accuracy: {_pct(svm_test.get('accuracy'))}):**")
        cm_svm_values = svm_test.get("confusion_matrix") or [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        cm_svm = pd.DataFrame(cm_svm_values, index=cm_labels_vi, columns=cm_cols_vi)
        st.dataframe(cm_svm, use_container_width=True)

    with col_cm2:
        st.markdown(f"**MobileNetV2 CNN (Accuracy: {_pct(cnn_test.get('accuracy'))}):**")
        cm_cnn_values = cnn_test.get("confusion_matrix") or [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        cm_cnn = pd.DataFrame(cm_cnn_values, index=cm_labels_vi, columns=cm_cols_vi)
        st.dataframe(cm_cnn, use_container_width=True)

# ----------------- TAB 4: GIỚI THIỆU -----------------
with tab_about:
    st.markdown(r"""
### 🍅 Dự án FreshLens làm gì?
FreshLens là hệ thống trợ lý thị giác máy tính và học máy hỗ trợ sàng lọc, phân loại độ chín cà chua theo 3 giai đoạn:
1. **Green (Xanh / chưa chín)**: Quả hoàn toàn có sắc tố xanh lục của chlorophyll.
2. **Turning (Đang chuyển màu)**: Giai đoạn biến đổi sắc tố sang vàng cam hoặc hồng đỏ ($\ge 10\%$ và $\le 90\%$ bề mặt quả).
3. **Ripe (Đỏ / chín)**: Quả chuyển hoàn toàn sang màu đỏ/hồng sẫm ($> 90\%$ bề mặt quả).

### 🛠️ Kiến trúc Pipeline
1. **Chuẩn hóa đầu vào:** Nhận ảnh, chuẩn hóa chiều xoay theo EXIF, resize bảo toàn tỷ lệ về kích thước làm việc tối đa 1600px.
2. **Kiểm tra chất lượng (Quality Assessment):** Đánh giá độ nét bằng toán tử Laplacian ($Var > 35$) và độ sáng kênh V ($45 < Brightness < 225$).
3. **Phân vùng quả (Segmentation):** Áp dụng GrabCut thu nhỏ kết hợp phép đóng/mở hình thái học (Morphological Opening/Closing) để tìm vùng liên thông lớn nhất.
4. **Trích xuất đặc trưng (Feature Extraction):** Tạo biểu đồ tần suất màu sắc HSV (18 bins H + 16 bins S + 16 bins V = **50 chiều đặc trưng**), chuẩn hóa $L_1$.
5. **Suy luận phân loại:**
   - **Mô hình Baseline:** `StandardScaler + Support Vector Machine (RBF Kernel)`
   - **Mô hình Deep Learning:** `MobileNetV2 Transfer Learning & Fine-tuning`

### ⚠️ Giới hạn kỹ thuật
- Hệ thống tối ưu cho kịch bản mỗi ảnh chứa **một quả đơn lẻ**.
- Không đánh giá được độ ngọt (Brix), độ mềm thịt quả hoặc dư lượng hóa chất bên trong.
- Các ảnh có độ tin cậy mô hình $< 65\%$ hoặc có cảnh báo chất lượng được gắn nhãn `needs_review` để nhân viên rà soát thủ công.
""")
