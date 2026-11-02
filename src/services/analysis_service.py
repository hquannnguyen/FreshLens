from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import (
    AnalysisRequest,
    AnalysisResult,
    PredictionResult,
    ProcessingConfig,
    QualityReport,
    SegmentationResult,
)
from ..features import color_histogram_features, visible_color_ratios
from ..image_pipeline import crop_to_mask, isolate_on_white, resize_for_processing, segment_tomato
from ..modeling import (
    load_cnn_manifest,
    load_cnn_model,
    load_manifest,
    load_model,
    predict_from_cnn,
    predict_from_features,
)
from ..settings import MODEL_PATH, MODEL_DIR
from ..quality import assess_image


class AnalysisService:
    """Service điều phối quy trình phân tích ảnh cà chua theo kiến trúc FreshLens.

    Thực hiện chuẩn hóa ảnh một lần, phân vùng một lần, và dùng cùng mặt nạ cho
    cả hiển thị, tính toán màu sắc và suy luận mô hình (SVM hoặc CNN).
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or MODEL_PATH
        self.cnn_path = MODEL_DIR / "freshlens_cnn.pt"

        self._cached_svm: Any | None = None
        self._cached_svm_mtime: float = 0.0
        self._svm_manifest = load_manifest()

        self._cached_cnn: Any | None = None
        self._cached_cnn_mtime: float = 0.0
        self._cnn_manifest = load_cnn_manifest()

    def get_svm_model(self):
        if not self.model_path.exists():
            return None
        current_mtime = self.model_path.stat().st_mtime
        if self._cached_svm is None or current_mtime != self._cached_svm_mtime:
            self._cached_svm = load_model(self.model_path)
            self._cached_svm_mtime = current_mtime
            self._svm_manifest = load_manifest()
        return self._cached_svm

    def get_cnn_model(self):
        if not self.cnn_path.exists():
            return None
        current_mtime = self.cnn_path.stat().st_mtime
        if self._cached_cnn is None or current_mtime != self._cached_cnn_mtime:
            self._cached_cnn = load_cnn_model(self.cnn_path)
            self._cached_cnn_mtime = current_mtime
            self._cnn_manifest = load_cnn_manifest()
        return self._cached_cnn

    @property
    def has_svm_model(self) -> bool:
        return self.model_path.exists()

    @property
    def has_cnn_model(self) -> bool:
        return self.cnn_path.exists()

    @property
    def has_model(self) -> bool:
        return self.has_svm_model or self.has_cnn_model

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        t0 = time.perf_counter()
        cfg = request.config or ProcessingConfig()
        issues: list[str] = []

        # 1. Kiểm tra ảnh đầu vào
        if request.image_bgr is None or not isinstance(request.image_bgr, np.ndarray) or request.image_bgr.size == 0:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return AnalysisResult(
                image_id=request.image_id,
                filename=request.filename,
                status="failed",
                duration_ms=duration_ms,
                issues=["Không thể đọc dữ liệu ảnh hoặc định dạng ảnh hỏng"],
            )

        # 2. Chuẩn hóa kích thước làm việc (chỉ resize một lần)
        work = resize_for_processing(request.image_bgr, max_side=cfg.max_side)
        h, w = work.shape[:2]

        # 3. Đánh giá chất lượng ảnh
        raw_quality = assess_image(work)
        quality = QualityReport(
            width=w,
            height=h,
            brightness=raw_quality["brightness"],
            sharpness=raw_quality["sharpness"],
            issues=list(raw_quality["issues"]),
        )
        issues.extend(quality.issues)

        # 4. Phân vùng quả (chỉ chạy một lần duy nhất)
        mask, used_method = segment_tomato(work, method=cfg.segmentation)
        fg_pixels = int(np.count_nonzero(mask))
        fg_ratio = float(fg_pixels / mask.size) if mask.size > 0 else 0.0

        bbox = None
        if fg_pixels > 0:
            points = cv2.findNonZero((mask > 0).astype(np.uint8))
            if points is not None:
                x, y, bw, bh = cv2.boundingRect(points)
                bbox = (x, y, bw, bh)

        seg_issues: list[str] = []
        is_valid_mask = cfg.min_mask_ratio <= fg_ratio <= cfg.max_mask_ratio and fg_pixels > 0
        if not is_valid_mask:
            seg_issues.append(
                f"Tỷ lệ vùng quả ({fg_ratio * 100:.1f}%) nằm ngoài ngưỡng cho phép "
                f"({cfg.min_mask_ratio * 100:.0f}% - {cfg.max_mask_ratio * 100:.0f}%)."
            )
            issues.extend(seg_issues)

        segmentation = SegmentationResult(
            mask=mask,
            bbox=bbox,
            method_used=used_method,
            foreground_ratio=fg_ratio,
            issues=seg_issues,
        )

        # Nếu phân vùng không đạt ngưỡng, từ chối gán nhãn
        if not is_valid_mask:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return AnalysisResult(
                image_id=request.image_id,
                filename=request.filename,
                status="rejected",
                quality=quality,
                segmentation=segmentation,
                prediction=PredictionResult(
                    label="",
                    confidence=None,
                    scores={},
                    mode="rejected",
                    model_id=None,
                ),
                color_ratios={},
                duration_ms=duration_ms,
                issues=issues,
                work_image=work,
                isolated_image=None,
                crop_image=None,
                crop_mask=None,
            )

        # 5. Tạo ảnh minh họa từ cùng một mặt nạ
        isolated = isolate_on_white(work, mask)
        crop, crop_mask = crop_to_mask(work, mask)
        ratios = visible_color_ratios(work, mask)

        # 6. Suy luận theo mô hình đã chọn
        use_cnn = cfg.model_type == "cnn" and self.has_cnn_model
        use_svm = cfg.model_type == "svm" and self.has_svm_model

        if request.mode != "color_preview" and (use_cnn or use_svm):
            if use_cnn:
                cnn_model = self.get_cnn_model()
                model_id = self._cnn_manifest.get("model_id", "freshlens_cnn_mobilenetv2_v1") if self._cnn_manifest else "freshlens_cnn"
                # CNN được huấn luyện trên ảnh chuẩn hóa đã tiền xử lý; truyền 'work' để khớp 100% với pipeline train/evaluation
                prediction = predict_from_cnn(cnn_model, work, model_id=model_id)
            else:
                svm_model = self.get_svm_model()
                model_id = self._svm_manifest.get("model_id", "freshlens_svm_baseline_v1") if self._svm_manifest else "freshlens_svm"
                features = color_histogram_features(work, mask)
                prediction = predict_from_features(svm_model, features, model_id=model_id)

            if prediction.confidence is not None and prediction.confidence < cfg.min_confidence:
                issues.append(
                    f"Độ tin cậy mô hình ({prediction.confidence * 100:.1f}%) dưới ngưỡng {cfg.min_confidence * 100:.0f}%"
                )
        else:
            red, green = ratios.get("red", 0.0), ratios.get("green", 0.0)
            if red >= 0.62:
                preview_label = "ripe"
            elif green >= 0.42 and red < 0.18:
                preview_label = "green"
            else:
                preview_label = "turning"

            prediction = PredictionResult(
                label=preview_label,
                confidence=None,
                scores={},
                mode="color_preview",
                model_id=None,
            )

        # 7. Xác định trạng thái tổng thể
        status = "needs_review" if len(issues) > 0 else "completed"
        duration_ms = (time.perf_counter() - t0) * 1000.0

        return AnalysisResult(
            image_id=request.image_id,
            filename=request.filename,
            status=status,
            quality=quality,
            segmentation=segmentation,
            prediction=prediction,
            color_ratios=ratios,
            duration_ms=duration_ms,
            issues=issues,
            work_image=work,
            isolated_image=isolated,
            crop_image=crop,
            crop_mask=crop_mask,
        )
