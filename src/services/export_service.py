from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np

from ..contracts import AnalysisResult
from ..settings import CLASS_LABELS


def export_csv_bytes(results: list[AnalysisResult]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "image_id",
        "filename",
        "status",
        "label",
        "label_vi",
        "confidence",
        "mode",
        "model_id",
        "method_used",
        "foreground_ratio",
        "brightness",
        "sharpness",
        "issues",
        "duration_ms",
    ])

    for r in results:
        label = r.prediction.label if r.prediction else ""
        label_vi_str = CLASS_LABELS.get(label, "") if label else ""
        confidence_str = f"{r.prediction.confidence:.4f}" if (r.prediction and r.prediction.confidence is not None) else ""
        mode_str = r.prediction.mode if r.prediction else ""
        model_id_str = r.prediction.model_id or "" if r.prediction else ""
        method_str = r.segmentation.method_used if r.segmentation else ""
        fg_ratio_str = f"{r.segmentation.foreground_ratio:.4f}" if r.segmentation else ""
        brightness_str = f"{r.quality.brightness:.2f}" if r.quality else ""
        sharpness_str = f"{r.quality.sharpness:.2f}" if r.quality else ""
        issues_str = "; ".join(r.issues)
        duration_str = f"{r.duration_ms:.1f}"

        writer.writerow([
            r.image_id,
            r.filename,
            r.status,
            label,
            label_vi_str,
            confidence_str,
            mode_str,
            model_id_str,
            method_str,
            fg_ratio_str,
            brightness_str,
            sharpness_str,
            issues_str,
            duration_str,
        ])

    return output.getvalue().encode("utf-8-sig")


def export_zip_bundle(results: list[AnalysisResult]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Thêm kết quả CSV
        csv_bytes = export_csv_bytes(results)
        zip_file.writestr("batch_results.csv", csv_bytes)

        # 2. Thêm ảnh đã xử lý với tiền tố image_id tránh trùng tên
        for r in results:
            stem = Path(r.filename).stem
            prefix = f"{r.image_id}_{stem}"

            if r.isolated_image is not None:
                success, buf = cv2.imencode(".png", r.isolated_image)
                if success:
                    zip_file.writestr(f"isolated/{prefix}_isolated.png", buf.tobytes())

            if r.crop_image is not None:
                success, buf = cv2.imencode(".png", r.crop_image)
                if success:
                    zip_file.writestr(f"crops/{prefix}_crop.png", buf.tobytes())

            if r.segmentation is not None and r.segmentation.mask is not None:
                success, buf = cv2.imencode(".png", r.segmentation.mask)
                if success:
                    zip_file.writestr(f"masks/{prefix}_mask.png", buf.tobytes())

    return zip_buffer.getvalue()

