from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import color_histogram_features
from src.image_pipeline import resize_for_processing, segment_tomato
from src.modeling import load_cnn_model
from src.settings import MODEL_DIR, OUTPUT_DIR

MAX_SIDE = 1600  # trùng cfg.max_side mặc định của AnalysisService


def _stats(values_ms: list[float]) -> dict:
    arr = np.asarray(values_ms, dtype=float)
    return {
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def main() -> None:
    svm_path = MODEL_DIR / "freshlens_svm.joblib"
    cnn_path = MODEL_DIR / "freshlens_cnn.pt"

    if not svm_path.exists() or not cnn_path.exists():
        raise FileNotFoundError("Thiếu artifact mô hình trong models/.")

    svm_size_bytes = svm_path.stat().st_size
    cnn_size_bytes = cnn_path.stat().st_size

    svm_model = joblib.load(svm_path)
    cnn_model = load_cnn_model(cnn_path)

    test_images = sorted((ROOT / "data/images/test").rglob("*.jpg"))
    if not test_images:
        raise FileNotFoundError("Không tìm thấy ảnh test trong data/images/test.")

    # Transform CNN trùng với eval_transform lúc huấn luyện
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Warmup để loại bỏ chi phí nạp lần đầu
    dummy = cv2.imread(str(test_images[0]))
    work = resize_for_processing(dummy, max_side=MAX_SIDE)
    mask, _ = segment_tomato(work, "grabcut")
    svm_model.predict_proba(color_histogram_features(work, mask).reshape(1, -1))
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        cnn_model(eval_transform(Image.fromarray(rgb)).unsqueeze(0))

    # Bộ đo thời gian từng thành phần
    svm_resize_ms, svm_seg_ms, svm_feat_ms, svm_infer_ms, svm_total_ms = [], [], [], [], []
    cnn_pre_ms, cnn_fwd_ms, cnn_total_ms, cnn_pipeline_ms = [], [], [], []

    for p in test_images:
        img = cv2.imread(str(p))
        if img is None:
            continue

        # --- Pipeline SVM (trùng AnalysisService: normalize 1 lần -> GrabCut -> HSV -> predict) ---
        t0 = time.perf_counter()
        work = resize_for_processing(img, max_side=MAX_SIDE)
        svm_resize_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        mask, _ = segment_tomato(work, "grabcut")
        svm_seg_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        feat = color_histogram_features(work, mask).reshape(1, -1)
        svm_feat_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        svm_model.predict_proba(feat)
        svm_infer_ms.append((time.perf_counter() - t0) * 1000.0)

        svm_total_ms.append(svm_resize_ms[-1] + svm_seg_ms[-1] + svm_feat_ms[-1] + svm_infer_ms[-1])

        # --- Pipeline CNN (trùng AnalysisService: normalize + GrabCut cho hiển thị + transform + forward) ---
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        tensor = eval_transform(Image.fromarray(rgb)).unsqueeze(0)
        cnn_pre_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = cnn_model(tensor)
            torch.softmax(out, dim=1)
        cnn_fwd_ms.append((time.perf_counter() - t0) * 1000.0)

        cnn_total_ms.append(cnn_pre_ms[-1] + cnn_fwd_ms[-1])
        cnn_pipeline_ms.append(svm_resize_ms[-1] + svm_seg_ms[-1] + cnn_pre_ms[-1] + cnn_fwd_ms[-1])

    benchmark_results = {
        "num_test_samples": len(svm_total_ms),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "Đo trên CPU, toàn bộ data/images/test, xử lý tuần tự từng ảnh.",
            "SVM: total_pipeline_ms gồm resize + GrabCut + trích xuất HSV + predict_proba (đúng luồng AnalysisService).",
            "CNN: total_pipeline_ms gồm resize + GrabCut (bắt buộc cho hiển thị/tách nền) + transform + forward + softmax.",
            "Độ trễ thuần mô hình (pure) KHÔNG bao gồm phân vùng; tổng thể (pipeline) đã bao gồm.",
        ],
        "svm": {
            "model_path": "models/freshlens_svm.joblib",
            "size_bytes": svm_size_bytes,
            "size_kb": round(svm_size_bytes / 1024, 2),
            "resize_ms": _stats(svm_resize_ms),
            "segmentation_grabcut_ms": _stats(svm_seg_ms),
            "feature_extraction_ms": _stats(svm_feat_ms),
            "pure_inference_ms": _stats(svm_infer_ms),
            "total_pipeline_ms": _stats(svm_total_ms),
        },
        "cnn": {
            "model_path": "models/freshlens_cnn.pt",
            "size_bytes": cnn_size_bytes,
            "size_mb": round(cnn_size_bytes / (1024 * 1024), 2),
            "preprocessing_transform_ms": _stats(cnn_pre_ms),
            "pure_forward_pass_ms": _stats(cnn_fwd_ms),
            "total_model_ms": _stats(cnn_total_ms),
            "total_pipeline_ms": _stats(cnn_pipeline_ms),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "inference_benchmark.json"
    out_file.write_text(json.dumps(benchmark_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {out_file.relative_to(ROOT)}:")
    print(json.dumps(benchmark_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
