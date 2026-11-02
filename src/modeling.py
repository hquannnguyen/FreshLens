from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .contracts import PredictionResult
from .features import color_histogram_features
from .image_pipeline import segment_tomato
from .settings import MODEL_DIR


CLASS_NAMES = ["green", "ripe", "turning"]


def load_model(path: Path):
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def load_manifest(manifest_path: Path | None = None) -> dict | None:
    path = manifest_path or (MODEL_DIR / "manifest.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_cnn_manifest(manifest_path: Path | None = None) -> dict | None:
    path = manifest_path or (MODEL_DIR / "manifest_cnn.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_cnn_model(model_path: Path | None = None):
    path = model_path or (MODEL_DIR / "freshlens_cnn.pt")
    if not path.exists():
        return None
    import torch
    import torchvision.models as models

    m = models.mobilenet_v2()
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(m.last_channel, len(CLASS_NAMES)),
    )
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    m.load_state_dict(state_dict)
    m.eval()
    return m


def predict_from_features(model, features: np.ndarray, model_id: str = "freshlens_svm_baseline_v1") -> PredictionResult:
    """Suy luận từ vector đặc trưng 50 chiều đã chuẩn hóa (cho SVM)."""
    feat_2d = features.reshape(1, -1) if features.ndim == 1 else features
    probabilities = model.predict_proba(feat_2d)[0]
    classes = [str(c) for c in model.classes_]
    scores = {c: float(p) for c, p in zip(classes, probabilities)}
    best_idx = int(np.argmax(probabilities))
    best_label = classes[best_idx]
    best_conf = float(probabilities[best_idx])

    return PredictionResult(
        label=best_label,
        confidence=best_conf,
        scores=scores,
        mode="trained_model",
        model_id=model_id,
    )


def predict_from_cnn(cnn_model, image_bgr: np.ndarray, model_id: str = "freshlens_cnn_mobilenetv2_v1") -> PredictionResult:
    """Suy luận trực tiếp bằng mạng nơ-ron tích chập Deep Learning (MobileNetV2)."""
    import torch
    from torchvision import transforms

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = eval_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = cnn_model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    scores = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
    best_idx = int(np.argmax(probs))
    best_label = CLASS_NAMES[best_idx]
    best_conf = float(probs[best_idx])

    return PredictionResult(
        label=best_label,
        confidence=best_conf,
        scores=scores,
        mode="trained_model",
        model_id=model_id,
    )


def predict_one(model, image_bgr: np.ndarray, mask: np.ndarray | None = None, segmentation: str = "grabcut") -> dict:
    """Hàm tương thích ngược nếu caller cũ gọi predict_one."""
    work = image_bgr.copy()
    if max(work.shape[:2]) > 1600:
        scale = 1600 / max(work.shape[:2])
        work = cv2.resize(work, (round(work.shape[1] * scale), round(work.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    if mask is None:
        mask, method = segment_tomato(work, segmentation)
    else:
        method = segmentation

    features = color_histogram_features(work, mask).reshape(1, -1)
    pred_res = predict_from_features(model, features)
    return {
        "label": pred_res.label,
        "confidence": pred_res.confidence,
        "probabilities": pred_res.scores,
        "mask": mask,
        "method": method,
    }


def build_classifier(c: float = 1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", probability=True, class_weight="balanced", random_state=42))
