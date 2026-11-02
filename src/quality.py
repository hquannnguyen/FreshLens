from __future__ import annotations

import cv2
import numpy as np

from .settings import MIN_IMAGE_SIDE


def assess_image(image_bgr: np.ndarray) -> dict:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(hsv[:, :, 2].mean())
    issues = []
    if min(h, w) < MIN_IMAGE_SIDE:
        issues.append(f"Cạnh ngắn nhất dưới {MIN_IMAGE_SIDE}px")
    if sharpness < 35:
        issues.append("Ảnh có thể bị nhòe")
    if brightness < 45:
        issues.append("Ảnh có thể quá tối")
    elif brightness > 225:
        issues.append("Ảnh có thể quá sáng")
    return {
        "width": w,
        "height": h,
        "sharpness": sharpness,
        "brightness": brightness,
        "issues": issues,
        "ok": not issues,
    }
