from __future__ import annotations

import cv2
import numpy as np


def color_histogram_features(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalized HSV histograms for the segmented fruit region."""
    if not np.any(mask):
        raise ValueError("Mặt nạ rỗng, không trích được đặc trưng.")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bins = ((0, 180, 18), (0, 256, 16), (0, 256, 16))
    parts = []
    for channel, (low, high, count) in enumerate(bins):
        hist = cv2.calcHist([hsv], [channel], mask, [count], [low, high]).flatten()
        total = float(hist.sum())
        parts.append(hist / total if total else hist)
    return np.concatenate(parts).astype(np.float32)


def visible_color_ratios(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Simple color evidence for a non-model preview; not a probability estimate."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    valid = mask > 0
    if not np.any(valid):
        return {"red": 0.0, "green": 0.0, "other": 0.0}
    hue, sat, val = cv2.split(hsv)
    colorful = valid & (sat >= 45) & (val >= 35)
    n = int(np.count_nonzero(colorful))
    if n == 0:
        return {"red": 0.0, "green": 0.0, "other": 1.0}
    red = colorful & ((hue <= 12) | (hue >= 168))
    green = colorful & (hue >= 30) & (hue <= 88)
    return {
        "red": float(np.count_nonzero(red) / n),
        "green": float(np.count_nonzero(green) / n),
        "other": float(max(0.0, 1.0 - np.count_nonzero(red) / n - np.count_nonzero(green) / n)),
    }
