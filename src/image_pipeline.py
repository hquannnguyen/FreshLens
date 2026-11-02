from __future__ import annotations

import cv2
import numpy as np


def resize_for_processing(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    k = max(3, int(round(min(h, w) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _largest_component(mask)


def _mask_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / mask.size)


def _segment_blue_background(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Blue/cyan capture backgrounds. Tunable for the group's actual capture setup.
    background = cv2.inRange(hsv, np.array([82, 35, 25]), np.array([142, 255, 255]))
    foreground = cv2.bitwise_not(background)
    return _clean_mask(foreground)


def _segment_grabcut(image: np.ndarray, iterations: int = 3, max_side: int = 400) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        margin_x, margin_y = max(1, sw // 30), max(1, sh // 30)
        rect = (margin_x, margin_y, max(1, sw - 2 * margin_x), max(1, sh - 2 * margin_y))
        mask_s = np.zeros((sh, sw), dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(small, mask_s, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
        foreground_small = np.where((mask_s == cv2.GC_FGD) | (mask_s == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        cleaned_small = _clean_mask(foreground_small)
        mask = cv2.resize(cleaned_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return _clean_mask(mask)

    margin_x, margin_y = max(1, w // 30), max(1, h // 30)
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))
    mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean_mask(foreground)


def segment_tomato(image: np.ndarray, method: str = "blue") -> tuple[np.ndarray, str]:
    """Return a cleaned 0/255 mask and the method actually used."""
    image = resize_for_processing(image)
    if method == "grabcut":
        mask = _segment_grabcut(image)
        return mask, "GrabCut"

    mask = _segment_blue_background(image)
    ratio = _mask_ratio(mask)
    if ratio < 0.05 or ratio > 0.95:
        fallback = _segment_grabcut(image)
        fallback_ratio = _mask_ratio(fallback)
        if 0.05 <= fallback_ratio <= 0.95:
            return fallback, "GrabCut (tự chuyển do nền không đạt ngưỡng)"
    return mask, "Ngưỡng nền xanh + morphology"


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return image.copy(), np.zeros(image.shape[:2], np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    pad = round(max(w, h) * padding)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def isolate_on_white(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    white = np.full_like(image, 255)
    alpha = (mask > 0)[:, :, None]
    return np.where(alpha, image, white)
