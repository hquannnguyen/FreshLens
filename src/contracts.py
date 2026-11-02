from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ProcessingConfig:
    max_side: int = 1600
    segmentation: str = "grabcut"
    model_type: str = "svm"  # 'svm' or 'cnn'
    feature_version: str = "hsv_hist_18_16_16_v1"
    min_confidence: float = 0.65
    min_side: int = 128
    min_mask_ratio: float = 0.05
    max_mask_ratio: float = 0.95
    min_brightness: float = 45.0
    max_brightness: float = 225.0
    min_sharpness: float = 35.0


@dataclass
class QualityReport:
    width: int
    height: int
    brightness: float
    sharpness: float
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


@dataclass
class SegmentationResult:
    mask: np.ndarray  # uint8 0/255
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h)
    method_used: str
    foreground_ratio: float
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return 0.05 <= self.foreground_ratio <= 0.95 and np.any(self.mask)


@dataclass
class PredictionResult:
    label: str  # 'green', 'turning', 'ripe', or ''
    confidence: float | None
    scores: dict[str, float] = field(default_factory=dict)
    mode: str = "trained_model"  # 'trained_model' | 'color_preview'
    model_id: str | None = None


@dataclass
class AnalysisRequest:
    image_id: str
    filename: str
    image_bgr: np.ndarray
    mode: str = "auto"  # 'auto', 'trained_model', 'color_preview'
    config: ProcessingConfig = field(default_factory=ProcessingConfig)


@dataclass
class AnalysisResult:
    image_id: str
    filename: str
    status: str  # 'completed' | 'needs_review' | 'rejected' | 'failed'
    quality: QualityReport | None = None
    segmentation: SegmentationResult | None = None
    prediction: PredictionResult | None = None
    color_ratios: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    issues: list[str] = field(default_factory=list)
    work_image: np.ndarray | None = None
    isolated_image: np.ndarray | None = None
    crop_image: np.ndarray | None = None
    crop_mask: np.ndarray | None = None

