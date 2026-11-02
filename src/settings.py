from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "images"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "freshlens_svm.joblib"

CLASS_LABELS = {
    "green": "Xanh / chưa chín",
    "turning": "Đang chuyển màu",
    "ripe": "Đỏ / chín",
}
REVIEW_CONFIDENCE = 0.65
MIN_IMAGE_SIDE = 128
