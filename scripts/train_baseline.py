from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import color_histogram_features
from src.image_pipeline import segment_tomato
from src.settings import DATA_DIR, MODEL_DIR, MODEL_PATH, OUTPUT_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _process_single_image(args: tuple[str, str, str]) -> tuple[np.ndarray | None, str, str]:
    path_str, class_name, segmentation = args
    image = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if image is None:
        return None, class_name, f"Bỏ qua ảnh đọc lỗi: {path_str}"
    try:
        mask, _ = segment_tomato(image, segmentation)
        feat = color_histogram_features(image, mask)
        return feat, class_name, ""
    except ValueError:
        return None, class_name, f"Bỏ qua ảnh không tạo được vùng quả: {path_str}"


def load_split(split: str, segmentation: str, workers: int = 4, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Thiếu thư mục dữ liệu: {split_dir}")

    cache_file = OUTPUT_DIR / f"features_{split}_{segmentation}.npz"
    if use_cache and cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        print(f"[{split.upper()}] Tải từ bộ nhớ đệm: {cache_file} ({len(data['y'])} mẫu)")
        return data["x"], data["y"]

    items: list[tuple[str, str, str]] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            items.append((str(path), class_dir.name, segmentation))

    if not items:
        raise ValueError(f"Không có ảnh hợp lệ trong split '{split}'.")

    print(f"[{split.upper()}] Đang trích xuất đặc trưng {len(items)} ảnh bằng {workers} luồng xử lý...")
    t0 = time.time()
    x, y = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for feat, class_name, err in executor.map(_process_single_image, items):
            if err:
                print(f"  {err}")
                continue
            if feat is not None:
                x.append(feat)
                y.append(class_name)

    elapsed = time.time() - t0
    print(f"[{split.upper()}] Hoàn thành trích xuất {len(x)}/{len(items)} ảnh trong {elapsed:.1f}s.")

    if not x:
        raise ValueError(f"Không có đặc trưng hợp lệ nào được trích xuất trong split '{split}'.")

    x_arr = np.stack(x)
    y_arr = np.asarray(y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, x=x_arr, y=y_arr)
    return x_arr, y_arr


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện baseline FreshLens (HSV histogram + SVM).")
    parser.add_argument("--segmentation", choices=["blue", "grabcut"], default="grabcut")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--no-cache", action="store_true", help="Không dùng cache đặc trưng đã trích xuất.")
    parser.add_argument("--skip-test", action="store_true", help="Chỉ huấn luyện nếu chưa chuẩn bị test split.")
    args = parser.parse_args()

    use_cache = not args.no_cache
    print(f"=== BẮT ĐẦU HUẤN LUYỆN BASELINE FRESHLENS ===")
    print(f"Phương pháp phân đoạn: {args.segmentation} | Số worker: {args.workers}")

    x_train, y_train = load_split("train", args.segmentation, workers=args.workers, use_cache=use_cache)
    x_val, y_val = load_split("val", args.segmentation, workers=args.workers, use_cache=use_cache)

    # Lựa chọn siêu tham số C bằng tập validation
    c_candidates = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    best_c = 3.0
    best_val_f1 = -1.0
    best_model = None

    print("\n--- Tinh chỉnh siêu tham số trên tập Validation ---")
    val_tuning_records = []
    for c in c_candidates:
        candidate_model = make_pipeline(
            StandardScaler(),
            SVC(C=c, kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        )
        candidate_model.fit(x_train, y_train)
        # Suy luận theo argmax(predict_proba) đồng nhất với AnalysisService và dự đoán có confidence
        val_proba = candidate_model.predict_proba(x_val)
        val_pred = candidate_model.classes_[np.argmax(val_proba, axis=1)]
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        val_acc = float(accuracy_score(y_val, val_pred))
        val_tuning_records.append({"C": c, "val_macro_f1": val_f1, "val_accuracy": val_acc})
        print(f"  C={c:4.1f} -> Val Macro-F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c
            best_model = candidate_model

    print(f"\n=> Chọn siêu tham số tối ưu: C={best_c} (Val Macro-F1 = {best_val_f1:.4f})")

    # Lưu model tốt nhất
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Đã lưu mô hình: {MODEL_PATH}")
    print(f"Số mẫu train: {len(y_train)}; lớp: {', '.join(sorted(set(y_train)))}")

    metrics = {
        "classes": sorted(set(y_train)),
        "train_size": int(len(y_train)),
        "segmentation": args.segmentation,
        "best_hyperparameters": {"C": best_c, "kernel": "rbf", "class_weight": "balanced"},
        "validation_tuning": val_tuning_records,
    }

    # Đánh giá trên val và test
    eval_splits = ["val"]
    if not args.skip_test:
        eval_splits.append("test")

    for split in eval_splits:
        if split == "val":
            x_eval, y_eval = x_val, y_val
        else:
            x_eval, y_eval = load_split(split, args.segmentation, workers=args.workers, use_cache=use_cache)

        eval_proba = best_model.predict_proba(x_eval)
        pred = best_model.classes_[np.argmax(eval_proba, axis=1)]
        split_metrics = {
            "size": int(len(y_eval)),
            "accuracy": float(accuracy_score(y_eval, pred)),
            "macro_f1": float(f1_score(y_eval, pred, average="macro", zero_division=0)),
            "classification_report": classification_report(y_eval, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_eval, pred, labels=best_model.classes_).tolist(),
            "labels": best_model.classes_.tolist(),
        }
        metrics[split] = split_metrics
        print(f"\n[{split.upper()}] — accuracy={split_metrics['accuracy']:.3f}, macro-F1={split_metrics['macro_f1']:.3f}")
        print(classification_report(y_eval, pred, zero_division=0))
        print("Confusion Matrix:")
        print("Labels:", best_model.classes_)
        print(np.array(split_metrics["confusion_matrix"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "baseline_metrics.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu đánh giá hoàn chỉnh: {report_path}")


if __name__ == "__main__":
    main()
