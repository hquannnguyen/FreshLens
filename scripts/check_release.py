"""Audit existing artifacts without training or overwriting checkpoints/metrics.

Run from project root: python scripts/check_release.py
Report: outputs/check_2026-11-02/checkpoint_check.json
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cnn import CLASS_NAMES, evaluate, get_transforms, load_dataset
from src.contracts import AnalysisRequest, ProcessingConfig
from src.modeling import load_cnn_model
from src.services.analysis_service import AnalysisService


def supporting_checks(out):
    import joblib

    report = {"dataset_counts": {}, "svm_prediction_rule": {}}
    with (ROOT / "data/annotations.csv").open(encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    for split in ["train", "val", "test"]:
        selected = [row for row in rows if row["split"] == split]
        report["dataset_counts"][split] = {
            "images": len(selected), "classes": dict(Counter(row["class"] for row in selected)),
            "raw_image_groups": len({row["session_id"] for row in selected}),
        }
    model = joblib.load(ROOT / "models/freshlens_svm.joblib")
    for split in ["val", "test"]:
        data = np.load(ROOT / f"outputs/features_{split}_grabcut.npz", allow_pickle=True)
        predicted = model.predict(data["x"])
        by_probability = model.classes_[np.argmax(model.predict_proba(data["x"]), axis=1)]
        report["svm_prediction_rule"][split] = {
            "samples": len(predicted),
            "predict_accuracy": accuracy_score(data["y"], predicted),
            "predict_macro_f1": f1_score(data["y"], predicted, average="macro"),
            "proba_argmax_accuracy": accuracy_score(data["y"], by_probability),
            "proba_argmax_macro_f1": f1_score(data["y"], by_probability, average="macro"),
            "changed_predictions": int(np.sum(predicted != by_probability)),
        }
    report["artifact_bytes"] = {path.name: path.stat().st_size for path in (ROOT / "models").iterdir() if path.is_file() and not path.name.startswith(".")}
    report["declared_torch_dependencies"] = [line for line in (ROOT / "requirements.txt").read_text().splitlines() if line.startswith(("torch", "torchvision"))]
    report["cnn_report_has_training_history"] = bool(json.loads((ROOT / "outputs/cnn_metrics.json").read_text()).get("history"))
    (out / "supporting_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    out = ROOT / "outputs" / "check_2026-11-02"
    out.mkdir(exist_ok=True)
    supporting_checks(out)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    report = {"kind": "read_only_checkpoint_audit", "cpu_threads": 4, "opencv_rng_seed": 42}

    def save():
        (out / "checkpoint_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    report["sha256_before"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["models/freshlens_cnn.pt", "models/manifest_cnn.json", "outputs/cnn_metrics.json", "models/freshlens_svm.joblib"]
    }
    recorded = json.loads((ROOT / "outputs/cnn_metrics.json").read_text(encoding="utf-8"))
    report["training_history_present"] = bool(recorded.get("history"))
    report["best_epoch_in_metrics"] = recorded.get("best_epoch")
    model = load_cnn_model()
    _, transform = get_transforms()
    report["raw_cnn"] = {}
    raw_predictions = {}
    for split in ["val", "test"]:
        images, labels = load_dataset(split, transform)
        start = time.perf_counter()
        measured = evaluate(model, images, labels, transform, torch.device("cpu"))
        raw_predictions[split] = measured["y_pred"]
        report["raw_cnn"][split] = {
            "samples": len(labels), "accuracy": measured["accuracy"], "macro_f1": measured["macro_f1"],
            "confusion_matrix": measured["confusion_matrix"],
            "matches_recorded_predictions": measured["y_pred"] == recorded[split]["y_pred"],
            "matches_recorded_metrics": bool(np.isclose(measured["macro_f1"], recorded[split]["macro_f1"]) and np.isclose(measured["accuracy"], recorded[split]["accuracy"])),
            "elapsed_seconds": time.perf_counter() - start,
        }
        print("RAW", split, report["raw_cnn"][split], flush=True)
        save()

    service = AnalysisService()
    service.get_cnn_model()
    config = ProcessingConfig(model_type="cnn", segmentation="grabcut")
    truth, predictions, rows = [], [], []
    for cls in CLASS_NAMES:
        for path in sorted((ROOT / "data/images/test" / cls).glob("*.jpg")):
            image = cv2.imread(str(path))
            try:
                result = service.analyze(AnalysisRequest(path.stem, path.name, image, "trained_model", config))
                pred = result.prediction.label if result.prediction and result.prediction.label else "__rejected__"
                row = {"filename": path.name, "truth": cls, "prediction": pred, "status": result.status,
                       "confidence": result.prediction.confidence if result.prediction else None,
                       "mask_ratio": result.segmentation.foreground_ratio if result.segmentation else None,
                       "duration_ms": result.duration_ms}
            except Exception as exc:
                pred = "__failed__"
                row = {"filename": path.name, "truth": cls, "prediction": pred, "status": "failed", "error": repr(exc)}
            truth.append(cls)
            predictions.append(pred)
            rows.append(row)
            if len(rows) % 30 == 0:
                print("SERVICE", len(rows), "/ 180", flush=True)

    accepted = [i for i, pred in enumerate(predictions) if pred in CLASS_NAMES]
    report["application_cnn"] = {
        "samples": len(rows), "accepted": len(accepted),
        "rejected": sum(x == "__rejected__" for x in predictions),
        "failed": sum(x == "__failed__" for x in predictions),
        "accuracy_including_rejections_as_incorrect": sum(a == b for a, b in zip(truth, predictions)) / len(rows),
        "conditional_accuracy_on_accepted": accuracy_score([truth[i] for i in accepted], [predictions[i] for i in accepted]),
        "macro_f1_on_accepted": f1_score([truth[i] for i in accepted], [predictions[i] for i in accepted], labels=CLASS_NAMES, average="macro", zero_division=0),
        "raw_vs_application_changed_predictions_on_accepted": sum(predictions[i] != raw_predictions["test"][i] for i in accepted),
        "confusion_matrix_with_rejected_and_failed": confusion_matrix(truth, predictions, labels=CLASS_NAMES + ["__rejected__", "__failed__"]).tolist(),
        "labels": CLASS_NAMES + ["__rejected__", "__failed__"],
        "rows": rows,
        "confidence_flag_violations": [row["filename"] for row in rows if row.get("confidence") is not None and row["confidence"] < 0.65 and row["status"] != "needs_review"],
    }
    report["artifacts_unchanged"] = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum for name, checksum in report["sha256_before"].items())
    save()
    print("APPLICATION", {k: v for k, v in report["application_cnn"].items() if k != "rows"}, flush=True)
    print("ARTIFACTS_UNCHANGED", report["artifacts_unchanged"], flush=True)


if __name__ == "__main__":
    main()
