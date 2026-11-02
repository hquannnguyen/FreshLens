"""Read-only follow-up audit of CNN application predictions against saved test results."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_cnn import CLASS_NAMES
from src.contracts import AnalysisRequest, ProcessingConfig
from src.services.analysis_service import AnalysisService


def main() -> None:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    artifact_paths = [ROOT / "models/freshlens_cnn.pt", ROOT / "outputs/cnn_metrics.json"]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths}
    recorded = json.loads((ROOT / "outputs/cnn_metrics.json").read_text(encoding="utf-8"))["test"]
    service = AnalysisService()
    cfg = ProcessingConfig(model_type="cnn", segmentation="grabcut")
    rows = []
    for label in CLASS_NAMES:
        for path in sorted((ROOT / "data/images/test" / label).glob("*.jpg")):
            result = service.analyze(
                AnalysisRequest(path.stem, path.name, cv2.imread(str(path)), "trained_model", cfg)
            )
            rows.append({
                "file": path.name,
                "truth": label,
                "service_prediction": result.prediction.label if result.prediction else None,
                "status": result.status,
            })
            if len(rows) % 30 == 0:
                print(f"Checked {len(rows)}/180", flush=True)

    for row, raw in zip(rows, recorded["y_pred"], strict=True):
        row["raw_prediction"] = raw
    accepted = [r for r in rows if r["service_prediction"] in CLASS_NAMES]
    rejected = [r for r in rows if r["service_prediction"] not in CLASS_NAMES]
    changed = [r for r in accepted if r["service_prediction"] != r["raw_prediction"]]
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths}
    report = {
        "total": len(rows),
        "accepted": len(accepted),
        "rejected": rejected,
        "changed_on_accepted": changed,
        "artifacts_unchanged": before == after,
        "artifact_sha256": after,
    }
    output = ROOT / "outputs/check_2026-11-02/fixes_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "artifact_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
