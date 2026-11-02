"""Software regression checks only; fixtures are not model evaluation data."""
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.train_cnn import get_transforms
from src.contracts import AnalysisRequest, AnalysisResult, PredictionResult, SegmentationResult
from src.features import color_histogram_features
from src.modeling import predict_from_cnn
from src.services.analysis_service import AnalysisService
from src.services.export_service import export_csv_bytes, export_zip_bundle


class ReleaseContracts(unittest.TestCase):
    def test_one_mask_is_shared_between_features_and_display(self):
        image = np.full((160, 160, 3), (25, 90, 160), np.uint8)
        mask = np.zeros((160, 160), np.uint8)
        mask[40:120, 40:120] = 255
        expected = color_histogram_features(image, mask)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.joblib"
            artifact.touch()
            service = AnalysisService(model_path=artifact)
            prediction = PredictionResult("green", 0.9, {"green": 0.9, "ripe": 0.05, "turning": 0.05})
            with patch("src.services.analysis_service.segment_tomato", return_value=(mask, "test_mask")) as segment, \
                 patch.object(service, "get_svm_model", return_value=object()), \
                 patch("src.services.analysis_service.predict_from_features", return_value=prediction) as infer:
                result = service.analyze(AnalysisRequest("check", "software_fixture.png", image, "trained_model"))
            segment.assert_called_once()
            np.testing.assert_allclose(infer.call_args.args[1], expected)
            np.testing.assert_array_equal(result.segmentation.mask, mask)
            np.testing.assert_array_equal(result.isolated_image[mask > 0], image[mask > 0])
            self.assertTrue(np.all(result.isolated_image[mask == 0] == 255))

    def test_empty_mask_does_not_get_a_default_label(self):
        image = np.zeros((160, 160, 3), np.uint8)
        with patch("src.services.analysis_service.segment_tomato", return_value=(np.zeros((160, 160), np.uint8), "empty")):
            result = AnalysisService().analyze(AnalysisRequest("bad_mask", "software_fixture.png", image, "color_preview"))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.prediction.label, "")

    def test_cnn_adapter_and_training_transform_agree_for_identical_input(self):
        class RecordingModel(torch.nn.Module):
            def forward(self, batch):
                self.seen = batch.clone()
                return torch.tensor([[1.0, 2.0, 3.0]])

        image = np.full((45, 75, 3), (20, 80, 190), np.uint8)
        model = RecordingModel().eval()
        prediction = predict_from_cnn(model, image)
        _, transform = get_transforms()
        expected = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        torch.testing.assert_close(model.seen, expected)
        self.assertEqual(prediction.label, "turning")
        self.assertAlmostEqual(sum(prediction.scores.values()), 1.0, places=6)
        self.assertAlmostEqual(prediction.confidence, prediction.scores[prediction.label])

    def test_exports_keep_failed_rows_and_duplicate_filenames_separate(self):
        image = np.zeros((12, 12, 3), np.uint8)
        mask = np.full((12, 12), 255, np.uint8)
        results = [AnalysisResult(
            image_id=ident, filename="same.jpg", status="completed", crop_image=image,
            isolated_image=image, segmentation=SegmentationResult(mask, (0, 0, 12, 12), "fixture", 1.0),
            prediction=PredictionResult("green", 0.8, {"green": 0.8, "ripe": 0.1, "turning": 0.1}, model_id="software_fixture"),
        ) for ident in ["one", "two"]]
        results.append(AnalysisResult("three", "broken.jpg", "failed", issues=["fixture error"]))
        rows = list(csv.DictReader(io.StringIO(export_csv_bytes(results).decode("utf-8-sig"))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["label"], "")
        with zipfile.ZipFile(io.BytesIO(export_zip_bundle(results))) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(len([x for x in names if x.startswith("crops/")]), 2)
            self.assertIn("batch_results.csv", names)


if __name__ == "__main__":
    unittest.main()
