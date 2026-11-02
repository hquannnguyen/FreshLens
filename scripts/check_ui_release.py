"""Reproduce UI cache/error-path checks without modifying application sources."""
from pathlib import Path
from unittest.mock import patch
import io
import json
import logging
import sys

import cv2
import torch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)
cv2.setNumThreads(1)


def button(at, label):
    return next(item for item in at.button if label in item.label)


def ids(at):
    return sorted({r.prediction.model_id for r in at.session_state["batch_results"] if r.prediction and r.prediction.model_id})


def main():
    report = {}
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    report["initial_app_exceptions"] = [item.message for item in at.exception]
    at.radio[0].set_value(at.radio[0].options[1]).run()
    button(at, "Thử nhanh").click().run()
    button(at, "Bắt đầu").click().run()
    report["initial_batch_size"] = len(at.session_state["batch_results"])
    report["initial_batch_model_ids"] = ids(at)
    model_select = next(x for x in at.selectbox if x.label == "Mô hình suy luận:")
    model_select.set_value(next(x for x in model_select.options if x.startswith("Deep Learning"))).run()
    report["batch_model_ids_after_select_cnn"] = ids(at)
    button(at, "Bắt đầu").click().run()
    report["batch_model_ids_after_explicit_rerun"] = ids(at)
    report["cnn_switch_produces_cnn_results"] = ids(at) == ["freshlens_cnn_mobilenetv2_v1"]
    button(at, "Thử nhanh").click().run()
    button(at, "Bắt đầu").click().run()
    report["batch_size_after_reload_same_count"] = len(at.session_state["batch_results"]) if "batch_results" in at.session_state else 0
    report["exceptions_after_batch_actions"] = [item.message for item in at.exception]

    import streamlit as st
    real_uploader = st.file_uploader

    class BrokenUpload(io.BytesIO):
        name = "broken.jpg"

    def uploader(*args, **kwargs):
        if kwargs.get("key") == "batch_uploader":
            return [BrokenUpload(b"not-an-image")]
        return real_uploader(*args, **kwargs)

    with patch("streamlit.file_uploader", side_effect=uploader):
        bad = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        bad.radio[0].set_value(bad.radio[0].options[1]).run()
        button(bad, "Bắt đầu").click().run()
        report["corrupt_upload_exceptions"] = [item.message for item in bad.exception]
        report["corrupt_upload_handled_without_app_crash"] = not bool(bad.exception)

    out = ROOT / "outputs/check_2026-11-02"
    out.mkdir(exist_ok=True)
    (out / "ui_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
