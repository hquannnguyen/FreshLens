"""Reproduce the same-filename/different-content batch cache collision with AppTest."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from unittest.mock import patch

import cv2
import torch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)
cv2.setNumThreads(1)


def main() -> None:
    green = sorted((ROOT / "data/images/test/green").glob("*.jpg"))[0].read_bytes()
    ripe = sorted((ROOT / "data/images/test/ripe").glob("*.jpg"))[0].read_bytes()
    current = {"content": green}

    class Upload(io.BytesIO):
        name = "same_filename.jpg"

    import streamlit as st
    original = st.file_uploader

    def uploader(*args, **kwargs):
        if kwargs.get("key") == "batch_uploader":
            return [Upload(current["content"])]
        return original(*args, **kwargs)

    with patch("streamlit.file_uploader", side_effect=uploader):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        at.radio[0].set_value(at.radio[0].options[1]).run()
        next(x for x in at.button if "Bắt đầu" in x.label).click().run()
        first = at.session_state["batch_results"][0]
        current["content"] = ripe
        at.run()
        second = at.session_state["batch_results"][0]

    report = {
        "different_input_bytes": green != ripe,
        "same_filename": True,
        "result_id_before": first.image_id,
        "result_id_after": second.image_id,
        "result_reused_after_content_change": first is second,
        "exceptions": [x.message for x in at.exception],
    }
    path = ROOT / "outputs/check_2026-11-02/batch_content_cache_check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
