"""Tests for Grad-CAM panel status and error recording."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class GradcamPanelStatusTests(unittest.TestCase):
    def test_partial_when_weights_but_no_overlays(self) -> None:
        from harchoc.gradcam_panel import render_gradcam_mosaic

        with tempfile.TemporaryDirectory() as td:
            crop = Path(td) / "crop.png"
            try:
                from PIL import Image  # type: ignore

                Image.new("RGB", (8, 8), color=(128, 64, 32)).save(crop)
            except Exception:
                self.skipTest("PIL unavailable")
            entries = [
                {
                    "error_type": "background",
                    "score": 0.5,
                    "crop_path": str(crop),
                    "image_path": str(crop),
                    "bbox": [0, 0, 8, 8],
                }
            ]
            out = Path(td) / "mosaic.png"
            with mock.patch(
                "harchoc.gradcam_panel._try_gradcam_overlay",
                return_value=False,
            ):
                result = render_gradcam_mosaic(
                    entries=entries,
                    out_path=out,
                    weights="models/fake.pt",
                )
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["gradcam_overlays"], 0)
            self.assertTrue(out.is_file())

    def test_overlay_failure_records_error(self) -> None:
        from harchoc.gradcam_panel import render_gradcam_mosaic

        with tempfile.TemporaryDirectory() as td:
            crop = Path(td) / "crop.png"
            try:
                from PIL import Image  # type: ignore

                Image.new("RGB", (8, 8), color=(64, 128, 32)).save(crop)
            except Exception:
                self.skipTest("PIL unavailable")
            entries = [
                {
                    "error_type": "localization",
                    "score": 0.7,
                    "crop_path": str(crop),
                    "image_path": str(crop),
                    "bbox": [0, 0, 8, 8],
                }
            ]
            out = Path(td) / "mosaic.png"

            def _fail(*_a: object, **_k: object) -> bool:
                raise RuntimeError("element 0 of tensors does not require grad")

            with mock.patch("harchoc.gradcam_panel._try_gradcam_overlay", side_effect=_fail):
                result = render_gradcam_mosaic(
                    entries=entries,
                    out_path=out,
                    weights="models/fake.pt",
                )
            self.assertEqual(result["status"], "partial")
            errors = result.get("gradcam_errors") or []
            self.assertGreaterEqual(len(errors), 1)
            self.assertEqual(errors[0]["panel_index"], 0)
            self.assertEqual(errors[0]["error_type"], "RuntimeError")
            self.assertIn("require grad", errors[0]["message"])

    def test_ok_without_weights(self) -> None:
        from harchoc.gradcam_panel import render_gradcam_mosaic

        with tempfile.TemporaryDirectory() as td:
            crop = Path(td) / "crop.png"
            try:
                from PIL import Image  # type: ignore

                Image.new("RGB", (4, 4)).save(crop)
            except Exception:
                self.skipTest("PIL unavailable")
            result = render_gradcam_mosaic(
                entries=[{"error_type": "fp", "score": 0.1, "crop_path": str(crop)}],
                out_path=Path(td) / "m.png",
                weights=None,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["gradcam_overlays"], 0)
            self.assertEqual(result.get("gradcam_errors"), [])

    def test_errors_capped_at_five(self) -> None:
        from harchoc.strict_ml import append_ml_error

        errors: list[dict[str, object]] = []
        for i in range(8):
            append_ml_error(errors, panel_index=i, exc=ValueError(f"e{i}"))
        self.assertEqual(len(errors), 5)
        self.assertEqual(errors[-1]["panel_index"], 4)


if __name__ == "__main__":
    unittest.main()
