"""Tests for reviewer2 §11 confusion + TIDE audit (CI-safe)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class Reviewer2ConfusionTideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_quiet = os.environ.get("HARCHOC_QUIET")
        os.environ["HARCHOC_QUIET"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._old_quiet is None:
            os.environ.pop("HARCHOC_QUIET", None)
        else:
            os.environ["HARCHOC_QUIET"] = cls._old_quiet

    def test_build_payload_parity(self) -> None:
        from harchoc.reviewer2_confusion_tide import build_reviewer2_confusion_tide_payload

        gt = {
            "images": [
                {
                    "image_id": "a",
                    "file_name": "a.jpg",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [20, 20, 30, 30], "category_id": 1},
                    ],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "a",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9},
                        {"bbox": [0, 0, 10, 10], "category_id": 1, "score": 0.8},
                        {"bbox": [50, 50, 60, 60], "category_id": 0, "score": 0.6},
                    ],
                }
            ]
        }
        payload = build_reviewer2_confusion_tide_payload(
            gt=gt, preds=preds, conf_thr=0.15
        )
        counts = payload["recomputed"]["counts"]
        self.assertTrue(payload["parity"]["confusion_iou_0_5_matches_error_counts"])
        self.assertEqual(
            payload["recomputed"]["derived_metrics"]["matrix_off_diagonal_cls"],
            counts["cls_confusion"],
        )
        self.assertEqual(payload["protocol"]["section11"]["iou"], 0.5)
        self.assertEqual(payload["recomputed"]["confusion_iou_0_3"]["match"]["iou"], 0.3)

    def test_experiment_dry_run(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "reviewer2.json"
            rc = main(["--dry-run", "reviewer2-confusion", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(obj.get("status"), "dry-run")
            self.assertIn("reproduce", obj)

    def test_experiment_config_merge(self) -> None:
        from scripts.experiment import main

        repo = Path(__file__).resolve().parents[1]
        cfg = repo / "configs/experiments/reviewer2_confusion_tide.json"
        if not cfg.is_file():
            self.skipTest("reviewer2_confusion_tide.json missing")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "reviewer2.json"
            rc = main(
                [
                    "--config",
                    str(cfg),
                    "--dry-run",
                    "reviewer2-confusion",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                obj.get("inputs", {}).get("gt_json"),
                "reports/hsp/gt_test.json",
            )


if __name__ == "__main__":
    unittest.main()
