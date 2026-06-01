from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _gt_doc() -> dict:
    return {
        "images": [
            {
                "image_id": "a",
                "annotations": [
                    {"bbox": [0, 0, 10, 10], "category_id": 0},
                    {"bbox": [20, 20, 30, 30], "category_id": 1},
                ],
            },
            {
                "image_id": "b",
                "annotations": [
                    {"bbox": [0, 0, 5, 5], "category_id": 0},
                ],
            },
        ]
    }


def _pred_doc(*, scores: list[float]) -> dict:
    return {
        "images": [
            {
                "image_id": "a",
                "detections": [
                    {"bbox": [0, 0, 10, 10], "category_id": 0, "score": scores[0]},
                    {"bbox": [20, 20, 30, 30], "category_id": 1, "score": scores[1]},
                ],
            },
            {
                "image_id": "b",
                "detections": [
                    {"bbox": [0, 0, 5, 5], "category_id": 0, "score": scores[2]},
                ],
            },
        ]
    }


class ReviewerCountingReportTests(unittest.TestCase):
    def test_per_class_totals_and_relative_error(self) -> None:
        from harchoc.reviewer_counting_report import (
            counting_block_for_pair,
            per_class_detection_totals,
            per_image_relative_error_summary,
        )
        from harchoc.detection_match import per_image_detection_counts

        gt = _gt_doc()
        preds = _pred_doc(scores=[0.9, 0.9, 0.9])
        per = per_image_detection_counts(gt=gt, preds=preds, conf_thr=0.5)
        rel = per_image_relative_error_summary(per)
        self.assertEqual(rel["mean"], 0.0)
        totals = per_class_detection_totals(gt=gt, preds=preds, conf_thr=0.5)
        self.assertEqual(totals["developed"]["n_gt"], 2)
        self.assertEqual(totals["aborted"]["n_gt"], 1)
        self.assertEqual(totals["developed"]["n_pred"], 2)
        block = counting_block_for_pair(gt=gt, preds=preds, conf_thr=0.5)
        self.assertEqual(block["pooled"]["mae"], 0.0)
        self.assertEqual(block["n_images"], 2)

    def test_build_report_from_paths(self) -> None:
        from harchoc.reviewer_counting_report import build_reviewer_counting_report

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            gt_path = base / "gt.json"
            pr_path = base / "preds.json"
            gt_path.write_text(json.dumps(_gt_doc()), encoding="utf-8")
            pr_path.write_text(json.dumps(_pred_doc(scores=[0.9, 0.9, 0.9])), encoding="utf-8")
            sweep = base / "sweep.json"
            sweep.write_text(
                json.dumps(
                    {
                        "schema_version": "threshold_sweep_run.v1",
                        "selected": {"row": {"conf_thr": 0.5}},
                        "match": {"iou": 0.3},
                    }
                ),
                encoding="utf-8",
            )
            out = build_reviewer_counting_report(
                locked_conf=0.5,
                locked_conf_from=str(sweep),
                test_gt_path=str(gt_path),
                test_preds_path=str(pr_path),
            )
            self.assertEqual(out["schema_version"], "reviewer_counting_report.v1")
            self.assertEqual(out["pooled"]["mae"], 0.0)
            self.assertIn("developed", out["per_class_totals"])

    def test_experiment_reviewer_counting_dry_run(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "reviewer.json"
            rc = main(
                [
                    "--dry-run",
                    "reviewer-counting",
                    "--out",
                    str(out),
                    "--gt-test",
                    str(Path(td) / "missing_gt.json"),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(obj.get("status"), "dry-run")
