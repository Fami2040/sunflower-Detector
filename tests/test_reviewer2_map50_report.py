"""Unit tests for reviewer2 mAP50 aggregate (no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class Reviewer2Map50ReportTests(unittest.TestCase):
    def test_build_cross_check_ok(self) -> None:
        from harchoc.reviewer2_map50_report import build_reviewer2_map50_computed

        m50 = 0.18028741415651692
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hsp = root / "reports/hsp"
            hsp.mkdir(parents=True)
            (hsp / "eval_test_map.json").write_text(
                json.dumps(
                    {
                        "mAP50": m50,
                        "mAP50_95": 0.06,
                        "export_only": False,
                        "imgsz": 1280,
                        "max_det": 3000,
                        "split_source": {"path": "data/splits/test.txt"},
                    }
                ),
                encoding="utf-8",
            )
            (hsp / "dual_metric.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "split": "val",
                                "counting": {"mae": 71.05},
                            },
                            {
                                "split": "test",
                                "detection": {"mAP50": m50, "mAP50_95": 0.06},
                                "counting": {
                                    "mae": 61.266,
                                    "rrmse": 0.1485,
                                    "n_images": 109,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (hsp / "eval_val.json").write_text(
                json.dumps({"export_only": True, "mAP50": None}),
                encoding="utf-8",
            )
            (hsp / "eval_test.json").write_text(
                json.dumps({"export_only": True, "mAP50": None}),
                encoding="utf-8",
            )

            payload = build_reviewer2_map50_computed(
                repo_root=root,
                eval_test_map="reports/hsp/eval_test_map.json",
                dual_metric="reports/hsp/dual_metric.json",
                eval_val="reports/hsp/eval_val.json",
                eval_test="reports/hsp/eval_test.json",
            )

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["cross_checks"]["dual_metric_test_map50_matches_eval_test_map"])
        self.assertEqual(payload["hsp_canonical"]["mAP50_display"], "0.180")
        self.assertEqual(payload["counting_at_locked_conf"]["test_mae_display"], 61.3)
        self.assertEqual(
            payload["val_test_gap_narrative"]["held_out_test_ranking_mAP50_hsp"],
            0.18,
        )
