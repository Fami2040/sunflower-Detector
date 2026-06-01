import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FpBudgetSweepTests(unittest.TestCase):
    def _tiny_gt_preds(self) -> tuple[dict, dict]:
        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [20, 20, 30, 30], "category_id": 0},
                    ],
                },
                {
                    "image_id": "img2",
                    "file_name": "/abs/img2.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.90},
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.80},
                        {"bbox": [50, 50, 60, 60], "category_id": 0, "score": 0.70},
                    ],
                },
                {
                    "image_id": "img2",
                    "file_name": "/abs/img2.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.60},
                        {"bbox": [15, 15, 25, 25], "category_id": 0, "score": 0.55},
                    ],
                },
            ]
        }
        return gt, preds

    def _sweep_rows(self) -> list[dict]:
        return [
            {"conf_thr": 0.10, "tp": 3, "fp": 3, "fn": 0, "precision": 0.5, "recall": 1.0, "f1": 0.667, "fp_per_image": 1.5},
            {"conf_thr": 0.20, "tp": 3, "fp": 2, "fn": 0, "precision": 0.6, "recall": 1.0, "f1": 0.75, "fp_per_image": 1.0},
            {"conf_thr": 0.30, "tp": 3, "fp": 1, "fn": 0, "precision": 0.75, "recall": 1.0, "f1": 0.857, "fp_per_image": 0.5},
            {"conf_thr": 0.85, "tp": 2, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.667, "f1": 0.8, "fp_per_image": 0.0},
        ]

    def test_build_fp_budget_sweep_payload_schema(self) -> None:
        from harchoc.fp_budget_sweep import FP_BUDGET_SWEEP_SCHEMA, build_fp_budget_sweep_payload

        gt, preds = self._tiny_gt_preds()
        rows = self._sweep_rows()
        payload = build_fp_budget_sweep_payload(
            gt=gt,
            preds=preds,
            rows=rows,
            iou_thr=0.5,
            category_aware=True,
            n_images=2,
            fp_budget_grid=[0.5, 1.0, 2.0],
        )
        self.assertEqual(payload["schema_version"], FP_BUDGET_SWEEP_SCHEMA)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["selection_comparison"]), 2)
        self.assertEqual(payload["selection_comparison"][0]["mode"], "min_count_mae")
        self.assertEqual(payload["selection_comparison"][1]["mode"], "best_f1")
        self.assertEqual(len(payload["fp_budget_grid"]), 3)
        self.assertTrue(payload["fp_budget_grid"][0]["feasible"])
        self.assertTrue(payload["fp_budget_grid"][1]["feasible"])
        self.assertTrue(payload["fp_budget_grid"][2]["feasible"])

    def test_constraints_picks_highest_f1_under_cap(self) -> None:
        from harchoc.fp_budget_sweep import build_fp_budget_sweep_payload

        gt, preds = self._tiny_gt_preds()
        payload = build_fp_budget_sweep_payload(
            gt=gt,
            preds=preds,
            rows=self._sweep_rows(),
            iou_thr=0.5,
            category_aware=True,
            n_images=2,
            fp_budget_grid=[1.0],
        )
        entry = payload["fp_budget_grid"][0]
        self.assertTrue(entry["feasible"])
        sel = entry["selected"]
        assert sel is not None
        self.assertAlmostEqual(float(sel["conf_thr"]), 0.30)
        self.assertAlmostEqual(float(sel["fp_per_image"]), 0.5)

    def test_load_sweep_rows_and_match(self) -> None:
        from harchoc.fp_budget_sweep import load_sweep_rows_and_match

        doc = {
            "schema_version": "threshold_sweep_run.v1",
            "match": {"iou": 0.3, "category_aware": True},
            "rows": self._sweep_rows(),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sweep.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            rows, iou, cat, _ = load_sweep_rows_and_match(path)
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(iou, 0.3)
        self.assertTrue(cat)

    def test_run_fp_budget_sweep_light_dry_run(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fp_budget_sweep.json"
            rc = main(["fp-budget-sweep", "--light", "--dry-run", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "fp_budget_sweep.v1")
            self.assertEqual(obj["status"], "dry-run")

    def test_run_fp_budget_sweep_light_real(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fp_budget_sweep.json"
            rc = main(
                [
                    "fp-budget-sweep",
                    "--light",
                    "--out",
                    str(out),
                    "--fp-budget-grid",
                    "0.5",
                    "--fp-budget-grid",
                    "1.5",
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["status"], "ok")
            self.assertEqual(len(obj["fp_budget_grid"]), 2)
            self.assertIn("count_mae", obj["selection_comparison"][0]["selected"])

    def test_experiment_config_fp_budget_sweep(self) -> None:
        from harchoc.experiment_config import load_config, script_section_from_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "configs/experiments/fp_budget_sweep.json")
        section = script_section_from_config(cfg, "fp_budget_sweep")
        self.assertEqual(section.get("out"), "reports/hsp/fp_budget_sweep.json")
        self.assertEqual(section.get("sweep_from"), "reports/hsp/threshold_val.json")

    def test_experiment_config_fp_budget_sweep_test(self) -> None:
        from harchoc.experiment_config import load_config, script_section_from_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "configs/experiments/fp_budget_sweep_test.json")
        section = script_section_from_config(cfg, "fp_budget_sweep")
        self.assertEqual(section.get("out"), "reports/hsp/fp_budget_sweep_test.json")
        self.assertEqual(section.get("sweep_from"), "reports/hsp/threshold_test_locked.json")
        self.assertEqual(section.get("locked_conf_from"), "reports/hsp/threshold_test_locked.json")
        self.assertEqual(section.get("summary_out"), "reports/hsp/fp_budget_sweep_test.md")

    def test_format_fp_budget_manuscript_md(self) -> None:
        from harchoc.fp_budget_sweep import build_fp_budget_sweep_payload, format_fp_budget_manuscript_md

        gt, preds = self._tiny_gt_preds()
        payload = build_fp_budget_sweep_payload(
            gt=gt,
            preds=preds,
            rows=self._sweep_rows(),
            iou_thr=0.5,
            category_aware=True,
            n_images=2,
            fp_budget_grid=[1.0],
            split_role="test",
        )
        locked = {"conf_thr": 0.30, "fp_per_image": 0.5, "count_mae": 0.5, "f1": 0.857}
        md = format_fp_budget_manuscript_md(payload, locked_row=locked, title="Test sweep")
        self.assertIn("# Test sweep", md)
        self.assertIn("min_count_mae", md)
        self.assertIn("Primary (locked)", md)
        self.assertIn("Constraint grid", md)

    def test_threshold_sweep_writes_fp_budget_sidecar(self) -> None:
        from scripts.threshold_sweep import main

        gt, preds = self._tiny_gt_preds()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            gt_path = td_path / "gt_val.json"
            preds_path = td_path / "preds_val.json"
            sweep_out = td_path / "sweep.json"
            fp_out = td_path / "fp_budget_sweep.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
            with patch("scripts.threshold_sweep.enforce_tuning_guardrails"):
                rc = main(
                    [
                        "--dataset-root",
                        str(td_path),
                        "--gt-json",
                        str(gt_path),
                        "--preds-json",
                        str(preds_path),
                        "--out",
                        str(sweep_out),
                        "--steps",
                        "4",
                        "--min",
                        "0.1",
                        "--max",
                        "0.4",
                        "--fp-budget-sweep-out",
                        str(fp_out),
                        "--fp-budget-grid",
                        "0.5",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(fp_out.is_file())
            obj = json.loads(fp_out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "fp_budget_sweep.v1")
            self.assertEqual(len(obj["fp_budget_grid"]), 1)


if __name__ == "__main__":
    unittest.main()
