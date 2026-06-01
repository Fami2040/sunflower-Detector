import tempfile
import unittest
from pathlib import Path


class ThresholdSweepSelectionTests(unittest.TestCase):
    def test_best_f1_tie_breaks_lower_threshold(self) -> None:
        from scripts.threshold_sweep import select_operating_point

        rows = [
            {"conf_thr": 0.20, "f1": 0.50, "precision": 0.60, "recall": 0.45, "fp_per_image": 0.10},
            {"conf_thr": 0.10, "f1": 0.50, "precision": 0.55, "recall": 0.50, "fp_per_image": 0.20},
            {"conf_thr": 0.30, "f1": 0.49, "precision": 0.70, "recall": 0.40, "fp_per_image": 0.05},
        ]
        sel = select_operating_point(rows, mode="best_f1")
        self.assertIsNotNone(sel)
        assert sel is not None
        self.assertAlmostEqual(float(sel["conf_thr"]), 0.10)

    def test_constraints_min_recall_then_best_f1(self) -> None:
        from scripts.threshold_sweep import select_operating_point

        rows = [
            {"conf_thr": 0.10, "f1": 0.40, "precision": 0.30, "recall": 0.90, "fp_per_image": 0.80},
            {"conf_thr": 0.20, "f1": 0.60, "precision": 0.60, "recall": 0.70, "fp_per_image": 0.40},
            {"conf_thr": 0.30, "f1": 0.65, "precision": 0.80, "recall": 0.60, "fp_per_image": 0.20},
        ]
        sel = select_operating_point(rows, mode="constraints", min_recall=0.65)
        self.assertIsNotNone(sel)
        assert sel is not None
        self.assertAlmostEqual(float(sel["conf_thr"]), 0.20)

    def test_constraints_max_fp_per_image(self) -> None:
        from scripts.threshold_sweep import select_operating_point

        rows = [
            {"conf_thr": 0.10, "f1": 0.70, "precision": 0.50, "recall": 0.90, "fp_per_image": 1.50},
            {"conf_thr": 0.20, "f1": 0.60, "precision": 0.55, "recall": 0.80, "fp_per_image": 0.20},
            {"conf_thr": 0.30, "f1": 0.65, "precision": 0.70, "recall": 0.65, "fp_per_image": 0.25},
        ]
        sel = select_operating_point(rows, mode="constraints", max_fp_per_image=0.22)
        self.assertIsNotNone(sel)
        assert sel is not None
        self.assertAlmostEqual(float(sel["conf_thr"]), 0.20)

    def test_min_count_mae_picks_lowest_mae(self) -> None:
        from unittest.mock import patch

        from harchoc.threshold_protocol import select_min_count_mae

        rows = [
            {"conf_thr": 0.20, "f1": 0.9},
            {"conf_thr": 0.10, "f1": 0.5},
            {"conf_thr": 0.30, "f1": 0.8},
        ]
        with patch("harchoc.threshold_protocol.counting_metrics_at_conf") as mock_count:
            mock_count.side_effect = [
                {"mae": 12.0},
                {"mae": 3.0},
                {"mae": 3.0},
            ]
            sel = select_min_count_mae(rows, gt={}, preds={}, iou_thr=0.5, category_aware=True)
        self.assertIsNotNone(sel)
        assert sel is not None
        self.assertAlmostEqual(float(sel["conf_thr"]), 0.10)
        self.assertAlmostEqual(float(sel["count_mae"]), 3.0)

    def test_constraints_no_feasible_returns_none(self) -> None:
        from scripts.threshold_sweep import select_operating_point

        rows = [
            {"conf_thr": 0.10, "f1": 0.70, "precision": 0.50, "recall": 0.90, "fp_per_image": 1.50},
            {"conf_thr": 0.20, "f1": 0.60, "precision": 0.55, "recall": 0.80, "fp_per_image": 0.90},
        ]
        sel = select_operating_point(rows, mode="constraints", max_fp_per_image=0.10)
        self.assertIsNone(sel)

    def test_threshold_sweep_val_config_select_min_count_mae(self) -> None:
        from harchoc.experiment_config import load_config, script_section_from_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "configs/experiments/threshold_sweep_val.json")
        sweep = script_section_from_config(cfg, "threshold_sweep")
        self.assertEqual(sweep.get("select"), "min_count_mae")
        self.assertEqual(cfg.get("schema_version"), "experiments.v1")

    def test_compact_csv_schema_and_rows(self) -> None:
        from scripts.threshold_sweep import write_compact_csv

        rows = [
            {"conf_thr": 0.1, "tp": 1, "fp": 2, "fn": 3, "precision": 0.2, "recall": 0.3, "f1": 0.24, "fp_per_image": 1.0},
            {"conf_thr": 0.2, "tp": 2, "fp": 1, "fn": 3, "precision": 0.66, "recall": 0.4, "f1": 0.5, "fp_per_image": 0.5},
        ]

        with tempfile.TemporaryDirectory() as td:
            out = write_compact_csv(Path(td) / "sweep.csv", rows)
            txt = out.read_text("utf-8").strip().splitlines()
            self.assertEqual(txt[0], "conf_thr,tp,fp,fn,precision,recall,f1,fp_per_image")
            self.assertEqual(len(txt), 3)


if __name__ == "__main__":
    unittest.main()
