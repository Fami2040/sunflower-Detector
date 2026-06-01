import json
import tempfile
import unittest
from pathlib import Path


class PlattTests(unittest.TestCase):
    def test_fit_platt_monotone_scores(self) -> None:
        from harchoc.platt import fit_platt

        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 0, 1, 1]
        m = fit_platt(scores=scores, targets=labels)
        self.assertGreater(m.predict_one(0.9), m.predict_one(0.1))


class MatrixSeedStatsTests(unittest.TestCase):
    def test_compare_runs_by_seed(self) -> None:
        from harchoc.matrix_seed_stats import compare_runs_by_seed

        doc = {
            "runs": [
                {"status": "ok", "name": "yolov8n", "run_name": "yolov8n_e100_s0", "mAP50": 0.8},
                {"status": "ok", "name": "yolov8n", "run_name": "yolov8n_e100_s1", "mAP50": 0.82},
            ]
        }
        out = compare_runs_by_seed(doc)
        self.assertEqual(out["status"], "ok")
        self.assertIn("yolov8n", out["models"])


class ThresholdSweepCalibrationTests(unittest.TestCase):
    def test_light_sweep_with_isotonic(self) -> None:
        import os
        from scripts.threshold_sweep import main

        os.environ["HARCHOC_ALLOW_BASE_PYTHON"] = "1"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sweep.json"
            rc = main(["--light", "--calibrate", "isotonic", "--out", str(out), "--steps", "5"])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["status"], "ok")
            self.assertEqual(obj.get("calibration", {}).get("mode"), "isotonic")


if __name__ == "__main__":
    unittest.main()
