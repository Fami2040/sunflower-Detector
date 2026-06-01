import json
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO / "tests" / "fixtures"


class MatrixSeedStatsHelperTests(unittest.TestCase):
    def test_count_mae_from_error_report(self) -> None:
        from harchoc.matrix_seed_stats import count_mae_from_doc, load_count_mae_from_artifact

        mae, source = load_count_mae_from_artifact(_FIXTURES / "error_s0_report.json", repo_root=_REPO)
        self.assertEqual(mae, 10.0)
        self.assertEqual(source, "error_report")

        doc = json.loads((_FIXTURES / "threshold_s1_locked.json").read_text(encoding="utf-8"))
        mae2, source2 = count_mae_from_doc(doc)
        self.assertEqual(mae2, 16.0)
        self.assertEqual(source2, "threshold_locked")

    def test_compare_runs_by_seed_with_cli_paths(self) -> None:
        from harchoc.matrix_seed_stats import compare_runs_by_seed

        train_doc = json.loads((_FIXTURES / "matrix_train_two_seeds.json").read_text(encoding="utf-8"))
        paths = {
            "yolov8n_e100_s0": str(_FIXTURES / "error_s0_report.json"),
            "yolov8n_e100_s1": str(_FIXTURES / "error_s1_report.json"),
        }
        out = compare_runs_by_seed(train_doc, count_mae_paths=paths, repo_root=_REPO)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["n_count_mae"], 2)
        self.assertAlmostEqual(out["count_mae_mean"], 12.0)
        model = out["models"]["yolov8n"]
        self.assertAlmostEqual(model["count_mae_mean"], 12.0)
        self.assertAlmostEqual(model["count_mae_std"], 2.8284271247461903, places=5)
        seeds = model["seeds"]
        self.assertEqual(seeds[0]["count_mae"], 10.0)
        self.assertEqual(seeds[1]["count_mae"], 14.0)

    def test_dry_run_placeholders(self) -> None:
        from harchoc.matrix_seed_stats import build_dry_run_matrix_seed_stats_v1

        payload = build_dry_run_matrix_seed_stats_v1(out="reports/x.json")
        self.assertEqual(payload["schema_version"], "matrix_seed_stats.v1")
        self.assertEqual(payload["status"], "dry-run")
        self.assertIsNone(payload["count_mae_mean"])
        self.assertIsNone(payload["count_mae_std"])


class MatrixSeedStatsCliTests(unittest.TestCase):
    def test_cli_dry_run_with_fixtures(self) -> None:
        import os

        from scripts.matrix_seed_stats import main

        os.environ["HARCHOC_ALLOW_BASE_PYTHON"] = "1"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "seed_stats.json"
            rc = main(
                [
                    "--dry-run",
                    "--train-out",
                    str(_FIXTURES / "matrix_train_two_seeds.json"),
                    "--count-mae-json",
                    f"yolov8n_e100_s0={_FIXTURES / 'error_s0_report.json'}",
                    "--count-mae-json",
                    f"yolov8n_e100_s1={_FIXTURES / 'error_s1_report.json'}",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(obj["schema_version"], "matrix_seed_stats.v1")
            self.assertEqual(obj["status"], "dry-run")
            self.assertAlmostEqual(obj["count_mae_mean"], 12.0)
            self.assertIsNotNone(obj["count_mae_std"])

    def test_hsp_baseline_single_seed_mae(self) -> None:
        from harchoc.matrix_seed_stats import compare_runs_by_seed

        train_doc = json.loads(
            (_FIXTURES / "matrix_train_hsp_baseline.json").read_text(encoding="utf-8")
        )
        out = compare_runs_by_seed(train_doc, repo_root=_REPO)
        self.assertEqual(out["n_count_mae"], 1)
        self.assertAlmostEqual(out["count_mae_mean"], 61.26605504587156)
        self.assertIsNone(out["count_mae_std"])
        model = out["models"]["yolov8m"]
        self.assertEqual(model["n_count_mae"], 1)
        self.assertAlmostEqual(model["count_mae_mean"], 61.26605504587156)

    def test_run_artifact_keys_on_train_row(self) -> None:
        from harchoc.matrix_seed_stats import compare_runs_by_seed

        train_doc = {
            "runs": [
                {
                    "status": "ok",
                    "name": "yolov8n",
                    "run_name": "yolov8n_e100_s0",
                    "error_test_report": str(_FIXTURES / "error_s0_report.json"),
                },
                {
                    "status": "ok",
                    "name": "yolov8n",
                    "run_name": "yolov8n_e100_s1",
                    "threshold_test_locked": str(_FIXTURES / "threshold_s1_locked.json"),
                },
            ]
        }
        out = compare_runs_by_seed(train_doc, repo_root=_REPO)
        self.assertEqual(out["n_count_mae"], 2)
        self.assertAlmostEqual(out["count_mae_mean"], 13.0)


if __name__ == "__main__":
    unittest.main()
