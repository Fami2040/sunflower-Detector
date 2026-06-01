from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _eval_doc(*, split_role: str, map50: float, map50_95: float) -> dict:
    return {
        "schema_version": "eval_run.v1",
        "status": "ok",
        "eval_target": {"split_role": split_role},
        "mAP50": map50,
        "mAP50_95": map50_95,
    }


def _sweep_val_doc(*, conf: float) -> dict:
    row = {
        "conf_thr": conf,
        "tp": 10,
        "fp": 2,
        "fn": 1,
        "precision": 0.83,
        "recall": 0.91,
        "f1": 0.87,
        "fp_per_image": 0.02,
    }
    return {
        "schema_version": "threshold_sweep_run.v1",
        "status": "ok",
        "selected": {"mode": "best_f1", "row": row},
    }


def _sweep_test_doc(*, conf: float, locked_mae: float | None = None) -> dict:
    row = {
        "conf_thr": conf,
        "tp": 8,
        "fp": 3,
        "fn": 2,
        "precision": 0.73,
        "recall": 0.80,
        "f1": 0.76,
        "fp_per_image": 0.03,
    }
    locked: dict = {"mode": "fixed_conf", "source": "reports/sweep_val.json", "row": row}
    if locked_mae is not None:
        locked["counting_metrics"] = {
            "mae": locked_mae,
            "rmse": locked_mae * 1.1,
            "rrmse": 0.04,
            "n_images": 109,
            "mae_ci": {"low": locked_mae - 5.0, "high": locked_mae + 5.0, "confidence": 0.95},
        }
    return {
        "schema_version": "threshold_sweep_run.v1",
        "status": "ok",
        "locked": locked,
    }


def _error_doc(*, mae: float) -> dict:
    return {
        "schema_version": "error_analysis_summary.v1",
        "status": "ok",
        "counting_metrics": {
            "mae": mae,
            "rmse": mae * 1.2,
            "rrmse": 0.05,
            "n_images": 4,
            "mae_ci": {"low": mae - 0.1, "high": mae + 0.1, "level": 0.95},
        },
    }


class DualMetricReportTests(unittest.TestCase):
    def test_prefers_locked_counting_metrics_on_test(self) -> None:
        from harchoc.dual_metric_report import build_dual_metric_report

        report = build_dual_metric_report(
            eval_val=_eval_doc(split_role="val", map50=0.97, map50_95=0.72),
            eval_test=_eval_doc(split_role="test", map50=0.79, map50_95=0.48),
            sweep_val=_sweep_val_doc(conf=0.42),
            sweep_test=_sweep_test_doc(conf=0.42, locked_mae=61.27),
            error_val=_error_doc(mae=1.2),
            error_test=_error_doc(mae=2.5),
        )
        rows = {r["split"]: r for r in report["rows"]}
        self.assertAlmostEqual(rows["test"]["counting"]["mae"], 61.27)
        self.assertAlmostEqual(rows["val"]["counting"]["mae"], 1.2)
        self.assertEqual(report["counting_sources"]["test"], "locked")
        self.assertEqual(report["counting_sources"]["val"], "error_analysis")

    def test_build_dual_metric_report(self) -> None:
        from harchoc.dual_metric_report import build_dual_metric_report

        report = build_dual_metric_report(
            eval_val=_eval_doc(split_role="val", map50=0.97, map50_95=0.72),
            eval_test=_eval_doc(split_role="test", map50=0.79, map50_95=0.48),
            sweep_val=_sweep_val_doc(conf=0.42),
            sweep_test=_sweep_test_doc(conf=0.42),
            error_val=_error_doc(mae=1.2),
            error_test=_error_doc(mae=2.5),
        )
        self.assertEqual(report["schema_version"], "dual_metric_report.v1")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["operating_point"]["selected_conf"], 0.42)
        self.assertEqual(report["operating_point"]["locked_conf"], 0.42)
        self.assertIn("in-training early-stop split", report["metric_roles"]["val"])
        self.assertIn("held-out manuscript split", report["metric_roles"]["test"])

        rows = {r["split"]: r for r in report["rows"]}
        self.assertIn("not generalization", rows["val"]["split_role_label"])
        self.assertIn("generalization", rows["test"]["split_role_label"])
        self.assertAlmostEqual(rows["val"]["detection"]["mAP50"], 0.97)
        self.assertAlmostEqual(rows["test"]["detection"]["mAP50"], 0.79)
        self.assertAlmostEqual(rows["val"]["counting"]["mae"], 1.2)
        self.assertAlmostEqual(rows["test"]["counting"]["mae"], 2.5)
        self.assertEqual(rows["val"]["operating_conf"], 0.42)
        self.assertEqual(rows["test"]["operating_conf"], 0.42)
        self.assertIn("mae_ci", rows["test"]["counting"])

    def test_merge_from_paths(self) -> None:
        from harchoc.dual_metric_report import merge_dual_metric_from_paths

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {
                "eval_val": root / "eval_val.json",
                "eval_test": root / "eval_test.json",
                "sweep_val": root / "sweep_val.json",
                "sweep_test": root / "sweep_test.json",
                "error_val": root / "error_val.json",
                "error_test": root / "error_test.json",
            }
            paths["eval_val"].write_text(
                json.dumps(_eval_doc(split_role="val", map50=0.9, map50_95=0.5)), encoding="utf-8"
            )
            paths["eval_test"].write_text(
                json.dumps(_eval_doc(split_role="test", map50=0.8, map50_95=0.4)), encoding="utf-8"
            )
            paths["sweep_val"].write_text(json.dumps(_sweep_val_doc(conf=0.35)), encoding="utf-8")
            paths["sweep_test"].write_text(json.dumps(_sweep_test_doc(conf=0.35)), encoding="utf-8")
            paths["error_val"].write_text(json.dumps(_error_doc(mae=1.0)), encoding="utf-8")
            paths["error_test"].write_text(json.dumps(_error_doc(mae=2.0)), encoding="utf-8")

            report = merge_dual_metric_from_paths(
                eval_val=str(paths["eval_val"]),
                eval_test=str(paths["eval_test"]),
                sweep_val=str(paths["sweep_val"]),
                sweep_test=str(paths["sweep_test"]),
                error_val=str(paths["error_val"]),
                error_test=str(paths["error_test"]),
            )
            self.assertEqual(report["inputs"]["sweep_val"], str(paths["sweep_val"]))

    def test_map_overlay_when_export_only(self) -> None:
        from harchoc.dual_metric_report import merge_dual_metric_from_paths

        export_only = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "export_only": True,
            "mAP50": None,
            "mAP50_95": None,
        }
        with_map = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "mAP50": 0.81,
            "mAP50_95": 0.52,
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ev_val = root / "eval_val.json"
            ev_test = root / "eval_test.json"
            ev_map = root / "eval_test_map.json"
            ev_val.write_text(
                json.dumps(_eval_doc(split_role="val", map50=0.9, map50_95=0.5)), encoding="utf-8"
            )
            ev_test.write_text(json.dumps(export_only), encoding="utf-8")
            sweep = root / "sweep.json"
            sweep.write_text(json.dumps(_sweep_val_doc(conf=0.35)), encoding="utf-8")
            err_v = root / "err_v.json"
            err_t = root / "err_t.json"
            err_v.write_text(json.dumps(_error_doc(mae=1.0)), encoding="utf-8")
            err_t.write_text(json.dumps(_error_doc(mae=2.0)), encoding="utf-8")

            report = merge_dual_metric_from_paths(
                eval_val=str(ev_val),
                eval_test=str(ev_test),
                sweep_val=str(sweep),
                error_val=str(err_v),
                error_test=str(err_t),
            )
            rows = {r["split"]: r for r in report["rows"]}
            self.assertEqual(rows["test"]["detection"], {})
            self.assertEqual(report["status"], "ok")
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("test", report["warnings"][0])

            ev_map.write_text(json.dumps(with_map), encoding="utf-8")
            report2 = merge_dual_metric_from_paths(
                eval_val=str(ev_val),
                eval_test=str(ev_test),
                sweep_val=str(sweep),
                error_val=str(err_v),
                error_test=str(err_t),
            )
            rows2 = {r["split"]: r for r in report2["rows"]}
            self.assertAlmostEqual(rows2["test"]["detection"]["mAP50"], 0.81)
            self.assertIn("eval_test_map", report2["inputs"])
            self.assertEqual(report2["status"], "ok")
            self.assertEqual(report2["warnings"], [])

    def test_eval_test_map_explicit_path(self) -> None:
        from harchoc.dual_metric_report import merge_dual_metric_from_paths

        export_only = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "export_only": True,
            "mAP50": None,
            "mAP50_95": None,
        }
        with_map = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "mAP50": 0.793,
            "mAP50_95": 0.481,
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {
                "eval_val": root / "eval_val.json",
                "eval_test": root / "eval_test.json",
                "eval_test_map": root / "eval_test_map.json",
                "sweep": root / "sweep.json",
                "error_val": root / "error_val.json",
                "error_test": root / "error_test.json",
            }
            paths["eval_val"].write_text(
                json.dumps(_eval_doc(split_role="val", map50=0.97, map50_95=0.72)), encoding="utf-8"
            )
            paths["eval_test"].write_text(json.dumps(export_only), encoding="utf-8")
            paths["eval_test_map"].write_text(json.dumps(with_map), encoding="utf-8")
            paths["sweep"].write_text(json.dumps(_sweep_val_doc(conf=0.15)), encoding="utf-8")
            paths["error_val"].write_text(json.dumps(_error_doc(mae=1.0)), encoding="utf-8")
            paths["error_test"].write_text(json.dumps(_error_doc(mae=61.3)), encoding="utf-8")

            report = merge_dual_metric_from_paths(
                eval_val=str(paths["eval_val"]),
                eval_test=str(paths["eval_test"]),
                eval_test_map=str(paths["eval_test_map"]),
                sweep_val=str(paths["sweep"]),
                error_val=str(paths["error_val"]),
                error_test=str(paths["error_test"]),
            )
            rows = {r["split"]: r for r in report["rows"]}
            self.assertAlmostEqual(rows["test"]["detection"]["mAP50"], 0.793)
            self.assertAlmostEqual(rows["test"]["detection"]["mAP50_95"], 0.481)
            self.assertEqual(report["inputs"]["eval_test_map"], str(paths["eval_test_map"]))
            self.assertEqual(report["warnings"], [])

    def test_experiment_dual_metric_with_eval_test_map(self) -> None:
        from scripts.experiment import main

        export_only = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "export_only": True,
            "mAP50": None,
            "mAP50_95": None,
        }
        with_map = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "mAP50": 0.793,
            "mAP50_95": 0.481,
        }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ev_val = root / "eval_val.json"
            ev_test = root / "eval_test.json"
            ev_map = root / "eval_test_map.json"
            sweep = root / "sweep.json"
            err_v = root / "err_v.json"
            err_t = root / "err_t.json"
            out = root / "dual_metric.json"
            ev_val.write_text(
                json.dumps(_eval_doc(split_role="val", map50=0.97, map50_95=0.72)), encoding="utf-8"
            )
            ev_test.write_text(json.dumps(export_only), encoding="utf-8")
            ev_map.write_text(json.dumps(with_map), encoding="utf-8")
            sweep.write_text(json.dumps(_sweep_val_doc(conf=0.15)), encoding="utf-8")
            err_v.write_text(json.dumps(_error_doc(mae=1.0)), encoding="utf-8")
            err_t.write_text(json.dumps(_error_doc(mae=61.3)), encoding="utf-8")

            rc = main(
                [
                    "dual-metric",
                    "--eval-val",
                    str(ev_val),
                    "--eval-test",
                    str(ev_test),
                    "--eval-test-map",
                    str(ev_map),
                    "--sweep",
                    str(sweep),
                    "--error-val",
                    str(err_v),
                    "--error-test",
                    str(err_t),
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            rows = {r["split"]: r for r in report["rows"]}
            self.assertAlmostEqual(rows["test"]["detection"]["mAP50"], 0.793)
            self.assertEqual(report["inputs"]["eval_test_map"], str(ev_map))

    def test_locked_conf_failure_warning(self) -> None:
        from harchoc.dual_metric_report import build_dual_metric_report

        with tempfile.TemporaryDirectory() as td:
            sweep_path = Path(td) / "sweep_val.json"
            sweep_path.write_text(
                json.dumps({"schema_version": "threshold_sweep_run.v1", "status": "ok"}),
                encoding="utf-8",
            )
            report = build_dual_metric_report(
                eval_val=_eval_doc(split_role="val", map50=0.9, map50_95=0.5),
                eval_test=_eval_doc(split_role="test", map50=0.8, map50_95=0.4),
                sweep_val={"schema_version": "threshold_sweep_run.v1", "status": "ok"},
                error_val=_error_doc(mae=1.0),
                error_test=_error_doc(mae=2.0),
                inputs={"sweep_val": str(sweep_path)},
            )
        self.assertIsNone(report["operating_point"]["locked_conf"])
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("conf_thr", report["warnings"][0])

    def test_empty_detection_strict_partial(self) -> None:
        import os

        from harchoc.dual_metric_report import build_dual_metric_report

        export_only = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "val"},
            "export_only": True,
            "mAP50": None,
            "mAP50_95": None,
        }
        try:
            os.environ["HARCHOC_STRICT_ML"] = "1"
            report = build_dual_metric_report(
                eval_val=export_only,
                eval_test=export_only,
                sweep_val=_sweep_val_doc(conf=0.35),
                error_val=_error_doc(mae=1.0),
                error_test=_error_doc(mae=2.0),
                inputs={"eval_val": "ev.json", "eval_test": "ev.json"},
            )
            self.assertEqual(report["status"], "partial")
            self.assertEqual(len(report["warnings"]), 2)
        finally:
            os.environ.pop("HARCHOC_STRICT_ML", None)

    def test_empty_detection_warns_without_strict(self) -> None:
        from harchoc.dual_metric_report import build_dual_metric_report

        export_only = {
            "schema_version": "eval_run.v1",
            "eval_target": {"split_role": "test"},
            "mAP50": None,
            "mAP50_95": None,
        }
        report = build_dual_metric_report(
            eval_val=_eval_doc(split_role="val", map50=0.9, map50_95=0.5),
            eval_test=export_only,
            sweep_val=_sweep_val_doc(conf=0.35),
            error_val=_error_doc(mae=1.0),
            error_test=_error_doc(mae=2.0),
            inputs={"eval_val": "v.json", "eval_test": "t.json"},
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("test", report["warnings"][0])

    def test_resolve_eval_via_append(self) -> None:
        from harchoc.dual_metric_report import merge_dual_metric_from_paths

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ev_val = root / "ev_val.json"
            ev_test = root / "ev_test.json"
            ev_val.write_text(json.dumps(_eval_doc(split_role="val", map50=0.5, map50_95=0.3)), encoding="utf-8")
            ev_test.write_text(json.dumps(_eval_doc(split_role="test", map50=0.4, map50_95=0.2)), encoding="utf-8")
            sweep = root / "sweep.json"
            sweep.write_text(json.dumps(_sweep_val_doc(conf=0.25)), encoding="utf-8")
            err_v = root / "err_v.json"
            err_t = root / "err_t.json"
            err_v.write_text(json.dumps(_error_doc(mae=0.5)), encoding="utf-8")
            err_t.write_text(json.dumps(_error_doc(mae=1.0)), encoding="utf-8")

            report = merge_dual_metric_from_paths(
                eval_paths=[str(ev_val), str(ev_test)],
                sweep_val=str(sweep),
                error_val=str(err_v),
                error_test=str(err_t),
            )
            self.assertEqual(len(report["rows"]), 2)

    def test_experiment_dual_metric_dry_run(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dual.json"
            rc = main(
                [
                    "dual-metric",
                    "--dry-run",
                    "--out",
                    str(out),
                    "--sweep",
                    "reports/sweep_val.json",
                    "--error-val",
                    "reports/error_val.json",
                    "--error-test",
                    "reports/error_test.json",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(obj["schema_version"], "dual_metric_report.v1")
            self.assertEqual(obj["status"], "dry-run")


if __name__ == "__main__":
    unittest.main()
