from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class ExperimentArgvTests(unittest.TestCase):
    def test_argv_for_dual_metric_matches_bundle_artifacts(self) -> None:
        from harchoc.experiment_argv import argv_for_dual_metric, dual_metric_fields_from_bundle_art
        from harchoc.manuscript_repro import build_manuscript_repro_chain, load_manuscript_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        bundle = load_manuscript_repro_bundle(repo / "configs/experiments/manuscript_repro_bundle.json")
        art = bundle["artifacts"]
        expected = argv_for_dual_metric(dual_metric_fields_from_bundle_art(art))
        steps = build_manuscript_repro_chain(bundle, repo_root=repo, skip_gpu_check=True)
        dual_step = next(argv for step_id, argv in steps if step_id == "dual_metric")
        self.assertEqual(dual_step[1:], expected)

    def test_argv_for_repro_steps_matches_build_chain(self) -> None:
        from harchoc.experiment_argv import argv_for_repro_steps
        from harchoc.manuscript_repro import build_manuscript_repro_chain, load_manuscript_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        bundle = load_manuscript_repro_bundle(repo / "configs/experiments/manuscript_repro_bundle.json")
        via_builder = build_manuscript_repro_chain(
            bundle, repo_root=repo, skip_gpu_check=True, include_test_map=True
        )
        via_argv = argv_for_repro_steps(
            bundle, repo_root=repo, skip_gpu_check=True, include_test_map=True
        )
        self.assertEqual(via_argv, via_builder)

    def test_experiment_repro_dry_run_uses_repro_chain(self) -> None:
        from harchoc.experiment_argv import argv_for_repro_steps
        from harchoc.manuscript_repro import load_manuscript_repro_bundle
        from scripts.experiment import main

        repo = Path(__file__).resolve().parents[1]
        bundle = load_manuscript_repro_bundle(repo / "configs/experiments/manuscript_repro_bundle.json")
        expected_steps = argv_for_repro_steps(bundle, repo_root=repo, skip_gpu_check=True)

        old_quiet = os.environ.get("HARCHOC_QUIET")
        os.environ["HARCHOC_QUIET"] = "1"
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--dry-run", "repro", "--skip-gpu-check"])
        finally:
            if old_quiet is None:
                os.environ.pop("HARCHOC_QUIET", None)
            else:
                os.environ["HARCHOC_QUIET"] = old_quiet

        self.assertEqual(rc, 0)
        out = buf.getvalue()
        for step_id, argv in expected_steps:
            self.assertIn(f"# {step_id}", out)
            self.assertIn(argv[-1] if argv else "", out)


class DualMetricBundleTests(unittest.TestCase):
    def test_dual_metric_dry_run_via_experiment(self) -> None:
        import json
        import tempfile

        from scripts.experiment import main

        old_quiet = os.environ.get("HARCHOC_QUIET")
        os.environ["HARCHOC_QUIET"] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "dual_metric.json"
                rc = main(
                    [
                        "--dry-run",
                        "dual-metric",
                        "--out",
                        str(out),
                        "--sweep",
                        "reports/hsp/threshold_val.json",
                        "--error-val",
                        "reports/hsp/error_val.json",
                        "--error-test",
                        "reports/hsp/error_test.json",
                    ]
                )
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("status"), "dry-run")
        finally:
            if old_quiet is None:
                os.environ.pop("HARCHOC_QUIET", None)
            else:
                os.environ["HARCHOC_QUIET"] = old_quiet


class SplitsArgvTests(unittest.TestCase):
    def test_argv_for_splits_dry_run_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_splits

        fields = {
            "dry_run": True,
            "manifest": "data/manifest.json",
            "default_dataset_name": "sunflower",
            "mode": "random",
            "out_dir": "data/splits",
            "ext": [".jpg", ".png"],
            "glob": "*.jpg",
            "seed": 42,
            "val_frac": 0.15,
            "test_frac": 0.15,
        }
        argv = argv_for_splits(fields)
        self.assertIn("--dry-run", argv)
        self.assertIn("--manifest", argv)
        self.assertIn("data/manifest.json", argv)
        self.assertIn("--default-dataset-name", argv)
        self.assertIn("sunflower", argv)
        self.assertIn("--mode", argv)
        self.assertIn("random", argv)
        self.assertIn("--out-dir", argv)
        self.assertIn("data/splits", argv)
        self.assertIn("--ext", argv)
        self.assertIn(".jpg", argv)
        self.assertIn(".png", argv)
        self.assertIn("--glob", argv)
        self.assertIn("--seed", argv)
        self.assertIn("42", argv)
        self.assertIn("--val-frac", argv)
        self.assertIn("0.15", argv)
        self.assertIn("--test-frac", argv)
        self.assertNotIn("--dataset-name", argv)
        self.assertNotIn("--dataset-root", argv)


class DescribeArgvTests(unittest.TestCase):
    def test_argv_for_describe_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_describe

        fields = {
            "dry_run": True,
            "manifest": "data/manifest.json",
            "default_dataset_name": "sunflower",
            "dataset_name": "sunflower_v2",
            "dataset_root": "/data/sunflower",
            "yolo_data_yaml": "/data/sunflower/data.yaml",
            "split": "val",
            "split_file": ["data/splits/val.txt", "data/splits/custom.txt"],
            "out": "reports/describe_val.json",
        }
        argv = argv_for_describe(fields)
        self.assertIn("--dry-run", argv)
        self.assertIn("--manifest", argv)
        self.assertIn("data/manifest.json", argv)
        self.assertIn("--default-dataset-name", argv)
        self.assertIn("sunflower", argv)
        self.assertIn("--dataset-name", argv)
        self.assertIn("sunflower_v2", argv)
        self.assertIn("--dataset-root", argv)
        self.assertIn("/data/sunflower", argv)
        self.assertIn("--yolo-data-yaml", argv)
        self.assertIn("/data/sunflower/data.yaml", argv)
        self.assertIn("--split", argv)
        self.assertIn("val", argv)
        self.assertIn("--split-file", argv)
        self.assertIn("data/splits/val.txt", argv)
        self.assertIn("data/splits/custom.txt", argv)
        self.assertIn("--out", argv)
        self.assertIn("reports/describe_val.json", argv)


class TrainArgvTests(unittest.TestCase):
    def test_argv_for_train_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_train

        fields = {
            "dry_run": True,
            "manifest": "data/manifest.json",
            "default_dataset_name": "sunflower",
            "dataset_name": "sunflower_v2",
            "dataset_root": "/data/sunflower",
            "yolo_data_yaml": "/data/sunflower/data.yaml",
            "config": "configs/experiments/train_yolov8m_baseline.json",
            "out_dir": "runs/train",
            "name": "yolov8m_baseline",
            "aug_config": "configs/aug/robustness_minimal.yaml",
            "skip_eval": True,
            "eval_out": "reports/eval_smoke.json",
        }
        argv = argv_for_train(fields)
        self.assertIn("--dry-run", argv)
        self.assertIn("--manifest", argv)
        self.assertIn("data/manifest.json", argv)
        self.assertIn("--default-dataset-name", argv)
        self.assertIn("sunflower", argv)
        self.assertIn("--dataset-name", argv)
        self.assertIn("sunflower_v2", argv)
        self.assertIn("--dataset-root", argv)
        self.assertIn("/data/sunflower", argv)
        self.assertIn("--yolo-data-yaml", argv)
        self.assertIn("/data/sunflower/data.yaml", argv)
        self.assertIn("--config", argv)
        self.assertIn("configs/experiments/train_yolov8m_baseline.json", argv)
        self.assertIn("--out-dir", argv)
        self.assertIn("runs/train", argv)
        self.assertIn("--name", argv)
        self.assertIn("yolov8m_baseline", argv)
        self.assertIn("--aug-config", argv)
        self.assertIn("configs/aug/robustness_minimal.yaml", argv)
        self.assertIn("--skip-eval", argv)
        self.assertIn("--eval-out", argv)
        self.assertIn("reports/eval_smoke.json", argv)


class BenchmarkArgvTests(unittest.TestCase):
    def test_argv_for_benchmark_defaults_dry_run(self) -> None:
        from harchoc.experiment_argv import argv_for_benchmark

        argv = argv_for_benchmark({})
        self.assertIn("--dry-run", argv)
        self.assertNotIn("--no-dry-run", argv)

    def test_argv_for_benchmark_no_dry_run_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_benchmark

        fields = {
            "dry_run": False,
            "manifest": "data/manifest.json",
            "bench_config": ["configs/bench/yolov8m.yaml"],
            "bench_dir": "configs/bench",
            "pattern": "yolov8*.yaml",
            "limit": 2,
            "out": "reports/hsp/matrix_train.json",
            "no_train": True,
            "no_eval": False,
            "eval_out": "reports/hsp/matrix_eval.json",
        }
        argv = argv_for_benchmark(fields)
        self.assertIn("--no-dry-run", argv)
        self.assertNotIn("--dry-run", argv)
        self.assertIn("--manifest", argv)
        self.assertIn("--bench-config", argv)
        self.assertIn("configs/bench/yolov8m.yaml", argv)
        self.assertIn("--bench-dir", argv)
        self.assertIn("--pattern", argv)
        self.assertIn("--limit", argv)
        self.assertIn("2", argv)
        self.assertIn("--out", argv)
        self.assertIn("--no-train", argv)
        self.assertNotIn("--no-eval", argv)
        self.assertIn("--eval-out", argv)

    def test_argv_for_benchmark_sahi_eval(self) -> None:
        from harchoc.experiment_argv import argv_for_benchmark

        argv = argv_for_benchmark({"dry_run": True, "sahi_eval": True, "out": "reports/sahi_plan.json"})
        self.assertIn("--sahi-eval", argv)
        self.assertIn("--dry-run", argv)
        self.assertIn("--out", argv)


class CvEvalArgvTests(unittest.TestCase):
    def test_argv_for_cv_eval_dry_run_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_cv_eval

        fields = {
            "dry_run": True,
            "manifest": "data/manifest.json",
            "dataset_name": "sunflower",
            "weights": "models/best2.pt",
            "folds": 5,
            "seed": 0,
            "splits_dir": "data/splits",
            "fold_metrics": ["reports/cv_eval/fold0.json", "reports/cv_eval/fold1.json"],
            "out": "reports/cv_eval/summary.json",
            "write_fold_splits": "data/splits/cv_folds",
        }
        argv = argv_for_cv_eval(fields)
        self.assertIn("--dry-run", argv)
        self.assertIn("--manifest", argv)
        self.assertIn("data/manifest.json", argv)
        self.assertIn("--dataset-name", argv)
        self.assertIn("sunflower", argv)
        self.assertIn("--weights", argv)
        self.assertIn("models/best2.pt", argv)
        self.assertIn("--folds", argv)
        self.assertIn("5", argv)
        self.assertIn("--seed", argv)
        self.assertIn("--splits-dir", argv)
        self.assertIn("--fold-metrics", argv)
        self.assertIn("reports/cv_eval/fold0.json", argv)
        self.assertIn("reports/cv_eval/fold1.json", argv)
        self.assertIn("--out", argv)
        self.assertIn("reports/cv_eval/summary.json", argv)
        self.assertIn("--write-fold-splits", argv)
        self.assertIn("data/splits/cv_folds", argv)


class DeployParityArgvTests(unittest.TestCase):
    def test_argv_for_deploy_parity_defaults(self) -> None:
        from harchoc.experiment_argv import argv_for_deploy_parity

        argv = argv_for_deploy_parity({})
        self.assertIn("--out", argv)
        self.assertIn("reports/hsp/deploy_hsp_parity.json", argv)


class GradcamArgvTests(unittest.TestCase):
    def test_argv_for_gradcam_defaults(self) -> None:
        from harchoc.experiment_argv import argv_for_gradcam

        argv = argv_for_gradcam({})
        self.assertIn("--out-dir", argv)
        self.assertIn("reports/figures", argv)
        self.assertIn("--meta-out", argv)
        self.assertIn("reports/figures/run.json", argv)
        self.assertIn("--error-report", argv)
        self.assertIn("reports/hsp/error_test_report.json", argv)
        self.assertIn("--panel-size", argv)
        self.assertIn("12", argv)
        self.assertIn("--figure", argv)
        self.assertIn("fig_gradcam_panel", argv)
        self.assertIn("--weights", argv)
        from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS

        self.assertIn(HSP_DETECTION_WEIGHTS, argv)

    def test_argv_for_gradcam_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_gradcam

        fields = {
            "out_dir": "figures/custom",
            "meta_out": "reports/custom/run.json",
            "error_report": "reports/hsp/error_val_report.json",
            "weights": "models/best2.pt",
            "panel_size": 8,
        }
        argv = argv_for_gradcam(fields)
        self.assertIn("--out-dir", argv)
        self.assertIn("figures/custom", argv)
        self.assertIn("--meta-out", argv)
        self.assertIn("reports/custom/run.json", argv)
        self.assertIn("--error-report", argv)
        self.assertIn("reports/hsp/error_val_report.json", argv)
        self.assertIn("--weights", argv)
        self.assertIn("models/best2.pt", argv)
        self.assertIn("--panel-size", argv)
        self.assertIn("8", argv)
        self.assertIn("--figure", argv)
        self.assertIn("fig_gradcam_panel", argv)


class MapCpuTests(unittest.TestCase):
    def test_argv_for_map_cpu_defaults(self) -> None:
        from harchoc.experiment_argv import argv_for_map_cpu

        argv = argv_for_map_cpu({})
        self.assertIn("--device", argv)
        self.assertIn("cpu", argv)
        self.assertIn("--max-det", argv)
        self.assertIn("3000", argv)
        self.assertIn("--imgsz", argv)
        self.assertIn("1280", argv)
        self.assertIn("--out", argv)
        self.assertIn("reports/hsp/eval_test_map.json", argv)
        self.assertIn("--split-file", argv)
        self.assertIn("data/splits/test.txt", argv)

    def test_map_cpu_dry_run_prints_argv(self) -> None:
        import io
        from contextlib import redirect_stdout

        from scripts.experiment import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["map-cpu", "--dry-run"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("# map-cpu", out)
        self.assertIn("scripts/eval.py", out)
        self.assertIn("--device cpu", out)
        self.assertIn("--max-det 3000", out)
        self.assertIn("--imgsz 1280", out)


class TuneSahiArgvTests(unittest.TestCase):
    def test_argv_for_tune_sahi_defaults(self) -> None:
        from harchoc.experiment_argv import argv_for_tune_sahi

        argv = argv_for_tune_sahi({})
        self.assertEqual(argv, ["test_sunflower_tune.png"])

    def test_argv_for_tune_sahi_fields(self) -> None:
        from harchoc.experiment_argv import argv_for_tune_sahi

        argv = argv_for_tune_sahi({"image": "data/heads/ref.png"})
        self.assertEqual(argv, ["data/heads/ref.png"])

    def test_tune_sahi_dry_run_prints_argv(self) -> None:
        import io
        from contextlib import redirect_stdout

        from scripts.experiment import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["tune-sahi", "--dry-run", "--image", "data/heads/ref.png"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("# tune-sahi", out)
        self.assertIn("scripts/experiment.py", out)
        self.assertIn("data/heads/ref.png", out)


if __name__ == "__main__":
    unittest.main()
