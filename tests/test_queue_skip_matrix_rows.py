"""Matrix row weight / checkpoint helpers (queue_skip_gates)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")


class TestMatrixRowWeightHelpers(unittest.TestCase):
    def test_matrix_row_missing_weights_skipped_when_no_run_dir(self) -> None:
        from harchoc.bench_config import load_bench_config
        from harchoc.queue_skip_gates import matrix_row_missing_weights_row

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cfg = load_bench_config(
                Path(__file__).resolve().parents[1] / "configs/bench/yolov10m_default.yaml"
            )
            row = matrix_row_missing_weights_row(cfg, tdp / "runs", backend="ultralytics")
            self.assertEqual(row["status"], "skipped_no_weights")
            self.assertEqual(row["reason"], "no_bench_run_weights")
            self.assertIsNone(row.get("weights"))

    def test_matrix_row_missing_weights_train_failed_when_run_dir_exists(self) -> None:
        from harchoc.bench_config import _bench_run_name, load_bench_config
        from harchoc.queue_skip_gates import matrix_row_missing_weights_row

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            runs = tdp / "runs"
            cfg = load_bench_config(
                Path(__file__).resolve().parents[1] / "configs/bench/yolov10m_default.yaml"
            )
            run_dir = runs / _bench_run_name(cfg)
            run_dir.mkdir(parents=True)
            (run_dir / "results.csv").write_text("epoch,loss\n", encoding="utf-8")
            row = matrix_row_missing_weights_row(cfg, runs, backend="ultralytics")
            self.assertEqual(row["status"], "train_failed")
            self.assertEqual(row["reason"], "train_run_without_best_pt")

    def test_normalize_matrix_train_row_maps_skipped_cache(self) -> None:
        from harchoc.bench_config import load_bench_config
        from harchoc.queue_skip_gates import normalize_matrix_train_row

        with tempfile.TemporaryDirectory() as td:
            cfg = load_bench_config(
                Path(__file__).resolve().parents[1] / "configs/bench/yolov8m_default.yaml"
            )
            out = normalize_matrix_train_row(
                {"status": "skipped", "reason": "weights_not_cached"},
                cfg,
                Path(td) / "runs",
            )
            self.assertEqual(out["status"], "skipped_no_weights")


class TestBenchmarkMatrixNoTrain(unittest.TestCase):
    @patch("scripts.benchmark_matrix._invoke_ultralytics_hsp_for_matrix")
    def test_no_train_eval_only_incremental_checkpoint(
        self, mock_hsp: object,
    ) -> None:
        from scripts.benchmark_matrix import main as benchmark_main

        mock_hsp.return_value = {
            "status": "ok",
            "test_count_mae": 42.0,
            "error_json": "reports/hsp/tiny_error.json",
        }
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True)
            (dataset_root / "data.yaml").write_text("path: .\n", encoding="utf-8")
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir()
            for name, model in (("a.yaml", "yolov8n.pt"), ("b.yaml", "yolo11n.pt")):
                (bench_dir / name).write_text(
                    "\n".join(
                        [
                            f"name: bench_{name}",
                            "backend: ultralytics",
                            f"model: {model}",
                            "groups: zoo_yolo_only",
                            "epochs: 1",
                            "seed: 0",
                            "infer:",
                            "  imgsz: 640",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )

            runs = tdp / "runs"
            from harchoc.bench_config import _bench_run_name, load_bench_config

            cfg_a = load_bench_config(bench_dir / "a.yaml")
            wdir = runs / _bench_run_name(cfg_a) / "weights"
            wdir.mkdir(parents=True)
            (wdir / "best.pt").write_bytes(b"fake")

            train_out = tdp / "matrix_train.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--group",
                    "zoo_yolo_only",
                    "--runs-dir",
                    str(runs),
                    "--train-out",
                    str(train_out),
                    "--no-dry-run",
                    "--no-train",
                ]
            )
            self.assertEqual(rc, 0)
            doc = json.loads(train_out.read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], "benchmark_matrix_train.v1")
            self.assertEqual(doc["status"], "eval")
            self.assertEqual(len(doc["runs"]), 2)
            by_name = {str(r["name"]): r for r in doc["runs"]}
            self.assertEqual(by_name["bench_a.yaml"]["status"], "ok")
            self.assertEqual(by_name["bench_b.yaml"]["status"], "skipped_no_weights")
            mock_hsp.assert_called_once()

    def test_dry_run_zoo_yolo_only_shows_bench_run_weights(self) -> None:
        from scripts.benchmark_matrix import main as benchmark_main

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir()
            os.environ["DATASET_ROOT"] = str(dataset_root)
            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")
            out = tdp / "matrix.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--group",
                    "zoo_yolo_only",
                    "--runs-dir",
                    str(repo / "runs/hsp_zoo"),
                    "--out",
                    str(out),
                    "--no-train",
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(obj["runs"]), 4)
            self.assertEqual(obj["selection"]["groups"], ["zoo_yolo_only"])
            v8 = next(r for r in obj["runs"] if "yolov8m" in r["config"]["path"])
            self.assertTrue(v8["execution"]["would_eval"])
            self.assertFalse(v8["execution"]["would_train"])
            self.assertIsNotNone(v8["resolved"].get("bench_run_weights"))


if __name__ == "__main__":
    unittest.main()
