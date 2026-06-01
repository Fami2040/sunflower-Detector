"""Tests for unified GPU queue skip gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestQueueSkipGates(unittest.TestCase):
    def test_hsp_eval_artifacts_verified(self) -> None:
        from harchoc.aug_smoke_runner import hsp_eval_artifacts_verified

        repo = Path(__file__).resolve().parents[1]
        err = repo / "reports/aug_smoke/aug_smoke_baseline_error.json"
        if not err.is_file():
            self.skipTest("baseline error json missing")
        self.assertTrue(
            hsp_eval_artifacts_verified(
                repo, run_name="aug_smoke_baseline", out_dir="reports/aug_smoke"
            )
        )

    def test_should_skip_eval_when_artifacts_exist_without_summary(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        err = repo / "reports/aug_smoke/aug_smoke_baseline_error.json"
        weights = repo / "runs/aug_smoke_baseline/weights/best.pt"
        if not err.is_file() or not weights.is_file():
            self.skipTest("baseline eval artifacts missing")

        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            summary = Path(td) / "fake_summary.json"
            job = {
                "id": "fake_eval",
                "kind": "aug_smoke",
                "eval_only": True,
                "run_name": "aug_smoke_baseline",
                "train_config": "configs/experiments/train_aug_smoke_baseline.json",
                "eval_out_dir": "reports/aug_smoke",
                "skip_if": {"summary": str(summary.relative_to(repo))},
            }
            skip, reason = should_skip_job(job, repo_root=repo)
            self.assertTrue(skip)
            self.assertIn("HSP eval artifacts complete", reason)
            self.assertTrue(summary.is_file())

    def test_enrich_matrix_train_run_from_hsp_artifacts(self) -> None:
        from harchoc.queue_skip_gates import enrich_matrix_train_run_from_artifacts

        repo = Path(__file__).resolve().parents[1]
        err = repo / "reports/hsp/yolo26m_e100_s0_error.json"
        if not err.is_file():
            self.skipTest("yolo26m_e100_s0_error.json missing")
        row = enrich_matrix_train_run_from_artifacts(
            repo,
            {
                "status": "ok",
                "run_name": "yolo26m_e100_s0",
                "weights": str(repo / "runs/hsp_zoo/yolo26m_e100_s0/weights/best.pt"),
            },
            runs_dir=repo / "runs/hsp_zoo",
        )
        self.assertIsNotNone(row.get("test_count_mae"))
        self.assertTrue(str(row.get("error_test_report") or "").endswith("_error.json"))

    def test_matrix_train_verified_fixture(self) -> None:
        from harchoc.queue_skip_gates import matrix_train_verified

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            train_out = tdp / "matrix_train.json"
            train_out.write_text(
                json.dumps(
                    {
                        "schema_version": "benchmark_matrix_train.v1",
                        "runs": [
                            {
                                "status": "ok",
                                "name": "yolov8m",
                                "config_path": str(
                                    (repo / "configs/bench/yolov8m_zoo.yaml").resolve()
                                ),
                                "weights": "/tmp/best.pt",
                                "test_count_mae": 61.3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bench = repo / "configs/bench/yolov8m_zoo.yaml"
            if not bench.is_file():
                self.skipTest("yolov8m_zoo bench missing")
            from harchoc.bench_config import load_bench_config

            cfg = load_bench_config(bench)
            group = cfg.groups[0] if cfg.groups else ""
            if not group:
                self.skipTest("no group on yolov8m bench")
            ok, reason = matrix_train_verified(repo, train_out, group)
            self.assertTrue(ok, reason)

    def test_matrix_train_verified_accepts_skipped_no_weights_zoo(self) -> None:
        import json
        import tempfile

        from harchoc.bench_config import load_bench_config
        from harchoc.queue_skip_gates import matrix_train_verified

        repo = Path(__file__).resolve().parents[1]
        configs = [
            repo / "configs/bench/yolov8m_default.yaml",
            repo / "configs/bench/yolov10m_default.yaml",
            repo / "configs/bench/yolo11m_default.yaml",
            repo / "configs/bench/yolo26m_default.yaml",
        ]
        runs: list[dict[str, object]] = []
        for pth in configs:
            cfg = load_bench_config(pth)
            if pth.name.startswith("yolov10m"):
                runs.append(
                    {
                        "status": "skipped_no_weights",
                        "reason": "no_bench_run_weights",
                        "config_path": str(pth.resolve()),
                        "name": cfg.name,
                        "weights": None,
                    }
                )
            else:
                runs.append(
                    {
                        "status": "ok",
                        "config_path": str(pth.resolve()),
                        "name": cfg.name,
                        "weights": str(repo / "models/best2.pt"),
                        "test_count_mae": 61.3,
                    }
                )
        with tempfile.TemporaryDirectory() as td:
            train_out = Path(td) / "matrix_train.json"
            train_out.write_text(
                json.dumps(
                    {
                        "schema_version": "benchmark_matrix_train.v1",
                        "runs": runs,
                    }
                ),
                encoding="utf-8",
            )
            ok, reason = matrix_train_verified(
                repo, train_out, "zoo_yolo_only", accept_skipped_no_weights=True
            )
        self.assertTrue(ok, reason)

    def test_should_skip_zoo_matrix_when_verified(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "zoo_matrix_p0_5",
            "kind": "zoo_matrix_train",
            "matrix_group": "zoo_yolo_only",
            "skip_if": {"accept_skipped_no_weights": True},
            "out": "reports/hsp/matrix_train.json",
        }
        skip, _ = should_skip_job(job, repo_root=repo)
        self.assertFalse(skip)


if __name__ == "__main__":
    unittest.main()
