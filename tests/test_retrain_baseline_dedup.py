"""Tests for zoo baseline retrain dedup (tensor-identical checkpoints)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harchoc.retrain_baseline_dedup import (
    annotate_train_compare_dedup_skips,
    checkpoint_max_tensor_diff,
    checkpoints_tensor_identical,
    infer_zoo_baseline_run_name,
    recipe_matches_zoo_baseline_ignoring_close_mosaic,
    resolve_dedup_baseline_weights,
    should_pre_skip_redundant_train,
)


class RetrainBaselineDedupTests(unittest.TestCase):
    def test_infer_zoo_baseline_run_name(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        name = infer_zoo_baseline_run_name(
            repo, "configs/experiments/train_yolo11m_close3_100ep.json"
        )
        self.assertEqual(name, "yolo11m_e100_s0")

    def test_recipe_matches_zoo_ignoring_close_mosaic(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        job = {
            "train_config": "configs/experiments/train_yolo11m_close3_100ep.json",
        }
        self.assertTrue(
            recipe_matches_zoo_baseline_ignoring_close_mosaic(job, repo_root=repo)
        )

    def test_v8m_close3_tensors_differ_from_zoo(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        zoo_w = repo / "runs/hsp_zoo/yolov8m_e100_s0/weights/best.pt"
        joined_w = repo / "runs/joined_close3_yolov8m_e100_s0/weights/best.pt"
        if not zoo_w.is_file() or not joined_w.is_file():
            self.skipTest("zoo/joined v8m weights missing")
        self.assertIsNotNone(checkpoint_max_tensor_diff(zoo_w, joined_w))
        self.assertFalse(checkpoints_tensor_identical(zoo_w, joined_w))

    def test_yolo11m_zoo_joined_tensor_identical_when_present(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        zoo_w = repo / "runs/hsp_zoo/yolo11m_e100_s0/weights/best.pt"
        joined_w = repo / "runs/joined_close3_yolo11m_e100_s0/weights/best.pt"
        if not zoo_w.is_file() or not joined_w.is_file():
            self.skipTest("zoo/joined 11m weights missing")
        self.assertEqual(checkpoint_max_tensor_diff(zoo_w, joined_w), 0.0)
        self.assertTrue(checkpoints_tensor_identical(zoo_w, joined_w))

    def test_should_pre_skip_11m_when_enabled(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        baseline = repo / "runs/hsp_zoo/yolo11m_e100_s0/weights/best.pt"
        if not baseline.is_file():
            self.skipTest("zoo 11m baseline missing")
        job = {
            "train_config": "configs/experiments/train_yolo11m_close3_100ep.json",
            "dedup_pre_skip_train": True,
        }
        ok, reason = should_pre_skip_redundant_train(
            job, repo_root=repo, baseline=baseline
        )
        self.assertTrue(ok)
        self.assertIn("close_mosaic", reason)

    def test_annotate_skips_job_when_weights_match_baseline(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        if not (
            (repo / "runs/hsp_zoo/yolo11m_e100_s0/weights/best.pt").is_file()
            and (repo / "runs/joined_close3_yolo11m_e100_s0/weights/best.pt").is_file()
        ):
            self.skipTest("weights missing")
        jobs = [
            {
                "id": "joined_close3_yolo11m_100ep",
                "kind": "train_compare",
                "run_name": "joined_close3_yolo11m_e100_s0",
                "train_config": "configs/experiments/train_yolo11m_close3_100ep.json",
                "dedup_baseline_run_name": "yolo11m_e100_s0",
            }
        ]
        out = annotate_train_compare_dedup_skips(
            jobs, repo_root=repo, defaults={"dedup_baseline_runs_dir": "runs/hsp_zoo"}
        )
        self.assertTrue(out[0].get("skip"))
        self.assertIn("tensor-identical", out[0].get("skip_reason", ""))

    def test_run_job_pre_skips_train_for_11m_dedup(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        baseline = repo / "runs/hsp_zoo/yolo11m_e100_s0/weights/best.pt"
        if not baseline.is_file():
            self.skipTest("zoo 11m baseline missing")
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "joined_close3_yolo11m_dedup_test",
                "kind": "train_compare",
                "train_config": "configs/experiments/train_yolo11m_close3_100ep.json",
                "run_name": "joined_close3_yolo11m_dedup_test_s0",
                "eval_out_dir": "reports/hsp",
                "dedup_baseline_run_name": "yolo11m_e100_s0",
                "dedup_pre_skip_train": True,
                "dedup_skip_eval_when_identical": True,
                "skip_eval": True,
            }
            defaults = {"dedup_baseline_runs_dir": "runs/hsp_zoo"}
            with mock.patch("harchoc.gpu_queue._run_subprocess_stage", return_value=0):
                result = run_job(
                    job,
                    repo_root=repo,
                    defaults=defaults,
                    dry_run=False,
                    min_free_mib=5500,
                    log_root=log_root,
                )
            ctx = result.get("context") or {}
            self.assertTrue(ctx.get("dedup_pre_skip_train"))
            self.assertTrue(ctx.get("dedup_baseline_identical"))
            train_log = log_root / job["id"] / "train.log"
            self.assertTrue(train_log.is_file())
            self.assertIn("train skipped", train_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
