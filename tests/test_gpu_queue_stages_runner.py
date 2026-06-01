"""Tests for harchoc.gpu_queue (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._gpu_queue_fixtures import load_manifest_with_index, write_pending_fixture_index


class GpuQueueStagesRunnerTests(unittest.TestCase):
    def test_build_aug_smoke_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {"id": "aug_smoke_S3", "kind": "aug_smoke", "smoke_id": "S3"}
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("dry_run", ids)
        self.assertIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)

    def test_should_skip_complete_summary(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        summary = repo / "reports/aug_smoke/s0_summary.json"
        if not summary.is_file():
            self.skipTest("s0_summary.json missing")
        job = {"id": "x", "skip_if": {"summary": "reports/aug_smoke/s0_summary.json"}}
        skip, reason = should_skip_job(job, repo_root=repo)
        self.assertTrue(skip)
        self.assertIn("complete", reason)

    def test_should_not_skip_stale_gpu_queue_job_transcript(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        transcript = repo / "reports/gpu_queue/jobs/rtdetr_queries_smoke.json"
        if not transcript.is_file():
            self.skipTest("rtdetr_queries_smoke job transcript missing")
        job = {
            "id": "rtdetr_queries_smoke",
            "kind": "rtdetr_smoke",
            "run_name": "rtdetr_queries_smoke_15ep",
            "skip_if": {
                "summary": "reports/gpu_queue/jobs/rtdetr_queries_smoke.json",
                "eval_error_json": "reports/gpu_queue/eval/rtdetr_queries_smoke_15ep_error.json",
            },
        }
        skip, _ = should_skip_job(job, repo_root=repo)
        self.assertFalse(skip)

    def test_should_skip_when_verified_summary_and_eval_exist(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        summary = repo / "reports/aug_smoke/s1_summary.json"
        err = repo / "reports/aug_smoke/aug_smoke_close3_error.json"
        if not summary.is_file() or not err.is_file():
            self.skipTest("s1 summary or error json missing")
        job = {
            "id": "aug_smoke_S1",
            "kind": "aug_smoke",
            "skip_if": {
                "summary": "reports/aug_smoke/s1_summary.json",
                "eval_error_json": "reports/aug_smoke/aug_smoke_close3_error.json",
            },
        }
        skip, reason = should_skip_job(job, repo_root=repo)
        self.assertTrue(skip)
        self.assertIn("complete", reason)

    def test_run_job_dry_run_status_not_complete(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "rtdetr_imgsz640",
                "kind": "train_compare",
                "train_config": "configs/experiments/train_rtdetr_imgsz640_smoke_15ep.json",
                "run_name": "rtdetr_imgsz640_smoke_15ep",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")

    def test_dry_run_preflight_job(self) -> None:
        from harchoc.gpu_queue import run_gpu_queue

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "mini.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "gpu_queue_manifest.v1",
                        "jobs": [{"id": "preflight", "kind": "preflight"}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("harchoc.gpu_queue._run_subprocess_stage", return_value=0):
                rc = run_gpu_queue(
                    manifest,
                    repo_root=repo,
                    dry_run=True,
                    state_path=Path(td) / "state.json",
                )
            self.assertEqual(rc, 0)

    def test_wait_gpu_free_dry_run(self) -> None:
        from harchoc.gpu_queue import wait_gpu_free

        info = wait_gpu_free(min_free_mib=5500, dry_run=True)
        self.assertEqual(info["status"], "dry_run")

    def test_adhoc_train_blocked_when_lock(self) -> None:
        from harchoc.gpu_exclusive import acquire_gpu_exclusive, release_gpu_exclusive

        repo = Path(__file__).resolve().parents[1]
        acquire_gpu_exclusive(repo_root=repo, owner="test")
        try:
            from harchoc.gpu_exclusive import adhoc_train_blocked

            self.assertTrue(adhoc_train_blocked(repo_root=repo))
        finally:
            release_gpu_exclusive(repo_root=repo)

    def test_run_job_train_compare_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "rtdetr_imgsz640",
                "kind": "train_compare",
                "train_config": "configs/experiments/train_rtdetr_imgsz640_smoke_15ep.json",
                "run_name": "rtdetr_imgsz640_smoke_15ep",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")

    def test_build_rtdetr_train_compare_includes_hsp_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "rtdetr_imgsz640",
            "kind": "train_compare",
            "train_config": "configs/experiments/train_rtdetr_imgsz640_smoke_15ep.json",
            "run_name": "rtdetr_imgsz640_smoke_15ep",
            "max_det": 300,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("eval_test", ids)
        eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
        self.assertEqual(eval_stage["meta"]["max_det"], 300)
        summary = next(s for s in stages if s["stage_id"] == "summary")
        self.assertEqual(summary["meta"]["summary_kind"], "rtdetr")

    def test_run_rtdetr_smoke_dry_run_writes_eval_chain(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            for job_id, cfg, name, max_det in (
                (
                    "rtdetr_queries_smoke",
                    "configs/experiments/train_rtdetr_queries_smoke_15ep.json",
                    "rtdetr_queries_smoke_15ep",
                    1024,
                ),
                (
                    "rtdetr_imgsz1280",
                    "configs/experiments/train_rtdetr_smoke_15ep.json",
                    "rtdetr_imgsz1280_smoke_15ep",
                    300,
                ),
            ):
                kind = "rtdetr_smoke" if job_id == "rtdetr_queries_smoke" else "train_compare"
                job = {
                    "id": job_id,
                    "kind": kind,
                    "train_config": cfg,
                    "run_name": name,
                    "max_det": max_det,
                }
                result = run_job(
                    job,
                    repo_root=repo,
                    defaults={},
                    dry_run=True,
                    min_free_mib=5500,
                    log_root=log_root,
                )
                self.assertEqual(result["status"], "dry_run_complete", job_id)
                eval_log = log_root / job_id / "eval_test.log"
                self.assertTrue(eval_log.is_file(), job_id)
                text = eval_log.read_text(encoding="utf-8")
                self.assertIn("eval_export", text, job_id)
                self.assertIn(str(max_det), text, job_id)

    def test_zoo_matrix_dry_run_includes_rtdetr_gate_stage(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {"id": "zoo_matrix_p0_5", "kind": "zoo_matrix_train", "matrix_group": "zoo_core"}
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("rtdetr_15ep_gate", ids)

    def test_run_job_aug_smoke_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {"id": "aug_smoke_S1", "kind": "aug_smoke", "smoke_id": "S1"}
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            eval_log = log_root / "aug_smoke_S1" / "eval_test.log"
            self.assertTrue(eval_log.is_file())
            self.assertIn("eval_export", eval_log.read_text(encoding="utf-8"))

    def test_run_job_vram_probe_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "vram_probe_rtdetr",
                "kind": "vram_probe",
                "train_config": "configs/experiments/train_batch_probe_rtdetr-l.json",
                "run_name": "batch_probe_rtdetr-l",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            self.assertTrue((log_root / "vram_probe_rtdetr" / "gpu_wait.log").is_file())

    def test_build_aug_sweep_15_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "aug_sweep_15_mosaic0",
            "kind": "aug_sweep_15",
            "train_config": "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json",
            "aug_config": "configs/aug/robustness_mosaic_off.yaml",
            "run_name": "aug_sweep_mosaic0_15ep",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        cfg_path = repo / "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json"
        self.assertTrue(cfg_path.is_file())

    def test_should_skip_when_weights_exist_for_train_only_job(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        weights = repo / "runs/amp_on_smoke_15ep/weights/best.pt"
        if not weights.is_file():
            self.skipTest("amp_on_smoke_15ep weights missing")
        job = {
            "id": "amp_smoke_15ep_on",
            "kind": "amp_smoke",
            "run_name": "amp_on_smoke_15ep",
            "skip_eval": True,
            "skip_if": {
                "summary": "reports/hsp/amp_on_smoke_15ep_summary.json",
                "weights_run_name": "amp_on_smoke_15ep",
            },
        }
        skip, reason = should_skip_job(job, repo_root=repo)
        self.assertTrue(skip)
        self.assertTrue(
            "weights exist" in reason or "summary complete" in reason,
            reason,
        )

    def test_should_skip_eval_only_when_summary_or_hsp_artifacts_complete(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        summary = repo / "reports/hsp/amp_on_smoke_15ep_summary.json"
        weights = repo / "runs/amp_on_smoke_15ep/weights/best.pt"
        if not weights.is_file():
            self.skipTest("amp_on_smoke_15ep weights missing")
        job = {
            "id": "amp_smoke_15ep_on_hsp_eval",
            "kind": "amp_smoke",
            "run_name": "amp_on_smoke_15ep",
            "eval_only": True,
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "eval_out_dir": "reports/hsp",
            "skip_if": {"summary": "reports/hsp/amp_on_smoke_15ep_summary.json"},
        }
        skip, reason = should_skip_job(job, repo_root=repo)
        if summary.is_file():
            self.assertTrue(skip, reason)
            self.assertTrue(
                "summary complete" in reason or "HSP eval artifacts complete" in reason,
                reason,
            )
        else:
            self.assertFalse(skip)

    def test_build_amp_smoke_stages_train_only_skips_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "amp_smoke_15ep_on",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "run_name": "amp_on_smoke_15ep",
            "eval_out_dir": "reports/hsp",
            "skip_eval": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertNotIn("eval_test", ids)
        self.assertIn("summary", ids)

    def test_build_amp_smoke_stages_includes_hsp_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "amp_smoke_15ep_on_hsp_eval",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "run_name": "amp_on_smoke_15ep",
            "eval_only": True,
            "eval_out_dir": "reports/hsp",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
        self.assertEqual(eval_stage.get("internal"), "smoke_hsp_eval")
        self.assertEqual(eval_stage["meta"]["out_dir"], "reports/hsp")

    def test_build_aug_smoke_s14_eval_only_max_det_300(self) -> None:
        from harchoc.gpu_queue import build_job_stages, expand_aug_smoke_jobs_from_index

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = write_pending_fixture_index(repo, Path(td), ("S14",))
            job = next(
                j
                for j in expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
                if j.get("smoke_id") == "S14"
            )
            self.assertTrue(job.get("eval_only"))
            self.assertEqual(job.get("max_det"), 300)
            self.assertEqual(job.get("weights_run_name"), "aug_smoke_close3")
            stages = build_job_stages(job, repo_root=repo)
            ids = [s["stage_id"] for s in stages]
            self.assertNotIn("train", ids)
            self.assertIn("eval_test", ids)
            eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
            self.assertEqual(eval_stage["meta"]["max_det"], 300)
            self.assertEqual(eval_stage["meta"]["weights_run_name"], "aug_smoke_close3")

    def test_run_job_aug_smoke_s14_dry_run_eval_max_det(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        weights = repo / "runs/aug_smoke_close3/weights/best.pt"
        if not weights.is_file():
            self.skipTest("S1 weights missing")
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "aug_smoke_S14",
                "kind": "aug_smoke",
                "smoke_id": "S14",
                "run_name": "aug_smoke_eval300",
                "weights_run_name": "aug_smoke_close3",
                "eval_only": True,
                "max_det": 300,
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            text = (log_root / "aug_smoke_S14" / "eval_test.log").read_text(encoding="utf-8")
            self.assertIn("--export-max-det", text)
            self.assertIn("300", text)
            self.assertIn("aug_smoke_close3", text)
            self.assertIn("aug_smoke_eval300_preds.json", text)

    def test_build_amp_smoke_eval_only_skips_train(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "amp_smoke_15ep_on_hsp_eval",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "run_name": "amp_on_smoke_15ep",
            "eval_only": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)

    def test_run_job_amp_smoke_eval_only_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "amp_smoke_15ep_on_hsp_eval",
                "kind": "amp_smoke",
                "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
                "run_name": "amp_on_smoke_15ep",
                "eval_only": True,
                "eval_out_dir": "reports/hsp",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            eval_log = log_root / "amp_smoke_15ep_on_hsp_eval" / "eval_test.log"
            self.assertTrue(eval_log.is_file())
            text = eval_log.read_text(encoding="utf-8")
            self.assertIn("eval_export", text)
            self.assertIn("reports/hsp/amp_on_smoke_15ep_preds.json", text)

    def test_build_sg_smoke_stages_train_only_skips_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "sg_yolo_nas_s_smoke",
            "kind": "sg_smoke",
            "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
            "run_name": "sg_yolo_nas_s_smoke_15ep",
            "eval_out_dir": "reports/aug_smoke",
            "skip_eval": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertNotIn("eval_test", ids)

    def test_build_sg_smoke_stages_includes_hsp_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "sg_yolo_nas_s_hsp_eval",
            "kind": "sg_smoke",
            "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
            "run_name": "sg_yolo_nas_s_smoke_15ep",
            "eval_only": True,
            "eval_out_dir": "reports/aug_smoke",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
        self.assertEqual(eval_stage.get("internal"), "smoke_hsp_eval")

    def test_build_sg_smoke_eval_only_skips_train(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "sg_yolo_nas_s_hsp_eval",
            "kind": "sg_smoke",
            "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
            "run_name": "sg_yolo_nas_s_smoke_15ep",
            "eval_only": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)

    def test_run_job_sg_smoke_eval_only_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "sg_yolo_nas_s_hsp_eval",
                "kind": "sg_smoke",
                "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
                "run_name": "sg_yolo_nas_s_smoke_15ep",
                "eval_only": True,
                "eval_out_dir": "reports/aug_smoke",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            eval_log = log_root / "sg_yolo_nas_s_hsp_eval" / "eval_test.log"
            self.assertTrue(eval_log.is_file())
            text = eval_log.read_text(encoding="utf-8")
            self.assertIn("error_analysis", text)
            self.assertIn("reports/aug_smoke/sg_yolo_nas_s_smoke_15ep_error.json", text)

    def test_infer_smoke_eval_backend(self) -> None:
        from harchoc.aug_smoke_runner import infer_smoke_eval_backend

        self.assertEqual(infer_smoke_eval_backend("runs/x/weights/best.pth"), "supergradients")
        self.assertEqual(infer_smoke_eval_backend("runs/x/weights/best.pt"), "ultralytics")

    def test_validate_missing_train_config_raises(self) -> None:
        from harchoc.gpu_queue import _validate_job_files

        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(FileNotFoundError):
            _validate_job_files(
                {
                    "id": "bad",
                    "kind": "train_compare",
                    "train_config": "configs/experiments/no_such_train.json",
                },
                repo,
            )


if __name__ == "__main__":
    unittest.main()
