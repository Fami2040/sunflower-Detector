"""GPU queue CLI dry-run (run_gpu_queue.py only; not experiment.py)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock


class RunGpuQueueCliTests(unittest.TestCase):
    def test_run_gpu_queue_dry_run_aug_pending(self) -> None:
        from scripts.run_gpu_queue import main as run_gpu_queue_main

        repo = Path(__file__).resolve().parents[1]
        manifest = repo / "configs/experiments/archive/gpu_queue_aug_pending.json"
        if not manifest.is_file():
            self.skipTest("gpu_queue_aug_pending.json missing")

        with mock.patch("harchoc.gpu_queue._run_subprocess_stage", return_value=0):
            with mock.patch("harchoc.gpu_queue.wait_gpu_free") as wg:
                wg.return_value = {"status": "dry_run"}
                rc = run_gpu_queue_main(["--manifest", str(manifest), "--dry-run"])
        self.assertEqual(rc, 0)


class RepairResumeStateTests(unittest.TestCase):
    def test_resume_after_failure_keeps_completed(self) -> None:
        from harchoc.gpu_queue_runner import repair_resume_state

        repo = Path(__file__).resolve().parents[1]
        manifest = repo / "configs/experiments/gpu_queue_post_zoo.json"
        state = {
            "manifest": str(manifest.resolve()),
            "dry_run": False,
            "completed": ["domain_tray_audit_refresh"],
            "failed": {"job_id": "finetune_weak_tray_1", "stage_id": "finetune"},
        }
        out = repair_resume_state(state, manifest_path=manifest, dry_run=False)
        self.assertEqual(out["completed"], ["domain_tray_audit_refresh"])
        self.assertIsNone(out.get("failed"))

    def test_relative_manifest_path_does_not_wipe_completed(self) -> None:
        from harchoc.gpu_queue_runner import repair_resume_state

        repo = Path(__file__).resolve().parents[1]
        manifest = repo / "configs/experiments/gpu_queue_post_zoo.json"
        state = {
            "manifest": "configs/experiments/gpu_queue_post_zoo.json",
            "dry_run": False,
            "completed": ["domain_tray_audit_refresh"],
            "failed": None,
        }
        out = repair_resume_state(state, manifest_path=manifest, dry_run=False)
        self.assertEqual(out["completed"], ["domain_tray_audit_refresh"])


if __name__ == "__main__":
    unittest.main()
