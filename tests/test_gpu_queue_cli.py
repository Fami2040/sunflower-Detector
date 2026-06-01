"""Tests for harchoc.gpu_queue (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._gpu_queue_fixtures import load_manifest_with_index, write_pending_fixture_index

class ExperimentGpuQueueCliTests(unittest.TestCase):
    def _dry_run_gpu_queue_aug_pending(self, *, invoke) -> tuple[int, str]:
        repo = Path(__file__).resolve().parents[1]
        manifest = repo / "configs/experiments/archive/gpu_queue_aug_pending.json"
        if not manifest.is_file():
            self.skipTest("gpu_queue_aug_pending.json missing")
        import io
        from contextlib import redirect_stdout

        with mock.patch("harchoc.gpu_queue._run_subprocess_stage", return_value=0):
            with mock.patch("harchoc.gpu_queue.wait_gpu_free") as wg:
                wg.return_value = {"status": "dry_run"}
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = invoke(manifest)
        return rc, buf.getvalue()

    def test_gpu_queue_dry_run_cli(self) -> None:
        from scripts.experiment import main

        rc, _ = self._dry_run_gpu_queue_aug_pending(
            invoke=lambda manifest: main(
                ["gpu-queue", "--manifest", str(manifest), "--dry-run"]
            )
        )
        self.assertEqual(rc, 0)

    def test_gpu_queue_dry_run_experiment_matches_run_gpu_queue(self) -> None:
        from scripts.experiment import main as experiment_main
        from scripts.run_gpu_queue import main as run_gpu_queue_main

        rc_exp, out_exp = self._dry_run_gpu_queue_aug_pending(
            invoke=lambda manifest: experiment_main(
                ["gpu-queue", "--manifest", str(manifest), "--dry-run"]
            )
        )
        rc_cli, out_cli = self._dry_run_gpu_queue_aug_pending(
            invoke=lambda manifest: run_gpu_queue_main(
                ["--manifest", str(manifest), "--dry-run"]
            )
        )
        self.assertEqual(rc_exp, rc_cli)
        self.assertEqual(out_exp, out_cli)


if __name__ == "__main__":
    unittest.main()
