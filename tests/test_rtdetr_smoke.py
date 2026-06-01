from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")


class RtdetrSmokeTests(unittest.TestCase):
    def test_smoke_config_requests_eval_skip(self) -> None:
        from harchoc.post_train_eval import post_train_eval_skipped
        from harchoc.train_config import load_train_config_json

        repo = Path(__file__).resolve().parents[1]
        merged = load_train_config_json(
            repo / "configs" / "experiments" / "train_rtdetr_smoke_15ep.json",
            repo_root=repo,
        )
        eval_section = merged.get("eval") if isinstance(merged.get("eval"), dict) else {}
        self.assertTrue(post_train_eval_skipped(cli_skip=False, eval_section=eval_section))

    @patch("scripts.rtdetr_smoke.subprocess.run")
    @patch("scripts.rtdetr_smoke._gpu_check_via_mamba")
    @patch("scripts.rtdetr_smoke.should_reexec_in_mamba_for_torch", return_value=False)
    def test_run_train_marks_complete_without_failure_phase(
        self,
        _reexec: object,
        mock_gpu: object,
        mock_run: object,
    ) -> None:
        from scripts.rtdetr_smoke import main as smoke_main

        mock_gpu.return_value = ({"status": "ok", "torch": {"cuda_available": True}}, 0)

        class _Proc:
            returncode = 0

        mock_run.return_value = _Proc()

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.json"
            rc = smoke_main(["--run-train", "--out", str(out)])
            self.assertEqual(rc, 0)
            report = json.loads(out.read_text("utf-8"))
            self.assertEqual(report["status"], "train_complete")
            self.assertTrue(report["post_train_eval_skipped"])
            self.assertNotIn("failure_phase", report)

            train_cmd = mock_run.call_args[0][0]
            self.assertIn("--skip-eval", train_cmd)

    @patch("scripts.rtdetr_smoke.subprocess.run")
    @patch("scripts.rtdetr_smoke._gpu_check_via_mamba")
    @patch("scripts.rtdetr_smoke.should_reexec_in_mamba_for_torch", return_value=False)
    def test_run_train_failure_sets_failure_phase(
        self,
        _reexec: object,
        mock_gpu: object,
        mock_run: object,
    ) -> None:
        from scripts.rtdetr_smoke import main as smoke_main

        mock_gpu.return_value = ({"status": "ok", "torch": {"cuda_available": True}}, 0)

        class _Proc:
            returncode = 1

        mock_run.return_value = _Proc()

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.json"
            rc = smoke_main(["--run-train", "--out", str(out)])
            self.assertEqual(rc, 1)
            report = json.loads(out.read_text("utf-8"))
            self.assertEqual(report["status"], "train_failed")
            self.assertEqual(report["failure_phase"], "train")


if __name__ == "__main__":
    unittest.main()
