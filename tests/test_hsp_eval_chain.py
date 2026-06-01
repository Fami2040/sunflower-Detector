"""Tests for harchoc.hsp_eval_chain (CI-safe, no GPU)."""

from __future__ import annotations

import unittest
from pathlib import Path


class HspEvalChainTests(unittest.TestCase):
    def test_prefix_paths(self) -> None:
        from harchoc.hsp_eval_chain import hsp_eval_prefix_paths

        repo = Path(__file__).resolve().parents[1]
        paths = hsp_eval_prefix_paths(repo, "smoke_run", "reports/aug_smoke")
        self.assertTrue(str(paths["gt"]).endswith("reports/aug_smoke/smoke_run_gt.json"))
        self.assertTrue(str(paths["error"]).endswith("_error.json"))

    def test_build_ultralytics_hsp_stages(self) -> None:
        from harchoc.hsp_eval_chain import build_ultralytics_hsp_stages

        repo = Path(__file__).resolve().parents[1]
        stages = build_ultralytics_hsp_stages(
            repo_root=repo,
            run_name="chain_test",
            weights="runs/x/weights/best.pt",
            out_dir="reports/aug_smoke",
        )
        self.assertEqual([s[0] for s in stages], ["eval_export", "error_analysis"])
        export_argv = stages[0][1]
        self.assertIn("scripts/eval.py", export_argv)
        self.assertIn("0.001", export_argv)
        self.assertIn("0.3", export_argv)
        error_argv = stages[1][1]
        self.assertIn("scripts/error_analysis.py", error_argv)
        self.assertIn("reports/aug_smoke/chain_test_error.json", error_argv)

    def test_build_error_analysis_argv_relative(self) -> None:
        from harchoc.hsp_eval_chain import build_error_analysis_argv

        repo = Path(__file__).resolve().parents[1]
        argv = build_error_analysis_argv(
            "reports/a_gt.json",
            "reports/a_preds.json",
            "reports/hsp/threshold_val.json",
            "reports/a_error.json",
            repo_root=repo,
        )
        self.assertEqual(argv[0], "scripts/error_analysis.py")
        self.assertIn("--locked-conf-from", argv)


if __name__ == "__main__":
    unittest.main()
