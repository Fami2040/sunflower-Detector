"""Tests for post-zoo reviewer2-repro chain (CI-safe)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestReviewer2Repro(unittest.TestCase):
    def test_load_bundle_schema(self) -> None:
        from harchoc.reviewer2_repro import load_reviewer2_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        bundle = load_reviewer2_repro_bundle(repo / "configs/experiments/reviewer2_repro.json")
        self.assertEqual(bundle["schema_version"], "reviewer2_repro_bundle.v1")
        self.assertIn("reviewer_counting", bundle["configs"])

    def test_build_chain_order(self) -> None:
        from harchoc.reviewer2_repro import build_reviewer2_repro_chain, load_reviewer2_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        bundle = load_reviewer2_repro_bundle(repo / "configs/experiments/reviewer2_repro.json")
        steps = build_reviewer2_repro_chain(bundle, repo_root=repo, global_dry_run=False)
        ids = [s[0] for s in steps]
        self.assertEqual(
            ids,
            [
                "reviewer_counting",
                "reviewer2_map50",
                "reviewer2_confusion",
                "reviewer2_paste_check",
            ],
        )
        self.assertTrue(any("reviewer-counting" in argv for _, argv in steps))

    def test_missing_exports_forces_confusion_dry_run(self) -> None:
        from harchoc.reviewer2_repro import build_reviewer2_repro_chain, load_reviewer2_repro_bundle

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "scripts").mkdir(parents=True)
            bundle = {
                "schema_version": "reviewer2_repro_bundle.v1",
                "hsp_artifacts": {
                    "gt_test": "reports/hsp/gt_test.json",
                    "preds_test": "reports/hsp/preds_test.json",
                },
                "configs": {},
            }
            steps = build_reviewer2_repro_chain(bundle, repo_root=repo)
            confusion_argv = dict(steps)["reviewer2_confusion"]
            self.assertIn("--dry-run", confusion_argv)
            counting_argv = dict(steps)["reviewer_counting"]
            self.assertIn("--dry-run", counting_argv)

    def test_run_chain_dry_run_no_subprocess(self) -> None:
        from harchoc.reviewer2_repro import load_reviewer2_repro_bundle, run_reviewer2_repro_chain

        repo = Path(__file__).resolve().parents[1]
        bundle = load_reviewer2_repro_bundle(repo / "configs/experiments/reviewer2_repro.json")
        runner = mock.Mock(return_value=0)
        rc = run_reviewer2_repro_chain(
            bundle,
            repo_root=repo,
            dry_run=True,
            run_argv=runner,
        )
        self.assertEqual(rc, 0)
        runner.assert_not_called()

    def test_experiment_reviewer2_repro_dry_run(self) -> None:
        from scripts.experiment import main

        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        with mock.patch("harchoc.repro_chain.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)
            rc = main(["--dry-run", "reviewer2-repro"])
        self.assertEqual(rc, 0)
        run_mock.assert_not_called()

    def test_experiment_repro_stage_post_zoo(self) -> None:
        from scripts.experiment import main

        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        with mock.patch("harchoc.repro_chain.subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)
            rc = main(["repro", "--stage", "post-zoo", "--dry-run"])
        self.assertEqual(rc, 0)
        run_mock.assert_not_called()

    def test_manuscript_bundle_has_post_zoo_block(self) -> None:
        from harchoc.manuscript_repro import load_manuscript_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        ms = load_manuscript_repro_bundle(repo / "configs/experiments/manuscript_repro_bundle.json")
        block = ms.get("post_zoo_reviewer2") or {}
        self.assertEqual(block.get("bundle"), "configs/experiments/reviewer2_repro.json")


if __name__ == "__main__":
    unittest.main()
