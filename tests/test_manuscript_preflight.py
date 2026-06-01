"""Tests for publication manuscript-preflight chain (CI-safe, mocked sub-steps)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestManuscriptPreflight(unittest.TestCase):
    def test_bundle_has_preflight_block(self) -> None:
        from harchoc.manuscript_repro import load_manuscript_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        ms = load_manuscript_repro_bundle(repo / "configs/experiments/manuscript_repro_bundle.json")
        pf = ms.get("manuscript_preflight") or {}
        self.assertEqual(pf.get("manifest"), "reports/manuscript/preflight_manifest.json")
        steps = pf.get("steps") or []
        for sid in (
            "reviewer2_repro",
            "figures_repro",
            "tables_repro",
            "docx_repro",
            "aug_compare",
            "backlog_narrative",
        ):
            self.assertIn(sid, steps)

    def test_run_preflight_dry_run_writes_manifest(self) -> None:
        from harchoc.manuscript_preflight import run_manuscript_preflight
        from harchoc.manuscript_repro import load_manuscript_repro_bundle

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports" / "manuscript").mkdir(parents=True)
            bundle = {
                "schema_version": "manuscript_repro_bundle.v1",
                "artifacts": {},
                "manuscript_preflight": {
                    "manifest": "reports/manuscript/preflight_manifest.json",
                    "steps": ["figures_repro", "backlog_narrative"],
                    "reviewer2": {"skip_if_missing_hsp_exports": True},
                    "figures": {"meta_out": "reports/figures/run.json"},
                    "backlog_narrative": {
                        "backlog": "backlog.md",
                        "out_md": "reports/manuscript/narrative_from_backlog.md",
                        "out_json": "reports/manuscript/backlog_narrative.json",
                    },
                },
            }
            (repo / "backlog.md").write_text("# Backlog\n\n## Now\n", encoding="utf-8")

            with mock.patch("harchoc.figures_repro.run_figures_repro", return_value=0) as fig_mock:
                with mock.patch(
                    "harchoc.backlog_narrative.run_backlog_narrative", return_value=0
                ) as bn_mock:
                    rc = run_manuscript_preflight(
                        bundle,
                        repo_root=repo,
                        dry_run=True,
                    )
            self.assertEqual(rc, 0)
            fig_mock.assert_called_once()
            bn_mock.assert_not_called()

            manifest_path = repo / "reports/manuscript/preflight_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "manuscript_preflight_manifest.v1")
            self.assertEqual(manifest["overall_status"], "dry_run")
            self.assertEqual(manifest["steps"]["figures_repro"]["status"], "dry_run")

    def test_reviewer2_skipped_when_exports_missing(self) -> None:
        from harchoc.manuscript_preflight import run_manuscript_preflight

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports" / "manuscript").mkdir(parents=True)
            bundle = {
                "schema_version": "manuscript_repro_bundle.v1",
                "post_zoo_reviewer2": {"bundle": "configs/experiments/reviewer2_repro.json"},
                "manuscript_preflight": {
                    "manifest": "reports/manuscript/preflight_manifest.json",
                    "steps": ["reviewer2_repro"],
                    "reviewer2": {
                        "bundle": "configs/experiments/reviewer2_repro.json",
                        "skip_if_missing_hsp_exports": True,
                    },
                },
            }
            r2_bundle = {
                "schema_version": "reviewer2_repro_bundle.v1",
                "hsp_artifacts": {
                    "gt_test": "reports/hsp/gt_test.json",
                    "preds_test": "reports/hsp/preds_test.json",
                },
                "configs": {},
            }

            with mock.patch(
                "harchoc.manuscript_preflight.load_reviewer2_repro_bundle",
                return_value=r2_bundle,
            ):
                with mock.patch(
                    "harchoc.manuscript_preflight.run_reviewer2_repro_chain"
                ) as chain_mock:
                    rc = run_manuscript_preflight(bundle, repo_root=repo, dry_run=False)
            self.assertEqual(rc, 0)
            chain_mock.assert_not_called()
            manifest = json.loads(
                (repo / "reports/manuscript/preflight_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["steps"]["reviewer2_repro"]["status"], "skipped")

    def test_experiment_manuscript_preflight_dry_run(self) -> None:
        from scripts.experiment import main

        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        with mock.patch("harchoc.manuscript_preflight.run_manuscript_preflight", return_value=0) as pf:
            rc = main(["manuscript-preflight", "--dry-run"])
        self.assertEqual(rc, 0)
        pf.assert_called_once()
        self.assertTrue(pf.call_args.kwargs.get("dry_run"))

    def test_experiment_repro_stage_preflight(self) -> None:
        from scripts.experiment import main

        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        with mock.patch("harchoc.manuscript_preflight.run_manuscript_preflight", return_value=0) as pf:
            rc = main(["repro", "--stage", "preflight", "--dry-run"])
        self.assertEqual(rc, 0)
        pf.assert_called_once()

    def test_experiment_repro_stage_full(self) -> None:
        from scripts.experiment import main

        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        with mock.patch(
            "harchoc.manuscript_preflight.run_publication_pipeline", return_value=0
        ) as pipe:
            rc = main(["repro", "--stage", "full", "--dry-run"])
        self.assertEqual(rc, 0)
        pipe.assert_called_once()
        self.assertTrue(pipe.call_args.kwargs.get("include_hsp"))
        self.assertTrue(pipe.call_args.kwargs.get("dry_run"))

    def test_run_repro_unknown_stage_raises(self) -> None:
        from scripts.experiment import _run_repro

        with self.assertRaises(ValueError):
            _run_repro({"stage": "not-a-stage"})


if __name__ == "__main__":
    unittest.main()
