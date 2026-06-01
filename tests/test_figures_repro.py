"""CI-safe tests for manuscript figures-repro manifest and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


class TestFiguresReproManifest(unittest.TestCase):
    def test_default_fields_use_hsp_paths(self) -> None:
        from harchoc.figures_repro import default_figures_repro_fields

        f = default_figures_repro_fields()
        self.assertEqual(f["out_dir"], "reports/figures")
        self.assertIn("split_drift_p0", f["split_drift_report"])
        self.assertTrue(f["journal_style"])

    def test_argv_for_figures_repro_journal_flag(self) -> None:
        from harchoc.experiment_argv import argv_for_figures_repro

        argv = argv_for_figures_repro({"journal_style": False})
        self.assertIn("--no-journal-style", argv)
        argv_on = argv_for_figures_repro({})
        self.assertIn("--journal-style", argv_on)

    def test_build_manifest_from_run_payload(self) -> None:
        from harchoc.figures_repro import build_figures_repro_manifest
        from harchoc.figure_style import FIGURE_DPI

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            png = repo / "reports" / "figures" / "fig_concept.png"
            png.parent.mkdir(parents=True)
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            run_payload = {
                "schema_version": "figures_run.v1",
                "rendered": {
                    "fig_concept": {
                        "status": "ok",
                        "out_path": str(png),
                        "figsize_inches": [7.0, 5.75],
                    }
                },
                "figures": [{"id": "fig_concept", "status": "ok", "paths": [str(png)]}],
            }
            manifest = build_figures_repro_manifest(
                repo_root=repo,
                run_payload=run_payload,
                journal_style=True,
                out_dir="reports/figures",
                run_json="reports/figures/run.json",
            )
            self.assertEqual(manifest["schema_version"], "figures_repro_manifest.v1")
            self.assertEqual(manifest["figure_dpi"], FIGURE_DPI)
            self.assertTrue(manifest["validation"]["ok"])
            self.assertEqual(len(manifest["files"]), 1)
            self.assertEqual(manifest["files"][0]["figure_id"], "fig_concept")
            self.assertGreater(manifest["files"][0]["size_bytes"], 0)
            self.assertIn("sha256", manifest["files"][0])

    def test_validate_required_figures_partial(self) -> None:
        from harchoc.figures_repro import validate_required_figures

        manifest = {
            "files": [
                {"figure_id": "fig_concept", "status": "ok", "size_bytes": 100},
            ],
            "audit": {"skipped": []},
        }
        errs = validate_required_figures(manifest, require_ids=("fig_concept",))
        self.assertEqual(errs, [])
        errs2 = validate_required_figures(manifest, require_ids=("fig_pr_curve",))
        self.assertEqual(len(errs2), 1)

    def test_figures_repro_dry_run(self) -> None:
        from harchoc.figures_repro import run_figures_repro

        repo = Path(__file__).resolve().parents[1]
        rc = run_figures_repro(
            {"figure": "fig_concept", "dry_run": True},
            repo_root=repo,
            dry_run=True,
        )
        self.assertEqual(rc, 0)

    def test_load_figures_repro_bundle(self) -> None:
        from harchoc.figures_repro import figures_repro_fields_from_bundle, load_figures_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        bundle = load_figures_repro_bundle(repo / "configs/experiments/figures_repro.json")
        fields = figures_repro_fields_from_bundle(bundle)
        self.assertEqual(fields["manifest_out"], "reports/figures/manifest.json")
        self.assertEqual(fields["figure"], "all")

    @unittest.skipUnless(_has_matplotlib(), "matplotlib not installed")
    def test_run_figures_repro_fig_concept_only(self) -> None:
        from harchoc.figures_repro import run_figures_repro

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            out_dir = repo / "figures"
            meta = repo / "run.json"
            manifest = repo / "manifest.json"
            rc = run_figures_repro(
                {
                    "out_dir": str(out_dir),
                    "meta_out": str(meta),
                    "manifest_out": str(manifest),
                    "figure": "fig_concept",
                    "error_report": "",
                    "journal_style": True,
                },
                repo_root=repo,
                dry_run=False,
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "fig_concept.png").is_file())
            self.assertTrue(manifest.is_file())
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(doc["schema_version"], "figures_repro_manifest.v1")
            self.assertTrue(doc["validation"]["ok"])


if __name__ == "__main__":
    unittest.main()
