"""Tests for manuscript docx figure/table reproduction (CI-safe)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestManuscriptDocxRepro(unittest.TestCase):
    def test_dry_run_writes_catalog(self) -> None:
        from harchoc.manuscript_docx_repro import run_manuscript_docx_repro

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports/manuscript/docx").mkdir(parents=True)
            payload = run_manuscript_docx_repro(repo, out_dir="reports/manuscript/docx", dry_run=True)
            self.assertTrue(payload.get("dry_run"))
            cat = repo / "reports/manuscript/docx/catalog.json"
            self.assertTrue(cat.is_file())

    def test_confusion_plots_from_fixture(self) -> None:
        from harchoc.manuscript_docx_repro import plot_confusion_matrix

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "abs.png"
            r = plot_confusion_matrix(
                [[10, 1, 2], [0, 8, 1], [3, 2, 0]],
                ["dev", "abort", "bg"],
                out_path=out,
                normalized=False,
                title="t",
            )
            self.assertEqual(r["status"], "ok")
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
