"""Smoke tests for fig_concept pipeline diagram (CPU, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestConceptDiagram(unittest.TestCase):
    def test_load_concept_headlines_defaults_without_json(self) -> None:
        from harchoc.concept_diagram import load_concept_headlines

        with tempfile.TemporaryDirectory() as td:
            h = load_concept_headlines(hsp_dir=Path(td))
        self.assertEqual(h["locked_conf"], "0.15")
        self.assertEqual(h["test_mae"], "61.3")

    def test_load_concept_headlines_from_fp_budget_json(self) -> None:
        from harchoc.concept_diagram import load_concept_headlines

        with tempfile.TemporaryDirectory() as td:
            hsp = Path(td)
            val_payload = {
                "selection_comparison": [
                    {
                        "mode": "min_count_mae",
                        "selected": {"conf_thr": 0.2, "count_mae": 80.5},
                    }
                ]
            }
            test_payload = {
                "selection_comparison": [
                    {
                        "mode": "min_count_mae",
                        "selected": {"conf_thr": 0.2, "count_mae": 55.2},
                    }
                ]
            }
            (hsp / "fp_budget_sweep.json").write_text(json.dumps(val_payload), encoding="utf-8")
            (hsp / "fp_budget_sweep_test.json").write_text(json.dumps(test_payload), encoding="utf-8")
            h = load_concept_headlines(hsp_dir=hsp)
        self.assertEqual(h["locked_conf"], "0.2")
        self.assertEqual(h["val_mae"], "80.5")
        self.assertEqual(h["test_mae"], "55.2")

    def test_emit_concept_diagram_writes_png(self) -> None:
        from harchoc.concept_diagram import emit_concept_diagram

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fig_concept.png"
            result = emit_concept_diagram(out_path=out, journal_style=True, include_svg=False)
            self.assertEqual(result.get("status"), "ok")
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 5000)
            figsize = result.get("figsize_inches")
            self.assertIsInstance(figsize, list)
            assert isinstance(figsize, list)
            self.assertGreaterEqual(figsize[0], 7.0)
            self.assertGreaterEqual(figsize[1], 5.0)
            headlines = result.get("headlines")
            self.assertIsInstance(headlines, dict)
            assert isinstance(headlines, dict)
            self.assertIn("test_mae", headlines)

    def test_emit_concept_diagram_journal_larger_than_legacy_height(self) -> None:
        """Redesign targets readable double-column layout (taller than 2.6 in)."""
        from harchoc.concept_diagram import emit_concept_diagram

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "fig_concept.png"
            result = emit_concept_diagram(out_path=out, journal_style=True, include_svg=False)
            figsize = result.get("figsize_inches")
            assert isinstance(figsize, list)
            self.assertGreater(figsize[1], 2.6)

    def test_make_figures_fig_concept_only(self) -> None:
        from scripts.make_figures import main

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "figures"
            meta = Path(td) / "run.json"
            rc = main(
                [
                    "--out-dir",
                    str(out_dir),
                    "--meta-out",
                    str(meta),
                    "--figure",
                    "fig_concept",
                    "--no-journal-style",
                ]
            )
            self.assertEqual(rc, 0)
            png = out_dir / "fig_concept.png"
            self.assertTrue(png.is_file())


if __name__ == "__main__":
    unittest.main()
