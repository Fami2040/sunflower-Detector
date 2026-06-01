"""Unit tests for journal figure style helpers."""

from __future__ import annotations

import unittest

from harchoc.figure_style import FIGURE_DPI, panel_label, savefig_kwargs


class TestFigureStyle(unittest.TestCase):
    def test_panel_labels(self) -> None:
        self.assertEqual(panel_label(0), "A")
        self.assertEqual(panel_label(25), "Z")
        self.assertEqual(panel_label(26), "AA")

    def test_savefig_kwargs_journal(self) -> None:
        kw = savefig_kwargs(journal_style=True)
        self.assertEqual(kw["dpi"], FIGURE_DPI)
        self.assertEqual(kw["bbox_inches"], "tight")

    def test_savefig_kwargs_legacy(self) -> None:
        kw = savefig_kwargs(journal_style=False)
        self.assertEqual(kw["dpi"], 120)


if __name__ == "__main__":
    unittest.main()
