"""Doc drift guards — EXPERIMENTS.md must not reintroduce stale queue/smoke content."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "docs" / "EXPERIMENTS.md"
DEDUP_ROOT_CAUSE = REPO / "reports" / "aug_smoke" / "dedup_root_cause.md"

# Substrings / patterns that indicate duplicated or obsolete doc state.
STALE_SUBSTRINGS = (
    "aug_sweep_15_mosaic",
    "8×100 ep",
    "8x100 ep",
)

STALE_PATTERNS = (
    re.compile(r"\| S6 \|.*\| complete \|", re.MULTILINE),
    re.compile(r"\| ID \| Run name \| Train config \| Aug YAML \| Status \|"),
)


class TestDocIndexParity(unittest.TestCase):
    def test_experiments_md_no_stale_strings(self) -> None:
        text = EXPERIMENTS.read_text(encoding="utf-8")
        for stale in STALE_SUBSTRINGS:
            with self.subTest(stale=stale):
                self.assertNotIn(
                    stale,
                    text,
                    f"Stale substring {stale!r} found in docs/EXPERIMENTS.md",
                )
        for pattern in STALE_PATTERNS:
            with self.subTest(pattern=pattern.pattern):
                self.assertIsNone(
                    pattern.search(text),
                    f"Stale pattern {pattern.pattern!r} matched docs/EXPERIMENTS.md",
                )

    def test_experiments_md_points_at_canonical_index_and_queue(self) -> None:
        text = EXPERIMENTS.read_text(encoding="utf-8")
        self.assertIn("aug_smoke_index.json", text)
        self.assertIn("gpu_queue_aug_pending.json", text)
        self.assertIn("./scripts/run_gpu_queue.sh", text)
        self.assertIn("backlog.md#runbook-gpu", text)

    def test_dedup_root_cause_canonical_path(self) -> None:
        self.assertTrue(
            DEDUP_ROOT_CAUSE.is_file(),
            "Canonical dedup notes must live at reports/aug_smoke/dedup_root_cause.md",
        )
        self.assertFalse(
            (REPO / "docs" / "dedup_root_cause.md").exists(),
            "docs/dedup_root_cause.md is redundant — use reports/aug_smoke/ only",
        )
        experiments = EXPERIMENTS.read_text(encoding="utf-8")
        backlog = (REPO / "backlog.md").read_text(encoding="utf-8")
        self.assertIn("reports/aug_smoke/dedup_root_cause.md", experiments)
        self.assertIn("reports/aug_smoke/dedup_root_cause.md", backlog)
        self.assertNotIn("docs/dedup_root_cause.md", experiments)
        self.assertNotIn("docs/dedup_root_cause.md", backlog)


if __name__ == "__main__":
    unittest.main()
