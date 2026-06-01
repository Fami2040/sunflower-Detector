"""Doc drift guards — EXPERIMENTS.md must not reintroduce stale queue/smoke content."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "docs" / "EXPERIMENTS.md"
AUG_SMOKE_INDEX = REPO / "configs" / "experiments" / "aug_smoke_index.json"
STALE_DEDUP_MD = "reports/aug_smoke/dedup_root_cause.md"
INDEX_REF = "configs/experiments/aug_smoke_index.json"

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

    def test_aug_dedup_canonical_registry_and_doc_links(self) -> None:
        """Preds/recipe dedup audit lives in aug_smoke_index equivalence_classes, not orphan markdown."""
        from harchoc.equivalence_index import parse_equivalence_classes

        index = json.loads(AUG_SMOKE_INDEX.read_text(encoding="utf-8"))
        classes = (index.get("equivalence_classes") or {}).get("classes") or []
        self.assertGreaterEqual(len(classes), 2, "equivalence_classes.classes required")

        class_ids = {tuple(sorted(c.get("smoke_ids") or [])) for c in classes}
        self.assertIn(tuple(sorted(["S0", "S1", "S13", "CLOSE25"])), class_ids)
        self.assertIn(tuple(sorted(["S3", "S6", "S7"])), class_ids)

        for cls in classes:
            with self.subTest(smoke_ids=cls.get("smoke_ids")):
                self.assertTrue(cls.get("canonical_smoke_id"))
                self.assertTrue(cls.get("preds_sha256"))
                self.assertGreaterEqual(len(cls.get("smoke_ids") or []), 2)

        audit_only, verified_preds, _ = parse_equivalence_classes(index)
        self.assertEqual(audit_only, {"S0", "S6", "S7", "S13", "CLOSE25"})
        self.assertIn(round(68.90825688073394, 12), verified_preds)
        self.assertIn(round(151.73394495412845, 12), verified_preds)

        for path in (EXPERIMENTS, REPO / "backlog.md", REPO / "docs" / "zoo_comparison_design.md"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(doc=path.name):
                self.assertIn(INDEX_REF, text)
                self.assertIn("equivalence_classes", text)
                self.assertNotIn(STALE_DEDUP_MD, text)
                self.assertNotIn("docs/dedup_root_cause.md", text)


if __name__ == "__main__":
    unittest.main()
