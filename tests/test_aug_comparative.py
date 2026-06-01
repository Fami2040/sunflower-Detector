"""Aug comparative analysis from aug_smoke_index fixture (CPU-only)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class AugComparativeTests(unittest.TestCase):
    def test_fixture_payload_rankings_rejected_narrative(self) -> None:
        from harchoc.aug_comparative import build_comparative_payload

        repo = Path(__file__).resolve().parents[1]
        fixture_root = repo / "tests/fixtures/aug_smoke_leaderboard"
        payload = build_comparative_payload(
            repo_root=fixture_root,
            index_path="index.json",
            out_dir="reports/aug_smoke",
        )
        self.assertEqual(payload["schema_version"], "aug_comparative_analysis.v1")
        ranked_ids = [r["smoke_id"] for r in payload["rankings"]]
        self.assertEqual(ranked_ids[0], "S1")
        self.assertNotIn("S0", ranked_ids)
        self.assertNotIn("S6", ranked_ids)
        rejected_ids = {a["smoke_id"] for a in payload["rejected_arms"]}
        self.assertIn("S2", rejected_ids)
        s2 = next(a for a in payload["rejected_arms"] if a["smoke_id"] == "S2")
        self.assertAlmostEqual(float(s2["test_count_mae"]), 147.41284403669724, places=3)
        bullets = payload.get("narrative_bullets") or []
        self.assertTrue(any("S1" in b for b in bullets))
        self.assertTrue(any("S2" in b for b in bullets))
        self.assertTrue(any("best2" in b.lower() for b in bullets))
        equiv = payload.get("equivalence_classes") or {}
        self.assertGreaterEqual(len(equiv.get("classes") or []), 1)

    def test_write_outputs_smoke(self) -> None:
        from harchoc.aug_comparative import write_aug_comparative_analysis

        repo = Path(__file__).resolve().parents[1]
        fixture_root = repo / "tests/fixtures/aug_smoke_leaderboard"
        out_dir = fixture_root / "reports/aug_smoke_compare"
        paths = write_aug_comparative_analysis(
            repo_root=fixture_root,
            index_path="index.json",
            out_dir=str(out_dir.relative_to(fixture_root)),
            write_figure=True,
        )
        self.assertTrue(paths["json"].is_file())
        obj = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(obj["schema_version"], "aug_comparative_analysis.v1")
        self.assertTrue(paths["markdown"].is_file())
        if "figure" in paths:
            self.assertTrue(paths["figure"].is_file())

    def test_real_repo_includes_s2_from_index(self) -> None:
        from harchoc.aug_comparative import build_comparative_payload

        repo = Path(__file__).resolve().parents[1]
        payload = build_comparative_payload(repo_root=repo)
        rejected_ids = {a["smoke_id"] for a in payload["rejected_arms"]}
        self.assertIn("S2", rejected_ids)
        self.assertIn("CLOSE10", rejected_ids)
        confirm = payload.get("reference_aug_confirm_100ep") or {}
        self.assertAlmostEqual(float(confirm["test_count_mae"]), 64.12844036697248, places=2)
        bullets = " ".join(payload.get("narrative_bullets") or [])
        self.assertIn("64.1", bullets)
        self.assertIn("61.3", bullets)


if __name__ == "__main__":
    unittest.main()
