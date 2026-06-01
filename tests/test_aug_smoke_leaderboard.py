"""Aug smoke leaderboard partitioning and equivalence-class ranking."""

from __future__ import annotations

import unittest
from pathlib import Path


class AugSmokeLeaderboardPartitionTests(unittest.TestCase):
    def test_real_repo_s1_ranked_once_s14_excluded_best_mae(self) -> None:
        from harchoc.aug_smoke_leaderboard import (
            build_leaderboard_payload,
            render_leaderboard_md,
        )

        repo = Path(__file__).resolve().parents[1]
        payload = build_leaderboard_payload(repo_root=repo)
        ranked_ids = [r["smoke_id"] for r in payload["ranked_rows"]]
        self.assertEqual(ranked_ids.count("S1"), 1)
        self.assertNotIn("S0", ranked_ids)
        self.assertNotIn("S13", ranked_ids)
        self.assertNotIn("S14", ranked_ids)
        audit_ids = {r["smoke_id"] for r in payload["audit_duplicate_rows"]}
        self.assertEqual(audit_ids, {"S0", "S6", "S7", "S13", "CLOSE25"})
        self.assertIn("S3", ranked_ids)
        self.assertEqual(ranked_ids.count("S3"), 1)
        self.assertNotIn("S6", ranked_ids)
        self.assertNotIn("S7", ranked_ids)
        eval_ids = {r["smoke_id"] for r in payload["eval_control_rows"]}
        self.assertNotIn("S14", ranked_ids)
        if "S14" in {r["smoke_id"] for r in payload["rows"]}:
            self.assertIn("S14", eval_ids)
        self.assertAlmostEqual(float(payload["best_smoke_mae"]), 68.90825688073394, places=3)
        md = render_leaderboard_md(payload)
        self.assertIn("## Audit / duplicate class", md)
        if eval_ids:
            self.assertIn("## Eval controls", md)
        clusters = payload.get("mae_clusters") or []
        dup = next((c for c in clusters if c.get("count", 0) >= 3), None)
        if dup:
            self.assertIsNotNone(dup.get("preds_sha256"))
            self.assertNotIn("unverified", (dup.get("interpretation") or "").lower())

    def test_fixture_partition_and_clusters(self) -> None:
        from harchoc.aug_smoke_leaderboard import (
            build_leaderboard_payload,
            find_mae_clusters,
            render_leaderboard_md,
        )

        repo = Path(__file__).resolve().parents[1]
        fixture_root = repo / "tests/fixtures/aug_smoke_leaderboard"
        payload = build_leaderboard_payload(
            repo_root=fixture_root,
            index_path="index.json",
            out_dir="reports/aug_smoke",
        )
        ranked_ids = [r["smoke_id"] for r in payload["ranked_rows"]]
        audit_ids = {r["smoke_id"] for r in payload["audit_duplicate_rows"]}
        self.assertIn("S1", ranked_ids)
        self.assertIn("S3", ranked_ids)
        self.assertNotIn("S0", ranked_ids)
        self.assertNotIn("S6", ranked_ids)
        self.assertNotIn("S7", ranked_ids)
        self.assertNotIn("S13", ranked_ids)
        self.assertNotIn("S14", ranked_ids)
        self.assertEqual(audit_ids, {"S0", "S6", "S7", "S13"})
        md = render_leaderboard_md(payload)
        self.assertIn("Duplicate MAE clusters", md)
        verified_preds = {
            round(float(c["test_count_mae"]), 12): str(c["preds_sha256"])
            for c in (payload.get("equivalence_classes") or {}).get("classes") or []
            if c.get("preds_sha256") is not None and c.get("test_count_mae") is not None
        }
        clusters = find_mae_clusters(
            payload["rows"],
            repo_root=fixture_root,
            verified_preds_by_mae=verified_preds,
        )
        self.assertGreaterEqual(len(clusters), 2)
        photo = next(
            (c for c in clusters if set(c.get("smoke_ids") or []) == {"S3", "S6", "S7"}),
            None,
        )
        self.assertIsNotNone(photo)
        assert photo is not None
        self.assertEqual(
            photo.get("preds_sha256"),
            "41e79d287721faf99c9de709d4578b09dd8f62af6d3fe00ee2cabece52387f4d",
        )


if __name__ == "__main__":
    unittest.main()
