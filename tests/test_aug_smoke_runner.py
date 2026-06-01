"""Tests for harchoc.aug_smoke_runner (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class AugSmokeRunnerTests(unittest.TestCase):
    def test_build_aug_smoke_eval_stages_paths(self) -> None:
        from harchoc.aug_smoke_runner import build_aug_smoke_eval_stages

        repo = Path(__file__).resolve().parents[1]
        stages = build_aug_smoke_eval_stages(
            repo_root=repo,
            run_name="aug_smoke_test",
            weights="runs/foo/weights/best.pt",
        )
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0][0], "eval_export")
        self.assertIn("scripts/eval.py", stages[0][1])
        self.assertTrue(any("aug_smoke_test_preds.json" in str(x) for x in stages[0][1]))

    def test_extract_count_mae_from_error_json(self) -> None:
        from harchoc.aug_smoke_runner import extract_count_mae

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "err.json"
            p.write_text(
                json.dumps({"counting_metrics": {"mae": 42.5, "mae_ci": {"point": 42.5}}}),
                encoding="utf-8",
            )
            mae, ci = extract_count_mae(p)
            self.assertAlmostEqual(mae or 0, 42.5)
            self.assertIsNotNone(ci)

    def test_artifact_fingerprints_on_existing_preds(self) -> None:
        from harchoc.aug_smoke_runner import artifact_fingerprints

        repo = Path(__file__).resolve().parents[1]
        preds = repo / "reports/aug_smoke/aug_smoke_close3_preds.json"
        weights = repo / "runs/aug_smoke_close3/weights/best.pt"
        if not preds.is_file() or not weights.is_file():
            self.skipTest("aug smoke artifacts missing")
        fp = artifact_fingerprints(weights=weights, preds_json=preds)
        self.assertEqual(
            fp["preds_json"]["sha256"],
            "ad6f1621d8c2c8a1c1db2000626f0fc17f9c19da83348aa92bbc1ba4862607e8",
        )
        self.assertNotEqual(fp["weights"]["sha256"], fp["preds_json"]["sha256"])


if __name__ == "__main__":
    unittest.main()
