"""Tests for finetune base selection after joined close3 study."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harchoc.finetune_base_selection import (
    build_finetune_base_selection,
    pick_finetune_base_candidate,
)


class TestFinetuneBaseSelection(unittest.TestCase):
    def test_pick_keeps_best2_when_retrain_worse(self) -> None:
        winner = pick_finetune_base_candidate(
            [
                {
                    "id": "JOINED_V8M",
                    "weights": "runs/joined_close3_yolov8m_e100_s0/weights/best.pt",
                    "test_count_mae": 64.13,
                }
            ],
            anchor_mae=61.3,
        )
        self.assertEqual(winner["id"], "best2")
        self.assertEqual(winner["weights"], "models/best2.pt")
        self.assertIn("keep_production", winner["reason"])

    def test_pick_retrain_when_beats_anchor(self) -> None:
        winner = pick_finetune_base_candidate(
            [
                {
                    "id": "HYPOTHETICAL",
                    "weights": "runs/hypothetical/weights/best.pt",
                    "test_count_mae": 58.0,
                }
            ],
            anchor_mae=61.3,
        )
        self.assertEqual(winner["id"], "HYPOTHETICAL")
        self.assertTrue(winner["beats_anchor"])

    def test_build_from_fixture_summaries(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        fixture = {
            "schema_version": "aug_smoke_summary.v1",
            "smoke_id": "JOINED_CLOSE3_YOLOV8M_100EP",
            "run_name": "joined_close3_yolov8m_e100_s0",
            "status": "complete",
            "train": {"weights": "runs/joined_close3_yolov8m_e100_s0/weights/best.pt"},
            "test_count_mae": 64.12844036697248,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summ_dir = root / "reports/gpu_queue/summaries"
            summ_dir.mkdir(parents=True)
            (summ_dir / "joined_close3_yolov8m_100ep.json").write_text(
                json.dumps(fixture), encoding="utf-8"
            )
            doc = build_finetune_base_selection(root, anchor_mae=61.3)
            self.assertEqual(doc["schema_version"], "finetune_base_selection.v1")
            self.assertEqual(doc["stage1_base_weights"], "models/best2.pt")
            self.assertEqual(len(doc["joined_close3_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
