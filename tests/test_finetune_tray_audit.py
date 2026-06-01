"""Tests for weak-tray audit plan."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class FinetuneTrayAuditTests(unittest.TestCase):
    def test_ranks_by_count_mae(self) -> None:
        from harchoc.finetune_tray_audit import build_weak_tray_plan

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "domain_count_mae.json"
            path.write_text(
                json.dumps(
                    {
                        "domains": [
                            {"tray_key": "a", "count_mae": 50.0},
                            {"tray_key": "b", "count_mae": 90.0},
                            {"tray_key": "c", "count_mae": 70.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = build_weak_tray_plan(count_mae_path=path, domain_eval_path=None, top_k=2)
            self.assertEqual(plan["recommended_tray_keys"], ["b", "c"])
            self.assertEqual(plan["status"], "ok")


if __name__ == "__main__":
    unittest.main()
