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

    def test_skips_trays_without_train_val_domain_splits(self) -> None:
        from harchoc.finetune_tray_audit import build_weak_tray_plan

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            domains = root / "domains"
            domains.mkdir()
            (domains / "train_ok-1.txt").write_text("images/train/x.jpg\n", encoding="utf-8")
            (domains / "val_ok-1.txt").write_text("images/val/y.jpg\n", encoding="utf-8")
            (domains / "test_only.txt").write_text("images/test/z.jpg\n", encoding="utf-8")

            count_mae = root / "domain_count_mae.json"
            count_mae.write_text(
                json.dumps(
                    {
                        "domains": [
                            {"tray_key": "only", "count_mae": 200.0},
                            {"tray_key": "ok-1", "count_mae": 90.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = build_weak_tray_plan(
                count_mae_path=count_mae,
                domain_eval_path=None,
                top_k=2,
                domains_dir=domains,
            )
            self.assertEqual(plan["recommended_tray_keys"], ["ok-1"])
            skipped = plan.get("skipped_not_finetunable") or []
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["tray_key"], "only")


if __name__ == "__main__":
    unittest.main()
