"""Tests for tray-targeted finetune split composition."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harchoc.domain_tags import tray_key_from_stem


class FinetuneTraySplitsTests(unittest.TestCase):
    def test_tray_adapt_composes_and_blocks_test_leak(self) -> None:
        from harchoc.finetune_tray_splits import compose_tray_adapt_splits

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "dataset"
            ds.mkdir()
            splits = root / "splits"
            domains = root / "domains"
            splits.mkdir()
            domains.mkdir()
            work = root / "work"

            img_train = "images/train/a_349-10-2.jpg"
            img_val = "images/val/b_349-10-2.jpg"
            img_test = "images/test/c_349-10-2.jpg"
            (domains / "train_349-10-2.txt").write_text(f"{img_train}\n", encoding="utf-8")
            (domains / "val_349-10-2.txt").write_text(f"{img_val}\n", encoding="utf-8")
            (splits / "test.txt").write_text(f"{img_test}\n", encoding="utf-8")

            plan = compose_tray_adapt_splits(
                ["349-10-2"],
                domains_dir=domains,
                splits_dir=splits,
                dataset_root=ds,
                work_dir=work,
            )
            self.assertEqual(plan.train_mode, "tray_adapt")
            self.assertEqual(plan.n_train, 2)
            self.assertTrue(plan.train_split_file.is_file())
            self.assertTrue(plan.val_split_file.is_file())

            (domains / "train_349-10-2.txt").write_text(f"{img_train}\n{img_test}\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                compose_tray_adapt_splits(
                    ["349-10-2"],
                    domains_dir=domains,
                    splits_dir=splits,
                    dataset_root=ds,
                    work_dir=work / "leak",
                )

    def test_lofo_pool_excludes_holdout_tray(self) -> None:
        from harchoc.finetune_tray_splits import compose_lofo_pool_splits

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "dataset"
            ds.mkdir()
            splits = root / "splits"
            splits.mkdir()
            work = root / "work"

            entries = [
                "images/train/349-10-2_a.jpg",
                "images/train/3a2-2_b.jpg",
                "images/val/349-10-2_c.jpg",
            ]
            (splits / "train.txt").write_text("\n".join(entries[:2]) + "\n", encoding="utf-8")
            (splits / "val.txt").write_text(entries[2] + "\n", encoding="utf-8")
            (splits / "test.txt").write_text("images/test/holdout.jpg\n", encoding="utf-8")

            plan = compose_lofo_pool_splits(
                ["349-10-2"],
                splits_dir=splits,
                dataset_root=ds,
                work_dir=work,
            )
            self.assertEqual(plan.train_mode, "lofo_pool")
            train_lines = plan.train_split_file.read_text(encoding="utf-8").splitlines()
            keys = {tray_key_from_stem(Path(line).stem) for line in train_lines if line.strip()}
            self.assertIn("3a2-2", keys)
            self.assertNotIn("349-10-2", keys)

    def test_tray_key_from_stem(self) -> None:
        self.assertEqual(tray_key_from_stem("349-10-2_aug0"), "349-10-2")


if __name__ == "__main__":
    unittest.main()
