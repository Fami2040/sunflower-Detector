import tempfile
import unittest
from pathlib import Path

from harchoc.split_leakage_audit import (
    assign_splits_by_group,
    audit_split_leakage,
    load_group_csv,
    normalize_stem,
    resolve_group_id,
)


class SplitLeakageAuditTests(unittest.TestCase):
    def test_normalize_stem_strips_aug_suffix(self) -> None:
        self.assertEqual(normalize_stem("plot42_aug"), "plot42")
        self.assertEqual(normalize_stem("plot42_flip_2"), "plot42")

    def test_stem_collision_detected_across_splits(self) -> None:
        splits = {
            "train": ["images/train/foo.jpg"],
            "val": ["images/val/foo_copy.jpg"],
            "test": [],
        }
        # same stem after normalization if we use identical stems
        splits["val"] = ["images/val/foo.jpg"]
        report = audit_split_leakage(splits)
        self.assertFalse(report["ok"])
        self.assertTrue(report["stem_collisions"])

    def test_normalized_stem_collision(self) -> None:
        splits = {
            "train": ["images/train/field_a.jpg"],
            "val": ["images/val/field_a_aug.jpg"],
            "test": [],
        }
        report = audit_split_leakage(splits)
        self.assertFalse(report["ok"])
        self.assertTrue(report["normalized_stem_collisions"])

    def test_assign_splits_by_group_keeps_members_together(self) -> None:
        items = [f"images/all/g{g}_i{i}.jpg" for g in range(3) for i in range(2)]
        splits = assign_splits_by_group(
            items,
            group_for=lambda rel: Path(rel).stem.rsplit("_", 1)[0],
            seed=7,
            val_frac=0.34,
            test_frac=0.0,
        )
        for g in range(3):
            seen: set[str] = set()
            for split_name, rels in splits.items():
                for rel in rels:
                    if rel.startswith(f"images/all/g{g}_"):
                        seen.add(split_name)
            self.assertEqual(len(seen), 1, f"group g{g} appeared in {seen}")

    def test_csv_group_key_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "groups.csv"
            csv_path.write_text(
                "image,group\n"
                "images/all/a1.jpg,G1\n"
                "images/all/a2.jpg,G1\n"
                "images/all/b1.jpg,G2\n",
                encoding="utf-8",
            )
            idx = load_group_csv(csv_path, dataset_root=root)
            self.assertEqual(resolve_group_id("images/all/a1.jpg", spec=f"csv:{csv_path.name}", csv_index=idx), "G1")

            splits = assign_splits_by_group(
                ["images/all/a1.jpg", "images/all/a2.jpg", "images/all/b1.jpg"],
                group_for=lambda rel: resolve_group_id(rel, spec=f"csv:{csv_path.name}", csv_index=idx),
                seed=0,
                val_frac=0.5,
                test_frac=0.0,
            )
            g1_where = [sn for sn, rels in splits.items() for r in rels if "a1" in r or "a2" in r]
            self.assertEqual(len(set(g1_where)), 1)

            report = audit_split_leakage(
                splits,
                group_key_spec=f"csv:{csv_path.name}",
                csv_index=idx,
            )
            self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
