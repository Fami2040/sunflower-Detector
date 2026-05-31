import os
import tempfile
import unittest
from pathlib import Path


class SplitToolingTests(unittest.TestCase):
    def _with_dataset_root(self, root: Path):
        class _Env:
            def __enter__(self_nonlocal):
                self_nonlocal._old = os.environ.get("DATASET_ROOT")
                os.environ["DATASET_ROOT"] = str(root)
                return self_nonlocal

            def __exit__(self_nonlocal, exc_type, exc, tb):
                if self_nonlocal._old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = self_nonlocal._old

        return _Env()

    def _mk_pair(self, root: Path, rel_img: str) -> None:
        img_path = root / rel_img
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"fake")

        rel_lbl = Path(rel_img)
        if rel_lbl.parts and rel_lbl.parts[0] == "images":
            rel_lbl = Path("labels", *rel_lbl.parts[1:])
        rel_lbl = rel_lbl.with_suffix(".txt")
        lbl_path = root / rel_lbl
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("0 0.5 0.5 1 1\n", "utf-8")

    def test_make_splits_from_folders_and_validate(self) -> None:
        from scripts.make_splits import main as make_main
        from scripts.validate_splits import main as validate_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            self._mk_pair(root, "images/train/a.jpg")
            self._mk_pair(root, "images/train/b.png")
            self._mk_pair(root, "images/val/c.jpg")

            with self._with_dataset_root(root):
                rc = make_main(["--mode", "from-folders"])
                self.assertEqual(rc, 0)

                splits_dir = root / "data" / "splits"
                self.assertTrue((splits_dir / "train.txt").exists())
                self.assertTrue((splits_dir / "val.txt").exists())
                self.assertTrue((splits_dir / "test.txt").exists())

                # validate should pass (test split can be empty but file exists)
                rc2 = validate_main(["--require-test"])
                self.assertEqual(rc2, 0)

    def test_make_splits_random_is_deterministic(self) -> None:
        from scripts.make_splits import main as make_main
        from scripts.validate_splits import main as validate_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            for i in range(10):
                self._mk_pair(root, f"images/all/img_{i:02d}.jpg")

            with self._with_dataset_root(root):
                rc = make_main(
                    [
                        "--mode",
                        "random",
                        "--glob",
                        "images/all/*.jpg",
                        "--seed",
                        "123",
                        "--val-frac",
                        "0.2",
                        "--test-frac",
                        "0.1",
                    ]
                )
                self.assertEqual(rc, 0)

                splits_dir = root / "data" / "splits"
                t1 = (splits_dir / "train.txt").read_text("utf-8")
                v1 = (splits_dir / "val.txt").read_text("utf-8")
                te1 = (splits_dir / "test.txt").read_text("utf-8")

                # re-run should match exactly
                rc2 = make_main(
                    [
                        "--mode",
                        "random",
                        "--glob",
                        "images/all/*.jpg",
                        "--seed",
                        "123",
                        "--val-frac",
                        "0.2",
                        "--test-frac",
                        "0.1",
                    ]
                )
                self.assertEqual(rc2, 0)
                self.assertEqual(t1, (splits_dir / "train.txt").read_text("utf-8"))
                self.assertEqual(v1, (splits_dir / "val.txt").read_text("utf-8"))
                self.assertEqual(te1, (splits_dir / "test.txt").read_text("utf-8"))

                self.assertEqual(validate_main(["--require-test"]), 0)

    def test_validate_splits_rtdetr_query_cap_warns_on_dense_labels(self) -> None:
        from scripts.validate_splits import main as validate_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            splits_dir = root / "data" / "splits"
            splits_dir.mkdir(parents=True)
            rel = "images/train/dense.jpg"
            img_path = root / rel
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(b"fake")
            lbl_path = root / "labels" / "train" / "dense.txt"
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.write_text("\n".join(["0 0.5 0.5 0.1 0.1"] * 5) + "\n", "utf-8")
            (splits_dir / "train.txt").write_text(rel + "\n", "utf-8")
            (splits_dir / "val.txt").write_text("", "utf-8")
            (splits_dir / "test.txt").write_text("", "utf-8")

            with self._with_dataset_root(root):
                rc = validate_main(
                    [
                        "--require-test",
                        "--check-rtdetr-query-cap",
                        "--num-queries",
                        "3",
                        "--documented-peak-gt-boxes",
                        "5",
                    ]
                )
                self.assertEqual(rc, 1)

    def test_validate_splits_fails_on_missing_label(self) -> None:
        from scripts.make_splits import main as make_main
        from scripts.validate_splits import main as validate_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            self._mk_pair(root, "images/train/a.jpg")
            self._mk_pair(root, "images/val/b.jpg")

            # Remove one label
            (root / "labels" / "val" / "b.txt").unlink()

            with self._with_dataset_root(root):
                self.assertEqual(make_main(["--mode", "from-folders"]), 0)
                self.assertNotEqual(validate_main(["--require-test"]), 0)

    def test_make_splits_random_group_key_prefix(self) -> None:
        from scripts.make_splits import main as make_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            # Same 8-char prefix -> same group
            for name in ("plotAAA_01.jpg", "plotAAA_02.jpg", "plotBBB_01.jpg", "plotBBB_02.jpg"):
                self._mk_pair(root, f"images/all/{name}")

            with self._with_dataset_root(root):
                rc = make_main(
                    [
                        "--mode",
                        "random",
                        "--glob",
                        "images/all/*.jpg",
                        "--seed",
                        "99",
                        "--val-frac",
                        "0.5",
                        "--test-frac",
                        "0.0",
                        "--group-key",
                        "prefix:7",
                    ]
                )
                self.assertEqual(rc, 0)

                splits_dir = root / "data" / "splits"
                train = (splits_dir / "train.txt").read_text("utf-8").splitlines()
                val = (splits_dir / "val.txt").read_text("utf-8").splitlines()

                def _prefix(rel: str) -> str:
                    return Path(rel).stem[:7]

                train_p = {_prefix(r) for r in train if r.strip()}
                val_p = {_prefix(r) for r in val if r.strip()}
                self.assertFalse(train_p & val_p)

    def test_validate_splits_audit_leakage_fails_on_collision(self) -> None:
        from scripts.validate_splits import main as validate_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            splits_dir = root / "data" / "splits"
            splits_dir.mkdir(parents=True)
            self._mk_pair(root, "images/train/leaf.jpg")
            self._mk_pair(root, "images/val/leaf_aug.jpg")
            (splits_dir / "train.txt").write_text("images/train/leaf.jpg\n", "utf-8")
            (splits_dir / "val.txt").write_text("images/val/leaf_aug.jpg\n", "utf-8")
            (splits_dir / "test.txt").write_text("", "utf-8")
            audit_out = root / "reports" / "leakage.json"

            with self._with_dataset_root(root):
                rc = validate_main(
                    [
                        "--require-test",
                        "--audit-leakage",
                        "--audit-leakage-out",
                        str(audit_out),
                    ]
                )
                self.assertEqual(rc, 3)
                self.assertTrue(audit_out.exists())
                self.assertIn('"ok": false', audit_out.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()

