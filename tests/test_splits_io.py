import tempfile
import unittest
from pathlib import Path

from harchoc.splits_io import (
    abs_paths_from_split_file,
    iter_split_list_lines,
    materialize_abs_split_list,
    read_split_list,
    resolve_split_entry,
)


class SplitsIoTests(unittest.TestCase):
    def test_iter_split_list_lines_skips_blanks_and_comments(self) -> None:
        lines = ["\n", "  images/a.png  ", "# comment", "#", "images/b.png"]
        self.assertEqual(list(iter_split_list_lines(lines)), ["images/a.png", "images/b.png"])

    def test_read_split_list_missing_ok(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.txt"
            self.assertEqual(read_split_list(missing, missing_ok=True), [])
            with self.assertRaises(FileNotFoundError):
                read_split_list(missing)

    def test_read_split_list_as_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "split.txt"
            p.write_text("# header\n\nimages/train/a.png\n", "utf-8")
            rels = read_split_list(p, as_paths=True)
            self.assertEqual(rels, [Path("images/train/a.png")])

    def test_resolve_split_entry_relative_and_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img = root / "images" / "train" / "a.png"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            rel = resolve_split_entry("images/train/a.png", dataset_root=root)
            self.assertEqual(rel, img.resolve())
            self.assertEqual(resolve_split_entry(img, dataset_root=root), img.resolve())

    def test_abs_paths_and_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img = root / "images" / "val" / "b.jpg"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            split_txt = Path(td) / "split.txt"
            split_txt.write_text("images/val/b.jpg\n", "utf-8")

            abs_list = abs_paths_from_split_file(split_source=split_txt, dataset_root=root)
            self.assertEqual(abs_list, [str(img.resolve())])

            out_txt = Path(td) / "out_abs.txt"
            materialize_abs_split_list(
                split_source=split_txt, dataset_root=root, out_path=out_txt
            )
            self.assertEqual(out_txt.read_text("utf-8"), f"{img.resolve()}\n")

    def test_train_eval_split_entry_helpers_match_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img = root / "images" / "test" / "a.jpg"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            split_txt = Path(td) / "split.txt"
            split_txt.write_text("# header\n\nimages/test/a.jpg\n", "utf-8")
            expected = f"{img.resolve()}\n"

            from scripts.eval import _val_entry_for_yaml
            from scripts.train import _split_entry_for_yaml

            eval_out = Path(td) / "eval_out"
            eval_txt = Path(
                _val_entry_for_yaml(dataset_root=root, val_source=split_txt, out_dir=eval_out)
            )
            self.assertEqual(eval_txt.name, "eval_val_abs_paths.txt")
            self.assertEqual(eval_txt.read_text("utf-8"), expected)

            train_out = Path(td) / "train_out"
            train_txt = Path(
                _split_entry_for_yaml(
                    dataset_root=root,
                    split_source=split_txt,
                    out_dir=train_out,
                    split_name="val",
                )
            )
            self.assertEqual(train_txt.name, "val_abs_paths.txt")
            self.assertEqual(train_txt.read_text("utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
