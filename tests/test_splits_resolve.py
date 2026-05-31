import tempfile
import unittest
from pathlib import Path

from harchoc.splits_io import resolve_splits_dir


class ResolveSplitsDirTests(unittest.TestCase):
    def test_prefers_dataset_splits_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            sd = root / "data" / "splits"
            sd.mkdir(parents=True)
            (sd / "val.txt").write_text("images/val/x.jpg\n", encoding="utf-8")
            got = resolve_splits_dir(dataset_root=root, splits_dir="data/splits")
            self.assertEqual(got, sd.resolve())


if __name__ == "__main__":
    unittest.main()
