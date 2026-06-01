import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")


class CocoDetectionExportTests(unittest.TestCase):
    def test_export_split_produces_coco_categories(self) -> None:
        from harchoc.coco_detection_export import export_split_to_coco_json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img = root / "images" / "train" / "a.jpg"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"\xff\xd8\xff\xe0")  # minimal jpeg header; PIL may fail → 1x1 ok
            lbl = root / "labels" / "train" / "a.txt"
            lbl.parent.mkdir(parents=True)
            lbl.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            split = root / "train.txt"
            split.write_text("images/train/a.jpg\n", encoding="utf-8")
            coco = export_split_to_coco_json(split_file=split, dataset_root=root)
            self.assertEqual(len(coco["categories"]), 2)
            self.assertGreaterEqual(len(coco["images"]), 1)
            self.assertGreaterEqual(len(coco["annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
