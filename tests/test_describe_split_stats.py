import json
import os
import tempfile
import unittest
from pathlib import Path

import struct


class DescribeSplitStatsTests(unittest.TestCase):
    def test_describe_split_computes_basic_stats(self) -> None:
        from scripts.describe_split import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img_dir = root / "images" / "train"
            lbl_dir = root / "labels" / "train"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            # Two tiny images with deterministic sizes.
            def _write_min_png(p: Path, w: int, h: int) -> None:
                # Minimal PNG with IHDR only (sufficient for our size reader).
                sig = b"\x89PNG\r\n\x1a\n"
                ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # RGB
                length = struct.pack(">I", len(ihdr_data))
                ctype = b"IHDR"
                crc = b"\x00\x00\x00\x00"  # not validated by reader
                p.write_bytes(sig + length + ctype + ihdr_data + crc)

            a = img_dir / "a.png"
            b = img_dir / "b.png"
            _write_min_png(a, 10, 20)
            _write_min_png(b, 30, 40)

            # YOLO labels: class cx cy w h (values not used, only class ids and count).
            (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", "utf-8")
            (lbl_dir / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n", "utf-8")

            out = Path(td) / "out.json"
            old_root = os.environ.get("DATASET_ROOT")
            old_yaml = os.environ.get("YOLO_DATA_YAML")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                os.environ.pop("YOLO_DATA_YAML", None)
                rc = main(["--split", "train", "--out", str(out)])
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
                if old_yaml is None:
                    os.environ.pop("YOLO_DATA_YAML", None)
                else:
                    os.environ["YOLO_DATA_YAML"] = old_yaml

            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["status"], "ok")
            self.assertEqual(obj["split"], "train")

            self.assertEqual(obj["images"]["count"], 2)
            self.assertEqual(obj["images"]["width"]["mean"], 20.0)
            self.assertEqual(obj["images"]["width"]["median"], 20.0)
            self.assertEqual(obj["images"]["height"]["mean"], 30.0)
            self.assertEqual(obj["images"]["height"]["median"], 30.0)
            self.assertEqual(obj["images"]["file_size_bytes"]["n"], 2)
            self.assertIsNotNone(obj["images"]["file_size_bytes"]["min"])

            self.assertEqual(obj["labels"]["boxes_per_image"]["n"], 2)
            self.assertEqual(obj["labels"]["boxes_per_image"]["mean"], 1.5)
            self.assertEqual(obj["labels"]["class_counts"], {"0": 2, "1": 1})

            self.assertEqual(obj["density"]["total_boxes"], 3)
            self.assertAlmostEqual(obj["density"]["total_megapixels"], 0.0014, places=7)
            self.assertAlmostEqual(obj["density"]["boxes_per_megapixel"], 3.0 / 0.0014, places=6)

    def test_describe_split_reads_dataset_root_relative_split_list(self) -> None:
        from scripts.describe_split import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img_dir = root / "images" / "test"
            lbl_dir = root / "labels" / "test"
            splits_dir = root / "data" / "splits"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            splits_dir.mkdir(parents=True, exist_ok=True)

            sig = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 12, 34, 8, 2, 0, 0, 0)
            length = struct.pack(">I", len(ihdr_data))
            (img_dir / "a.png").write_bytes(sig + length + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00")
            (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", "utf-8")
            (splits_dir / "test.txt").write_text("images/test/a.png\n", "utf-8")

            out = Path(td) / "out.json"
            old_root = os.environ.get("DATASET_ROOT")
            old_yaml = os.environ.get("YOLO_DATA_YAML")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                os.environ.pop("YOLO_DATA_YAML", None)
                rc = main(
                    [
                        "--split",
                        "test",
                        "--split-file",
                        str(splits_dir / "test.txt"),
                        "--out",
                        str(out),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
                if old_yaml is None:
                    os.environ.pop("YOLO_DATA_YAML", None)
                else:
                    os.environ["YOLO_DATA_YAML"] = old_yaml

            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["images"]["count"], 1)
            self.assertEqual(obj["images"]["width"]["mean"], 12.0)


if __name__ == "__main__":
    unittest.main()

