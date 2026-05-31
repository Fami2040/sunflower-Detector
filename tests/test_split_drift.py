import json
import os
import struct
import tempfile
import unittest
from pathlib import Path


def _write_min_png(p: Path, w: int, h: int) -> None:
    # Minimal PNG with IHDR only (sufficient for our size reader).
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # RGB
    length = struct.pack(">I", len(ihdr_data))
    ctype = b"IHDR"
    crc = b"\x00\x00\x00\x00"  # not validated by reader
    p.write_bytes(sig + length + ctype + ihdr_data + crc)


class SplitDriftTests(unittest.TestCase):
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

    def test_split_drift_uses_split_lists_and_compares(self) -> None:
        from scripts.split_drift import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"

            # Create images + labels.
            for split, w, h, classes in [
                ("train", 10, 10, [0, 0]),
                ("val", 20, 10, [0, 1, 1]),
                ("test", 10, 20, []),
            ]:
                img_dir = root / "images" / split
                lbl_dir = root / "labels" / split
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)

                img = img_dir / "a.png"
                _write_min_png(img, w, h)
                if classes:
                    lines = "\n".join([f"{c} 0.5 0.5 0.1 0.1" for c in classes]) + "\n"
                    (lbl_dir / "a.txt").write_text(lines, "utf-8")
                else:
                    (lbl_dir / "a.txt").write_text("", "utf-8")

            # Split lists under data/splits.
            splits_dir = root / "data" / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "train.txt").write_text("images/train/a.png\n", "utf-8")
            (splits_dir / "val.txt").write_text("images/val/a.png\n", "utf-8")
            (splits_dir / "test.txt").write_text("images/test/a.png\n", "utf-8")

            out = Path(td) / "out.json"
            with self._with_dataset_root(root):
                rc = main(["--out", str(out)])
            self.assertEqual(rc, 0)

            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["status"], "ok")
            self.assertEqual(obj["script"], "split_drift")

            self.assertEqual(obj["splits"]["train"]["images"]["count"], 1)
            self.assertEqual(obj["splits"]["val"]["images"]["count"], 1)
            self.assertEqual(obj["splits"]["test"]["images"]["count"], 1)

            # Verify resolution differences made it into split stats.
            self.assertEqual(obj["splits"]["train"]["images"]["width"]["mean"], 10.0)
            self.assertEqual(obj["splits"]["val"]["images"]["width"]["mean"], 20.0)
            self.assertEqual(obj["splits"]["test"]["images"]["height"]["mean"], 20.0)

            # Verify class counts differ between splits.
            self.assertEqual(obj["splits"]["train"]["labels"]["class_counts"], {"0": 2})
            self.assertEqual(obj["splits"]["val"]["labels"]["class_counts"], {"0": 1, "1": 2})
            self.assertEqual(obj["splits"]["test"]["labels"]["class_counts"], {})

            # Pairwise comparison sanity.
            tv = obj["comparisons"]["train_vs_val"]
            self.assertAlmostEqual(tv["images"]["width_mean_ratio"], 2.0, places=7)  # 20 / 10
            self.assertAlmostEqual(tv["labels"]["boxes_per_image_mean_ratio"], 1.5, places=7)  # 3 / 2
            self.assertGreaterEqual(tv["labels"]["class_dist_l1"], 0.0)
            self.assertIsNotNone(tv["labels"]["class_jsd_nats"])

    def test_split_drift_with_ks_records_payload(self) -> None:
        from scripts.split_drift import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            for split, w, h, classes in [
                ("train", 10, 10, [0, 0]),
                ("val", 20, 10, [0, 1, 1]),
            ]:
                img_dir = root / "images" / split
                lbl_dir = root / "labels" / split
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                img = img_dir / "a.png"
                _write_min_png(img, w, h)
                lines = "\n".join([f"{c} 0.5 0.5 0.1 0.1" for c in classes]) + "\n"
                (lbl_dir / "a.txt").write_text(lines, "utf-8")

            splits_dir = root / "data" / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "train.txt").write_text("images/train/a.png\n", "utf-8")
            (splits_dir / "val.txt").write_text("images/val/a.png\n", "utf-8")
            (splits_dir / "test.txt").write_text("", "utf-8")

            out = Path(td) / "out.json"
            with self._with_dataset_root(root):
                rc = main(["--out", str(out), "--with-ks", "--ks-limit", "10"])
            self.assertEqual(rc, 0)

            obj = json.loads(out.read_text("utf-8"))
            tv = obj["comparisons"]["train_vs_val"]
            self.assertIn("width_ks", tv["images"])
            self.assertTrue(tv["images"]["width_ks"]["available"])
            self.assertIn("boxes_per_image_ks", tv["labels"])
            self.assertTrue(tv["labels"]["boxes_per_image_ks"]["available"])

    def test_split_drift_acceptance_in_report(self) -> None:
        from scripts.split_drift import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            for split, w, h in [("train", 10, 10), ("val", 10, 10)]:
                img_dir = root / "images" / split
                lbl_dir = root / "labels" / split
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                img = img_dir / "a.png"
                _write_min_png(img, w, h)
                (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", "utf-8")
            splits_dir = root / "data" / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "train.txt").write_text("images/train/a.png\n", "utf-8")
            (splits_dir / "val.txt").write_text("images/val/a.png\n", "utf-8")
            (splits_dir / "test.txt").write_text("images/val/a.png\n", "utf-8")
            out = Path(td) / "out.json"
            with self._with_dataset_root(root):
                rc = main(["--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertIn("acceptance", obj)
            self.assertIn(obj["acceptance"]["status"], ("ok", "warn", "fail"))

    def test_split_drift_extended_block(self) -> None:
        from scripts.split_drift import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            cases = [
                ("train", "100-1-1-a", 100, 100, [(0, 0.1, 0.1), (0, 0.1, 0.1)]),
                ("train", "100-1-1-b", 100, 100, [(0, 0.1, 0.1)]),
                ("val", "200-2-1-c", 100, 100, [(1, 0.5, 0.5)]),
                ("test", "300-3-1-d", 100, 100, []),
            ]
            for split, stem, w, h, boxes in cases:
                img_dir = root / "images" / split
                lbl_dir = root / "labels" / split
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                img = img_dir / f"{stem}.png"
                _write_min_png(img, w, h)
                lines = "\n".join(f"{c} 0.5 0.5 {bw} {bh}" for c, bw, bh in boxes)
                (lbl_dir / f"{stem}.txt").write_text(lines + ("\n" if lines else ""), "utf-8")

            splits_dir = root / "data" / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / "train.txt").write_text(
                "images/train/100-1-1-a.png\nimages/train/100-1-1-b.png\n", "utf-8"
            )
            (splits_dir / "val.txt").write_text("images/val/200-2-1-c.png\n", "utf-8")
            (splits_dir / "test.txt").write_text("images/test/300-3-1-d.png\n", "utf-8")

            out = Path(td) / "out.json"
            with self._with_dataset_root(root):
                rc = main(["--out", str(out), "--extended"])
            self.assertEqual(rc, 0)

            obj = json.loads(out.read_text("utf-8"))
            ext = obj["extended"]
            train = ext["per_split"]["train"]
            val = ext["per_split"]["val"]

            self.assertAlmostEqual(train["per_class_boxes_per_image_mean"]["0"], 1.5, places=5)
            self.assertAlmostEqual(val["per_class_boxes_per_image_mean"]["1"], 1.0, places=5)
            self.assertEqual(train["bbox_area_px_quantiles"]["n"], 3)
            self.assertEqual(val["bbox_area_px_quantiles"]["n"], 1)
            self.assertEqual(train["images_per_tray"]["n_trays"], 1)
            self.assertEqual(train["images_per_tray"]["per_tray_counts"]["mean"], 2.0)
            self.assertEqual(val["images_per_tray"]["n_trays"], 1)

            tv = ext["comparisons"]["train_vs_val"]
            self.assertIn("per_class_boxes_per_image_mean_ratio", tv)
            self.assertIsNotNone(tv["bbox_area_px_q50_ratio"])
            self.assertGreater(tv["bbox_area_px_q50_ratio"], 1.0)
            self.assertEqual(tv["tray_key_jaccard"], 0.0)

    def test_build_extended_drift_block_unit(self) -> None:
        from harchoc.split_drift_extended import build_extended_drift_block, collect_split_extended_stats

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img_dir = root / "images" / "train"
            lbl_dir = root / "labels" / "train"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            img = img_dir / "349-10-2-img.png"
            _write_min_png(img, 50, 40)
            (lbl_dir / "349-10-2-img.txt").write_text("0 0.5 0.5 0.2 0.25\n", "utf-8")

            stats = collect_split_extended_stats(
                dataset_root=root,
                split_list=["images/train/349-10-2-img.png"],
            )
            self.assertEqual(stats["n_images_scanned"], 1)
            self.assertAlmostEqual(stats["per_class_boxes_per_image_mean"]["0"], 1.0)
            area = 0.2 * 0.25 * 50 * 40
            self.assertAlmostEqual(stats["bbox_area_px_quantiles"]["q50"], area, places=3)
            self.assertEqual(stats["images_per_tray"]["tray_keys"], ["349-10-2"])

            block = build_extended_drift_block(
                dataset_root=root,
                split_lists={"train": ["images/train/349-10-2-img.png"], "val": [], "test": []},
            )
            self.assertIn("comparisons", block)
            self.assertIn("train", block["per_split"])


if __name__ == "__main__":
    unittest.main()

