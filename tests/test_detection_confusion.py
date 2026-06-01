"""Tests for harchoc.detection_confusion (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class DetectionConfusionTests(unittest.TestCase):
    def test_tp_and_fn(self) -> None:
        from harchoc.detection_confusion import ConfusionMatrixAccumulator

        acc = ConfusionMatrixAccumulator()
        acc.update_image(
            [{"bbox": [0, 0, 10, 10], "category_id": 0}],
            [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9}],
            conf_thr=0.15,
            iou_thr=0.5,
        )
        acc.update_image(
            [{"bbox": [0, 0, 10, 10], "category_id": 1}],
            [],
            conf_thr=0.15,
            iou_thr=0.5,
        )
        self.assertEqual(acc.matrix[0][0], 1)
        self.assertEqual(acc.matrix[1][2], 1)
        self.assertEqual(acc.stats["tp"], 1)
        self.assertEqual(acc.stats["fn"], 1)

    def test_cross_class_confusion(self) -> None:
        from harchoc.detection_confusion import ConfusionMatrixAccumulator

        acc = ConfusionMatrixAccumulator()
        acc.update_image(
            [{"bbox": [0, 0, 10, 10], "category_id": 0}],
            [{"bbox": [0, 0, 10, 10], "category_id": 1, "score": 0.9}],
            conf_thr=0.15,
            iou_thr=0.5,
        )
        self.assertEqual(acc.matrix[0][1], 1)
        self.assertEqual(acc.stats["cls_confusion"], 1)

    def test_background_fp(self) -> None:
        from harchoc.detection_confusion import ConfusionMatrixAccumulator

        acc = ConfusionMatrixAccumulator()
        acc.update_image(
            [],
            [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9}],
            conf_thr=0.15,
            iou_thr=0.5,
        )
        self.assertEqual(acc.matrix[2][0], 1)
        self.assertEqual(acc.stats["fp"], 1)

    def test_confusion_matrix_from_exports(self) -> None:
        from harchoc.detection_confusion import confusion_matrix_from_exports

        gt = {
            "images": [
                {
                    "image_id": "a",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "a",
                    "detections": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.5}],
                }
            ]
        }
        acc = confusion_matrix_from_exports(gt, preds, conf_thr=0.15, iou_thr=0.5)
        self.assertEqual(acc.matrix[0][0], 1)
        self.assertEqual(acc.n_images, 1)

    def test_parity_with_error_analysis_counts(self) -> None:
        from harchoc.detection_confusion import confusion_matrix_from_exports
        from harchoc.error_analysis_core import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "a",
                    "file_name": "a.jpg",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [20, 20, 30, 30], "category_id": 1},
                    ],
                },
                {
                    "image_id": "b",
                    "file_name": "b.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "a",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9},
                        {"bbox": [0, 0, 10, 10], "category_id": 1, "score": 0.8},
                        {"bbox": [20, 20, 30, 30], "category_id": 1, "score": 0.7},
                        {"bbox": [50, 50, 60, 60], "category_id": 0, "score": 0.6},
                    ],
                },
                {
                    "image_id": "b",
                    "detections": [],
                },
            ]
        }
        summary = analyze_errors(gt=gt, preds=preds, conf_thr=0.15, iou_thr=0.5, iou_bg_thr=0.1)
        acc = confusion_matrix_from_exports(
            gt, preds, conf_thr=0.15, iou_thr=0.5, iou_bg_thr=0.1
        )
        counts = summary["counts"]
        self.assertEqual(acc.stats["tp"], counts["tp"])
        self.assertEqual(acc.stats["cls_confusion"], counts["cls_confusion"])
        self.assertEqual(acc.stats["dupe"], counts["dupe"])
        self.assertEqual(acc.stats["fn"], counts["fn"])
        self.assertEqual(acc.stats["fp"], counts["fp"])

    def test_parse_confusion_matrix_splits(self) -> None:
        from harchoc.detection_confusion import parse_confusion_matrix_splits

        repo = Path("/repo")
        splits = parse_confusion_matrix_splits("test,train", repo_root=repo)
        self.assertEqual(set(splits), {"test", "train"})
        self.assertEqual(splits["test"], Path("/repo/data/splits/test.txt"))

    def test_confusion_matrix_out_path(self) -> None:
        from harchoc.detection_confusion import confusion_matrix_out_path

        p = confusion_matrix_out_path(Path("reports/hsp/yolo26m_e100_s0"), "test")
        self.assertEqual(p, Path("reports/hsp/yolo26m_e100_s0_test_confusion.json"))

    def test_multi_split_loads_model_once(self) -> None:
        import sys

        from harchoc.detection_confusion import confusion_matrix_multi_split

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "dataset"
            img_dir = ds / "images" / "train"
            lbl_dir = ds / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            img = img_dir / "a.jpg"
            img.write_bytes(b"\xff\xd8\xff\xd9")
            (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            split = root / "train.txt"
            split.write_text("images/train/a.jpg\n", encoding="utf-8")
            weights = root / "w.pt"
            weights.write_bytes(b"x")

            fake_model = mock.MagicMock()
            fake_model.predict.return_value = [mock.MagicMock(boxes=None)]
            fake_ultra = mock.MagicMock()
            fake_ultra.YOLO.return_value = fake_model

            with mock.patch.dict(sys.modules, {"ultralytics": fake_ultra}):
                with mock.patch(
                    "harchoc.eval_export.iter_split_image_paths",
                    return_value=[("a", img, "images/train/a.jpg")],
                ):
                    with mock.patch(
                        "harchoc.eval_export.read_image_size",
                        return_value=(100, 100),
                    ):
                        out = confusion_matrix_multi_split(
                            weights=weights,
                            splits={"train": split, "test": split},
                            dataset_root=ds,
                            conf_thr=0.15,
                            iou_thr=0.5,
                            device="cpu",
                        )
            self.assertEqual(set(out), {"train", "test"})
            self.assertEqual(fake_ultra.YOLO.call_count, 1)
            self.assertEqual(fake_model.predict.call_count, 2)

    def test_eval_dry_run_includes_confusion_splits(self) -> None:
        from scripts.eval import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dry.json"
            rc = main(
                [
                    "--dry-run",
                    "--confusion-matrix-only",
                    "--confusion-matrix-out",
                    "reports/hsp/run",
                    "--confusion-matrix-splits",
                    "test,train",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(obj["confusion_matrix_splits"], "test,train")

    def test_eval_confusion_from_exports(self) -> None:
        from scripts.eval import main

        gt = {
            "images": [
                {
                    "image_id": "a",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "a",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.5}
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gt_path = root / "gt.json"
            preds_path = root / "preds.json"
            cm_path = root / "cm.json"
            run_out = root / "eval_run.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
            weights = root / "w.pt"
            weights.write_bytes(b"x")
            rc = main(
                [
                    "--confusion-matrix-only",
                    "--confusion-from-exports",
                    "--export-gt-json",
                    str(gt_path),
                    "--export-preds-json",
                    str(preds_path),
                    "--confusion-matrix-out",
                    str(cm_path),
                    "--weights",
                    str(weights),
                    "--out",
                    str(run_out),
                ]
            )
            self.assertEqual(rc, 0)
            cm = json.loads(cm_path.read_text(encoding="utf-8"))
            self.assertEqual(cm["stats"]["tp"], 1)
            self.assertEqual(cm["match"]["iou"], 0.3)
            run_doc = json.loads(run_out.read_text(encoding="utf-8"))
            self.assertTrue(run_doc.get("confusion_from_exports"))


if __name__ == "__main__":
    unittest.main()
