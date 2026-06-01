import json
import tempfile
import unittest
from pathlib import Path


class ThresholdLockTests(unittest.TestCase):
    def test_load_locked_conf_from_selected(self) -> None:
        from harchoc.threshold_lock import load_locked_conf

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sweep.json"
            p.write_text(
                json.dumps(
                    {
                        "selected": {"row": {"conf_thr": 0.42, "f1": 0.9}},
                    }
                ),
                encoding="utf-8",
            )
            self.assertAlmostEqual(load_locked_conf(p), 0.42)

    def test_metrics_row_at_conf(self) -> None:
        from harchoc.threshold_lock import metrics_row_at_conf

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
                    "detections": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.8}],
                }
            ]
        }
        row = metrics_row_at_conf(
            gt=gt,
            preds=preds,
            conf_thr=0.5,
            iou_thr=0.5,
            category_aware=True,
            n_images=1,
        )
        self.assertEqual(row["tp"], 1)
        self.assertAlmostEqual(row["conf_thr"], 0.5)


class ThresholdSweepLockedConfTests(unittest.TestCase):
    def test_locked_conf_from_val_sweep(self) -> None:
        from scripts.threshold_sweep import main

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
                    "detections": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.8}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset_root = td_path / "dataset"
            dataset_root.mkdir()
            gt_path = td_path / "gt.json"
            preds_path = td_path / "preds.json"
            val_sweep = td_path / "val.json"
            out = td_path / "test_locked.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
            val_sweep.write_text(
                json.dumps({"selected": {"row": {"conf_thr": 0.5}}}),
                encoding="utf-8",
            )
            rc = main(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--gt-json",
                    str(gt_path),
                    "--preds-json",
                    str(preds_path),
                    "--locked-conf-from",
                    str(val_sweep),
                    "--out",
                    str(out),
                    "--steps",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertIn("locked", obj)
            self.assertAlmostEqual(obj["locked"]["row"]["conf_thr"], 0.5)
            self.assertEqual(obj["locked"]["row"]["tp"], 1)
            self.assertIn("counting_metrics", obj["locked"])
            self.assertEqual(obj["locked"]["counting_metrics"]["n_images"], 1)

    def test_load_locked_match_iou(self) -> None:
        from harchoc.threshold_lock import load_locked_match_iou

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sweep.json"
            p.write_text(
                json.dumps(
                    {
                        "selected": {"nms_iou": {"selected": 0.45}, "row": {"conf_thr": 0.3}},
                        "match": {"iou": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            locked_iou = load_locked_match_iou(p)
            self.assertIsNotNone(locked_iou)
            assert locked_iou is not None
            self.assertAlmostEqual(locked_iou, 0.45)


class GradcamPanelTests(unittest.TestCase):
    def test_plan_from_report(self) -> None:
        from harchoc.gradcam_panel import plan_gradcam_panel

        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "fp_crops": {
                            "results": [
                                {
                                    "status": "ok",
                                    "error_type": "background",
                                    "score": 0.9,
                                    "crop_path": "/tmp/a.png",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            plan = plan_gradcam_panel(report_path=report, max_panels=12)
            self.assertEqual(plan["n_selected"], 1)


if __name__ == "__main__":
    unittest.main()
