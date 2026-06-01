import json
import tempfile
import unittest
from pathlib import Path


class ThresholdProtocolTests(unittest.TestCase):
    def test_infer_split_role_from_paths(self) -> None:
        from harchoc.threshold_protocol import infer_split_role

        role, _ = infer_split_role(gt_json="reports/hsp/gt_val.json", preds_json="reports/hsp/preds_val.json")
        self.assertEqual(role, "val")
        role2, _ = infer_split_role(split_file="data/splits/test.txt")
        self.assertEqual(role2, "test")

    def test_enforce_blocks_test_tuning(self) -> None:
        from harchoc.threshold_protocol import enforce_tuning_guardrails

        with self.assertRaises(SystemExit):
            enforce_tuning_guardrails("test", locked_conf_from=None, iou_grid=None)

    def test_locked_conf_allows_test(self) -> None:
        from harchoc.threshold_protocol import enforce_tuning_guardrails

        enforce_tuning_guardrails(
            "test",
            locked_conf_from="reports/hsp/threshold_val.json",
            iou_grid=[0.4, 0.5],
        )

    def test_build_iou_grid_explicit(self) -> None:
        from harchoc.threshold_protocol import build_iou_grid

        g = build_iou_grid(iou=0.5, iou_grid=[0.3, 0.5, 0.7])
        self.assertEqual(g, [0.3, 0.5, 0.7])

    def test_build_iou_grid_range(self) -> None:
        from harchoc.threshold_protocol import build_iou_grid

        g = build_iou_grid(iou=0.5, iou_min=0.4, iou_max=0.6, iou_steps=3)
        self.assertEqual(len(g), 3)
        self.assertAlmostEqual(g[0], 0.4)
        self.assertAlmostEqual(g[-1], 0.6)

    def test_counting_metrics_at_conf(self) -> None:
        from harchoc.threshold_protocol import counting_metrics_at_conf

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [20, 20, 30, 30], "category_id": 0},
                    ],
                },
                {"image_id": "img2", "annotations": []},
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "detections": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9}],
                },
                {
                    "image_id": "img2",
                    "detections": [
                        {"bbox": [5, 5, 7, 7], "category_id": 0, "score": 0.6},
                        {"bbox": [8, 8, 10, 10], "category_id": 0, "score": 0.55},
                    ],
                },
            ]
        }
        cm = counting_metrics_at_conf(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5)
        self.assertAlmostEqual(cm["mae"], 1.5)
        self.assertEqual(cm["n_images"], 2)

    def test_select_best_iou_from_grid(self) -> None:
        from harchoc.threshold_protocol import select_best_iou_from_grid

        grid = [
            {"iou": 0.3, "selected_row": {"f1": 0.5}},
            {"iou": 0.5, "selected_row": {"f1": 0.7}},
            {"iou": 0.7, "selected_row": {"f1": 0.6}},
        ]
        iou, _ = select_best_iou_from_grid(grid)
        self.assertAlmostEqual(iou, 0.5)

    def test_load_locked_conf_prefers_locked_on_test_sweep(self) -> None:
        from harchoc.threshold_lock import load_locked_conf

        fixture = {
            "selected": {
                "mode": "best_f1",
                "row": {"conf_thr": 0.10, "f1": 0.62},
            },
            "locked": {
                "mode": "fixed_conf",
                "row": {"conf_thr": 0.15, "f1": 0.61},
                "source": "reports/hsp/threshold_val.json",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "threshold_test_locked.json"
            p.write_text(json.dumps(fixture), encoding="utf-8")
            self.assertAlmostEqual(load_locked_conf(p), 0.15)

    def test_load_locked_conf_falls_back_to_selected_on_val_sweep(self) -> None:
        from harchoc.threshold_lock import load_locked_conf

        fixture = {"selected": {"mode": "best_f1", "row": {"conf_thr": 0.42, "f1": 0.9}}}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "threshold_val.json"
            p.write_text(json.dumps(fixture), encoding="utf-8")
            self.assertAlmostEqual(load_locked_conf(p), 0.42)

    def test_load_locked_conf_from_hsp_test_locked_artifact(self) -> None:
        from harchoc.threshold_lock import load_locked_conf

        p = Path("reports/hsp/threshold_test_locked.json")
        if not p.is_file():
            self.skipTest("HSP threshold_test_locked.json not present")
        self.assertAlmostEqual(load_locked_conf(p), 0.15, places=2)


class ThresholdSweepProtocolIntegrationTests(unittest.TestCase):
    def test_rejects_test_split_tuning(self) -> None:
        from scripts.threshold_sweep import main

        gt = {"images": [{"image_id": "a", "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}]}]}
        preds = {
            "images": [
                {"image_id": "a", "detections": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.8}]}
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            dataset_root = td_path / "dataset"
            dataset_root.mkdir()
            gt_path = td_path / "gt_test.json"
            preds_path = td_path / "preds_test.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--gt-json",
                        str(gt_path),
                        "--preds-json",
                        str(preds_path),
                        "--out",
                        str(td_path / "out.json"),
                        "--steps",
                        "3",
                    ]
                )

    def test_iou_grid_stores_nms_iou(self) -> None:
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
            gt_path = td_path / "gt_val.json"
            preds_path = td_path / "preds_val.json"
            out = td_path / "sweep.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
            rc = main(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--gt-json",
                    str(gt_path),
                    "--preds-json",
                    str(preds_path),
                    "--iou-grid",
                    "0.45",
                    "--iou-grid",
                    "0.55",
                    "--out",
                    str(out),
                    "--steps",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertIn("nms_iou", obj["selected"])
            self.assertIn("selected", obj["selected"]["nms_iou"])

    def test_locked_test_includes_counting_metrics(self) -> None:
        from scripts.threshold_sweep import main

        gt = {
            "images": [
                {
                    "image_id": "a",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [20, 20, 30, 30], "category_id": 0},
                    ],
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
            val_sweep = td_path / "val.json"
            val_sweep.write_text(json.dumps({"selected": {"row": {"conf_thr": 0.5}}}), encoding="utf-8")
            gt_path = td_path / "gt_test.json"
            preds_path = td_path / "preds_test.json"
            out = td_path / "locked.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
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
            self.assertIn("counting_metrics", obj["locked"])
            self.assertAlmostEqual(obj["locked"]["counting_metrics"]["mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
