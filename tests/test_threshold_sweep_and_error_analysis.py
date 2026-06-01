import unittest


class ThresholdSweepAndErrorAnalysisTests(unittest.TestCase):
    def test_threshold_sweep_test_guardrail(self) -> None:
        import tempfile
        from pathlib import Path

        import json

        from scripts.threshold_sweep import main

        gt = {"images": [{"image_id": "a", "annotations": []}]}
        preds = {"images": [{"image_id": "a", "detections": []}]}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            gt_path = td_path / "gt_test.json"
            preds_path = td_path / "preds_test.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            preds_path.write_text(json.dumps(preds), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--dataset-root",
                        str(td_path),
                        "--gt-json",
                        str(gt_path),
                        "--preds-json",
                        str(preds_path),
                        "--out",
                        str(td_path / "o.json"),
                    ]
                )

    def test_threshold_sweep_counts(self) -> None:
        from harchoc.detection_match import match_counts_for_threshold

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
                {
                    "image_id": "img2",
                    "file_name": "/abs/img2.jpg",
                    "annotations": [],
                },
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.90},  # TP
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.80},  # FP
                    ],
                },
                {
                    "image_id": "img2",
                    "file_name": "/abs/img2.jpg",
                    "detections": [{"bbox": [5, 5, 7, 7], "category_id": 0, "score": 0.60}],  # FP
                },
            ]
        }

        c1 = match_counts_for_threshold(gt=gt, preds=preds, conf_thr=0.50, iou_thr=0.5, category_aware=True)
        self.assertEqual(c1, {"tp": 1, "fp": 2, "fn": 0})

        c2 = match_counts_for_threshold(gt=gt, preds=preds, conf_thr=0.85, iou_thr=0.5, category_aware=True)
        self.assertEqual(c2, {"tp": 1, "fp": 0, "fn": 0})

    def test_error_analysis_taxonomy_and_top_fp(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
                {
                    "image_id": "img2",
                    "file_name": "/abs/img2.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                },
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.90},  # TP
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.70},  # FP
                    ],
                },
                {
                    "image_id": "img2",
                    "file_name": "/abs/img2.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 1, "score": 0.95},  # confusion (wrong class)
                    ],
                },
            ]
        }

        out = analyze_errors(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5)
        self.assertEqual(out["counts"]["tp"], 1)
        self.assertEqual(out["counts"]["fp"], 1)
        self.assertEqual(out["counts"]["fn"], 1)  # img2 GT unmatched (same-class matching)
        self.assertEqual(out["counts"]["cls_confusion"], 1)

        self.assertEqual(out["fp_breakdown"]["classification"], 1)
        self.assertEqual(out["fp_breakdown"]["background"], 1)
        self.assertEqual(out["fp_breakdown"]["localization"], 0)

        # Explicit reviewer-facing split: localization vs classification
        self.assertEqual(out["error_taxonomy"]["localization"]["fp"], 1)
        self.assertEqual(out["error_taxonomy"]["localization"]["fp_background"], 1)
        self.assertEqual(out["error_taxonomy"]["localization"]["fp_low_iou"], 0)
        self.assertEqual(out["error_taxonomy"]["localization"]["fn"], 1)
        self.assertEqual(out["error_taxonomy"]["classification"]["fp"], 1)
        self.assertEqual(out["error_taxonomy"]["classification"]["fp_cross_class_confusion"], 1)

        top = out["top_fp_images"]
        self.assertTrue(len(top) >= 2)
        self.assertEqual(top[0]["image_id"], "img1")
        self.assertEqual(top[0]["fp"], 1)
        self.assertEqual(top[0]["file_name"], "/abs/img1.jpg")

    def test_error_analysis_localization_fp_breakdown(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [1, 1, 9, 9], "category_id": 0, "score": 0.90},  # TP @ iou>=0.5
                        {"bbox": [5, 5, 15, 15], "category_id": 0, "score": 0.80},  # localization FP (t_b<=IoU<t_f)
                    ],
                }
            ]
        }

        out = analyze_errors(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5)
        self.assertEqual(out["counts"]["tp"], 1)
        self.assertEqual(out["counts"]["fp"], 1)
        self.assertEqual(out["counts"]["cls_confusion"], 0)
        self.assertEqual(out["fp_breakdown"]["localization"], 1)
        self.assertEqual(out["error_taxonomy"]["localization"]["fp_low_iou"], 1)
        self.assertEqual(out["error_taxonomy"]["classification"]["fp"], 0)

    def test_export_fp_crops_skips_without_pil(self) -> None:
        from scripts.error_analysis import export_topk_fp_crops

        fp_examples = [
            {
                "image_id": "img1",
                "file_name": "/abs/missing.jpg",
                "bbox": [0, 0, 10, 10],
                "category_id": 0,
                "score": 0.9,
                "error_type": "background",
            }
        ]
        out = export_topk_fp_crops(fp_examples=fp_examples, out_dir="/tmp/does-not-matter", topk=1, dataset_root=None)
        self.assertIn(out["status"], ("ok", "skipped"))
        if out["status"] == "skipped":
            self.assertIn("PIL not available", str(out.get("reason") or ""))

    def test_error_analysis_dupe_bucket(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.95},
                        {"bbox": [1, 1, 9, 9], "category_id": 0, "score": 0.85},
                    ],
                }
            ]
        }
        out = analyze_errors(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5)
        self.assertEqual(out["counts"]["tp"], 1)
        self.assertEqual(out["counts"]["dupe"], 1)
        self.assertEqual(out["fp_breakdown"]["dupe"], 1)

    def test_error_analysis_background_iou_tb(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        # Small overlap (~0.05) — background with t_b=0.1, not localization
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.90},
                        {"bbox": [9, 9, 19, 19], "category_id": 0, "score": 0.80},
                    ],
                }
            ]
        }
        out = analyze_errors(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5, iou_bg_thr=0.1)
        self.assertEqual(out["counts"]["tp"], 1)
        self.assertEqual(out["fp_breakdown"]["background"], 1)
        self.assertEqual(out["fp_breakdown"]["localization"], 0)

    def test_error_taxonomy_area_strata_and_conf_grid_helpers(self) -> None:
        from harchoc.error_taxonomy import (
            area_stratum_label,
            bbox_sqrt_area,
            build_bbox_area_strata,
            build_conf_taxonomy_grid,
            conf_bin_label,
        )

        self.assertAlmostEqual(bbox_sqrt_area([0, 0, 10, 10]), 10.0)
        self.assertEqual(area_stratum_label(10.0), "small")
        self.assertEqual(area_stratum_label(50.0), "medium")
        self.assertEqual(area_stratum_label(100.0), "large")

        events = [
            {"error_type": "tp", "bbox": [0, 0, 10, 10], "score": 0.9},
            {"error_type": "fp_background", "bbox": [0, 0, 50, 50], "score": 0.6},
            {"error_type": "fn", "bbox": [0, 0, 100, 100], "score": None},
        ]
        strata = build_bbox_area_strata(events)
        self.assertEqual(strata["by_error_type"]["tp"]["small"], 1)
        self.assertEqual(strata["by_error_type"]["fp_background"]["medium"], 1)
        self.assertEqual(strata["by_error_type"]["fn"]["large"], 1)

        self.assertEqual(conf_bin_label(0.9), "[0.75,1.00]")
        self.assertEqual(conf_bin_label(0.6), "[0.50,0.75)")
        grid = build_conf_taxonomy_grid(events)
        self.assertEqual(grid["counts"]["[0.75,1.00]"]["tp"], 1)
        self.assertEqual(grid["counts"]["[0.50,0.75)"]["fp_background"], 1)
        self.assertNotIn("fn", grid["error_types"])

    def test_error_analysis_bbox_area_strata_and_conf_grid(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [0, 0, 100, 100], "category_id": 0},
                    ],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.90},
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.70},
                    ],
                }
            ]
        }
        out = analyze_errors(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5)
        self.assertIn("bbox_area_strata", out)
        self.assertIn("conf_taxonomy_grid", out)

        strata = out["bbox_area_strata"]
        self.assertEqual(strata["method"], "sqrt_area")
        self.assertEqual(strata["by_error_type"]["tp"]["small"], 1)
        self.assertEqual(strata["by_error_type"]["fp_background"]["small"], 1)
        self.assertEqual(strata["by_error_type"]["fn"]["large"], 1)

        grid = out["conf_taxonomy_grid"]
        self.assertEqual(grid["counts"]["[0.75,1.00]"]["tp"], 1)
        self.assertEqual(grid["counts"]["[0.50,0.75)"]["fp_background"], 1)
        self.assertEqual(sum(grid["counts"]["[0.75,1.00]"].values()), 1)

    def test_error_analysis_counting_metrics(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "a.jpg",
                    "annotations": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0},
                        {"bbox": [20, 20, 30, 30], "category_id": 0},
                    ],
                },
                {
                    "image_id": "img2",
                    "file_name": "b.jpg",
                    "annotations": [],
                },
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "a.jpg",
                    "detections": [{"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9}],
                },
                {
                    "image_id": "img2",
                    "file_name": "b.jpg",
                    "detections": [
                        {"bbox": [5, 5, 7, 7], "category_id": 0, "score": 0.6},
                        {"bbox": [8, 8, 10, 10], "category_id": 0, "score": 0.55},
                    ],
                },
            ]
        }
        out = analyze_errors(gt=gt, preds=preds, conf_thr=0.25, iou_thr=0.5)
        cm = out["counting_metrics"]
        self.assertEqual(cm["n_images"], 2)
        # errors: img1 -1, img2 +2 => MAE = (1+2)/2 = 1.5
        self.assertAlmostEqual(cm["mae"], 1.5)
        self.assertIsNotNone(cm["rrmse"])
        self.assertIn("mae_ci", cm)

    def test_error_analysis_ambiguous_fp_crosstab(self) -> None:
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.90},
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.20},
                        {"bbox": [40, 40, 50, 50], "category_id": 0, "score": 0.80},
                    ],
                }
            ]
        }
        out = analyze_errors(
            gt=gt,
            preds=preds,
            conf_thr=0.15,
            ambiguity_conf_low=0.15,
            ambiguity_conf_high=0.30,
            iou_thr=0.5,
        )
        xt = out["ambiguous_fp_crosstab"]
        self.assertIn("by_bucket", xt)
        self.assertEqual(xt["by_bucket"]["tp"]["not_ambiguous"], 1)
        self.assertEqual(xt["by_bucket"]["background"]["ambiguous"], 1)
        self.assertEqual(xt["by_bucket"]["background"]["not_ambiguous"], 1)
        self.assertIn("tide_bucket_summary", out)
        self.assertEqual(out["tide_bucket_summary"]["buckets"]["Bkg"], 2)
        self.assertIn("counting_metrics_excl_ambiguous_band", out)

    def test_tide_summary_helpers(self) -> None:
        from harchoc.tide_summary import build_ambiguous_fp_crosstab, build_tide_bucket_summary

        xt = build_ambiguous_fp_crosstab(
            [
                {"ambiguous": True, "flags": ["low_conf_band"], "bucket": "background"},
                {"ambiguous": False, "flags": [], "bucket": "tp"},
            ],
            conf_band=[0.15, 0.30],
        )
        self.assertEqual(xt["totals"]["n_ambiguous"], 1)
        self.assertEqual(xt["by_flag"]["low_conf_band"]["background"], 1)

        tide = build_tide_bucket_summary(
            counts={"tp": 10, "fn": 2, "fp": 0, "cls_confusion": 0, "dupe": 0},
            fp_breakdown={"background": 3, "localization": 5, "classification": 1, "dupe": 0},
            map50=0.793,
        )
        self.assertEqual(tide["dominant_bucket"], "Loc")
        self.assertTrue(tide["localization_dominates_classification"])
        self.assertIn("delta_ap_estimate", tide)

    def test_export_coco_predictions_for_tide(self) -> None:
        from harchoc.tide_summary import export_coco_gt_for_tide, export_coco_predictions_for_tide, try_run_tidecv

        gt = {
            "images": [
                {
                    "image_id": "example_0001",
                    "file_name": "images/val/example_0001.jpg",
                    "annotations": [{"bbox": [10.0, 20.0, 50.0, 60.0], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "example_0001",
                    "file_name": "images/val/example_0001.jpg",
                    "detections": [
                        {"bbox": [12.0, 22.0, 48.0, 58.0], "category_id": 0, "score": 0.85},
                        {"bbox": [0.0, 0.0, 5.0, 5.0], "category_id": 1, "score": 0.05},
                    ],
                }
            ]
        }

        coco_gt = export_coco_gt_for_tide(gt)
        self.assertEqual(len(coco_gt["images"]), 1)
        self.assertEqual(coco_gt["images"][0]["id"], "example_0001")
        self.assertEqual(len(coco_gt["annotations"]), 1)
        self.assertEqual(coco_gt["annotations"][0]["bbox"], [10.0, 20.0, 40.0, 40.0])
        self.assertIn("segmentation", coco_gt["annotations"][0])
        self.assertEqual(len(coco_gt["categories"]), 2)

        coco_preds = export_coco_predictions_for_tide(gt=gt, preds=preds)
        self.assertEqual(len(coco_preds), 2)
        row = coco_preds[0]
        self.assertEqual(row["image_id"], "example_0001")
        self.assertEqual(row["category_id"], 0)
        self.assertAlmostEqual(row["score"], 0.85)
        self.assertEqual(row["bbox"], [12.0, 22.0, 36.0, 36.0])

        tidecv_out = try_run_tidecv(gt=gt, preds=preds)
        self.assertIsNotNone(tidecv_out)
        assert tidecv_out is not None
        self.assertIn(tidecv_out["status"], ("skipped", "ok", "error"))
        self.assertIn("adapter_ok", tidecv_out)
        self.assertIn("adapter", tidecv_out)
        self.assertEqual(tidecv_out["adapter"]["n_predictions"], 2)
        self.assertEqual(tidecv_out["adapter"]["n_gt_annotations"], 1)

    def test_build_tidecv_compare_skipped_when_tidecv_missing(self) -> None:
        from unittest.mock import patch

        from harchoc.tide_summary import build_tidecv_compare, build_tide_bucket_summary, try_run_tidecv

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "img1.jpg",
                    "annotations": [{"bbox": [0.0, 0.0, 10.0, 10.0], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "img1.jpg",
                    "detections": [{"bbox": [1.0, 1.0, 9.0, 9.0], "category_id": 0, "score": 0.9}],
                }
            ]
        }
        proxy = build_tide_bucket_summary(
            counts={"tp": 1, "fn": 0, "fp": 0, "cls_confusion": 0, "dupe": 0},
            fp_breakdown={"background": 0, "localization": 0, "classification": 0, "dupe": 0},
        )

        with patch.dict("sys.modules", {"tidecv": None, "tidecv.datasets": None}):
            tidecv_out = try_run_tidecv(gt=gt, preds=preds)
        assert tidecv_out is not None
        self.assertEqual(tidecv_out["status"], "skipped")
        self.assertTrue(tidecv_out["adapter_ok"])
        self.assertEqual(tidecv_out["reason"], "tidecv not installed")
        self.assertEqual(tidecv_out["adapter"]["n_predictions"], 1)
        self.assertEqual(tidecv_out["adapter"]["n_gt_annotations"], 1)

        compare = build_tidecv_compare(tide_bucket_summary=proxy, tidecv_result=tidecv_out)
        self.assertEqual(compare["status"], "skipped")
        self.assertTrue(compare["adapter_ok"])
        self.assertEqual(compare["skipped_reason"], "tidecv not installed")
        self.assertEqual(compare["adapter"]["n_predictions"], 1)
        self.assertIsNone(compare["tidecv"]["ap50"])
        self.assertIn("delta_ap_share", compare["proxy"])

    def test_build_tidecv_compare_ok_with_official_delta_ap(self) -> None:
        from harchoc.tide_summary import build_tidecv_compare, build_tide_bucket_summary

        proxy = build_tide_bucket_summary(
            counts={"tp": 8, "fn": 2, "fp": 0, "cls_confusion": 0, "dupe": 0},
            fp_breakdown={"background": 3, "localization": 5, "classification": 1, "dupe": 0},
            map50=0.8,
        )
        tidecv_out = {
            "status": "ok",
            "adapter_ok": True,
            "adapter": {"n_images": 1, "n_gt_annotations": 10, "n_predictions": 12},
            "ap50": 0.75,
            "delta_ap": {"Loc": 0.04, "Bkg": 0.02, "Cls": 0.01, "Miss": 0.03, "Dupe": 0.0},
        }
        compare = build_tidecv_compare(tide_bucket_summary=proxy, tidecv_result=tidecv_out)
        self.assertEqual(compare["status"], "ok")
        self.assertTrue(compare["adapter_ok"])
        self.assertIsNone(compare["skipped_reason"])
        self.assertAlmostEqual(compare["tidecv"]["ap50"], 0.75)
        self.assertIn("Loc", compare["comparison"])
        self.assertAlmostEqual(compare["comparison"]["Loc"]["tidecv_delta_ap"], 0.04)

    def test_error_analysis_tidecv_writes_compare_sidecar(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from scripts.error_analysis import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "summary.json"
            report = Path(td) / "error_test_report.json"
            compare = Path(td) / "error_test_tidecv_compare.json"
            rc = main(
                [
                    "--light",
                    "--tidecv",
                    "--out",
                    str(out),
                    "--report",
                    str(report),
                    "--tidecv-compare-out",
                    str(compare),
                ]
            )
            self.assertEqual(rc, 0)
            report_obj = json.loads(report.read_text("utf-8"))
            self.assertIn("tidecv", report_obj)
            self.assertEqual(report_obj["tidecv"]["status"], "skipped")
            self.assertIn("tidecv_compare", report_obj)
            self.assertFalse(report_obj["tidecv_compare"]["adapter_ok"] is False)
            self.assertEqual(report_obj["tidecv_compare"]["skipped_reason"], "tidecv not installed")

            compare_obj = json.loads(compare.read_text("utf-8"))
            self.assertEqual(compare_obj["schema_version"], "tidecv_compare.v1")
            self.assertEqual(compare_obj["status"], "skipped")
            self.assertTrue(compare_obj["adapter_ok"])
            self.assertGreaterEqual(compare_obj["adapter"]["n_predictions"], 1)
            self.assertGreaterEqual(compare_obj["adapter"]["n_gt_annotations"], 0)

    def test_ambiguous_panel_select(self) -> None:
        from harchoc.ambiguous_panel import select_ambiguous_panel_entries

        entries = [
            {"status": "ok", "crop_path": "/a.png", "error_type": "localization", "score": 0.7},
            {"status": "ok", "crop_path": "/b.png", "error_type": "background", "score": 0.2},
            {"status": "ok", "crop_path": "/c.png", "error_type": "background", "score": 0.9},
        ]
        picked = select_ambiguous_panel_entries(entries, conf_band=[0.15, 0.30], max_panels=2)
        self.assertEqual(len(picked), 2)
        tags = {p.get("panel_tag") for p in picked}
        self.assertIn("localization_fp", tags)
        self.assertIn("low_conf_band", tags)

    def test_error_analysis_schema_ambiguous_fp_crosstab(self) -> None:
        from harchoc.error_analysis_schema import (
            ERROR_ANALYSIS_SUMMARY_V1,
            validate_ambiguous_fp_crosstab,
            validate_error_analysis_payload,
        )
        from harchoc.tide_summary import build_ambiguous_fp_crosstab
        from scripts.error_analysis import analyze_errors

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "file_name": "/abs/img1.jpg",
                    "detections": [
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.20},
                    ],
                }
            ]
        }
        summary = analyze_errors(gt=gt, preds=preds, conf_thr=0.15)
        payload = {
            "schema_version": ERROR_ANALYSIS_SUMMARY_V1,
            "status": "ok",
            **{k: summary[k] for k in (
                "ambiguous_summary",
                "ambiguous_fp_crosstab",
                "tide_bucket_summary",
                "counting_metrics",
                "counting_metrics_excl_ambiguous_band",
                "error_taxonomy",
                "fp_breakdown",
            )},
        }
        validate_error_analysis_payload(payload, schema_version=ERROR_ANALYSIS_SUMMARY_V1)
        validate_ambiguous_fp_crosstab(build_ambiguous_fp_crosstab([], conf_band=[0.15, 0.30]))

    def test_eval_export_yolo_label_to_xyxy(self) -> None:
        from harchoc.eval_export import yolo_label_line_to_xyxy

        parsed = yolo_label_line_to_xyxy("0 0.5 0.5 0.2 0.2", img_w=100, img_h=100)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        cls, (x1, y1, x2, y2) = parsed
        self.assertEqual(cls, 0)
        self.assertAlmostEqual(x1, 40.0)
        self.assertAlmostEqual(y1, 40.0)
        self.assertAlmostEqual(x2, 60.0)
        self.assertAlmostEqual(y2, 60.0)

    def test_export_fp_crops_smoke_with_pil(self) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            raise unittest.SkipTest("PIL not installed")

        import tempfile
        from pathlib import Path

        from scripts.error_analysis import export_topk_fp_crops

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            img_path = td_p / "img.png"
            out_dir = td_p / "out"

            im = Image.new("RGB", (32, 32), color=(255, 0, 0))
            im.save(img_path)

            fp_examples = [
                {
                    "image_id": "img1",
                    "file_name": str(img_path),
                    "bbox": [4, 4, 20, 20],
                    "category_id": 0,
                    "score": 0.9,
                    "error_type": "background",
                }
            ]

            out = export_topk_fp_crops(fp_examples=fp_examples, out_dir=out_dir, topk=1, dataset_root=None)
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["exported"], 1)
            self.assertTrue(Path(out["results"][0]["crop_path"]).exists())


if __name__ == "__main__":
    unittest.main()

