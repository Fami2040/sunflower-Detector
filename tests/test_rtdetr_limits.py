import json
import os
import tempfile
import unittest
from pathlib import Path

from harchoc.rtdetr_limits import (
    SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE,
    is_rtdetr_model,
    rtdetr_eval_max_det,
    rtdetr_fields_from_train_json,
    validate_rtdetr_infer_max_det,
    validate_rtdetr_query_cap,
)


class RtdetrLimitsTests(unittest.TestCase):
    def test_is_rtdetr_model(self) -> None:
        self.assertTrue(is_rtdetr_model("rtdetr-l.pt"))
        self.assertFalse(is_rtdetr_model("yolov8m.pt"))

    def test_fails_without_accept_truncation(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            validate_rtdetr_query_cap(
                model="rtdetr-l.pt",
                train_json={
                    "num_queries": 300,
                    "documented_peak_gt_boxes_per_image": 1015,
                },
                train_json_path="train.json",
                fail=True,
            )
        self.assertIn("num_queries=300", str(ctx.exception))
        self.assertIn("1015", str(ctx.exception))

    def test_passes_with_accept_truncation(self) -> None:
        warnings = validate_rtdetr_query_cap(
            model="rtdetr-l.pt",
            train_json={
                "num_queries": 300,
                "documented_peak_gt_boxes_per_image": SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE,
                "accept_rtdetr_query_truncation": True,
            },
            train_json_path="train.json",
            fail=True,
        )
        self.assertEqual(len(warnings), 1)

    def test_committed_rtdetr_bench_json_policy_fields(self) -> None:
        from harchoc.train_config import load_train_config_json

        repo = Path(__file__).resolve().parents[1]
        path = repo / "configs" / "experiments" / "train_bench_rtdetr-l.json"
        merged = load_train_config_json(path, repo_root=repo)
        fields = rtdetr_fields_from_train_json(merged, path=str(path))
        self.assertEqual(fields["num_queries"], 300)
        self.assertEqual(
            fields["documented_peak_gt_boxes_per_image"],
            SUNFLOWER_DOCUMENTED_PEAK_GT_BOXES_PER_IMAGE,
        )
        self.assertTrue(fields["accept_rtdetr_query_truncation"])

    def test_rtdetr_smoke_15ep_merges_bench_and_epochs(self) -> None:
        from harchoc.train_config import load_train_config_json

        repo = Path(__file__).resolve().parents[1]
        merged = load_train_config_json(
            repo / "configs" / "experiments" / "train_rtdetr_smoke_15ep.json",
            repo_root=repo,
        )
        self.assertEqual(merged["model"], "rtdetr-l.pt")
        self.assertEqual(merged["epochs"], 15)
        self.assertEqual(merged["imgsz"], 1280)
        self.assertEqual(merged["batch"], 1)
        self.assertTrue(merged["accept_rtdetr_query_truncation"])

    def test_committed_rtdetr_bench_json_loads_in_matrix(self) -> None:
        from harchoc.bench_config import load_bench_config

        repo = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo / "configs" / "bench" / "rtdetr_l_default.yaml")
        self.assertEqual(cfg.model, "rtdetr-l.pt")

    def test_rtdetr_bench_fails_without_ack_in_train_json(self) -> None:
        from harchoc.bench_config import load_bench_config

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "train_bench_rtdetr-l.json"
            bad.write_text(
                json.dumps(
                    {
                        "model": "rtdetr-l.pt",
                        "num_queries": 300,
                        "documented_peak_gt_boxes_per_image": 1015,
                    }
                ),
                "utf-8",
            )
            bench = Path(td) / "rtdetr.yaml"
            bench.write_text(
                "\n".join(
                    [
                        "name: rtdetr_bad",
                        "backend: ultralytics",
                        "model: rtdetr-custom.pt",
                        f"train_config: {bad}",
                        "",
                    ]
                ),
                "utf-8",
            )
            with self.assertRaises(SystemExit):
                load_bench_config(bench)

    def test_warn_only_env(self) -> None:
        old = os.environ.get("HARCHOC_RTDETR_QUERY_CAP")
        try:
            os.environ["HARCHOC_RTDETR_QUERY_CAP"] = "warn"
            warnings = validate_rtdetr_query_cap(
                model="rtdetr-l.pt",
                train_json={"num_queries": 300, "documented_peak_gt_boxes_per_image": 1015},
                train_json_path="train.json",
                fail=True,
            )
            self.assertEqual(len(warnings), 1)
        finally:
            if old is None:
                os.environ.pop("HARCHOC_RTDETR_QUERY_CAP", None)
            else:
                os.environ["HARCHOC_RTDETR_QUERY_CAP"] = old

    def test_peak_override_env(self) -> None:
        old_peak = os.environ.get("HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE")
        try:
            os.environ["HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE"] = "50"
            warnings = validate_rtdetr_query_cap(
                model="rtdetr-l.pt",
                train_json={"num_queries": 300, "documented_peak_gt_boxes_per_image": 1015},
                train_json_path="train.json",
                fail=True,
            )
            self.assertEqual(warnings, [])
        finally:
            if old_peak is None:
                os.environ.pop("HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE", None)
            else:
                os.environ["HARCHOC_RTDETR_PEAK_GT_BOXES_PER_IMAGE"] = old_peak


    def test_queries_smoke_model_yaml_is_rtdetr(self) -> None:
        from harchoc.rtdetr_limits import is_rtdetr_model

        self.assertTrue(is_rtdetr_model("configs/models/rtdetr-l_nq1024.yaml"))

    def test_rtdetr_eval_max_det_matches_num_queries(self) -> None:
        self.assertEqual(rtdetr_eval_max_det(300), 300)
        self.assertEqual(rtdetr_eval_max_det(1024), 1024)

    def test_validate_rtdetr_infer_max_det_passes_when_aligned(self) -> None:
        warnings = validate_rtdetr_infer_max_det(
            model="rtdetr-l.pt",
            infer_max_det=300,
            train_json={"num_queries": 300},
            train_json_path="train.json",
            cfg_path="bench.yaml",
            fail=True,
        )
        self.assertEqual(warnings, [])

    def test_validate_rtdetr_infer_max_det_fails_when_mismatch(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            validate_rtdetr_infer_max_det(
                model="rtdetr-l.pt",
                infer_max_det=3000,
                train_json={"num_queries": 300},
                train_json_path="train.json",
                cfg_path="bench.yaml",
                fail=True,
            )
        self.assertIn("infer.max_det=3000", str(ctx.exception))
        self.assertIn("num_queries=300", str(ctx.exception))

    def test_committed_rtdetr_bench_yaml_infer_max_det(self) -> None:
        from harchoc.bench_config import load_bench_config

        repo = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo / "configs" / "bench" / "rtdetr_l_default.yaml")
        self.assertEqual(cfg.infer.get("max_det"), 300)

    def test_committed_rtdetr_nq1024_bench_yaml_infer_max_det(self) -> None:
        from harchoc.bench_config import load_bench_config

        repo = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo / "configs" / "bench" / "rtdetr_l_nq1024.yaml")
        self.assertEqual(cfg.infer.get("max_det"), 1024)
        self.assertEqual(cfg.model, "configs/models/rtdetr-l_nq1024.yaml")


if __name__ == "__main__":
    unittest.main()
