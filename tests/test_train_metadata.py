import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TrainMetadataTests(unittest.TestCase):
    def test_collect_repo_split_files_hashes(self) -> None:
        from harchoc.run_metadata import collect_repo_split_files

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            splits = repo / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "train.txt").write_text("images/train/a.jpg\n", "utf-8")
            (splits / "val.txt").write_text("images/val/a.jpg\n", "utf-8")
            (splits / "test.txt").write_text("images/test/a.jpg\n", "utf-8")

            rec = collect_repo_split_files(repo)
            self.assertEqual(rec["splits_dir"], str(splits))
            for split in ("train", "val", "test"):
                self.assertTrue(rec["files"][split]["exists"])
                self.assertIsInstance(rec["files"][split].get("sha256"), str)

    def test_train_dry_run_includes_split_hashes_and_roles(self) -> None:
        from scripts.train import main

        repo_root = Path(__file__).resolve().parents[1]
        cfg = repo_root / "configs" / "experiments" / "train_yolov8m_baseline.json"
        self.assertTrue(cfg.is_file())

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_dir = tdp / "runs"
            name = "meta_unit"
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)

            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(out_dir),
                        "--name",
                        name,
                        "--config",
                        str(cfg),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

            self.assertEqual(rc, 0)
            meta = json.loads((out_dir / name / "meta.json").read_text("utf-8"))
            self.assertIn("split_roles", meta)
            self.assertIn("ultralytics_train", meta)
            self.assertIsInstance(meta["ultralytics_train"].get("forwarded_keys"), list)
            self.assertEqual(meta["split_roles"]["ultralytics_val"]["role"], "validation")
            self.assertEqual(meta["split_roles"]["post_train_eval"]["role"], "test")

            repo_splits = meta["run_metadata"].get("repo_splits", {})
            for split in ("train", "val", "test"):
                rec = repo_splits.get("files", {}).get(split, {})
                if rec.get("exists"):
                    self.assertIsInstance(rec.get("sha256"), str)

            cfg_out = json.loads((out_dir / name / "config.json").read_text("utf-8"))
            self.assertEqual(cfg_out["train"]["epochs"], 100)
            self.assertEqual(cfg_out["train"]["imgsz"], 1280)
            self.assertEqual(cfg_out["train"]["seed"], 0)

    def test_aug_smoke_s9_dry_run_inline_baseline_defaults(self) -> None:
        from scripts.train import main

        repo_root = Path(__file__).resolve().parents[1]
        cfg = repo_root / "configs" / "experiments" / "train_aug_s9_no_aug_yaml_smoke.json"
        self.assertTrue(cfg.is_file())

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_dir = tdp / "runs"
            name = "aug_smoke_no_aug_yaml"
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)

            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(out_dir),
                        "--name",
                        name,
                        "--config",
                        str(cfg),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

            self.assertEqual(rc, 0)
            cfg_out = json.loads((out_dir / name / "config.json").read_text("utf-8"))
            self.assertIsNone(cfg_out.get("aug_config"))
            train = cfg_out["train"]
            self.assertIsNone(train.get("aug_config"))
            self.assertEqual(train["mosaic"], 0.1)
            self.assertNotIn("close_mosaic", train)

    def test_aug_config_merges_ultralytics_keys(self) -> None:
        from harchoc.aug_config import merge_aug_yaml

        repo_root = Path(__file__).resolve().parents[1]
        aug = repo_root / "configs" / "aug" / "robustness_minimal.yaml"
        self.assertTrue(aug.is_file())
        merged = merge_aug_yaml({"mosaic": 0.5}, aug)
        self.assertEqual(merged["mixup"], 0.0)
        self.assertEqual(merged["close_mosaic"], 15)
        self.assertEqual(merged["mosaic"], 0.1)

    def test_rtdetr_smoke_15ep_dry_run_passes_query_cap_guard(self) -> None:
        from scripts.train import main

        repo_root = Path(__file__).resolve().parents[1]
        cfg = repo_root / "configs" / "experiments" / "train_rtdetr_smoke_15ep.json"
        self.assertTrue(cfg.is_file())

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_dir = tdp / "runs"
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(out_dir),
                        "--name",
                        "rtdetr_smoke_15ep",
                        "--config",
                        str(cfg),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
            self.assertEqual(rc, 0)

    def test_rtdetr_train_dry_run_fails_without_accept_truncation(self) -> None:
        from scripts.train import main

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            bad = tdp / "rtdetr_bad.json"
            bad.write_text(
                json.dumps(
                    {
                        "model": "rtdetr-l.pt",
                        "epochs": 1,
                        "num_queries": 300,
                        "documented_peak_gt_boxes_per_image": 1015,
                    }
                ),
                "utf-8",
            )
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "--dry-run",
                            "--out-dir",
                            str(tdp / "runs"),
                            "--name",
                            "bad",
                            "--config",
                            str(bad),
                        ]
                    )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

    def test_train_bench_config_forwards_cache_in_meta_dry_run(self) -> None:
        from scripts.train import main

        repo_root = Path(__file__).resolve().parents[1]
        cfg = repo_root / "configs" / "experiments" / "train_bench_yolov8m.json"
        self.assertTrue(cfg.is_file())

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_dir = tdp / "runs"
            name = "bench_cache_meta"
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)

            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(out_dir),
                        "--name",
                        name,
                        "--config",
                        str(cfg),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

            self.assertEqual(rc, 0)
            meta = json.loads((out_dir / name / "meta.json").read_text("utf-8"))
            forwarded = meta["ultralytics_train"]["forwarded_keys"]
            self.assertIn("cache", forwarded)

    def test_train_dry_run_no_warnings_without_strict_ml(self) -> None:
        from scripts.train import main

        repo_root = Path(__file__).resolve().parents[1]
        cfg = repo_root / "configs" / "experiments" / "train_yolov8m_baseline.json"
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            old_root = os.environ.get("DATASET_ROOT")
            old_strict = os.environ.get("HARCHOC_STRICT_ML")
            try:
                os.environ.pop("HARCHOC_STRICT_ML", None)
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(tdp / "runs"),
                        "--name",
                        "no_strict",
                        "--config",
                        str(cfg),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
                if old_strict is None:
                    os.environ.pop("HARCHOC_STRICT_ML", None)
                else:
                    os.environ["HARCHOC_STRICT_ML"] = old_strict
            self.assertEqual(rc, 0)
            meta = json.loads((tdp / "runs" / "no_strict" / "meta.json").read_text("utf-8"))
            self.assertNotIn("warnings", meta)

    def test_runtime_versions_strict_records_import_failure(self) -> None:
        import builtins

        from scripts.train import _runtime_versions

        real_import = builtins.__import__

        def _block_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(os.environ, {"HARCHOC_STRICT_ML": "1"}):
            warnings: list[str] = []
            with mock.patch("builtins.__import__", side_effect=_block_torch):
                versions = _runtime_versions(warnings=warnings)
            self.assertIsNone(versions["torch"])
            self.assertTrue(any("import torch" in w for w in warnings))

    def test_val_metrics_summary_extracts_map_keys(self) -> None:
        from scripts.train import _val_metrics_summary

        raw = {
            "metrics/precision(B)": 0.9,
            "metrics/mAP50-95(B)": 0.5,
            "train/box_loss": 1.2,
        }
        summary = _val_metrics_summary(raw)
        self.assertIsNotNone(summary)
        self.assertIn("metrics/mAP50-95(B)", summary)
        self.assertNotIn("train/box_loss", summary)

    def test_train_dry_run_forwards_freeze_from_config(self) -> None:
        from harchoc.train_kwargs import ultralytics_train_kwargs
        from scripts.train import main

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cfg_path = tdp / "freeze_train.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "model": "yolov8m.pt",
                        "epochs": 5,
                        "freeze_backbone": True,
                        "unfreeze_epoch": 10,
                    }
                ),
                "utf-8",
            )
            out_dir = tdp / "runs"
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(
                    [
                        "--dry-run",
                        "--out-dir",
                        str(out_dir),
                        "--name",
                        "freeze_meta",
                        "--config",
                        str(cfg_path),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
            self.assertEqual(rc, 0)
            meta = json.loads((out_dir / "freeze_meta" / "meta.json").read_text("utf-8"))
            forwarded = meta["ultralytics_train"]["forwarded_keys"]
            self.assertIn("freeze", forwarded)
            policy = meta["freeze_policy"]
            self.assertTrue(policy.get("ultralytics_freeze_honored"))
            self.assertEqual(policy.get("freeze"), 10)
            self.assertFalse(policy.get("unfreeze_epoch_honored"))
            kwargs = ultralytics_train_kwargs(
                {"freeze": 10, "epochs": 5}, data_yaml="/tmp/data.yaml", run_name="x"
            )
            self.assertEqual(kwargs["freeze"], 10)


if __name__ == "__main__":
    unittest.main()
