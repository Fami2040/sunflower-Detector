import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")

from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from scripts.benchmark_matrix import main as benchmark_main


class BenchmarkMatrixTests(unittest.TestCase):
    def test_dry_run_includes_all_repo_bench_configs_and_routes_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            # Ensure dataset resolution follows env precedence.
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            # Dummy manifest path (ignored due to DATASET_ROOT precedence).
            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            repo_root = Path(__file__).resolve().parents[1]
            bench_dir = repo_root / "configs" / "bench"
            self.assertTrue(bench_dir.is_dir())
            cfg_paths = sorted(p for p in bench_dir.glob("*.yaml") if not p.name.startswith("_"))
            self.assertGreaterEqual(len(cfg_paths), 26)

            out_path = tdp / "reports" / "benchmarks" / "matrix.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())

            obj = json.loads(out_path.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "benchmark_matrix.v1")
            self.assertEqual(obj["status"], "plan")
            self.assertTrue(obj["dry_run"])
            self.assertIn("generated_at", obj)

            self.assertIn("dataset", obj)
            self.assertIn("description", obj["dataset"])
            self.assertIn(str(dataset_root), obj["dataset"]["description"])

            self.assertIn("runs", obj)
            self.assertEqual(len(obj["runs"]), len(cfg_paths))

            by_basename = {Path(r["config"]["path"]).name: r for r in obj["runs"]}
            for p in cfg_paths:
                self.assertIn(p.name, by_basename)

            for run in obj["runs"]:
                self.assertEqual(run["schema_version"], "benchmark_run.v1")
                self.assertIn("config", run)
                self.assertIn("planned", run)
                self.assertIn("resolved", run)
                self.assertIn("execution", run)
                self.assertIn("eval", run)
                self.assertIn("matrix_metadata", run)
                meta = run["matrix_metadata"]
                for key in (
                    "backend",
                    "nms_free",
                    "num_queries",
                    "accept_rtdetr_query_truncation",
                    "rect",
                    "infer_max_det",
                    "train_max_det",
                ):
                    self.assertIn(key, meta, msg=f"missing matrix_metadata.{key} in {run['config']['path']}")
                self.assertFalse(run["execution"]["would_train"])
                self.assertFalse(run["execution"]["would_eval"])
                self.assertIn("backend", run["resolved"])
                self.assertIn("backend_available", run["resolved"])
                self.assertIn("weights", run["resolved"])

            rtdetr_meta = by_basename["rtdetr_l_default.yaml"]["matrix_metadata"]
            self.assertTrue(rtdetr_meta["nms_free"])
            self.assertEqual(rtdetr_meta["num_queries"], 300)
            self.assertTrue(rtdetr_meta["accept_rtdetr_query_truncation"])
            self.assertFalse(rtdetr_meta["rect"])
            self.assertEqual(rtdetr_meta["infer_max_det"], 300)
            self.assertEqual(rtdetr_meta["train_max_det"], 3000)

            yolo_meta = by_basename["yolov8m_default.yaml"]["matrix_metadata"]
            self.assertFalse(yolo_meta["nms_free"])
            self.assertIsNone(yolo_meta["num_queries"])
            self.assertIsNone(yolo_meta["accept_rtdetr_query_truncation"])

            # Backend routing selection must work in dry-run without imports.
            self.assertEqual(by_basename["yolo_nas_s_default.yaml"]["resolved"]["backend"], "supergradients")
            self.assertEqual(by_basename["yolo_nas_s_default.yaml"]["planned"]["model_id"], "yolo_nas_s")

            # Legacy YOLOv8 configs without explicit backend should default to ultralytics.
            self.assertEqual(by_basename["yolov8n_default.yaml"]["resolved"]["backend"], "ultralytics")

    def test_config_json_can_drive_benchmark_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir(parents=True, exist_ok=True)
            (bench_dir / "one.json").write_text(
                json.dumps(
                    {
                        "name": "one",
                        "task": "detect",
                        "backend": "ultralytics",
                        "model": "models/missing.pt",
                        "infer": {"imgsz": 640},
                        "notes": "test",
                    }
                )
                + "\n",
                "utf-8",
            )

            out_path = tdp / "reports" / "benchmarks" / "matrix.json"
            cfg_path = tdp / "exp.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset": {"manifest": str(manifest_path), "default_dataset_name": "sunflower-cvat-1093"},
                        "benchmark": {
                            "dry_run": True,
                            "bench_dir": str(bench_dir),
                            "pattern": "*.json",
                            "out": str(out_path),
                        },
                    }
                )
                + "\n",
                "utf-8",
            )

            rc = benchmark_main(["--config", str(cfg_path)])
            self.assertEqual(rc, 0)

            obj = json.loads(out_path.read_text("utf-8"))
            self.assertEqual(obj["status"], "plan")
            self.assertTrue(obj["dry_run"])
            self.assertEqual(len(obj["runs"]), 1)
            self.assertEqual(Path(obj["runs"][0]["config"]["path"]).name, "one.json")

    def test_dry_run_marks_missing_supergradients_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            repo_root = Path(__file__).resolve().parents[1]
            bench_dir = repo_root / "configs" / "bench"
            out_path = tdp / "reports" / "benchmarks" / "matrix.json"

            def fake_find_spec(name: str, *args: object, **kwargs: object) -> object | None:
                if name == "super_gradients":
                    return None
                # Treat ultralytics as "present" for the purpose of this unit test.
                if name == "ultralytics":
                    return object()
                return object()

            with patch("harchoc.model_zoo.importlib.util.find_spec", new=fake_find_spec):
                rc = benchmark_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--bench-dir",
                        str(bench_dir),
                        "--out",
                        str(out_path),
                    ]
                )
            self.assertEqual(rc, 0)

            obj = json.loads(out_path.read_text("utf-8"))
            by_basename = {Path(r["config"]["path"]).name: r for r in obj["runs"]}
            nas = by_basename["yolo_nas_s_default.yaml"]
            self.assertEqual(nas["resolved"]["backend"], "supergradients")
            self.assertFalse(nas["resolved"]["backend_available"])
            self.assertEqual(nas["resolved"]["backend_missing_reason"], "missing_dependency:super_gradients")

    def test_group_filter_limits_selected_configs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            repo_root = Path(__file__).resolve().parents[1]
            bench_dir = repo_root / "configs" / "bench"
            out_path = tdp / "reports" / "benchmarks" / "matrix.json"

            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--group",
                    "yolov8_scales",
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)

            obj = json.loads(out_path.read_text("utf-8"))
            basenames = {Path(r["config"]["path"]).name for r in obj["runs"]}
            self.assertIn("yolov8n_default.yaml", basenames)
            self.assertIn("yolov8s_default.yaml", basenames)
            self.assertNotIn("yolo_nas_s_default.yaml", basenames)

    def test_list_groups_outputs_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            repo_root = Path(__file__).resolve().parents[1]
            bench_dir = repo_root / "configs" / "bench"
            out_path = tdp / "reports" / "benchmarks" / "groups.json"

            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--list-groups",
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out_path.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "benchmark_groups.v1")
            names = {g["name"] for g in obj["groups"]}
            self.assertIn("yolov8_scales", names)
            self.assertIn("sota_2026", names)

    def test_invalid_bench_config_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir(parents=True, exist_ok=True)
            # backend resolves to ultralytics by default; missing model should error.
            (bench_dir / "bad.yaml").write_text("name: bad\nepochs: 10\n", "utf-8")

            out_path = tdp / "reports" / "benchmarks" / "matrix.json"
            with self.assertRaises(SystemExit) as ctx:
                benchmark_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--bench-dir",
                        str(bench_dir),
                        "--out",
                        str(out_path),
                    ]
                )
            self.assertIn("Missing required field `model`", str(ctx.exception))

    def test_budget_caps_are_enforced_via_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir(parents=True, exist_ok=True)
            (bench_dir / "too_big.yaml").write_text(
                "\n".join(
                    [
                        "name: too_big",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 100",
                        "infer:",
                        "  imgsz: 640",
                        "",
                    ]
                ),
                "utf-8",
            )

            out_path = tdp / "reports" / "benchmarks" / "matrix.json"
            old_epochs = os.environ.get("HARCHOC_MAX_EPOCHS")
            try:
                os.environ["HARCHOC_MAX_EPOCHS"] = "50"
                with self.assertRaises(SystemExit) as ctx:
                    benchmark_main(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--bench-dir",
                            str(bench_dir),
                            "--out",
                            str(out_path),
                        ]
                    )
                self.assertIn("exceeds HARCHOC_MAX_EPOCHS", str(ctx.exception))
            finally:
                if old_epochs is None:
                    os.environ.pop("HARCHOC_MAX_EPOCHS", None)
                else:
                    os.environ["HARCHOC_MAX_EPOCHS"] = old_epochs

    def test_max_batch_cap_enforced_on_train_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir(parents=True, exist_ok=True)
            train_cfg = tdp / "train_high_batch.json"
            train_cfg.write_text(
                json.dumps({"batch": 4, "epochs": 1, "imgsz": 1280}) + "\n",
                encoding="utf-8",
            )
            (bench_dir / "yolov8n.yaml").write_text(
                "\n".join(
                    [
                        "name: yolov8n",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 1",
                        f"train_config: {train_cfg}",
                        "infer:",
                        "  imgsz: 1280",
                        "",
                    ]
                ),
                "utf-8",
            )

            out_path = tdp / "reports" / "benchmarks" / "matrix.json"
            old_batch = os.environ.get("HARCHOC_MAX_BATCH")
            try:
                os.environ["HARCHOC_MAX_BATCH"] = "1"
                with self.assertRaises(SystemExit) as ctx:
                    benchmark_main(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--bench-dir",
                            str(bench_dir),
                            "--out",
                            str(out_path),
                        ]
                    )
                self.assertIn("exceeds HARCHOC_MAX_BATCH", str(ctx.exception))
            finally:
                if old_batch is None:
                    os.environ.pop("HARCHOC_MAX_BATCH", None)
                else:
                    os.environ["HARCHOC_MAX_BATCH"] = old_batch

    @patch("scripts.benchmark_matrix._invoke_ultralytics_hsp_for_matrix")
    @patch("scripts.benchmark_matrix._invoke_train_for_bench")
    def test_no_dry_run_chains_test_eval_after_train(self, mock_invoke, mock_hsp_eval) -> None:
        mock_invoke.return_value = {
            "status": "ok",
            "returncode": 0,
            "run_name": "tiny",
            "weights": HSP_DETECTION_WEIGHTS,
        }
        mock_hsp_eval.return_value = {
            "status": "ok",
            "split": "test",
            "mAP50": 0.81,
            "mAP50_95": 0.42,
            "eval_out": "/tmp/test_eval.json",
            "test_count_mae": 61.3,
        }
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            (dataset_root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 2\n", "utf-8")
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir()
            (bench_dir / "tiny.yaml").write_text(
                "\n".join(
                    [
                        "name: tiny",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 1",
                        "infer:",
                        "  imgsz: 640",
                        "",
                    ]
                ),
                "utf-8",
            )

            out_path = tdp / "matrix.json"
            train_out = tdp / "matrix_train.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--out",
                    str(out_path),
                    "--train-out",
                    str(train_out),
                    "--no-dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            mock_invoke.assert_called_once()
            mock_hsp_eval.assert_called_once()
            train_obj = json.loads(train_out.read_text("utf-8"))
            self.assertEqual(train_obj["schema_version"], "benchmark_matrix_train.v1")
            run0 = train_obj["runs"][0]
            self.assertEqual(run0["mAP50"], 0.81)
            self.assertEqual(run0["mAP50_95"], 0.42)
            self.assertIn("test_eval", run0)

    @patch("scripts.benchmark_matrix._invoke_train_for_bench")
    def test_no_dry_run_calls_train_when_would_train(self, mock_invoke) -> None:
        mock_invoke.return_value = {"status": "ok", "returncode": 0}
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir()
            (bench_dir / "tiny.yaml").write_text(
                "\n".join(
                    [
                        "name: tiny",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 1",
                        "infer:",
                        "  imgsz: 640",
                        "",
                    ]
                ),
                "utf-8",
            )

            out_path = tdp / "matrix.json"
            train_out = tdp / "matrix_train.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--out",
                    str(out_path),
                    "--train-out",
                    str(train_out),
                    "--no-dry-run",
                    "--no-eval",
                ]
            )
            self.assertEqual(rc, 0)
            mock_invoke.assert_called_once()
            self.assertTrue(train_out.exists())
            train_obj = json.loads(train_out.read_text("utf-8"))
            self.assertEqual(train_obj["status"], "train")
            self.assertEqual(len(train_obj["runs"]), 1)

    def test_dry_run_does_not_write_train_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir()
            (bench_dir / "tiny.yaml").write_text(
                "name: tiny\nbackend: ultralytics\nmodel: yolov8n.pt\nepochs: 1\n",
                "utf-8",
            )

            out_path = tdp / "matrix.json"
            train_out = tdp / "matrix_train.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-dir",
                    str(bench_dir),
                    "--out",
                    str(out_path),
                    "--train-out",
                    str(train_out),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertFalse(train_out.exists())
            obj = json.loads(out_path.read_text("utf-8"))
            self.assertTrue(obj["dry_run"])


    def test_bench_matrix_metadata_rtdetr_from_committed_train_json(self) -> None:
        from harchoc.bench_config import bench_matrix_metadata, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "rtdetr_l_default.yaml")
        meta = bench_matrix_metadata(cfg)
        self.assertEqual(meta["backend"], "ultralytics")
        self.assertTrue(meta["nms_free"])
        self.assertEqual(meta["num_queries"], 300)
        self.assertTrue(meta["accept_rtdetr_query_truncation"])
        self.assertFalse(meta["rect"])
        self.assertEqual(meta["infer_max_det"], 300)
        self.assertEqual(meta["train_max_det"], 3000)

    def test_bench_matrix_metadata_yolo_nms_model(self) -> None:
        from harchoc.bench_config import bench_matrix_metadata, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8m_default.yaml")
        meta = bench_matrix_metadata(cfg)
        self.assertEqual(meta["backend"], "ultralytics")
        self.assertFalse(meta["nms_free"])
        self.assertIsNone(meta["num_queries"])
        self.assertIsNone(meta["accept_rtdetr_query_truncation"])
        self.assertFalse(meta["rect"])
        self.assertEqual(meta["train_max_det"], 3000)

    def test_bench_run_name_uses_model_epochs_seed_pattern(self) -> None:
        from scripts.benchmark_matrix import _bench_run_name, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8n_default.yaml")
        old = os.environ.get("HARCHOC_BENCH_USE_LEGACY_NAME")
        try:
            os.environ.pop("HARCHOC_BENCH_USE_LEGACY_NAME", None)
            self.assertEqual(_bench_run_name(cfg), "yolov8n_e100_s0")
        finally:
            if old is not None:
                os.environ["HARCHOC_BENCH_USE_LEGACY_NAME"] = old

    def test_bench_to_train_config_keeps_train_max_det_from_committed_json(self) -> None:
        from pathlib import Path

        from scripts.benchmark_matrix import _bench_to_train_config, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8m_default.yaml")
        doc = _bench_to_train_config(cfg, weights_path="/tmp/yolov8m.pt")
        self.assertEqual(doc["train"].get("max_det"), 3000)
        self.assertEqual(doc.get("eval", {}).get("max_det"), 3000)

    def test_all_ultralytics_bench_configs_have_train_bench_recipe(self) -> None:
        from harchoc.bench_config import _load_bench_train_raw, load_bench_config, select_backend

        repo_root = Path(__file__).resolve().parents[1]
        bench_dir = repo_root / "configs" / "bench"
        for pth in sorted(bench_dir.glob("*.yaml")):
            if pth.name.startswith("_"):
                continue
            cfg = load_bench_config(pth)
            if select_backend(cfg) != "ultralytics":
                continue
            try:
                _load_bench_train_raw(cfg)
            except FileNotFoundError as exc:
                self.fail(f"missing train bench recipe for {pth.name}: {exc}")

    def test_supergradients_bench_resolves_train_bench_json(self) -> None:
        from scripts.benchmark_matrix import _resolve_bench_train_config_path, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolo_nas_s_default.yaml")
        resolved = _resolve_bench_train_config_path(cfg)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertTrue(resolved.name.endswith("train_bench_yolo_nas_s.json"))

    def test_bench_yaml_matches_implicit_train_bench_recipe(self) -> None:
        from harchoc.bench_config import _infer_imgsz, _load_bench_train_raw, load_bench_config, select_backend

        repo_root = Path(__file__).resolve().parents[1]
        bench_dir = repo_root / "configs" / "bench"
        for pth in sorted(bench_dir.glob("*.yaml")):
            if pth.name.startswith("_"):
                continue
            cfg = load_bench_config(pth)
            if cfg.train_config:
                continue
            if select_backend(cfg) == "external":
                continue
            try:
                raw = _load_bench_train_raw(cfg)
            except FileNotFoundError:
                self.fail(f"missing train bench recipe for implicit train_config in {pth.name}")
            from harchoc.config_coerce import as_dict

            eval_section = as_dict(raw.get("eval"))

            self.assertEqual(
                cfg.epochs,
                raw.get("epochs"),
                msg=f"epochs mismatch for {pth.name}",
            )
            self.assertEqual(
                cfg.patience,
                raw.get("patience"),
                msg=f"patience mismatch for {pth.name}",
            )
            self.assertEqual(
                cfg.seed,
                raw.get("seed"),
                msg=f"seed mismatch for {pth.name}",
            )
            imgsz = _infer_imgsz(cfg)
            self.assertEqual(
                imgsz,
                raw.get("imgsz"),
                msg=f"infer.imgsz mismatch for {pth.name}",
            )
            infer = cfg.infer if isinstance(cfg.infer, dict) else {}
            infer_max_det = infer.get("max_det")
            self.assertEqual(
                infer_max_det,
                eval_section.get("max_det"),
                msg=f"infer.max_det mismatch for {pth.name}",
            )

    @patch("scripts.eval.main")
    def test_invoke_test_eval_for_bench_passes_imgsz(self, mock_eval_main) -> None:
        from scripts.benchmark_matrix import _invoke_test_eval_for_bench, load_bench_config

        mock_eval_main.return_value = 0
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8m_default.yaml")
        with tempfile.TemporaryDirectory() as td:
            eval_out = Path(td) / "eval.json"
            eval_out.write_text(json.dumps({"mAP50": 0.5, "mAP50_95": 0.3}), "utf-8")
            _invoke_test_eval_for_bench(
                cfg=cfg,
                weights=HSP_DETECTION_WEIGHTS,
                manifest=str(repo_root / "data" / "manifest.json"),
                default_dataset_name="default",
                dataset_env=None,
                train_doc={"imgsz": 1280, "eval": {"device": "cpu"}},
                eval_out=eval_out,
            )
        argv = mock_eval_main.call_args[0][0]
        self.assertIn("--imgsz", argv)
        idx = argv.index("--imgsz")
        self.assertEqual(argv[idx + 1], "1280")
        self.assertIn("--device", argv)
        self.assertIn("cpu", argv)
        self.assertIn("--export-device", argv)

    def test_invoke_test_eval_for_bench_skips_when_eval_skip(self) -> None:
        from scripts.benchmark_matrix import _invoke_test_eval_for_bench, load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8m_default.yaml")
        out = _invoke_test_eval_for_bench(
            cfg=cfg,
            weights=HSP_DETECTION_WEIGHTS,
            manifest=str(repo_root / "data" / "manifest.json"),
            default_dataset_name="default",
            dataset_env=None,
            train_doc={"eval": {"skip": True}},
        )
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["reason"], "eval.skip")

    @patch("scripts.benchmark_matrix._invoke_test_eval_for_bench")
    @patch("scripts.benchmark_matrix._invoke_train_for_bench")
    def test_no_dry_run_skips_chained_eval_when_eval_skip(
        self, mock_invoke, mock_eval
    ) -> None:
        from harchoc.train_config import load_train_config_json

        mock_invoke.return_value = {
            "status": "ok",
            "returncode": 0,
            "run_name": "rtdetr_smoke",
            "weights": HSP_DETECTION_WEIGHTS,
        }
        repo_root = Path(__file__).resolve().parents[1]
        smoke_doc = load_train_config_json(
            repo_root / "configs" / "experiments" / "train_rtdetr_smoke_15ep.json",
            repo_root=repo_root,
        )
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            (dataset_root / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\nnc: 2\n",
                "utf-8",
            )
            os.environ["DATASET_ROOT"] = str(dataset_root)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            bench_dir = tdp / "bench"
            bench_dir.mkdir()
            (bench_dir / "rtdetr.yaml").write_text(
                "\n".join(
                    [
                        "name: rtdetr_l",
                        "backend: ultralytics",
                        "model: rtdetr-l.pt",
                        "epochs: 15",
                        "train_config: configs/experiments/train_rtdetr_smoke_15ep.json",
                        "infer:",
                        "  imgsz: 1280",
                        "",
                    ]
                ),
                "utf-8",
            )

            with patch(
                "scripts.benchmark_matrix._bench_to_train_config",
                return_value=smoke_doc,
            ):
                rc = benchmark_main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--bench-config",
                        str(bench_dir / "rtdetr.yaml"),
                        "--out",
                        str(tdp / "matrix.json"),
                        "--train-out",
                        str(tdp / "matrix_train.json"),
                        "--no-dry-run",
                    ]
                )
            self.assertEqual(rc, 0)
            mock_eval.assert_not_called()
            train_obj = json.loads((tdp / "matrix_train.json").read_text("utf-8"))
            self.assertNotIn("test_eval", train_obj["runs"][0])

    @patch("harchoc.ultralytics_eval.run_val")
    def test_ultralytics_eval_one_warns_on_results_dict_failure(self, mock_run_val) -> None:
        from scripts.benchmark_matrix import _ultralytics_eval_one, load_bench_config

        class _BadMetrics:
            @property
            def results_dict(self) -> dict[str, object]:
                raise RuntimeError("results_dict unavailable")

        mock_run_val.return_value = _BadMetrics()
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8n_default.yaml")
        err = io.StringIO()
        with patch.object(sys, "stderr", err):
            out = _ultralytics_eval_one(
                cfg=cfg,
                dataset_yaml=repo_root / "data" / "data.yaml",
                weights=Path(HSP_DETECTION_WEIGHTS),
            )
        self.assertIsNone(out["results"])
        err_obj = out.get("results_dict_error")
        self.assertIsInstance(err_obj, dict)
        assert isinstance(err_obj, dict)
        self.assertIn("results_dict unavailable", err_obj.get("msg", ""))
        self.assertIn("results_dict unavailable", err.getvalue())

    @patch("harchoc.ultralytics_eval.run_val")
    def test_ultralytics_eval_one_strict_raises_on_results_dict_failure(self, mock_run_val) -> None:
        from scripts.benchmark_matrix import _ultralytics_eval_one, load_bench_config

        class _BadMetrics:
            @property
            def results_dict(self) -> dict[str, object]:
                raise RuntimeError("results_dict unavailable")

        mock_run_val.return_value = _BadMetrics()
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8n_default.yaml")
        old = os.environ.get("HARCHOC_STRICT_ML")
        try:
            os.environ["HARCHOC_STRICT_ML"] = "1"
            with self.assertRaises(RuntimeError) as ctx:
                _ultralytics_eval_one(
                    cfg=cfg,
                    dataset_yaml=repo_root / "data" / "data.yaml",
                    weights=Path(HSP_DETECTION_WEIGHTS),
                )
            self.assertIn("results_dict unavailable", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("HARCHOC_STRICT_ML", None)
            else:
                os.environ["HARCHOC_STRICT_ML"] = old

    def test_aggregate_seeds_writes_stats(self) -> None:
        from scripts.benchmark_matrix import main as benchmark_main

        repo_root = Path(__file__).resolve().parents[1]
        fixtures = repo_root / "tests" / "fixtures"

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            train_out = tdp / "matrix_train.json"
            train_out.write_text(
                json.dumps(
                    {
                        "schema_version": "benchmark_matrix_train.v1",
                        "runs": [
                            {
                                "status": "ok",
                                "name": "yolov8n",
                                "run_name": "yolov8n_e100_s0",
                                "mAP50": 0.8,
                                "mAP50_95": 0.4,
                                "error_test_report": str(fixtures / "error_s0_report.json"),
                            },
                            {
                                "status": "ok",
                                "name": "yolov8n",
                                "run_name": "yolov8n_e100_s1",
                                "mAP50": 0.82,
                                "mAP50_95": 0.41,
                                "threshold_test_locked": str(fixtures / "threshold_s1_locked.json"),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stats_out = tdp / "seed_stats.json"
            rc = benchmark_main(
                [
                    "--aggregate-seeds",
                    "--train-out",
                    str(train_out),
                    "--seed-stats-out",
                    str(stats_out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(stats_out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "benchmark_matrix_seed_stats.v1")
            self.assertEqual(obj["status"], "ok")
            self.assertIn("yolov8n", obj["models"])
            self.assertEqual(obj["n_count_mae"], 2)
            self.assertAlmostEqual(obj["count_mae_mean"], 13.0)
            model = obj["models"]["yolov8n"]
            self.assertAlmostEqual(model["count_mae_mean"], 13.0)
            self.assertIsNotNone(model["count_mae_std"])

    def test_bench_defaults_include_merges_epochs(self) -> None:
        from harchoc.bench_config import load_bench_config

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8m_default.yaml")
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.patience, 50)
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.infer.get("imgsz"), 1280)

    def test_ci_validate_zoo_matrix(self) -> None:
        """CI gate (no GPU): zoo manifest + bench YAML parity via library and CLI."""
        from harchoc.zoo_matrix_scaffold import validate_zoo_matrix

        report = validate_zoo_matrix()
        self.assertEqual(report.mode, "validate")
        self.assertEqual([], report.errors, msg="\n".join(report.errors))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "zoo_report.json"
            rc = benchmark_main(
                [
                    "--validate-zoo",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "zoo_matrix_scaffold_report.v1")
            self.assertEqual(obj["mode"], "validate")
            self.assertEqual(obj.get("errors") or [], [])

    def test_scaffold_zoo_skips_hand_authored_rtdetr(self) -> None:
        from harchoc.zoo_matrix_scaffold import scaffold_zoo_matrix

        repo_root = Path(__file__).resolve().parents[1]
        rtdetr_path = repo_root / "configs" / "bench" / "rtdetr_l_nq1024.yaml"
        before = rtdetr_path.read_text(encoding="utf-8")
        report = scaffold_zoo_matrix(repo_root=repo_root, write=True)
        after = rtdetr_path.read_text(encoding="utf-8")
        self.assertEqual(before, after)
        self.assertGreaterEqual(report.bench_skipped, 8)

    def test_scaffold_zoo_does_not_write_runtime_ultralytics_train_bench(self) -> None:
        from harchoc.zoo_matrix_scaffold import scaffold_zoo_matrix

        repo_root = Path(__file__).resolve().parents[1]
        runtime_path = repo_root / "configs" / "experiments" / "train_bench_yolo11s.json"
        existed_before = runtime_path.is_file()
        report = scaffold_zoo_matrix(repo_root=repo_root, write=True)
        self.assertFalse(runtime_path.is_file(), msg="runtime ultralytics train_bench must not be scaffolded")
        if existed_before:
            runtime_path.unlink(missing_ok=True)
        self.assertGreaterEqual(report.train_skipped, 1)

    def test_aggregate_seeds_null_mae_without_artifacts(self) -> None:
        from scripts.benchmark_matrix import main as benchmark_main

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            train_out = tdp / "matrix_train.json"
            train_out.write_text(
                json.dumps(
                    {
                        "schema_version": "benchmark_matrix_train.v1",
                        "runs": [
                            {
                                "status": "ok",
                                "name": "yolov8n",
                                "run_name": "yolov8n_e100_s0",
                                "mAP50": 0.8,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stats_out = tdp / "seed_stats.json"
            rc = benchmark_main(
                [
                    "--aggregate-seeds",
                    "--train-out",
                    str(train_out),
                    "--seed-stats-out",
                    str(stats_out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(stats_out.read_text("utf-8"))
            self.assertEqual(obj["n_count_mae"], 0)
            self.assertIsNone(obj["count_mae_mean"])
            self.assertIsNone(obj["count_mae_std"])


if __name__ == "__main__":
    unittest.main()

