import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")

from harchoc.bench_config import load_bench_config
from harchoc.experiment_argv import argv_for_benchmark
from harchoc.experiment_config import load_config, script_section_from_config
from harchoc.sahi_matrix import (
    deploy_default_sahi_params,
    expand_bench_configs_with_sahi,
    parse_sahi_eval_params,
    parse_sahi_rows_config,
    resolve_sahi_rows_for_bench,
    sahi_params_from_infer,
)
from scripts.benchmark_matrix import main as benchmark_main


class SahiMatrixTests(unittest.TestCase):
    def test_deploy_default_matches_run_infer_once(self) -> None:
        p = deploy_default_sahi_params()
        self.assertEqual(p.slice_size, 500)
        self.assertEqual(p.overlap, 0.35)
        self.assertEqual(p.nms_iou, 0.50)
        self.assertEqual(p.conf_fertilized, 0.06)
        self.assertEqual(p.conf_unfertilized, 0.04)

    def test_parse_sahi_eval_params_validates_overlap(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            parse_sahi_eval_params({"slice_size": 500, "overlap": 1.0, "nms_iou": 0.5})
        self.assertIn("overlap", str(ctx.exception))

    def test_sahi_params_from_infer_tiling_only(self) -> None:
        p = sahi_params_from_infer({"tiling": "sahi", "imgsz": 1280})
        self.assertEqual(p, deploy_default_sahi_params())

    def test_sahi_params_from_infer_nested_block(self) -> None:
        p = sahi_params_from_infer(
            {
                "tiling": "sahi",
                "sahi": {"slice_size": 512, "overlap": 0.32, "nms_iou": 0.58, "label": "custom"},
            }
        )
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.slice_size, 512)
        self.assertEqual(p.label, "custom")

    def test_resolve_sahi_rows_prefers_bench_infer_over_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench_path = Path(td) / "one.yaml"
            bench_path.write_text(
                "\n".join(
                    [
                        "name: one",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 1",
                        "infer:",
                        "  imgsz: 1280",
                        "  tiling: sahi",
                        "  sahi_slice_size: 560",
                        "  sahi_overlap: 0.30",
                        "  sahi_nms_iou: 0.55",
                        "  sahi_label: bench_row",
                        "",
                    ]
                ),
                "utf-8",
            )
            cfg = load_bench_config(bench_path)
        matrix_rows = parse_sahi_rows_config(
            [{"label": "matrix_row", "slice_size": 640, "overlap": 0.28, "nms_iou": 0.45}]
        )
        rows = resolve_sahi_rows_for_bench(cfg, matrix_rows=matrix_rows, sahi_eval=True)
        self.assertEqual(len(rows), 1)
        row0 = rows[0]
        self.assertIsNotNone(row0)
        assert row0 is not None
        self.assertEqual(row0.label, "bench_row")
        self.assertEqual(row0.slice_size, 560)

    def test_expand_bench_configs_cross_product(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo_root / "configs" / "bench" / "yolov8n_default.yaml")
        matrix_rows = parse_sahi_rows_config(
            [
                {"label": "a", "slice_size": 500, "overlap": 0.35, "nms_iou": 0.5},
                {"label": "b", "slice_size": 512, "overlap": 0.35, "nms_iou": 0.58},
            ]
        )
        expanded = expand_bench_configs_with_sahi([cfg], matrix_rows=matrix_rows, sahi_eval=True)
        self.assertEqual(len(expanded), 2)
        sahi_a = expanded[0][1]
        sahi_b = expanded[1][1]
        self.assertIsNotNone(sahi_a)
        self.assertIsNotNone(sahi_b)
        assert sahi_a is not None
        assert sahi_b is not None
        self.assertEqual(sahi_a.label, "a")
        self.assertEqual(sahi_b.label, "b")

    def test_dry_run_sahi_eval_writes_plan_json(self) -> None:
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
            (bench_dir / "one.yaml").write_text(
                "\n".join(
                    [
                        "name: one",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 1",
                        "infer:",
                        "  imgsz: 1280",
                        "",
                    ]
                ),
                "utf-8",
            )

            out_path = tdp / "sahi_matrix.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-config",
                    str(bench_dir / "one.yaml"),
                    "--sahi-eval",
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out_path.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "sahi_matrix_eval.v1")
            self.assertEqual(obj["eval_protocol"], "sahi")
            self.assertEqual(len(obj["runs"]), 1)
            run = obj["runs"][0]
            self.assertEqual(run["eval"]["protocol"], "sahi")
            self.assertIn("sahi", run["planned"])
            self.assertEqual(run["planned"]["sahi"]["slice_size"], 500)

    def test_experiments_v1_sahi_matrix_eval_config_drives_plan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "configs" / "experiments" / "sahi_matrix_eval_plan_dry.json"
        section = script_section_from_config(load_config(str(cfg_path)), "benchmark_matrix")
        self.assertTrue(section.get("sahi_eval"))
        self.assertIn("sahi_rows", section)

        argv = argv_for_benchmark(section)
        self.assertIn("--sahi-eval", argv)

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)

            manifest_path = tdp / "manifest.json"
            manifest_path.write_text(json.dumps({"datasets": []}), "utf-8")

            out_path = tdp / "plan.json"
            cfg_path_local = tdp / "exp.json"
            cfg_path_local.write_text(cfg_path.read_text("utf-8"), "utf-8")
            obj = json.loads(cfg_path_local.read_text("utf-8"))
            obj["dataset"]["manifest"] = str(manifest_path)
            obj["run"]["out"] = str(out_path)
            cfg_path_local.write_text(json.dumps(obj), "utf-8")

            rc = benchmark_main(["--config", str(cfg_path_local)])
            self.assertEqual(rc, 0)
            plan = json.loads(out_path.read_text("utf-8"))
            self.assertEqual(plan["schema_version"], "sahi_matrix_eval.v1")
            self.assertGreater(len(plan["runs"]), 1)
            labels = {r["planned"]["sahi"]["label"] for r in plan["runs"]}
            self.assertIn("deploy_default", labels)
            self.assertIn("recall_512", labels)


if __name__ == "__main__":
    unittest.main()
