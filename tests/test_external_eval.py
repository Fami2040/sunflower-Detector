import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")


class ExternalEvalTests(unittest.TestCase):
    def test_external_results_to_detections_sorts_and_caps(self) -> None:
        from harchoc.external_detector_eval import external_results_to_detections

        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        labels = torch.tensor([[1, 0]])
        boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]])
        scores = torch.tensor([[0.2, 0.9]])

        dets = external_results_to_detections(labels, boxes, scores, max_det=1)
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(float(dets[0]["score"]), 0.9)

    def test_eval_hsp_for_bench_missing_weights(self) -> None:
        from harchoc.external_detector_eval import eval_hsp_for_bench

        with tempfile.TemporaryDirectory() as td:
            out = eval_hsp_for_bench(
                source_id="rtdetrv2_l",
                model_id="rtdetrv2_l",
                weights=Path(td) / "missing.pth",
                dataset_root=Path(td),
            )
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["reason"], "weights_not_found")

    @patch("harchoc.external_detector_eval.export_hsp_gt_preds_json")
    @patch("harchoc.external_detector_eval._run_error_analysis")
    @patch("harchoc.external_detector_eval.external_bench_availability", return_value=(True, None))
    @patch("harchoc.external_detector_eval.resolve_external_repo_path")
    @patch("harchoc.external_detector_eval.entry_for_bench")
    def test_eval_hsp_for_bench_returns_count_mae(
        self,
        mock_entry,
        mock_repo,
        _mock_avail,
        mock_error,
        mock_export,
    ) -> None:
        from harchoc.external_detector_eval import eval_hsp_for_bench

        repo_root = Path(__file__).resolve().parents[1]
        mock_entry.return_value = type(
            "E",
            (),
            {"train_stack": "rtdetrv2_pytorch", "config_relpath": "configs/rtdetrv2/x.yml"},
        )()
        mock_repo.return_value = repo_root / "external" / "RT-DETR" / "rtdetrv2_pytorch"
        mock_export.return_value = {"gt_json": "g", "preds_json": "p"}

        def _write_error(_rr: Path, argv: list[str], _env: dict[str, str]) -> int:
            out_idx = argv.index("--out") + 1
            err_path = Path(argv[out_idx])
            if not err_path.is_absolute():
                err_path = repo_root / err_path
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(
                json.dumps({"counting_metrics": {"mae": 42.5, "mae_ci": {"low": 40.0, "high": 45.0}}}),
                encoding="utf-8",
            )
            return 0

        mock_error.side_effect = _write_error

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            weights = tdp / "best.pth"
            weights.write_bytes(b"x")
            run_dir = tdp / "rtdetrv2_l_e100_s0"
            run_dir.mkdir()
            (run_dir / "harchoc_train_overlay.yml").write_text("task: detect\n", encoding="utf-8")
            split_dir = repo_root / "data" / "splits"
            if not (split_dir / "test.txt").is_file():
                self.skipTest("data/splits/test.txt missing")

            out = eval_hsp_for_bench(
                source_id="rtdetrv2_l",
                model_id="rtdetrv2_l",
                weights=weights,
                dataset_root=tdp,
                run_dir=run_dir,
                repo_root=repo_root,
            )
        self.assertEqual(out["status"], "ok")
        self.assertAlmostEqual(float(out["test_count_mae"]), 42.5)
        mock_export.assert_called_once()
        mock_error.assert_called_once()

    @patch("scripts.benchmark_matrix._invoke_test_eval_for_external")
    @patch("scripts.benchmark_matrix._invoke_train_for_bench")
    def test_matrix_chains_external_hsp_eval(self, mock_train, mock_eval) -> None:
        from scripts.benchmark_matrix import main as benchmark_main

        mock_train.return_value = {
            "status": "ok",
            "returncode": 0,
            "run_name": "rtdetrv2_l_e100_s0",
            "weights": "/tmp/best.pth",
            "backend": "external",
        }
        mock_eval.return_value = {
            "status": "ok",
            "split": "test",
            "test_count_mae": 55.0,
            "error_json": "/tmp/error_test.json",
        }

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
            (bench_dir / "ext.yaml").write_text(
                "\n".join(
                    [
                        "name: bench_rtdetrv2_l_default",
                        "backend: external",
                        "source_id: rtdetrv2_l",
                        "model_id: rtdetrv2_l",
                        "epochs: 100",
                        "seed: 0",
                        "infer:",
                        "  imgsz: 1280",
                        "  max_det: 3000",
                        "",
                    ]
                ),
                "utf-8",
            )

            train_out = tdp / "matrix_train.json"
            rc = benchmark_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--bench-config",
                    str(bench_dir / "ext.yaml"),
                    "--out",
                    str(tdp / "matrix.json"),
                    "--train-out",
                    str(train_out),
                    "--no-dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            mock_eval.assert_called_once()
            train_obj = json.loads(train_out.read_text("utf-8"))
            run0 = train_obj["runs"][0]
            self.assertEqual(run0["test_count_mae"], 55.0)
            self.assertIn("test_eval", run0)
            self.assertEqual(run0["error_test_report"], "/tmp/error_test.json")


if __name__ == "__main__":
    unittest.main()
