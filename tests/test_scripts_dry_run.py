import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class ScriptDryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_quiet = os.environ.get("HARCHOC_QUIET")
        os.environ["HARCHOC_QUIET"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._old_quiet is None:
            os.environ.pop("HARCHOC_QUIET", None)
        else:
            os.environ["HARCHOC_QUIET"] = cls._old_quiet

    def _assert_dry_run_writes(self, main_fn, argv: list[str], out_path: Path) -> None:
        rc = main_fn(argv)
        self.assertEqual(rc, 0)
        self.assertTrue(out_path.exists())
        obj = json.loads(out_path.read_text("utf-8"))
        self.assertEqual(obj.get("status"), "dry-run")
        sv = obj.get("schema_version")
        self.assertIsInstance(sv, str)
        self.assertTrue(str(sv).endswith(".v1"), msg=f"Unexpected schema_version: {sv!r}")

    def test_eval_dry_run(self) -> None:
        from scripts.eval import main

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "data" / "splits").mkdir(parents=True, exist_ok=True)
            (base / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")
            # Avoid depending on a checked-out data/manifest.json by forcing a local dataset root.
            (base / "dataset").mkdir(parents=True, exist_ok=True)
            out = base / "eval.json"

            old_cwd = Path.cwd()
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.chdir(base)
                os.environ["DATASET_ROOT"] = str((base / "dataset").resolve())
                self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)
            finally:
                os.chdir(old_cwd)
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

    def test_eval_dry_run_uses_detection_model_env(self) -> None:
        from scripts.eval import main

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "data" / "splits").mkdir(parents=True, exist_ok=True)
            (base / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")
            (base / "dataset").mkdir(parents=True, exist_ok=True)
            out = base / "eval.json"
            old = os.environ.get("DETECTION_MODEL")
            old_root = os.environ.get("DATASET_ROOT")
            old_cwd = Path.cwd()
            try:
                os.chdir(base)
                os.environ["DATASET_ROOT"] = str((base / "dataset").resolve())
                os.environ["DETECTION_MODEL"] = "/tmp/does-not-need-to-exist.pt"
                rc = main(["--dry-run", "--out", str(out)])
            finally:
                os.chdir(old_cwd)
                if old is None:
                    os.environ.pop("DETECTION_MODEL", None)
                else:
                    os.environ["DETECTION_MODEL"] = old
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("weights"), "/tmp/does-not-need-to-exist.pt")

    def test_eval_dry_run_split_file_default_if_present(self) -> None:
        from scripts.eval import main

        with tempfile.TemporaryDirectory() as td:
            # Make a fake repo-like cwd with data/splits/test.txt
            base = Path(td)
            (base / "data" / "splits").mkdir(parents=True, exist_ok=True)
            (base / "data" / "splits" / "test.txt").write_text("images/val/a.jpg\n", "utf-8")
            (base / "dataset").mkdir(parents=True, exist_ok=True)
            out = base / "eval.json"

            old_cwd = Path.cwd()
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.chdir(base)
                os.environ["DATASET_ROOT"] = str((base / "dataset").resolve())
                rc = main(["--dry-run", "--out", str(out)])
            finally:
                os.chdir(old_cwd)
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("split_source", {}).get("kind"), "split_file")
            self.assertEqual(obj.get("split_source", {}).get("path"), "data/splits/test.txt")

    def test_describe_split_dry_run(self) -> None:
        from scripts.describe_split import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "split.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)

    def test_threshold_sweep_dry_run(self) -> None:
        from scripts.threshold_sweep import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sweep.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)

    def test_error_analysis_dry_run(self) -> None:
        from scripts.error_analysis import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "err.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)

    def test_threshold_sweep_light_mode(self) -> None:
        from scripts.threshold_sweep import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sweep.json"
            rc = main(["--light", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("status"), "ok")
            self.assertTrue(obj.get("light"))
            self.assertGreaterEqual(len(obj.get("rows") or []), 1)

    def test_error_analysis_light_mode(self) -> None:
        from scripts.error_analysis import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "summary.json"
            report = Path(td) / "report.json"
            rc = main(["--light", "--out", str(out), "--report", str(report)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("status"), "ok")
            self.assertTrue(obj.get("light"))
            self.assertIn("fp_breakdown", obj)

    def test_cv_eval_dry_run(self) -> None:
        from scripts.cv_eval import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cv.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)

    def test_finetune_dry_run(self) -> None:
        from scripts.finetune import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)

    def test_eval_domains_dry_run(self) -> None:
        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "domains.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--out", str(out)], out)

    def test_make_figures_dry_run(self) -> None:
        from scripts.make_figures import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "meta.json"
            self._assert_dry_run_writes(main, ["--dry-run", "--meta-out", str(out)], out)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("schema_version"), "figures_run.v1")
            self.assertEqual(obj.get("script"), "make_figures")
            self.assertIsInstance(obj.get("figures"), list)
            self.assertGreaterEqual(len(obj.get("figures") or []), 1)

    def test_check_gpu_sanity_subcommand(self) -> None:
        from scripts.check_gpu import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "gpu.json"
            rc = main(["sanity", "--dry-run", "--out", str(out)])
            self.assertEqual(rc, 0)

    def test_check_gpu_smoke_ultralytics_subcommand(self) -> None:
        from scripts.check_gpu import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.json"
            rc = main(["smoke-ultralytics", "--dry-run", "--out", str(out)])
            self.assertEqual(rc, 0)

    def test_matrix_seed_stats_dry_run(self) -> None:
        from scripts.matrix_seed_stats import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "seed_stats.json"
            self._assert_dry_run_writes(
                main,
                ["--dry-run", "--out", str(out)],
                out,
            )
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("schema_version"), "matrix_seed_stats.v1")
            self.assertIsNone(obj.get("count_mae_mean"))
            self.assertIsNone(obj.get("count_mae_std"))

    def test_run_meta_dry_run(self) -> None:
        from scripts.run_meta import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "meta.json"
            rc = main(["--dry-run", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("schema_version"), "run_meta.v1")
            self.assertEqual(obj.get("script"), "run_meta")
            self.assertEqual(obj.get("status"), "dry-run")

    def test_pipeline_request_dry_run_writes_contract(self) -> None:
        from scripts.pipeline_request import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "req.json"
            img = Path(__file__).parent / "assets" / "sunflower.ppm"
            rc = main(["--dry-run", "--image", str(img), "--request-id", "req-123", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("schema_version"), "pipeline_request.v1")
            self.assertEqual(obj.get("script"), "pipeline_request")
            self.assertEqual(obj.get("status"), "dry-run")
            self.assertEqual(obj.get("request_id"), "req-123")
            self.assertEqual(obj.get("input", {}).get("image"), str(img))
            self.assertIsInstance(obj.get("availability", {}).get("modules"), dict)

    def test_pick_sample_images_prints_paths(self) -> None:
        from scripts.pick_sample_images import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img_dir = root / "images" / "train"
            img_dir.mkdir(parents=True, exist_ok=True)
            (img_dir / "a.jpg").write_bytes(b"fake")
            (img_dir / "b.png").write_bytes(b"fake")

            buf = StringIO()
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                with redirect_stdout(buf):
                    rc = main(["--split", "train", "--n", "2"])
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old

            self.assertEqual(rc, 0)
            lines = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            for ln in lines:
                self.assertTrue(Path(ln).is_absolute())
                self.assertTrue(Path(ln).exists())


if __name__ == "__main__":
    unittest.main()

