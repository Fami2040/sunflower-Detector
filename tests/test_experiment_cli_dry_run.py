import json
import os
import tempfile
import unittest
from pathlib import Path


class ExperimentCliDryRunTests(unittest.TestCase):
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

    def test_splits_dry_run(self) -> None:
        from scripts.experiment import main

        rc = main(["--dry-run", "splits"])
        self.assertEqual(rc, 0)

    def test_validate_splits_dry_run(self) -> None:
        from scripts.experiment import main

        rc = main(["--dry-run", "validate-splits"])
        self.assertEqual(rc, 0)

    def test_describe_dry_run_writes_json(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "describe.json"
            rc = main(["--dry-run", "describe", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("status"), "dry-run")
            self.assertEqual(obj.get("script"), "describe_split")

    def test_eval_dry_run_writes_json(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "data" / "splits").mkdir(parents=True, exist_ok=True)
            (base / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")
            # Avoid depending on repo data/manifest.json when cwd is the temp dir.
            (base / "dataset").mkdir(parents=True, exist_ok=True)

            out = base / "eval.json"
            old_cwd = Path.cwd()
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.chdir(base)
                os.environ["DATASET_ROOT"] = str((base / "dataset").resolve())
                rc = main(["--dry-run", "eval", "--out", str(out)])
            finally:
                os.chdir(old_cwd)
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("status"), "dry-run")
            self.assertEqual(obj.get("script"), "eval")

    def test_threshold_sweep_dry_run(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sweep.json"
            rc = main(["--dry-run", "threshold-sweep", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("status"), "dry-run")
            self.assertEqual(obj.get("script"), "threshold_sweep")

    def test_benchmark_dry_run_writes_plan(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)

            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                out = tdp / "matrix.json"
                rc = main(["--dry-run", "benchmark", "--out", str(out)])
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old

            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("status"), "plan")
            self.assertTrue(obj.get("dry_run"))

    def test_aug_compare_dry_run_subcommand_flag(self) -> None:
        from scripts.experiment import main

        rc = main(["aug-compare", "--dry-run"])
        self.assertEqual(rc, 0)

    def test_backlog_narrative_dry_run_subcommand_flag(self) -> None:
        from scripts.experiment import main

        rc = main(["backlog-narrative", "--dry-run"])
        self.assertEqual(rc, 0)

    def test_backlog_narrative_global_dry_run(self) -> None:
        from scripts.experiment import main

        rc = main(["--dry-run", "backlog-narrative"])
        self.assertEqual(rc, 0)

    def test_train_dry_run_writes_run_files(self) -> None:
        from scripts.experiment import main

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_dir = tdp / "runs"
            name = "unit_test_run"

            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(dataset_root)
                rc = main(["--dry-run", "train", "--out-dir", str(out_dir), "--name", name])
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old

            self.assertEqual(rc, 0)
            run_dir = out_dir / name
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "meta.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())

            cfg = json.loads((run_dir / "config.json").read_text("utf-8"))
            self.assertEqual(cfg.get("schema_version"), "train_config.v1")
            self.assertEqual(cfg.get("status"), "dry-run")
            self.assertEqual(cfg.get("script"), "train")


if __name__ == "__main__":
    unittest.main()

