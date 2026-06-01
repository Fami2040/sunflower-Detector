import tempfile
import unittest
from pathlib import Path


class ExperimentConfigNormalizeTests(unittest.TestCase):
    def test_canonical_experiment_specs_normalize(self) -> None:
        from harchoc.experiment_config import load_config, normalize_experiment_spec

        repo_root = Path(__file__).resolve().parents[1]
        cfg_dir = repo_root / "configs" / "experiments"
        cfg_paths = sorted([p for p in cfg_dir.glob("*.json")])
        self.assertGreaterEqual(len(cfg_paths), 3)

        with tempfile.TemporaryDirectory() as td:
            dataset_root = Path(td) / "dataset"
            (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
            (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
            (dataset_root / "images" / "test").mkdir(parents=True, exist_ok=True)

            for p in cfg_paths:
                raw = load_config(p)
                if raw.get("schema_version") != "experiments.v1":
                    continue
                run_raw = raw.get("run")
                if not isinstance(run_raw, dict) or not str(run_raw.get("kind") or "").strip():
                    self.fail(
                        f"{p} declares experiments.v1 but has no run.kind "
                        "(use a top-level section block without schema_version, e.g. reviewer_counting.json)"
                    )
                raw.setdefault("dataset", {})
                self.assertIsInstance(raw["dataset"], dict)
                raw["dataset"]["dataset_env"] = {
                    "DATASET_ROOT": str(dataset_root),
                    "YOLO_DATA_YAML": "",
                    "DATASET_NAME": "",
                }

                resolved = normalize_experiment_spec(raw, repo_root=repo_root)
                self.assertEqual(resolved.get("schema_version"), "experiments.v1", msg=str(p))
                self.assertEqual(Path(resolved["dataset"]["root"]), dataset_root.resolve(), msg=str(p))
                self.assertIn(
                    resolved["run"]["kind"],
                    {
                        "eval",
                        "benchmark_matrix",
                        "sahi_matrix_eval",
                        "split_drift",
                        "threshold_sweep",
                        "error_analysis",
                        "cv_eval",
                        "fp_budget_sweep",
                    },
                    msg=str(p),
                )
                # Common: output path is normalized to an absolute path when present.
                if resolved["run"].get("out") is not None:
                    self.assertTrue(str(resolved["run"]["out"]).startswith(str(repo_root.resolve())), msg=str(p))

    def test_normalize_uses_dataset_env_root(self) -> None:
        from harchoc.experiment_config import normalize_config

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td) / "repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "data").mkdir(parents=True, exist_ok=True)
            (repo_root / "data" / "manifest.json").write_text('{"datasets": []}\n', "utf-8")

            dataset_root = Path(td) / "dataset"
            (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)

            rcfg = normalize_config(
                {
                    "dataset_env": {
                        "DATASET_ROOT": str(dataset_root),
                        "YOLO_DATA_YAML": "",
                        "DATASET_NAME": "",
                    }
                },
                repo_root=repo_root,
            )
            self.assertEqual(Path(rcfg["dataset"]["root"]), dataset_root.resolve())
            self.assertEqual(rcfg["splits"]["val"]["source"]["kind"], "dir")
            self.assertEqual(Path(rcfg["splits"]["val"]["path"]), (dataset_root / "images" / "val").resolve())

    def test_normalize_defaults_split_file_when_present(self) -> None:
        from harchoc.experiment_config import normalize_config

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td) / "repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "data" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (repo_root / "data" / "manifest.json").write_text('{"datasets": []}\n', "utf-8")
            (repo_root / "data" / "splits").mkdir(parents=True, exist_ok=True)
            (repo_root / "data" / "splits" / "test.txt").write_text("images/val/a.jpg\n", "utf-8")

            dataset_root = Path(td) / "dataset"
            dataset_root.mkdir(parents=True, exist_ok=True)

            rcfg = normalize_config(
                {
                    "dataset_env": {
                        "DATASET_ROOT": str(dataset_root),
                        "YOLO_DATA_YAML": "",
                        "DATASET_NAME": "",
                    }
                },
                repo_root=repo_root,
            )

            self.assertEqual(rcfg["split_source"]["kind"], "split_file")
            self.assertEqual(Path(rcfg["split_source"]["path"]), (repo_root / "data" / "splits" / "test.txt").resolve())
            self.assertEqual(rcfg["default_dataset_name"], "sunflower-cvat-1093")
            self.assertEqual(Path(rcfg["manifest"]), (repo_root / "data" / "manifest.json").resolve())


if __name__ == "__main__":
    unittest.main()

