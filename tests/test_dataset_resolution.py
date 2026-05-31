import os
import tempfile
import unittest
from pathlib import Path

from harchoc.datasets import resolve_dataset


class DatasetResolutionTests(unittest.TestCase):
    def test_dataset_root_env_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env = {"DATASET_ROOT": td, "YOLO_DATA_YAML": "", "DATASET_NAME": ""}
            spec = resolve_dataset(environ=env)
            self.assertEqual(spec.root, Path(td).resolve())

    def test_dataset_root_arg_overrides_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            explicit = Path(td) / "explicit_root"
            explicit.mkdir(parents=True, exist_ok=True)
            env = {"DATASET_ROOT": str(Path(td) / "env_root"), "YOLO_DATA_YAML": "", "DATASET_NAME": ""}
            spec = resolve_dataset(dataset_root=str(explicit), environ=env)
            self.assertEqual(spec.root, explicit.resolve())

    def test_yolo_data_yaml_fallback_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            yaml_path = Path(td) / "data.yaml"
            yaml_path.write_text("path: .\n", "utf-8")
            env = {"DATASET_ROOT": "", "YOLO_DATA_YAML": str(yaml_path), "DATASET_NAME": ""}
            spec = resolve_dataset(environ=env)
            self.assertEqual(spec.root, yaml_path.parent.resolve())
            self.assertEqual(spec.yolo_data_yaml, yaml_path.resolve())

    def test_dataset_name_arg_selects_manifest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Follow repo convention: manifest at data/manifest.json so repo-root is mp.parent.parent.
            mp = base / "data" / "manifest.json"
            mp.parent.mkdir(parents=True, exist_ok=True)

            (base / "datasets" / "a").mkdir(parents=True, exist_ok=True)
            (base / "datasets" / "b").mkdir(parents=True, exist_ok=True)

            mp.write_text(
                '{"datasets":[{"name":"a","extracted_paths":["datasets/a"]},{"name":"b","extracted_paths":["datasets/b"]}]}',
                "utf-8",
            )

            spec = resolve_dataset(manifest_path=mp, dataset_name="b", environ={"DATASET_ROOT": "", "YOLO_DATA_YAML": "", "DATASET_NAME": ""})
            self.assertEqual(spec.root, (base / "datasets" / "b").resolve())
            self.assertEqual(spec.name, "b")


if __name__ == "__main__":
    unittest.main()

