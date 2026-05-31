import os
import tempfile
import unittest
from pathlib import Path

from harchoc.data_yaml import (
    ensure_data_yaml,
    read_class_names,
    require_data_yaml_path,
    resolve_data_yaml_path,
)


class DataYamlTests(unittest.TestCase):
    def test_resolve_explicit_then_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            explicit = root / "custom.yaml"
            explicit.write_text("path: .\n", "utf-8")
            (root / "data.yaml").write_text("path: .\n", "utf-8")

            self.assertEqual(
                resolve_data_yaml_path(dataset_root=root, explicit_yaml=explicit, use_env=False),
                explicit.resolve(),
            )
            self.assertEqual(
                resolve_data_yaml_path(dataset_root=root, explicit_yaml=None, use_env=False),
                (root / "data.yaml").resolve(),
            )

    def test_resolve_yolo_data_yaml_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_yaml = Path(td) / "from_env.yaml"
            env_yaml.write_text("path: .\n", "utf-8")
            prev = os.environ.pop("YOLO_DATA_YAML", None)
            try:
                self.assertIsNone(
                    resolve_data_yaml_path(
                        dataset_root=Path(td) / "missing_root",
                        use_env=True,
                        explicit_yaml=None,
                    )
                )
                os.environ["YOLO_DATA_YAML"] = str(env_yaml)
                found = resolve_data_yaml_path(
                    dataset_root=Path(td) / "missing_root", use_env=True
                )
                self.assertEqual(found, env_yaml.resolve())
                with self.assertRaises(FileNotFoundError):
                    os.environ["YOLO_DATA_YAML"] = str(Path(td) / "nope.yaml")
                    resolve_data_yaml_path(
                        dataset_root=Path(td) / "missing_root", use_env=True
                    )
            finally:
                if prev is None:
                    os.environ.pop("YOLO_DATA_YAML", None)
                else:
                    os.environ["YOLO_DATA_YAML"] = prev

    def test_require_honors_explicit_when_missing_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            explicit = root / "configured.yaml"
            got = require_data_yaml_path(dataset_root=root, explicit_yaml=explicit)
            self.assertEqual(got, explicit.resolve())

    def test_ensure_generates_minimal_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)
            labels = root / "labels" / "train"
            labels.mkdir(parents=True)
            (labels / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", "utf-8")

            path = ensure_data_yaml(dataset_root=root, yolo_data_yaml=None)
            self.assertTrue(Path(path).is_file())
            text = Path(path).read_text("utf-8")
            self.assertIn("nc: 2", text)
            self.assertIn("images/train", text)

    def test_read_class_names_from_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                "path: .\nnc: 2\nnames:\n  0: seed\n  1: hull\n",
                "utf-8",
            )
            self.assertEqual(read_class_names(dataset_root=root), ["seed", "hull"])

    def test_read_class_names_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(read_class_names(dataset_root=Path(td)), ["developed", "aborted"])


if __name__ == "__main__":
    unittest.main()
