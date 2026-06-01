import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class EvalConfigResolutionTests(unittest.TestCase):
    def test_missing_test_split_errors(self) -> None:
        from scripts.eval import main

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            os.chdir(td_path)
            root = td_path / "dataset"
            (root / "images" / "val").mkdir(parents=True, exist_ok=True)

            out = td_path / "eval.json"
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                with self.assertRaises(SystemExit) as ctx:
                    main(["--dry-run", "--out", str(out)])
            finally:
                os.chdir(old_cwd)
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old

            self.assertIn("Refusing to evaluate on a non-test split", str(ctx.exception))

    def test_weights_default_is_hsp_best2_when_env_unset(self) -> None:
        from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        old_weights = os.environ.pop("DETECTION_MODEL", None)
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td)
                (td_path / "data" / "splits").mkdir(parents=True, exist_ok=True)
                (td_path / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")
                root = td_path / "dataset"
                root.mkdir(parents=True, exist_ok=True)
                os.environ["DATASET_ROOT"] = str(root)
                out = td_path / "eval.json"
                rc = main(["--dry-run", "--out", str(out)])
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("weights"), HSP_DETECTION_WEIGHTS)
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root
            if old_weights is not None:
                os.environ["DETECTION_MODEL"] = old_weights

    def test_weights_default_uses_detection_model_env(self) -> None:
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        old_weights = os.environ.get("DETECTION_MODEL")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td)

                # Provide required test split.
                (td_path / "data" / "splits").mkdir(parents=True, exist_ok=True)
                (td_path / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")

                # Dataset root may be minimal for dry-run.
                root = td_path / "dataset"
                root.mkdir(parents=True, exist_ok=True)
                os.environ["DATASET_ROOT"] = str(root)
                os.environ["DETECTION_MODEL"] = str(td_path / "w.pt")

                out = td_path / "eval.json"
                rc = main(["--dry-run", "--out", str(out)])
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("weights"), str(td_path / "w.pt"))
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root
            if old_weights is None:
                os.environ.pop("DETECTION_MODEL", None)
            else:
                os.environ["DETECTION_MODEL"] = old_weights

    def test_minimal_yaml_parsing_and_label_nc_inference(self) -> None:
        from harchoc.yaml_minimal import parse_names_and_nc
        from scripts.eval import _infer_nc_from_labels

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.yaml"
            p.write_text(
                "\n".join(
                    [
                        "nc: 2",
                        "names:",
                        "  0: developed",
                        "  1: aborted",
                        "",
                    ]
                ),
                "utf-8",
            )
            names, nc = parse_names_and_nc(p)
            self.assertEqual(nc, 2)
            self.assertEqual(names, {0: "developed", 1: "aborted"})

            root = Path(td) / "dataset"
            (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", "utf-8")
            (root / "labels" / "val" / "b.txt").write_text("1 0.4 0.4 0.2 0.2\n", "utf-8")
            inferred = _infer_nc_from_labels(dataset_root=root, split_dir=(Path(td) / "val"))
            self.assertEqual(inferred, 2)

    def test_uses_repo_data_splits_test_txt_when_present(self) -> None:
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td)

                # Create tracked-style split file in the repo-like temp CWD.
                (td_path / "data" / "splits").mkdir(parents=True, exist_ok=True)
                (td_path / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")

                # Dataset root can be anywhere; split selection should not fall back to val.
                root = td_path / "dataset"
                (root / "images" / "val").mkdir(parents=True, exist_ok=True)
                os.environ["DATASET_ROOT"] = str(root)

                out = td_path / "eval.json"
                rc = main(["--dry-run", "--out", str(out)])
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("split_source", {}).get("kind"), "split_file")
                self.assertEqual(obj.get("split_source", {}).get("path"), "data/splits/test.txt")
                self.assertEqual(obj.get("eval_target", {}).get("split_role"), "test")
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root

    def test_imgsz_included_in_dry_run_payload(self) -> None:
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td)
                (td_path / "data" / "splits").mkdir(parents=True, exist_ok=True)
                (td_path / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")
                root = td_path / "dataset"
                root.mkdir(parents=True, exist_ok=True)
                os.environ["DATASET_ROOT"] = str(root)

                out = td_path / "eval.json"
                rc = main(["--dry-run", "--out", str(out), "--imgsz", "1280"])
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("imgsz"), 1280)
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root

    def test_max_det_included_in_dry_run_payload(self) -> None:
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td)
                (td_path / "data" / "splits").mkdir(parents=True, exist_ok=True)
                (td_path / "data" / "splits" / "test.txt").write_text("images/test/0001.jpg\n", "utf-8")
                root = td_path / "dataset"
                root.mkdir(parents=True, exist_ok=True)
                os.environ["DATASET_ROOT"] = str(root)

                out = td_path / "eval.json"
                rc = main(["--dry-run", "--out", str(out), "--max-det", "300"])
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("max_det"), 300)
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root

    def test_non_dry_run_out_path_before_eval_data_yaml(self) -> None:
        """Regression: finetune tray eval hit UnboundLocalError on out_path."""
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td_path)
                root = td_path / "dataset"
                (root / "images" / "test").mkdir(parents=True)
                (root / "labels" / "test").mkdir(parents=True)
                (root / "data.yaml").write_text(
                    "nc: 2\nnames:\n  0: developed\n  1: aborted\n",
                    encoding="utf-8",
                )
                split_file = td_path / "data" / "splits" / "tray.txt"
                split_file.parent.mkdir(parents=True, exist_ok=True)
                split_file.write_text("images/test/a.jpg\n", encoding="utf-8")
                weights = td_path / "model.pt"
                weights.write_bytes(b"stub")
                out = td_path / "reports" / "nested" / "eval.json"
                os.environ["DATASET_ROOT"] = str(root)
                with mock.patch(
                    "scripts.eval._write_eval_data_yaml",
                    return_value=td_path / "eval_data.yaml",
                ) as mock_yaml:
                    rc = main(
                        [
                            "--out",
                            str(out),
                            "--weights",
                            str(weights),
                            "--split-file",
                            str(split_file),
                            "--export-only",
                        ]
                    )
                self.assertEqual(rc, 0)
                mock_yaml.assert_called_once()
                self.assertEqual(mock_yaml.call_args.kwargs["out_dir"], out.parent)
                self.assertTrue(out.is_file())
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root

    def test_explicit_split_file_overrides_default(self) -> None:
        from scripts.eval import main

        old_cwd = os.getcwd()
        old_dataset_root = os.environ.get("DATASET_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                os.chdir(td)
                root = td_path / "dataset"
                root.mkdir(parents=True, exist_ok=True)
                os.environ["DATASET_ROOT"] = str(root)

                # Make a default file, but also pass an explicit one.
                (td_path / "data" / "splits").mkdir(parents=True, exist_ok=True)
                (td_path / "data" / "splits" / "test.txt").write_text("images/test/default.jpg\n", "utf-8")

                explicit = td_path / "my_split.txt"
                explicit.write_text("images/test/explicit.jpg\n", "utf-8")

                out = td_path / "eval.json"
                rc = main(["--dry-run", "--out", str(out), "--split-file", str(explicit)])
                self.assertEqual(rc, 0)
                obj = json.loads(out.read_text("utf-8"))
                self.assertEqual(obj.get("split_source", {}).get("kind"), "split_file")
                self.assertEqual(obj.get("split_source", {}).get("path"), str(explicit))
                self.assertEqual(obj.get("eval_target", {}).get("split_role"), "test")
        finally:
            os.chdir(old_cwd)
            if old_dataset_root is None:
                os.environ.pop("DATASET_ROOT", None)
            else:
                os.environ["DATASET_ROOT"] = old_dataset_root


if __name__ == "__main__":
    unittest.main()

