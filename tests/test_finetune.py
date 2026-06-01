from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class FinetuneTests(unittest.TestCase):
    def test_dry_run_writes_schema(self) -> None:
        from scripts.finetune import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            rc = main(["--dry-run", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(obj.get("schema_version"), "finetune_run.v1")
            self.assertEqual(obj.get("status"), "dry-run")
            policy = obj.get("transfer_policy", {})
            self.assertTrue(policy.get("ultralytics_freeze_honored"))
            self.assertEqual(policy.get("freeze"), 10)

    def test_dry_run_includes_tray_eval_plan(self) -> None:
        from scripts.finetune import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            rc = main(
                [
                    "--dry-run",
                    "--out",
                    str(out),
                    "--tray-key",
                    "349-10-2",
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            plan = obj.get("tray_eval_plan", {})
            self.assertTrue(plan.get("enabled"))
            before_cmds = plan.get("before", {}).get("commands", [])
            roles = {c["role"] for c in before_cmds}
            self.assertEqual(roles, {"tray", "val", "test"})
            self.assertTrue(all("argv" in c for c in before_cmds))
            self.assertIn("test", obj.get("tray_eval_before", {}))
            self.assertIn("349-10-2", obj.get("tray_eval_before", {}))
            after_paths = obj.get("tray_eval_after", {})
            self.assertIn("test", after_paths)

    def test_dry_run_no_tray_eval(self) -> None:
        from scripts.finetune import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            rc = main(["--dry-run", "--no-tray-eval", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(obj.get("tray_eval_plan", {}).get("enabled"))
            self.assertIsNone(obj.get("tray_eval_before"))

    @patch("scripts.eval.main", return_value=0)
    @patch("scripts.train.main", return_value=0)
    def test_train_invokes_train_main(self, mock_train: MagicMock, mock_eval: MagicMock) -> None:
        from scripts.finetune import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            rc = main(
                [
                    "--dataset-root",
                    str(Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted" / "dataset"),
                    "--out",
                    str(out),
                    "--name",
                    "finetune_test",
                    "--out-dir",
                    str(Path(td) / "runs"),
                    "--no-tray-eval",
                ]
            )
            if rc != 0 and not Path(
                Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted" / "dataset"
            ).is_dir():
                self.skipTest("dataset root missing")
            mock_train.assert_called_once()

    @patch("scripts.eval.main", return_value=0)
    @patch("scripts.train.main", return_value=0)
    def test_tray_eval_before_train(self, mock_train: MagicMock, mock_eval: MagicMock) -> None:
        from scripts.finetune import main

        dataset_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted" / "dataset"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            runs = Path(td) / "runs"
            (runs / "finetune_tray" / "weights").mkdir(parents=True)
            (runs / "finetune_tray" / "weights" / "best.pt").write_bytes(b"")
            rc = main(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--out",
                    str(out),
                    "--name",
                    "finetune_tray",
                    "--out-dir",
                    str(runs),
                    "--tray-key",
                    "349-10-2",
                ]
            )
            if rc != 0 and not dataset_root.is_dir():
                self.skipTest("dataset root missing")
            self.assertGreaterEqual(mock_eval.call_count, 1)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNotNone(obj.get("tray_eval_before"))

    @patch("scripts.train.main", return_value=0)
    def test_train_temp_config_includes_freeze(self, mock_train: MagicMock) -> None:
        from scripts.finetune import main

        captured: dict[str, object] = {}

        def _capture_train(argv: list[str]) -> int:
            cfg_idx = argv.index("--config")
            captured["train_doc"] = json.loads(
                Path(argv[cfg_idx + 1]).read_text(encoding="utf-8")
            )
            return 0

        mock_train.side_effect = _capture_train

        dataset_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted" / "dataset"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune.json"
            rc = main(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--out",
                    str(out),
                    "--name",
                    "finetune_freeze",
                    "--out-dir",
                    str(Path(td) / "runs"),
                    "--no-tray-eval",
                ]
            )
            if rc != 0 and not dataset_root.is_dir():
                self.skipTest("dataset root missing")
            train_doc = captured.get("train_doc")
            self.assertIsInstance(train_doc, dict)
            assert isinstance(train_doc, dict)
            self.assertTrue(train_doc.get("freeze_backbone"))
            self.assertEqual(train_doc.get("unfreeze_epoch"), 10)

    def test_stage2_dry_run_uses_stage_configs(self) -> None:
        from scripts.finetune import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "finetune_s2.json"
            rc = main(["--dry-run", "--stage", "2", "--out", str(out)])
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("finetune_tray_stage2.json", obj.get("config", ""))
            policy = obj.get("transfer_policy", {})
            self.assertEqual(policy.get("freeze"), 0)


if __name__ == "__main__":
    unittest.main()
