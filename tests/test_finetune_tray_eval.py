from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class FinetuneTrayEvalTests(unittest.TestCase):
    def test_build_commands_roles(self) -> None:
        from harchoc.finetune_tray_eval import build_tray_eval_commands

        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            cmds = build_tray_eval_commands(
                phase="before",
                weights="models/best2.pt",
                tray_keys=["abc-1"],
                reports_dir=reports,
                domains_dir=Path("data/domains"),
                splits_dir=Path("data/splits"),
                manifest="data/manifest.json",
                default_dataset_name="sunflower",
                dataset_name=None,
                dataset_root=None,
                yolo_data_yaml=None,
                eval_section={"device": "cpu"},
                train_imgsz=1280,
            )
            roles = {c["role"] for c in cmds}
            self.assertEqual(roles, {"tray", "val", "test"})
            tray_cmd = next(c for c in cmds if c["role"] == "tray")
            self.assertIn("data/domains/test_abc-1.txt", tray_cmd["split_file"])
            self.assertIn("--device", tray_cmd["argv"])
            self.assertIn("cpu", tray_cmd["argv"])


if __name__ == "__main__":
    unittest.main()
