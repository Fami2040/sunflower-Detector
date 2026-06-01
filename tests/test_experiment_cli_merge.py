"""Golden tests for harchoc.experiment_cli config merge helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


class ExperimentCliMergeTests(unittest.TestCase):
    def test_pick_cli_or_section_and_dataset(self) -> None:
        from harchoc.experiment_cli import (
            apply_dataset_args,
            pick_cli_or_section,
            pick_cli_or_dataset,
            section_and_dataset_from_config,
        )

        args = Namespace(out="cli_out.json", manifest="data/manifest.json")
        section = {"out": "cfg_out.json", "steps": 19}
        self.assertEqual(
            pick_cli_or_section(args, "out", section_cfg=section, default="default.json"),
            "cli_out.json",
        )
        args2 = Namespace(out="default.json")
        self.assertEqual(
            pick_cli_or_section(args2, "out", section_cfg=section, default="default.json"),
            "cfg_out.json",
        )
        self.assertEqual(
            pick_cli_or_dataset(
                Namespace(manifest="cli_only.json"),
                "manifest",
                dataset_cfg={"manifest": "from_cfg.json"},
                default="data/manifest.json",
            ),
            "cli_only.json",
        )
        self.assertEqual(
            pick_cli_or_dataset(
                Namespace(manifest="data/manifest.json"),
                "manifest",
                dataset_cfg={"manifest": "from_cfg.json"},
                default="data/manifest.json",
            ),
            "from_cfg.json",
        )
        self.assertEqual(
            pick_cli_or_dataset(
                Namespace(manifest="data/manifest.json"),
                "default_dataset_name",
                dataset_cfg={"default_dataset_name": "sunflower-cvat-1093"},
                default="sunflower-cvat-1093",
            ),
            "sunflower-cvat-1093",
        )

        config_obj = {
            "schema_version": "experiments.v1",
            "dataset": {"default_dataset_name": "sunflower-cvat-1093"},
            "run": {"kind": "threshold_sweep", "out": "reports/hsp/threshold_val.json"},
        }
        section_cfg, dataset_cfg = section_and_dataset_from_config(config_obj, "threshold_sweep")
        self.assertEqual(section_cfg.get("out"), "reports/hsp/threshold_val.json")
        apply_dataset_args(args2, dataset_cfg)
        self.assertEqual(args2.default_dataset_name, "sunflower-cvat-1093")

    def test_merge_config_objects_chain(self) -> None:
        from harchoc.experiment_cli import merge_config_objects

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base.json"
            overlay = Path(td) / "overlay.json"
            base.write_text(
                json.dumps(
                    {
                        "schema_version": "experiments.v1",
                        "run": {"kind": "error_analysis", "conf": 0.25},
                    }
                ),
                encoding="utf-8",
            )
            overlay.write_text(
                json.dumps({"run": {"conf": 0.15, "out": "reports/hsp/error_test_report.json"}}),
                encoding="utf-8",
            )
            merged = merge_config_objects([str(base), str(overlay)])
            run = merged.get("run")
            self.assertIsInstance(run, dict)
            self.assertEqual(run.get("conf"), 0.15)
            self.assertEqual(run.get("out"), "reports/hsp/error_test_report.json")


if __name__ == "__main__":
    unittest.main()
