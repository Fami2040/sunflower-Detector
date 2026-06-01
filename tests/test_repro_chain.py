"""Tests for harchoc.repro_chain (CI-safe)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harchoc.repro_chain import (
    format_repro_cmd,
    hsp_test_exports_present,
    load_json_bundle,
    overall_step_status,
    reproduce_command_block,
    run_argv_chain,
    step_record,
)


class TestReproChain(unittest.TestCase):
    def test_load_json_bundle_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "b.json"
            p.write_text('{"schema_version": "test.v1", "x": 1}', encoding="utf-8")
            obj = load_json_bundle(p, schema_version="test.v1")
            self.assertEqual(obj["x"], 1)
            with self.assertRaises(ValueError):
                load_json_bundle(p, schema_version="other.v1")

    def test_hsp_exports_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports/hsp").mkdir(parents=True)
            (repo / "reports/hsp/gt_test.json").write_text("{}", encoding="utf-8")
            self.assertFalse(
                hsp_test_exports_present(repo, {"gt_test": "reports/hsp/gt_test.json", "preds_test": "reports/hsp/preds_test.json"})
            )
            (repo / "reports/hsp/preds_test.json").write_text("{}", encoding="utf-8")
            self.assertTrue(
                hsp_test_exports_present(repo, {"gt_test": "reports/hsp/gt_test.json", "preds_test": "reports/hsp/preds_test.json"})
            )

    def test_run_argv_chain_dry_run(self) -> None:
        runner = mock.Mock(return_value=0)
        rc = run_argv_chain(
            [("a", ["scripts/foo.py"]), ("b", ["scripts/bar.py"])],
            repo_root=".",
            dry_run=True,
            run_argv=runner,
        )
        self.assertEqual(rc, 0)
        runner.assert_not_called()

    def test_run_argv_chain_returns_first_failure(self) -> None:
        runner = mock.Mock(side_effect=[0, 3])
        rc = run_argv_chain(
            [("a", ["scripts/foo.py"]), ("b", ["scripts/bar.py"])],
            repo_root=".",
            dry_run=False,
            run_argv=runner,
            fail_label="test",
        )
        self.assertEqual(rc, 3)
        self.assertEqual(runner.call_count, 2)

    def test_overall_step_status(self) -> None:
        steps = {
            "a": step_record(status="ok"),
            "b": step_record(status="skipped"),
        }
        self.assertEqual(overall_step_status(steps), "partial")

    def test_reproduce_command_block(self) -> None:
        block = reproduce_command_block(ci_dry_run=["dry"], local=["run"])
        self.assertEqual(block["ci_safe_dry_run"], ["dry"])
        self.assertEqual(block["local"], ["run"])

    def test_format_repro_cmd_no_mamba(self) -> None:
        s = format_repro_cmd(["scripts/experiment.py", "repro"], mamba=False)
        self.assertIn("experiment.py", s)


if __name__ == "__main__":
    unittest.main()
