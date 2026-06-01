"""CI tests for zoo_core RT-DETR 15-ep smoke gate (no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class RtdetrZooGateTests(unittest.TestCase):
    def test_gates_fail_without_summaries(self) -> None:
        from harchoc.rtdetr_zoo_gate import check_zoo_core_rtdetr_15ep_gates, zoo_core_rtdetr_gates_passed

        repo = Path(__file__).resolve().parents[1]
        statuses = check_zoo_core_rtdetr_15ep_gates(repo_root=repo)
        self.assertEqual(len(statuses), 2)
        self.assertFalse(zoo_core_rtdetr_gates_passed(repo_root=repo))
        self.assertFalse(any(s["passed"] for s in statuses))

    def test_gate_passes_with_verified_summary_and_error_json(self) -> None:
        from harchoc.rtdetr_zoo_gate import ZOO_CORE_RTDETR_15EP_GATES, rtdetr_15ep_gate_status

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            gate = ZOO_CORE_RTDETR_15EP_GATES[0]
            summary = repo / gate["summary"]
            summary.parent.mkdir(parents=True, exist_ok=True)
            err = repo / gate["eval_error_json"]
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(
                json.dumps({"counting_metrics": {"mae": 88.0, "mae_ci": {"point": 88.0}}}),
                encoding="utf-8",
            )
            summary.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "test_count_mae": 88.0,
                        "test_eval": {"error_json": str(err.relative_to(repo))},
                    }
                ),
                encoding="utf-8",
            )
            st = rtdetr_15ep_gate_status(repo_root=repo, gate=gate)
            self.assertTrue(st["passed"])

    def test_bench_matrix_row_id_strips_default_suffix(self) -> None:
        from harchoc.rtdetr_zoo_gate import bench_matrix_row_id

        self.assertEqual(
            bench_matrix_row_id(bench_path=Path("configs/bench/rtdetr_x_default.yaml")),
            "rtdetr_x",
        )
        self.assertEqual(
            bench_matrix_row_id(bench_path=Path("configs/bench/rtdetr_l_nq1024.yaml")),
            "rtdetr_l_nq1024",
        )
