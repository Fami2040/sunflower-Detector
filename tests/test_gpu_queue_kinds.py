"""GPU queue kinds: domain audit refresh, finetune skip gates, head ROI smoke (CI-safe)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class GpuQueuePostZooKindsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]

    def test_domain_tray_audit_refresh_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        job = {
            "id": "domain_tray_audit_refresh",
            "kind": "domain_tray_audit_refresh",
            "mamba": False,
            "catalog": "reports/domains/catalog.json",
            "domains_dir": "data/domains",
            "domain_eval": "reports/domains/domain_eval.json",
            "count_mae": "reports/domains/domain_count_mae.json",
            "weak_plan_out": "reports/domains/weak_tray_plan.json",
            "top_k": 3,
            "global_mae": 61.3,
            "device": "0",
        }
        stages = build_job_stages(
            job,
            repo_root=self.repo,
            defaults={"locked_conf_from": "reports/hsp/threshold_val.json"},
        )
        ids = [s["stage_id"] for s in stages]
        self.assertEqual(
            ids,
            [
                "write_domain_splits",
                "gpu_wait",
                "merge_tray_count_mae",
                "domain_tray_audit",
            ],
        )
        by_stage = {s["stage_id"]: s for s in stages}
        write_argv = by_stage["write_domain_splits"]["argv"]
        self.assertTrue(any("eval_domains.py" in str(a) for a in write_argv))
        self.assertIn("--write-domain-splits", write_argv)
        merge_argv = by_stage["merge_tray_count_mae"]["argv"]
        self.assertIn("--merge-tray-count-mae", merge_argv)
        self.assertIn("reports/hsp/threshold_val.json", merge_argv)
        audit_argv = by_stage["domain_tray_audit"]["argv"]
        self.assertTrue(any("experiment.py" in str(a) for a in audit_argv))
        self.assertIn("domain-tray-audit", audit_argv)
        self.assertIn("reports/domains/weak_tray_plan.json", audit_argv)
        self.assertFalse(stages[0].get("mamba"))

    def test_head_roi_eval_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        job = {
            "id": "head_roi_eval_smoke",
            "kind": "head_roi_eval",
            "eval_only": True,
            "mamba": True,
            "out": "reports/hsp/head_roi_eval_smoke.json",
            "device": "0",
            "weights": "models/best2.pt",
        }
        stages = build_job_stages(
            job,
            repo_root=self.repo,
            defaults={"locked_conf_from": "reports/hsp/threshold_val.json"},
        )
        self.assertEqual([s["stage_id"] for s in stages], ["gpu_wait", "eval"])
        argv = next(s["argv"] for s in stages if s["stage_id"] == "eval")
        self.assertTrue(any("head_roi_eval.py" in str(a) for a in argv))
        self.assertNotIn("--dry-run", argv)
        self.assertIn("--locked-conf-from", argv)
        self.assertIn("reports/hsp/threshold_val.json", argv)
        self.assertIn("reports/hsp/head_roi_eval_smoke.json", argv)
        self.assertIn("--device", argv)
        self.assertIn("0", argv)

    def test_head_roi_mask_removes_outside_center(self) -> None:
        from harchoc.head_roi_eval import apply_head_roi_mask_to_preds

        gt = {
            "images": [
                {
                    "image_id": "img1",
                    "width": 1000,
                    "height": 1000,
                    "annotations": [
                        {"bbox": [100, 100, 200, 200], "category_id": 0},
                        {"bbox": [300, 300, 400, 400], "category_id": 1},
                    ],
                }
            ]
        }
        preds = {
            "images": [
                {
                    "image_id": "img1",
                    "detections": [
                        {"bbox": [150, 150, 160, 160], "category_id": 0, "score": 0.9},
                        {"bbox": [900, 900, 910, 910], "category_id": 0, "score": 0.8},
                    ],
                }
            ]
        }
        masked, stats = apply_head_roi_mask_to_preds(preds, gt, margin_frac=0.02)
        dets = masked["images"][0]["detections"]
        self.assertEqual(len(dets), 1)
        self.assertEqual(stats["n_preds_removed"], 1)

    def test_shell_kind_argv(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        job = {
            "id": "shell_smoke",
            "kind": "shell",
            "eval_only": True,
            "argv": ["scripts/eval.py", "--dry-run"],
        }
        stages = build_job_stages(job, repo_root=self.repo)
        self.assertEqual([s["stage_id"] for s in stages], ["run"])
        self.assertIn("--dry-run", stages[0]["argv"])

    def test_skip_if_missing_plan_empty(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            plan = repo / "reports/domains/weak_tray_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text(
                json.dumps({"recommended_tray_keys": [], "status": "empty"}),
                encoding="utf-8",
            )
            job = {
                "id": "finetune_x",
                "kind": "finetune_tray",
                "tray_key": "3a5-9",
                "skip_if_missing_plan": True,
                "weak_plan": "reports/domains/weak_tray_plan.json",
            }
            skip, reason = should_skip_job(job, repo_root=repo)
            self.assertTrue(skip)
            self.assertIn("missing or empty", reason)

    def test_skip_if_missing_plan_present(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        fixture = self.repo / "tests/fixtures/finetune/weak_tray_plan.json"
        if not fixture.is_file():
            self.skipTest("weak_tray_plan fixture missing")
        job = {
            "id": "finetune_x",
            "kind": "finetune_tray",
            "tray_key": "3a5-9",
            "skip_if_missing_plan": True,
            "weak_plan": str(fixture.relative_to(self.repo)),
        }
        skip, reason = should_skip_job(job, repo_root=self.repo)
        self.assertFalse(skip)
        self.assertEqual(reason, "")

    def test_skip_if_weak_plan_present(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        fixture = self.repo / "tests/fixtures/finetune/weak_tray_plan.json"
        if not fixture.is_file():
            self.skipTest("weak_tray_plan fixture missing")
        job = {
            "id": "domain_tray_audit_refresh",
            "kind": "domain_tray_audit_refresh",
            "skip_if": {"weak_plan": str(fixture.relative_to(self.repo))},
        }
        skip, reason = should_skip_job(job, repo_root=self.repo)
        self.assertTrue(skip)
        self.assertIn("weak_tray_plan present", reason)


if __name__ == "__main__":
    unittest.main()
