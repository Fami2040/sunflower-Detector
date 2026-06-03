from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "finetune"


class FinetunePipelineTests(unittest.TestCase):
    def test_merge_finetune_eval_section_honors_harchoc_export_device_over_cpu_config(
        self,
    ) -> None:
        from harchoc.finetune_pipeline import merge_finetune_eval_section

        with patch.dict(os.environ, {"HARCHOC_EXPORT_DEVICE": "0"}, clear=False):
            out = merge_finetune_eval_section(
                {"device": "cpu", "export_device": "cpu"},
                locked_conf_from="reports/hsp/threshold_val.json",
                hsp_counting=True,
            )
        self.assertEqual(out["device"], "0")
        self.assertEqual(out["export_device"], "0")

    def test_tide_guidance_tray_adapt(self) -> None:
        from harchoc.finetune_pipeline import finetune_tide_guidance

        g = finetune_tide_guidance(_FIXTURES / "tide_bucket_summary.json", tray_key="tray-a")
        self.assertEqual(g["status"], "ok")
        self.assertEqual(g["recommended_train_mode"], "tray_adapt")

    def test_tide_guidance_miss_defers(self) -> None:
        from harchoc.finetune_pipeline import finetune_tide_guidance

        tide = {
            "buckets": {"Miss": 80, "Loc": 5, "Bkg": 5, "Cls": 5, "Dupe": 0},
            "n_errors": 95,
            "dominant_bucket": "Miss",
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tide.json"
            p.write_text(json.dumps(tide), encoding="utf-8")
            g = finetune_tide_guidance(p)
        self.assertEqual(g["recommended_train_mode"], "defer_finetune")

    def test_build_finetune_outcome_deltas(self) -> None:
        from harchoc.finetune_pipeline import build_finetune_outcome

        repo = Path(__file__).resolve().parents[1]
        before = {
            "tray-a": {"tray": {"count_mae": 100.0}},
            "test": {"count_mae": 61.0},
        }
        after = {
            "tray-a": {"tray": {"count_mae": 80.0}},
            "test": {"count_mae": 63.0},
        }
        out = build_finetune_outcome(
            tray_keys=["tray-a"],
            tray_eval_before=before,
            tray_eval_after=after,
            repo_root=repo,
            global_mae=61.3,
            gate_pct=0.10,
            weak_plan=json.loads((_FIXTURES / "weak_tray_plan.json").read_text(encoding="utf-8")),
            tide_guidance={"status": "ok"},
        )
        self.assertEqual(out["schema_version"], "finetune_outcome.v1")
        hold = out["tray_holdout"][0]
        self.assertEqual(hold["delta_mae"], -20.0)
        self.assertTrue(hold["improved"])
        self.assertTrue(out["canonical_gate"]["passed"])

    def test_hsp_export_argv_on_commands(self) -> None:
        from harchoc.finetune_tray_eval import build_tray_eval_commands

        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            cmds = build_tray_eval_commands(
                phase="before",
                weights="models/best2.pt",
                tray_keys=["abc"],
                reports_dir=reports,
                domains_dir=Path("data/domains"),
                splits_dir=Path("data/splits"),
                manifest="data/manifest.json",
                default_dataset_name="sunflower",
                dataset_name=None,
                dataset_root=None,
                yolo_data_yaml=None,
                eval_section={"device": "cpu", "hsp_counting": True},
                train_imgsz=1280,
                locked_conf_from="reports/hsp/threshold_val.json",
                hsp_counting=True,
            )
            tray = next(c for c in cmds if c.get("role") == "tray")
            argv = tray["argv"]
            self.assertIn("--export-gt-json", argv)
            self.assertIn("--export-preds-json", argv)
            self.assertIn("--locked-conf-from", argv)
            self.assertIn("hsp_before_tray_abc_gt.json", " ".join(argv))

    @patch("scripts.train.main", return_value=0)
    @patch("scripts.eval.main", return_value=0)
    @patch("harchoc.finetune_tray_eval._run_error_analysis", return_value=0)
    def test_live_run_records_outcome(
        self,
        mock_err: MagicMock,
        mock_eval: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        from scripts.finetune import main

        def _fake_eval(argv: list[str]) -> int:
            out_i = argv.index("--out") + 1
            out_p = Path(argv[out_i])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            gt_i = argv.index("--export-gt-json") + 1
            preds_i = argv.index("--export-preds-json") + 1
            Path(argv[gt_i]).write_text('{"images":[]}', encoding="utf-8")
            Path(argv[preds_i]).write_text('{"images":[]}', encoding="utf-8")
            out_p.write_text(
                json.dumps({"counting_metrics": {"mae": 50.0}, "mAP50": 0.3}),
                encoding="utf-8",
            )
            err_p = Path(argv[gt_i]).with_name(Path(argv[gt_i]).name.replace("_gt.json", "_error.json"))
            err_p.write_text(
                json.dumps({"counting_metrics": {"mae": 50.0}}),
                encoding="utf-8",
            )
            return 0

        mock_eval.side_effect = _fake_eval

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ds_root = td_path / "dataset"
            ds_root.mkdir()
            weak = td_path / "weak.json"
            weak.write_text((_FIXTURES / "weak_tray_plan.json").read_text(encoding="utf-8"), encoding="utf-8")
            (td_path / "domains").mkdir(parents=True)
            (td_path / "domains" / "test_tray-a.txt").write_text("images/test/x.jpg\n", encoding="utf-8")
            (td_path / "domains" / "val_tray-a.txt").write_text("images/val/x.jpg\n", encoding="utf-8")
            (td_path / "domains" / "train_tray-a.txt").write_text("images/train/x.jpg\n", encoding="utf-8")
            (td_path / "splits").mkdir()
            (td_path / "splits" / "test.txt").write_text("images/test/x.jpg\n", encoding="utf-8")
            out = td_path / "finetune.json"
            runs = td_path / "runs"
            (runs / "finetune_debug" / "weights").mkdir(parents=True)
            (runs / "finetune_debug" / "weights" / "best.pt").write_bytes(b"")
            rc = main(
                [
                    "--dataset-root",
                    str(ds_root),
                    "--domains-dir",
                    str(td_path / "domains"),
                    "--splits-dir",
                    str(td_path / "splits"),
                    "--from-weak-plan",
                    "--weak-plan",
                    str(weak),
                    "--tray-key",
                    "tray-a",
                    "--out",
                    str(out),
                    "--name",
                    "finetune_debug",
                    "--out-dir",
                    str(runs),
                    "--debug",
                    "--no-hsp-counting",
                ]
            )
            self.assertEqual(rc, 0, out.read_text(encoding="utf-8") if out.is_file() else "")
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("finetune_outcome", obj)
            self.assertEqual(obj.get("train_mode"), "tray_adapt")


class FinetuneGpuQueueTests(unittest.TestCase):
    def test_resolve_locked_conf_prefers_test_locked(self) -> None:
        from harchoc.finetune_pipeline import (
            THRESHOLD_TEST_LOCKED_PATH,
            resolve_finetune_locked_conf,
        )

        with tempfile.TemporaryDirectory() as td:
            rr = Path(td)
            (rr / "reports/hsp").mkdir(parents=True)
            locked = rr / THRESHOLD_TEST_LOCKED_PATH
            locked.parent.mkdir(parents=True, exist_ok=True)
            locked.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_finetune_locked_conf(rr), THRESHOLD_TEST_LOCKED_PATH)

    def test_build_finetune_queue_argv(self) -> None:
        from harchoc.finetune_pipeline import build_finetune_queue_argv

        repo = Path(__file__).resolve().parents[1]
        weak = repo / "tests/fixtures/finetune/weak_tray_plan.json"
        argv = build_finetune_queue_argv(
            {
                "id": "finetune_tray_demo",
                "tray_key": "349-10-2",
                "stage": 1,
                "weak_plan": str(weak),
                "debug": True,
            },
            repo_root=repo,
            dry_run=True,
        )
        self.assertIn("--dry-run", argv)
        self.assertIn("--tray-eval", argv)
        self.assertIn("--from-weak-plan", argv)
        self.assertIn("--hsp-counting", argv)
        self.assertIn("--debug", argv)
        self.assertIn("--config", argv)
        self.assertEqual(
            argv[argv.index("--config") + 1],
            "configs/experiments/finetune_tray_stage1.json",
        )
        self.assertIn("--transfer-config", argv)
        self.assertEqual(
            argv[argv.index("--transfer-config") + 1],
            "configs/transfer/finetune_stage1.yaml",
        )
        self.assertIn("--train-mode", argv)
        self.assertEqual(argv[argv.index("--train-mode") + 1], "tray_adapt")
        self.assertIn("--tide-summary", argv)
        tk = argv.index("--tray-key")
        self.assertEqual(argv[tk + 1], "349-10-2")
        out_i = argv.index("--out")
        self.assertIn("finetune_queue_finetune_tray_demo", argv[out_i + 1])

    def test_resolve_finetune_base_weights_stage2(self) -> None:
        from harchoc.finetune_pipeline import resolve_finetune_base_weights

        bw = resolve_finetune_base_weights(
            {"tray_key": "3a5-9", "stage1_run_name": "finetune_3a5-9_s1"},
            stage=2,
        )
        self.assertEqual(bw, "runs/transfer/finetune_3a5-9_s1/weights/best.pt")

    def test_resolve_finetune_base_weights_stage2_ignores_defaults_base(self) -> None:
        from harchoc.finetune_pipeline import resolve_finetune_base_weights

        bw = resolve_finetune_base_weights(
            {"tray_key": "3a5-9", "stage1_run_name": "finetune_3a5-9_s1", "stage": 2},
            stage=2,
            defaults={"base_weights": "models/best2.pt"},
        )
        self.assertEqual(bw, "runs/transfer/finetune_3a5-9_s1/weights/best.pt")

    def test_gpu_queue_build_finetune_tray_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "finetune_tray_s1",
            "kind": "finetune_tray",
            "tray_key": "tray-a",
            "stage": 1,
            "weak_plan": "tests/fixtures/finetune/weak_tray_plan.json",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertEqual(ids, ["dry_run", "gpu_wait", "finetune", "summary"])
        dry_argv = stages[0]["argv"]
        self.assertTrue(any("finetune.py" in str(a) for a in dry_argv))
        self.assertIn("--dry-run", dry_argv)
        self.assertIn("--config", dry_argv)
        self.assertTrue(any("finetune_tray_stage1.json" in str(a) for a in dry_argv))
        live_argv = next(s["argv"] for s in stages if s["stage_id"] == "finetune")
        self.assertNotIn("--dry-run", live_argv)
        summary = next(s for s in stages if s["stage_id"] == "summary")
        self.assertEqual(summary.get("internal"), "finetune_summary")

    def test_post_zoo_finetune_not_skipped_when_domains_plan_actionable(self) -> None:
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest, should_skip_job

        repo = Path(__file__).resolve().parents[1]
        fixture = repo / "tests/fixtures/finetune/weak_tray_plan.json"
        if not fixture.is_file():
            self.skipTest("weak_tray_plan fixture missing")
        manifest = load_gpu_queue_manifest(
            repo / "configs/experiments/gpu_queue_post_zoo.json", repo_root=repo
        )
        defaults = manifest.get("defaults") or {}
        fin_jobs = [j for j in manifest["jobs"] if j.get("kind") == "finetune_tray"]

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            plan_rel = "reports/domains/weak_tray_plan.json"
            plan = tmp / plan_rel
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

            for job in fin_jobs:
                skip, reason = should_skip_job(job, repo_root=tmp)
                self.assertFalse(skip, f"{job['id']}: {reason}")

            job = fin_jobs[0]
            stages = build_job_stages(job, repo_root=tmp, defaults=defaults)
            dry_argv = stages[0]["argv"]
            self.assertIn("--from-weak-plan", dry_argv)
            wp_i = dry_argv.index("--weak-plan")
            self.assertEqual(dry_argv[wp_i + 1], plan_rel)


if __name__ == "__main__":
    unittest.main()
