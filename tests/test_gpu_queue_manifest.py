"""Tests for harchoc.gpu_queue (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._gpu_queue_fixtures import load_manifest_with_index, write_pending_fixture_index


class GpuQueueManifestTests(unittest.TestCase):
    def test_load_full_manifest_expands_aug_smokes_from_index(self) -> None:
        from harchoc.gpu_queue import expand_aug_smoke_jobs_from_index, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        full = repo / "configs/experiments/archive/gpu_queue_full.json"
        m = load_gpu_queue_manifest(full, repo_root=repo)
        self.assertTrue(m.get("aug_smoke_from_index"))
        ids = {j.get("id") for j in m["jobs"]}
        self.assertNotIn("aug_smoke_S4", ids)
        self.assertNotIn("aug_smoke_S14", ids)
        self.assertNotIn("aug_smoke_S0", ids)
        self.assertEqual(expand_aug_smoke_jobs_from_index(repo_root=repo), [])

        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = write_pending_fixture_index(repo, Path(td), ("S4", "S14"))
            pending = expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
            pending_ids = {j.get("smoke_id") for j in pending}
            self.assertIn("S4", pending_ids)
            self.assertIn("S14", pending_ids)
            self.assertNotIn("S0", pending_ids)
            m_fix = load_manifest_with_index(
                repo, template=full, index_rel=index_rel, tmp_dir=Path(td)
            )
            fix_ids = {j.get("id") for j in m_fix["jobs"]}
            self.assertIn("aug_smoke_S4", fix_ids)
            self.assertIn("aug_smoke_S14", fix_ids)

    def test_load_full_manifest(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_full.json")
        self.assertEqual(m["schema_version"], "gpu_queue_manifest.v1")
        self.assertGreaterEqual(len(m["jobs"]), 14)
        by_id = {j["id"]: j for j in m["jobs"]}
        amp_eval = by_id["amp_smoke_15ep_on_hsp_eval"]
        self.assertTrue(amp_eval.get("eval_only"))
        self.assertEqual(amp_eval["backlog"], ["P1-AMP-HSP-EVAL"])
        self.assertEqual(
            amp_eval["summary_path"], "reports/hsp/amp_on_smoke_15ep_summary.json"
        )
        sg_eval = by_id["sg_yolo_nas_s_hsp_eval"]
        self.assertTrue(sg_eval.get("eval_only"))
        self.assertEqual(sg_eval["backlog"], ["P1-SG-HSP-EVAL"])
        self.assertTrue(by_id["amp_smoke_15ep_on"].get("skip_eval"))
        self.assertTrue(by_id["sg_yolo_nas_s_smoke"].get("skip_eval"))

    def test_aug_smoke_index_queue_parity(self) -> None:
        from harchoc.aug_smoke_runner import aug_smoke_index_queue_parity_errors

        repo = Path(__file__).resolve().parents[1]
        errors = aug_smoke_index_queue_parity_errors(repo_root=repo)
        self.assertEqual(errors, [], msg="; ".join(errors))

    def test_aug_sweep_15_aug_config_wiring(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_full.json")
        by_id = {j["id"]: j for j in m["jobs"]}
        self.assertNotIn("aug_sweep_15_mosaic0", by_id)
        self.assertEqual(
            by_id["aug_sweep_15_close10"]["aug_config"],
            "configs/aug/robustness_smoke_close10.yaml",
        )
        self.assertEqual(
            by_id["aug_sweep_15_close10"]["train_config"],
            "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json",
        )
        self.assertEqual(
            by_id["aug_sweep_15_close15"]["aug_config"],
            "configs/aug/robustness_smoke_close15.yaml",
        )
        self.assertEqual(
            by_id["aug_sweep_15_close25"]["aug_config"],
            "configs/aug/robustness_smoke_close25.yaml",
        )
        self.assertEqual(
            by_id["aug_sweep_15_close25"]["train_config"],
            "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json",
        )

    def test_sweeps_15ep_index_matches_full_queue(self) -> None:
        from harchoc.aug_smoke_runner import load_aug_smoke_index
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")
        sweeps = index.get("sweeps_15ep") or {}
        arms = {str(a["id"]): a for a in (sweeps.get("arms") or [])}
        self.assertEqual(set(arms), {"close10", "close15", "close25"})

        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_full.json")
        by_id = {j["id"]: j for j in m["jobs"]}
        for arm_id, arm in arms.items():
            qid = str(arm.get("queue_job_id") or "")
            self.assertIn(qid, by_id, msg=arm_id)
            job = by_id[qid]
            self.assertEqual(job.get("train_config"), arm.get("train_config"))
            self.assertEqual(job.get("aug_config"), arm.get("aug_config"))
            self.assertEqual(job.get("run_name"), arm.get("run_name"))
        close15 = by_id["aug_sweep_15_close15"]
        self.assertTrue(close15.get("skip"))
        self.assertIn("recipe duplicate", close15.get("skip_reason") or "")

    def test_load_aug_pending_manifest_expands_gpu_pending(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        aug_pending = repo / "configs/experiments/archive/gpu_queue_aug_pending.json"
        m = load_gpu_queue_manifest(aug_pending, repo_root=repo)
        ids = {j.get("id") for j in m["jobs"]}
        self.assertIn("preflight", ids)
        self.assertNotIn("aug_smoke_S0", ids)
        self.assertEqual(ids, {"preflight"})

        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = write_pending_fixture_index(repo, Path(td), ("S4", "S14"))
            m_fix = load_manifest_with_index(
                repo, template=aug_pending, index_rel=index_rel, tmp_dir=Path(td)
            )
            fix_ids = {j.get("id") for j in m_fix["jobs"]}
            self.assertIn("preflight", fix_ids)
            self.assertIn("aug_smoke_S4", fix_ids)
            self.assertIn("aug_smoke_S14", fix_ids)
            self.assertNotIn("aug_smoke_S0", fix_ids)
            s14 = next(j for j in m_fix["jobs"] if j.get("id") == "aug_smoke_S14")
            self.assertTrue(s14.get("eval_only"))
            self.assertNotEqual(s14.get("skip"), True)

    def test_amp_smoke_recipe_fingerprint_differs_from_s1(self) -> None:
        """amp_smoke_15ep_on is not recipe-equivalent to complete S1 (distinct train path)."""
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.gpu_queue import _job_train_recipe_fingerprint
        from harchoc.train_config import effective_train_recipe_fingerprint

        repo = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")
        fp_s1 = effective_train_recipe_fingerprint(
            resolve_aug_smoke_train_raw(find_smoke_entry(index, "S1"), repo_root=repo),
            repo_root=repo,
        )
        amp_job = {
            "id": "amp_smoke_15ep_on",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
        }
        fp_amp = _job_train_recipe_fingerprint(amp_job, repo_root=repo)
        self.assertIsNotNone(fp_amp)
        self.assertNotEqual(fp_amp, fp_s1)

    def test_amp_smoke_not_recipe_deduped_against_s1(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_full.json", repo_root=repo)
        amp = next(j for j in m["jobs"] if j.get("id") == "amp_smoke_15ep_on")
        self.assertNotEqual(amp.get("skip"), True)
        reason = str(amp.get("skip_reason") or "")
        self.assertNotIn("recipe duplicate of complete smoke S1", reason)

    def test_s14_eval_only_not_recipe_deduped_as_s1(self) -> None:
        from harchoc.gpu_queue import _job_train_recipe_fingerprint, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        full = repo / "configs/experiments/archive/gpu_queue_full.json"
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = write_pending_fixture_index(repo, Path(td), ("S14",))
            m = load_manifest_with_index(
                repo, template=full, index_rel=index_rel, tmp_dir=Path(td)
            )
            s14 = next(j for j in m["jobs"] if j.get("id") == "aug_smoke_S14")
            self.assertTrue(s14.get("eval_only"))
            self.assertIsNone(_job_train_recipe_fingerprint(s14, repo_root=repo))
            self.assertNotEqual(s14.get("skip"), True)

    def test_load_aug_confirm_manifest(self) -> None:
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_aug_confirm.json")
        self.assertEqual(len(m["jobs"]), 1)
        job = m["jobs"][0]
        self.assertEqual(job["id"], "aug_confirm_winner_100ep")
        self.assertEqual(job["kind"], "aug_sweep_100")
        self.assertEqual(
            job["train_config"], "configs/experiments/train_aug_winner_100ep.json"
        )
        self.assertEqual(job["run_name"], "aug_confirm_winner_100ep")
        self.assertEqual(
            job["summary_path"],
            "reports/aug_smoke/aug_confirm_winner_100ep_summary.json",
        )
        self.assertEqual(job["env"]["HARCHOC_MAX_EPOCHS"], "100")
        self.assertEqual(
            m["defaults"]["locked_conf_from"], "reports/hsp/threshold_val.json"
        )
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        eval_meta = next(s for s in stages if s["stage_id"] == "eval_test")["meta"]
        self.assertEqual(eval_meta["run_name"], "aug_confirm_winner_100ep")
        self.assertEqual(eval_meta["out_dir"], "reports/aug_smoke")
        summary_meta = next(s for s in stages if s["stage_id"] == "summary")["meta"]
        self.assertEqual(
            summary_meta["summary_path"],
            "reports/aug_smoke/aug_confirm_winner_100ep_summary.json",
        )

    def test_load_aug_close_phase_a_manifest(self) -> None:
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_aug_close_phase_a.json")
        by_id = {j["id"]: j for j in m["jobs"]}
        self.assertIn("preflight", by_id)
        for jid in ("aug_sweep_15_close10", "aug_sweep_15_close25"):
            job = by_id[jid]
            self.assertEqual(job["kind"], "aug_sweep_15")
            self.assertIn("aug_config", job)
            self.assertIn("skip_if", job)
            stages = build_job_stages(job, repo_root=repo)
            ids = [s["stage_id"] for s in stages]
            self.assertEqual(ids, ["dry_run", "gpu_wait", "train", "eval_test", "summary"])
            train_argv = next(s for s in stages if s["stage_id"] == "train")["argv"]
            self.assertIn("--aug-config", train_argv)

    def test_load_aug_close_100ep_manifest(self) -> None:
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_aug_close_100ep.json")
        by_id = {j["id"]: j for j in m["jobs"]}
        for jid, cfg, run_name in (
            (
                "aug_sweep_100_close10",
                "configs/experiments/train_aug_close10_100ep.json",
                "aug_sweep_close10_100ep",
            ),
            (
                "aug_sweep_100_close25",
                "configs/experiments/train_aug_close25_100ep.json",
                "aug_sweep_close25_100ep",
            ),
            (
                "aug_schedule_patience25_100ep",
                "configs/experiments/train_aug_schedule_patience25_100ep.json",
                "aug_schedule_patience25_100ep",
            ),
        ):
            job = by_id[jid]
            self.assertEqual(job["kind"], "aug_sweep_100")
            self.assertEqual(job["train_config"], cfg)
            self.assertEqual(job["run_name"], run_name)
            self.assertTrue(job.get("skip"), msg=jid)
            self.assertIn("skip_if", job)
            stages = build_job_stages(job, repo_root=repo)
            self.assertIn("eval_test", [s["stage_id"] for s in stages])

    def test_aug_close_100ep_train_configs_validate_schedule(self) -> None:
        from harchoc.train_config import load_train_config_json, validate_epochs_patience_close_mosaic

        repo = Path(__file__).resolve().parents[1]
        for name in (
            "train_aug_close10_100ep.json",
            "train_aug_close25_100ep.json",
            "train_aug_schedule_patience25_100ep.json",
        ):
            cfg = load_train_config_json(repo / "configs/experiments" / name, repo_root=repo)
            validate_epochs_patience_close_mosaic(cfg, repo_root=repo, label=name)

        close10 = load_train_config_json(
            repo / "configs/experiments/train_aug_close10_100ep.json", repo_root=repo
        )
        close25 = load_train_config_json(
            repo / "configs/experiments/train_aug_close25_100ep.json", repo_root=repo
        )
        self.assertEqual(close10.get("epochs"), 100)
        self.assertEqual(close25.get("epochs"), 100)
        self.assertEqual(close10.get("patience"), 30)
        sched = load_train_config_json(
            repo / "configs/experiments/train_aug_schedule_patience25_100ep.json",
            repo_root=repo,
        )
        self.assertEqual(sched.get("patience"), 25)

    def test_full_manifest_aug_confirm_winner_skipped(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(
            repo / "configs/experiments/archive/gpu_queue_full.json", repo_root=repo
        )
        job = next(j for j in m["jobs"] if j.get("id") == "aug_confirm_winner_100ep")
        self.assertTrue(job.get("skip"))
        self.assertIn("gpu_queue_aug_confirm", str(job.get("skip_reason") or ""))

    def test_post_zoo_manifest_finetune_and_domain_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(
            repo / "configs/experiments/gpu_queue_post_zoo.json", repo_root=repo
        )
        self.assertEqual(
            m["defaults"]["weak_plan"], "reports/domains/weak_tray_plan.json"
        )
        by_id = {j["id"]: j for j in m["jobs"]}
        domain = by_id["domain_tray_audit_refresh"]
        self.assertEqual(domain["weak_plan_out"], "reports/domains/weak_tray_plan.json")
        self.assertTrue(domain.get("merge_mamba"))
        dom_stages = [s["stage_id"] for s in build_job_stages(domain, repo_root=repo)]
        self.assertEqual(
            dom_stages,
            [
                "write_domain_splits",
                "gpu_wait",
                "merge_tray_count_mae",
                "domain_tray_audit",
            ],
        )
        merge_stage = next(
            s for s in build_job_stages(domain, repo_root=repo) if s["stage_id"] == "merge_tray_count_mae"
        )
        self.assertTrue(merge_stage.get("mamba"))
        self.assertIn("0", merge_stage["argv"])
        fin = by_id["finetune_weak_tray_1"]
        self.assertEqual(fin.get("device"), "0")
        self.assertTrue(fin.get("mamba"))
        self.assertEqual(
            fin["train_config"], "configs/experiments/finetune_tray_stage1.json"
        )
        hp = fin.get("hyperparams") or {}
        self.assertEqual(hp.get("epochs"), 25)
        self.assertEqual(hp.get("freeze_backbone"), True)
        fin_argv = build_job_stages(fin, repo_root=repo)[0]["argv"]
        self.assertTrue(any("finetune_tray_stage1.json" in str(a) for a in fin_argv))
        zoo = load_gpu_queue_manifest(
            repo / "configs/experiments/archive/gpu_queue_full.json", repo_root=repo
        )
        zjob = next(j for j in zoo["jobs"] if j.get("id") == "zoo_matrix_p0_5")
        mh = zjob.get("matrix_hyperparams") or {}
        self.assertEqual(mh.get("epochs"), 100)
        self.assertEqual(mh.get("imgsz"), 1280)
        self.assertEqual(len(zjob.get("bench_configs") or []), 4)

    def test_full_manifest_gpu_execution_tier_order(self) -> None:
        """Tier 1 RT-DETR block → Tier 2 eval/sweeps → P0-5 zoo_matrix before cv_fold (backlog § GPU execution tiers)."""
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(
            repo / "configs/experiments/archive/gpu_queue_full.json", repo_root=repo
        )
        ids = [j["id"] for j in m["jobs"]]

        def idx(job_id: str) -> int:
            self.assertIn(job_id, ids, msg=f"missing job {job_id}")
            return ids.index(job_id)

        tier1_rtdetr = (
            "vram_probe_rtdetr",
            "rtdetr_queries_smoke",
            "rtdetr_imgsz640",
            "rtdetr_imgsz1280",
        )
        tier2_eval = ("amp_smoke_15ep_on_hsp_eval", "sg_yolo_nas_s_hsp_eval")
        tier2_close = ("aug_sweep_15_close10", "aug_sweep_15_close25")

        for i in range(len(tier1_rtdetr) - 1):
            self.assertLess(idx(tier1_rtdetr[i]), idx(tier1_rtdetr[i + 1]))

        last_rtdetr = idx(tier1_rtdetr[-1])
        for job_id in tier2_eval + tier2_close:
            self.assertGreater(idx(job_id), last_rtdetr, msg=f"{job_id} must follow RT-DETR block")

        self.assertGreater(idx("zoo_matrix_p0_5"), idx(tier2_close[-1]))
        self.assertGreater(idx("cv_fold_train"), idx("zoo_matrix_p0_5"))



if __name__ == "__main__":
    unittest.main()
