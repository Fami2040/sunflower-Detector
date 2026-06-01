"""Tests for harchoc.gpu_queue (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class GpuQueueManifestTests(unittest.TestCase):
    def _write_pending_fixture_index(
        self,
        repo: Path,
        tmp_dir: Path,
        pending_ids: tuple[str, ...],
        *,
        include_equivalence_classes: bool = True,
    ) -> str:
        """Clone production index with *pending_ids* as gpu_pending; return repo-relative path."""
        from harchoc.aug_smoke_runner import load_aug_smoke_index

        prod = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")
        pending_set = {s.upper() for s in pending_ids}
        smokes: list[dict] = []
        for entry in prod.get("smokes") or []:
            e = dict(entry)
            sid = str(e.get("id") or "").upper()
            if sid in pending_set:
                e["status"] = "gpu_pending"
            smokes.append(e)
        obj: dict = {"schema_version": "aug_smoke_index.v1", "smokes": smokes}
        if include_equivalence_classes and prod.get("equivalence_classes"):
            obj["equivalence_classes"] = prod["equivalence_classes"]
        path = tmp_dir / "aug_smoke_index_fixture.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        return str(path.relative_to(repo))

    def _load_manifest_with_index(
        self,
        repo: Path,
        *,
        template: Path,
        index_rel: str,
        tmp_dir: Path,
    ) -> dict:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        base = json.loads(template.read_text(encoding="utf-8"))
        base["aug_smoke_index"] = index_rel
        manifest = tmp_dir / "manifest.json"
        manifest.write_text(json.dumps(base), encoding="utf-8")
        return load_gpu_queue_manifest(manifest, repo_root=repo)

    def test_load_full_manifest_expands_aug_smokes_from_index(self) -> None:
        from harchoc.gpu_queue import expand_aug_smoke_jobs_from_index, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        full = repo / "configs/experiments/gpu_queue_full.json"
        m = load_gpu_queue_manifest(full, repo_root=repo)
        self.assertTrue(m.get("aug_smoke_from_index"))
        ids = {j.get("id") for j in m["jobs"]}
        self.assertNotIn("aug_smoke_S4", ids)
        self.assertNotIn("aug_smoke_S14", ids)
        self.assertNotIn("aug_smoke_S0", ids)
        self.assertEqual(expand_aug_smoke_jobs_from_index(repo_root=repo), [])

        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(repo, Path(td), ("S4", "S14"))
            pending = expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
            pending_ids = {j.get("smoke_id") for j in pending}
            self.assertIn("S4", pending_ids)
            self.assertIn("S14", pending_ids)
            self.assertNotIn("S0", pending_ids)
            m_fix = self._load_manifest_with_index(
                repo, template=full, index_rel=index_rel, tmp_dir=Path(td)
            )
            fix_ids = {j.get("id") for j in m_fix["jobs"]}
            self.assertIn("aug_smoke_S4", fix_ids)
            self.assertIn("aug_smoke_S14", fix_ids)

    def test_load_full_manifest(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_full.json")
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
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_full.json")
        by_id = {j["id"]: j for j in m["jobs"]}
        self.assertNotIn("aug_sweep_15_mosaic0", by_id)
        self.assertEqual(
            by_id["aug_sweep_15_close10"]["aug_config"],
            "configs/aug/robustness_smoke_close10.yaml",
        )
        self.assertEqual(
            by_id["aug_sweep_15_close10"]["train_config"],
            "configs/experiments/train_aug_close10_sweep_smoke_15ep.json",
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
            "configs/experiments/train_aug_close25_sweep_smoke_15ep.json",
        )

    def test_sweeps_15ep_index_matches_full_queue(self) -> None:
        from harchoc.aug_smoke_runner import load_aug_smoke_index
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")
        sweeps = index.get("sweeps_15ep") or {}
        arms = {str(a["id"]): a for a in (sweeps.get("arms") or [])}
        self.assertEqual(set(arms), {"close10", "close15", "close25"})

        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_full.json")
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

    def test_complete_recipe_owners_maps_close3_to_first_complete(self) -> None:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.gpu_queue import _complete_recipe_owners
        from harchoc.train_config import effective_train_recipe_fingerprint

        repo = Path(__file__).resolve().parents[1]
        owners = _complete_recipe_owners(repo_root=repo)
        index = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")
        fp = effective_train_recipe_fingerprint(
            resolve_aug_smoke_train_raw(find_smoke_entry(index, "S1"), repo_root=repo),
            repo_root=repo,
        )
        self.assertIsNotNone(fp)
        self.assertIn(fp, owners)
        self.assertIn(owners[fp], {"S0", "S1"})

    def _index_entry_recipe_fingerprint(self, entry: dict, *, repo: Path) -> str:
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import effective_train_recipe_fingerprint

        return effective_train_recipe_fingerprint(
            resolve_aug_smoke_train_raw(entry, repo_root=repo),
            repo_root=repo,
        )

    def _enrich_aug_smoke_job_from_index(self, job: dict, *, repo: Path) -> dict:
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import (
            resolve_aug_smoke_aug_config,
            resolve_aug_smoke_train_config_path,
        )

        entry = find_smoke_entry(
            load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json"),
            str(job.get("smoke_id") or ""),
        )
        enriched = dict(job)
        tc = resolve_aug_smoke_train_config_path(entry, repo_root=repo)
        enriched["train_config"] = tc
        aug = resolve_aug_smoke_aug_config(entry, repo_root=repo, train_config_path=tc)
        if aug:
            enriched["aug_config"] = aug
        return enriched

    def test_pending_gpu_pending_smokes_have_unique_fingerprints_vs_complete(self) -> None:
        """S4/S6–S9 train recipes must not duplicate any verified complete smoke."""
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.gpu_queue import (
            _complete_recipe_owners,
            _job_train_recipe_fingerprint,
            expand_aug_smoke_jobs_from_index,
        )

        repo = Path(__file__).resolve().parents[1]
        pending_train_ids = ("S4", "S6", "S7", "S8", "S9")
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(repo, Path(td), pending_train_ids)
            index = load_aug_smoke_index(repo / index_rel)
            owners = _complete_recipe_owners(repo_root=repo, index_path=index_rel)
            complete_fps = set(owners)

            pending_fps: dict[str, str] = {}
            for sid in pending_train_ids:
                entry = find_smoke_entry(index, sid)
                self.assertEqual(entry.get("status"), "gpu_pending", msg=sid)
                fp = self._index_entry_recipe_fingerprint(entry, repo=repo)
                pending_fps[sid] = fp
                self.assertNotIn(
                    fp,
                    complete_fps,
                    msg=f"{sid} recipe fp={fp} duplicates complete smoke {owners.get(fp)}",
                )

            self.assertEqual(
                len(set(pending_fps.values())),
                len(pending_fps),
                msg="pending train smokes must test distinct hypotheses",
            )

            expanded = {
                str(j["smoke_id"]): j
                for j in expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
                if not j.get("eval_only")
            }
            audit_pending = {"S6", "S7"}
            for sid in pending_train_ids:
                job = expanded[sid]
                self.assertEqual(job["id"], f"aug_smoke_{sid}")
                if sid in audit_pending:
                    self.assertTrue(job.get("skip"))
                    reason = str(job.get("skip_reason") or "")
                    self.assertIn("audit-only equivalence class", reason)
                    self.assertIn("canonical S3", reason)
                    self.assertIn("41e79d28", reason)
                    continue
                enriched = self._enrich_aug_smoke_job_from_index(job, repo=repo)
                self.assertEqual(
                    _job_train_recipe_fingerprint(enriched, repo_root=repo),
                    pending_fps[sid],
                )

    def test_expand_aug_smoke_skips_audit_only_gpu_pending(self) -> None:
        from harchoc.gpu_queue import expand_aug_smoke_jobs_from_index

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(
                repo, Path(td), ("S0", "S6", "S13", "S4")
            )
            by_sid = {
                str(j["smoke_id"]): j
                for j in expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
            }
            for sid in ("S0", "S6", "S13"):
                job = by_sid[sid]
                self.assertTrue(job.get("skip"), msg=sid)
                reason = str(job.get("skip_reason") or "")
                self.assertIn("audit-only equivalence class", reason)
                self.assertIn("preds_sha256=", reason)
            self.assertIn("canonical S1", by_sid["S0"]["skip_reason"])
            self.assertIn("ad6f1621", by_sid["S0"]["skip_reason"])
            self.assertIn("canonical S3", by_sid["S6"]["skip_reason"])
            self.assertIn("41e79d28", by_sid["S6"]["skip_reason"])
            self.assertIn("canonical S1", by_sid["S13"]["skip_reason"])
            self.assertNotEqual(by_sid["S4"].get("skip"), True)

    def test_job_train_recipe_fingerprint_resolves_from_index_only(self) -> None:
        """Expanded aug_smoke jobs omit train_config; fingerprint loads from index."""
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.gpu_queue import (
            _job_train_recipe_fingerprint,
            expand_aug_smoke_jobs_from_index,
        )

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(repo, Path(td), ("S4",))
            index = load_aug_smoke_index(repo / index_rel)
            job = next(
                j
                for j in expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
                if j.get("smoke_id") == "S4"
            )
            self.assertNotIn("train_config", job)
            expected = self._index_entry_recipe_fingerprint(find_smoke_entry(index, "S4"), repo=repo)
            self.assertEqual(_job_train_recipe_fingerprint(job, repo_root=repo), expected)

    def test_prune_dry_run_log_stubs(self) -> None:
        from harchoc.gpu_queue import _prune_dry_run_log_stubs

        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            log_root.mkdir()
            (log_root / "preflight").mkdir()
            (log_root / "preflight" / "check_gpu.log").write_text("ok\n", encoding="utf-8")
            real = log_root / "aug_smoke_S0"
            real.mkdir()
            (real / "train.log").write_bytes(b"x" * 2048)
            stub = log_root / "amp_smoke_15ep_on"
            stub.mkdir()
            (stub / "gpu_wait.log").write_text('{"status": "dry_run"}\n', encoding="utf-8")
            (log_root / "zoo_matrix_p0_5").mkdir()
            removed = _prune_dry_run_log_stubs(log_root)
            self.assertEqual(sorted(removed), ["amp_smoke_15ep_on", "zoo_matrix_p0_5"])
            self.assertTrue((log_root / "preflight").is_dir())
            self.assertTrue((log_root / "aug_smoke_S0").is_dir())
            self.assertFalse(stub.exists())

    def test_s0_s1_s13_recipe_fingerprint_equivalence(self) -> None:
        """S0/S1 share effective recipe fp; S13 is patience-only (index equivalence class)."""
        from harchoc.aug_smoke_runner import find_smoke_entry, load_aug_smoke_index
        from harchoc.aug_smoke_train import resolve_aug_smoke_train_raw
        from harchoc.train_config import EFFECTIVE_RECIPE_KEYS, effective_train_aug_merged

        repo = Path(__file__).resolve().parents[1]
        index = load_aug_smoke_index(repo / "configs/experiments/aug_smoke_index.json")

        def _merged_subset(sid: str) -> dict:
            entry = find_smoke_entry(index, sid)
            cfg = resolve_aug_smoke_train_raw(entry, repo_root=repo)
            merged = effective_train_aug_merged(cfg, repo_root=repo)
            return {k: merged.get(k) for k in EFFECTIVE_RECIPE_KEYS}

        fp_s0 = self._index_entry_recipe_fingerprint(find_smoke_entry(index, "S0"), repo=repo)
        fp_s1 = self._index_entry_recipe_fingerprint(find_smoke_entry(index, "S1"), repo=repo)
        fp_s13 = self._index_entry_recipe_fingerprint(find_smoke_entry(index, "S13"), repo=repo)

        self.assertEqual(fp_s0, fp_s1)

        s0_keys = _merged_subset("S0")
        s13_keys = _merged_subset("S13")
        self.assertNotEqual(fp_s13, fp_s0)
        self.assertEqual(
            {k: v for k, v in s13_keys.items() if k != "patience"},
            {k: v for k, v in s0_keys.items() if k != "patience"},
        )
        self.assertEqual(s0_keys["patience"], 12)
        self.assertEqual(s13_keys["patience"], 5)

        equiv = index.get("equivalence_classes") or {}
        class_ids = {
            tuple(sorted(c.get("smoke_ids") or []))
            for c in (equiv.get("classes") or [])
        }
        self.assertIn(tuple(sorted(["S0", "S1", "S13", "CLOSE25"])), class_ids)

    def test_complete_recipe_owners_uses_summary_has_count_mae(self) -> None:
        import inspect

        from harchoc.equivalence_index import summary_has_count_mae
        from harchoc.gpu_queue import _complete_recipe_owners

        src = inspect.getsource(_complete_recipe_owners)
        self.assertIn("summary_has_count_mae", src)
        self.assertNotIn("extract_count_mae", src)

        repo = Path(__file__).resolve().parents[1]
        transcript = {"status": "complete", "stages": [{"stage_id": "train"}]}
        self.assertFalse(summary_has_count_mae(transcript, repo))
        with_mae = {"status": "complete", "test_count_mae": 1.0}
        self.assertTrue(summary_has_count_mae(with_mae, repo))

    def test_filter_duplicate_train_recipes_skips_known_dupes(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_full.json", repo_root=repo)
        by_id = {j["id"]: j for j in m["jobs"]}
        dup = by_id["aug_sweep_15_close15"]
        self.assertTrue(dup.get("skip"))
        self.assertIn("recipe duplicate of complete smoke", dup.get("skip_reason") or "")

    def test_complete_preds_sha_owners_maps_photometric_cluster_to_s3(self) -> None:
        from harchoc.gpu_queue import _complete_preds_sha_owners

        repo = Path(__file__).resolve().parents[1]
        owners = _complete_preds_sha_owners(repo_root=repo)
        photometric_sha = "41e79d287721faf99c9de709d4578b09dd8f62af6d3fe00ee2cabece52387f4d"
        self.assertEqual(owners.get(photometric_sha), "S3")

    def test_filter_duplicate_preds_sha_skips_s6_against_s3_owner(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        repo = Path(__file__).resolve().parents[1]
        s6 = repo / "reports/aug_smoke/s6_summary.json"
        if not s6.is_file():
            self.skipTest("s6_summary.json missing")
        job = {
            "id": "aug_smoke_S6",
            "kind": "aug_smoke",
            "smoke_id": "S6",
            "skip_if": {"summary": "reports/aug_smoke/s6_summary.json", "index_status": "complete"},
        }
        out = _filter_duplicate_preds_sha([job], repo_root=repo)
        self.assertTrue(out[0].get("skip"))
        reason = str(out[0].get("skip_reason") or "")
        self.assertIn("preds duplicate of complete smoke S3", reason)
        self.assertIn("41e79d28", reason)

    def test_filter_duplicate_preds_sha_skips_s7_against_s3_owner(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        repo = Path(__file__).resolve().parents[1]
        s7 = repo / "reports/aug_smoke/s7_summary.json"
        if not s7.is_file():
            self.skipTest("s7_summary.json missing")
        job = {
            "id": "aug_smoke_S7",
            "kind": "aug_smoke",
            "smoke_id": "S7",
            "skip_if": {"summary": "reports/aug_smoke/s7_summary.json", "index_status": "complete"},
        }
        out = _filter_duplicate_preds_sha([job], repo_root=repo)
        self.assertTrue(out[0].get("skip"))
        self.assertIn("preds duplicate of complete smoke S3", out[0].get("skip_reason") or "")

    def test_filter_duplicate_preds_sha_does_not_skip_unique_preds(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        repo = Path(__file__).resolve().parents[1]
        s8 = repo / "reports/aug_smoke/s8_summary.json"
        if not s8.is_file():
            self.skipTest("s8_summary.json missing")
        job = {
            "id": "aug_smoke_S8",
            "kind": "aug_smoke",
            "smoke_id": "S8",
            "skip_if": {"summary": "reports/aug_smoke/s8_summary.json", "index_status": "complete"},
        }
        out = _filter_duplicate_preds_sha([job], repo_root=repo)
        self.assertNotEqual(out[0].get("skip"), True)

    def test_job_preds_sha_for_dedup_prefers_index_equivalence(self) -> None:
        """index_preds from equivalence_classes resolves SHA before summary exists."""
        from harchoc.gpu_queue import _index_preds_sha_by_smoke_id, _job_preds_sha_for_dedup

        photometric_sha = "41e79d287721faf99c9de709d4578b09dd8f62af6d3fe00ee2cabece52387f4d"
        index = {
            "equivalence_classes": {
                "classes": [{"smoke_ids": ["S3", "S6"], "preds_sha256": photometric_sha}]
            }
        }
        index_preds = _index_preds_sha_by_smoke_id(index)
        self.assertEqual(index_preds["S6"], photometric_sha)
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            job = {
                "kind": "aug_smoke",
                "smoke_id": "S6",
                "skip_if": {"summary": "reports/aug_smoke/s6_summary.json"},
            }
            sha = _job_preds_sha_for_dedup(job, repo_root=repo, index_preds=index_preds)
            self.assertEqual(sha, photometric_sha)

    def test_filter_duplicate_preds_sha_uses_index_equivalence_when_summary_missing(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports" / "aug_smoke").mkdir(parents=True)
            index = {
                "schema_version": "aug_smoke_index.v1",
                "smokes": [
                    {
                        "id": "S3",
                        "status": "complete",
                        "summary": "reports/aug_smoke/s3_summary.json",
                        "aug_config": "configs/aug/robustness_photometric_only.yaml",
                    },
                    {"id": "S6", "status": "gpu_pending", "summary": "reports/aug_smoke/s6_summary.json"},
                ],
                "equivalence_classes": {
                    "classes": [
                        {
                            "smoke_ids": ["S3", "S6"],
                            "preds_sha256": "41e79d287721faf99c9de709d4578b09dd8f62af6d3fe00ee2cabece52387f4d",
                        }
                    ]
                },
            }
            (repo / "index.json").write_text(json.dumps(index), encoding="utf-8")
            photometric_sha = "41e79d287721faf99c9de709d4578b09dd8f62af6d3fe00ee2cabece52387f4d"
            (repo / "reports/aug_smoke/s3_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "test_count_mae": 151.73,
                        "artifacts": {"preds_json": {"sha256": photometric_sha}},
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "id": "aug_smoke_S6",
                "kind": "aug_smoke",
                "smoke_id": "S6",
                "skip_if": {"summary": "reports/aug_smoke/s6_summary.json", "index_status": "complete"},
            }
            out = _filter_duplicate_preds_sha([job], repo_root=repo, index_path="index.json")
            self.assertTrue(out[0].get("skip"))
            self.assertIn("preds duplicate of complete smoke S3", out[0].get("skip_reason") or "")

    def test_filter_duplicate_preds_sha_skips_second_manifest_dup(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        repo = Path(__file__).resolve().parents[1]
        s3 = repo / "reports/aug_smoke/s3_summary.json"
        if not s3.is_file():
            self.skipTest("s3_summary.json missing")
        base = {
            "kind": "aug_smoke",
            "smoke_id": "S3",
            "skip_if": {"summary": "reports/aug_smoke/s3_summary.json", "index_status": "complete"},
        }
        jobs = [
            {"id": "aug_smoke_S3_a", **base},
            {"id": "aug_smoke_S3_b", **base},
        ]
        out = _filter_duplicate_preds_sha(jobs, repo_root=repo)
        skipped = [j for j in out if j.get("skip")]
        self.assertEqual(len(skipped), 1)
        self.assertIn("duplicate preds in manifest", skipped[0].get("skip_reason") or "")

    def test_s14_eval_only_not_preds_deduped_as_s1(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "aug_smoke_S14",
            "kind": "aug_smoke",
            "smoke_id": "S14",
            "eval_only": True,
            "skip_if": {"summary": "reports/aug_smoke/s14_summary.json", "index_status": "complete"},
        }
        out = _filter_duplicate_preds_sha([job], repo_root=repo)
        self.assertNotEqual(out[0].get("skip"), True)

    def test_filter_duplicate_preds_sha_skips_close25_against_s1_owner(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        repo = Path(__file__).resolve().parents[1]
        s1 = repo / "reports/aug_smoke/s1_summary.json"
        if not s1.is_file():
            self.skipTest("s1_summary.json missing")
        job = {
            "id": "aug_sweep_15_close25",
            "kind": "aug_sweep_15",
            "summary_path": "reports/aug_smoke/sweep_close25_15ep_summary.json",
            "skip_if": {"summary": "reports/aug_smoke/sweep_close25_15ep_summary.json"},
        }
        out = _filter_duplicate_preds_sha([job], repo_root=repo)
        self.assertTrue(out[0].get("skip"))
        reason = str(out[0].get("skip_reason") or "")
        self.assertRegex(reason, r"preds duplicate of complete smoke S[01]")
        self.assertIn("ad6f1621", reason)

    def test_filter_duplicate_preds_sha_skips_close25_when_summary_exists_manifest_dup(
        self,
    ) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        baseline_sha = "ad6f1621d8c2c8a1c1db2000626f0fc17f9c19da83348aa92bbc1ba4862607e8"
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports" / "aug_smoke").mkdir(parents=True)
            summary_rel = "reports/aug_smoke/sweep_close25_15ep_summary.json"
            (repo / summary_rel).write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "smoke_id": "CLOSE25",
                        "test_count_mae": 68.9,
                        "artifacts": {"preds_json": {"sha256": baseline_sha}},
                    }
                ),
                encoding="utf-8",
            )
            index = {
                "schema_version": "aug_smoke_index.v1",
                "smokes": [],
                "sweeps_15ep": {
                    "arms": [
                        {
                            "id": "close25",
                            "status": "complete",
                            "queue_job_id": "aug_sweep_15_close25",
                            "summary": summary_rel,
                        }
                    ]
                },
            }
            (repo / "index.json").write_text(json.dumps(index), encoding="utf-8")
            base = {
                "kind": "aug_sweep_15",
                "summary_path": summary_rel,
                "skip_if": {"summary": summary_rel},
            }
            jobs = [
                {"id": "aug_sweep_15_close25_a", **base},
                {"id": "aug_sweep_15_close25_b", **base},
            ]
            out = _filter_duplicate_preds_sha(jobs, repo_root=repo, index_path="index.json")
            skipped = [j for j in out if j.get("skip")]
            self.assertEqual(len(skipped), 1)
            self.assertIn("duplicate preds in manifest", skipped[0].get("skip_reason") or "")

    def test_job_preds_sha_for_dedup_close25_index_equivalence_before_summary(self) -> None:
        """CLOSE25 in equivalence_classes resolves SHA before sweep summary exists."""
        from harchoc.gpu_queue import _index_preds_sha_by_smoke_id, _job_preds_sha_for_dedup

        baseline_sha = "ad6f1621d8c2c8a1c1db2000626f0fc17f9c19da83348aa92bbc1ba4862607e8"
        index = {
            "equivalence_classes": {
                "classes": [
                    {
                        "smoke_ids": ["S1", "CLOSE25"],
                        "preds_sha256": baseline_sha,
                    }
                ]
            },
            "sweeps_15ep": {
                "arms": [
                    {
                        "id": "close25",
                        "queue_job_id": "aug_sweep_15_close25",
                        "summary": "reports/aug_smoke/sweep_close25_15ep_summary.json",
                    }
                ]
            },
        }
        index_preds = _index_preds_sha_by_smoke_id(index)
        self.assertEqual(index_preds["CLOSE25"], baseline_sha)
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            job = {
                "kind": "aug_sweep_15",
                "id": "aug_sweep_15_close25",
                "summary_path": "reports/aug_smoke/sweep_close25_15ep_summary.json",
            }
            sha = _job_preds_sha_for_dedup(
                job, repo_root=repo, index_preds=index_preds, index=index
            )
            self.assertEqual(sha, baseline_sha)

    def test_filter_duplicate_preds_sha_skips_aug_sweep_15_against_smoke_owner(self) -> None:
        from harchoc.gpu_queue import _filter_duplicate_preds_sha

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "reports" / "aug_smoke").mkdir(parents=True)
            baseline_sha = "ad6f1621d8c2c8a1c1db2000626f0fc17f9c19da83348aa92bbc1ba4862607e8"
            index = {
                "schema_version": "aug_smoke_index.v1",
                "smokes": [
                    {
                        "id": "S1",
                        "status": "complete",
                        "summary": "reports/aug_smoke/s1_summary.json",
                        "aug_config": "configs/aug/robustness_smoke_close3.yaml",
                    }
                ],
                "sweeps_15ep": {
                    "arms": [
                        {
                            "id": "close25",
                            "status": "gpu_pending",
                            "queue_job_id": "aug_sweep_15_close25",
                            "summary": "reports/aug_smoke/sweep_close25_15ep_summary.json",
                        }
                    ]
                },
                "equivalence_classes": {
                    "classes": [
                        {
                            "smoke_ids": ["S1", "CLOSE25"],
                            "preds_sha256": baseline_sha,
                        }
                    ]
                },
            }
            (repo / "index.json").write_text(json.dumps(index), encoding="utf-8")
            (repo / "reports/aug_smoke/s1_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "test_count_mae": 68.9,
                        "artifacts": {"preds_json": {"sha256": baseline_sha}},
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "id": "aug_sweep_15_close25",
                "kind": "aug_sweep_15",
                "summary_path": "reports/aug_smoke/sweep_close25_15ep_summary.json",
            }
            out = _filter_duplicate_preds_sha([job], repo_root=repo, index_path="index.json")
            self.assertTrue(out[0].get("skip"))
            self.assertIn("preds duplicate of complete smoke S1", out[0].get("skip_reason") or "")

    def test_load_aug_pending_manifest_expands_gpu_pending(self) -> None:
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        aug_pending = repo / "configs/experiments/gpu_queue_aug_pending.json"
        m = load_gpu_queue_manifest(aug_pending, repo_root=repo)
        ids = {j.get("id") for j in m["jobs"]}
        self.assertIn("preflight", ids)
        self.assertNotIn("aug_smoke_S0", ids)
        self.assertEqual(ids, {"preflight"})

        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(repo, Path(td), ("S4", "S14"))
            m_fix = self._load_manifest_with_index(
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
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_full.json", repo_root=repo)
        amp = next(j for j in m["jobs"] if j.get("id") == "amp_smoke_15ep_on")
        self.assertNotEqual(amp.get("skip"), True)
        reason = str(amp.get("skip_reason") or "")
        self.assertNotIn("recipe duplicate of complete smoke S1", reason)

    def test_s14_eval_only_not_recipe_deduped_as_s1(self) -> None:
        from harchoc.gpu_queue import _job_train_recipe_fingerprint, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        full = repo / "configs/experiments/gpu_queue_full.json"
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(repo, Path(td), ("S14",))
            m = self._load_manifest_with_index(
                repo, template=full, index_rel=index_rel, tmp_dir=Path(td)
            )
            s14 = next(j for j in m["jobs"] if j.get("id") == "aug_smoke_S14")
            self.assertTrue(s14.get("eval_only"))
            self.assertIsNone(_job_train_recipe_fingerprint(s14, repo_root=repo))
            self.assertNotEqual(s14.get("skip"), True)

    def test_load_aug_confirm_manifest(self) -> None:
        from harchoc.gpu_queue import build_job_stages, load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_aug_confirm.json")
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
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_aug_close_phase_a.json")
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
        m = load_gpu_queue_manifest(repo / "configs/experiments/gpu_queue_aug_close_100ep.json")
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
            repo / "configs/experiments/gpu_queue_full.json", repo_root=repo
        )
        job = next(j for j in m["jobs"] if j.get("id") == "aug_confirm_winner_100ep")
        self.assertTrue(job.get("skip"))
        self.assertIn("gpu_queue_aug_confirm", str(job.get("skip_reason") or ""))

    def test_full_manifest_gpu_execution_tier_order(self) -> None:
        """Tier 1 RT-DETR block → Tier 2 eval/sweeps → P0-5 zoo_matrix before cv_fold (backlog § GPU execution tiers)."""
        from harchoc.gpu_queue import load_gpu_queue_manifest

        repo = Path(__file__).resolve().parents[1]
        m = load_gpu_queue_manifest(
            repo / "configs/experiments/gpu_queue_full.json", repo_root=repo
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

    def test_build_aug_smoke_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {"id": "aug_smoke_S3", "kind": "aug_smoke", "smoke_id": "S3"}
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("dry_run", ids)
        self.assertIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)

    def test_should_skip_complete_summary(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        summary = repo / "reports/aug_smoke/s0_summary.json"
        if not summary.is_file():
            self.skipTest("s0_summary.json missing")
        job = {"id": "x", "skip_if": {"summary": "reports/aug_smoke/s0_summary.json"}}
        skip, reason = should_skip_job(job, repo_root=repo)
        self.assertTrue(skip)
        self.assertIn("complete", reason)

    def test_should_not_skip_stale_gpu_queue_job_transcript(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        transcript = repo / "reports/gpu_queue/jobs/rtdetr_queries_smoke.json"
        if not transcript.is_file():
            self.skipTest("rtdetr_queries_smoke job transcript missing")
        job = {
            "id": "rtdetr_queries_smoke",
            "kind": "rtdetr_smoke",
            "run_name": "rtdetr_queries_smoke_15ep",
            "skip_if": {
                "summary": "reports/gpu_queue/jobs/rtdetr_queries_smoke.json",
                "eval_error_json": "reports/gpu_queue/eval/rtdetr_queries_smoke_15ep_error.json",
            },
        }
        skip, _ = should_skip_job(job, repo_root=repo)
        self.assertFalse(skip)

    def test_should_skip_when_verified_summary_and_eval_exist(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        summary = repo / "reports/aug_smoke/s1_summary.json"
        err = repo / "reports/aug_smoke/aug_smoke_close3_error.json"
        if not summary.is_file() or not err.is_file():
            self.skipTest("s1 summary or error json missing")
        job = {
            "id": "aug_smoke_S1",
            "kind": "aug_smoke",
            "skip_if": {
                "summary": "reports/aug_smoke/s1_summary.json",
                "eval_error_json": "reports/aug_smoke/aug_smoke_close3_error.json",
            },
        }
        skip, reason = should_skip_job(job, repo_root=repo)
        self.assertTrue(skip)
        self.assertIn("complete", reason)

    def test_run_job_dry_run_status_not_complete(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "rtdetr_imgsz640",
                "kind": "train_compare",
                "train_config": "configs/experiments/train_rtdetr_imgsz640_smoke_15ep.json",
                "run_name": "rtdetr_imgsz640_smoke_15ep",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")

    def test_dry_run_preflight_job(self) -> None:
        from harchoc.gpu_queue import run_gpu_queue

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "mini.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "gpu_queue_manifest.v1",
                        "jobs": [{"id": "preflight", "kind": "preflight"}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("harchoc.gpu_queue._run_subprocess_stage", return_value=0):
                rc = run_gpu_queue(
                    manifest,
                    repo_root=repo,
                    dry_run=True,
                    state_path=Path(td) / "state.json",
                )
            self.assertEqual(rc, 0)

    def test_wait_gpu_free_dry_run(self) -> None:
        from harchoc.gpu_queue import wait_gpu_free

        info = wait_gpu_free(min_free_mib=5500, dry_run=True)
        self.assertEqual(info["status"], "dry_run")

    def test_adhoc_train_blocked_when_lock(self) -> None:
        from harchoc.gpu_exclusive import acquire_gpu_exclusive, release_gpu_exclusive

        repo = Path(__file__).resolve().parents[1]
        acquire_gpu_exclusive(repo_root=repo, owner="test")
        try:
            from harchoc.gpu_exclusive import adhoc_train_blocked

            self.assertTrue(adhoc_train_blocked(repo_root=repo))
        finally:
            release_gpu_exclusive(repo_root=repo)

    def test_run_job_train_compare_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "rtdetr_imgsz640",
                "kind": "train_compare",
                "train_config": "configs/experiments/train_rtdetr_imgsz640_smoke_15ep.json",
                "run_name": "rtdetr_imgsz640_smoke_15ep",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")

    def test_build_rtdetr_train_compare_includes_hsp_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "rtdetr_imgsz640",
            "kind": "train_compare",
            "train_config": "configs/experiments/train_rtdetr_imgsz640_smoke_15ep.json",
            "run_name": "rtdetr_imgsz640_smoke_15ep",
            "max_det": 300,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("eval_test", ids)
        eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
        self.assertEqual(eval_stage["meta"]["max_det"], 300)
        summary = next(s for s in stages if s["stage_id"] == "summary")
        self.assertEqual(summary["meta"]["summary_kind"], "rtdetr")

    def test_run_rtdetr_smoke_dry_run_writes_eval_chain(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            for job_id, cfg, name, max_det in (
                (
                    "rtdetr_queries_smoke",
                    "configs/experiments/train_rtdetr_queries_smoke_15ep.json",
                    "rtdetr_queries_smoke_15ep",
                    1024,
                ),
                (
                    "rtdetr_imgsz1280",
                    "configs/experiments/train_rtdetr_smoke_15ep.json",
                    "rtdetr_imgsz1280_smoke_15ep",
                    300,
                ),
            ):
                kind = "rtdetr_smoke" if job_id == "rtdetr_queries_smoke" else "train_compare"
                job = {
                    "id": job_id,
                    "kind": kind,
                    "train_config": cfg,
                    "run_name": name,
                    "max_det": max_det,
                }
                result = run_job(
                    job,
                    repo_root=repo,
                    defaults={},
                    dry_run=True,
                    min_free_mib=5500,
                    log_root=log_root,
                )
                self.assertEqual(result["status"], "dry_run_complete", job_id)
                eval_log = log_root / job_id / "eval_test.log"
                self.assertTrue(eval_log.is_file(), job_id)
                text = eval_log.read_text(encoding="utf-8")
                self.assertIn("eval_export", text, job_id)
                self.assertIn(str(max_det), text, job_id)

    def test_zoo_matrix_dry_run_includes_rtdetr_gate_stage(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {"id": "zoo_matrix_p0_5", "kind": "zoo_matrix_train", "matrix_group": "zoo_core"}
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("rtdetr_15ep_gate", ids)

    def test_run_job_aug_smoke_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {"id": "aug_smoke_S1", "kind": "aug_smoke", "smoke_id": "S1"}
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            eval_log = log_root / "aug_smoke_S1" / "eval_test.log"
            self.assertTrue(eval_log.is_file())
            self.assertIn("eval_export", eval_log.read_text(encoding="utf-8"))

    def test_run_job_vram_probe_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "vram_probe_rtdetr",
                "kind": "vram_probe",
                "train_config": "configs/experiments/train_batch_probe_rtdetr-l.json",
                "run_name": "batch_probe_rtdetr-l",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            self.assertTrue((log_root / "vram_probe_rtdetr" / "gpu_wait.log").is_file())

    def test_build_aug_sweep_15_stages(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "aug_sweep_15_mosaic0",
            "kind": "aug_sweep_15",
            "train_config": "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json",
            "aug_config": "configs/aug/robustness_mosaic_off.yaml",
            "run_name": "aug_sweep_mosaic0_15ep",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        cfg_path = repo / "configs/experiments/train_aug_mosaic_sweep_smoke_15ep.json"
        self.assertTrue(cfg_path.is_file())

    def test_should_skip_when_weights_exist_for_train_only_job(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        weights = repo / "runs/amp_on_smoke_15ep/weights/best.pt"
        if not weights.is_file():
            self.skipTest("amp_on_smoke_15ep weights missing")
        job = {
            "id": "amp_smoke_15ep_on",
            "kind": "amp_smoke",
            "run_name": "amp_on_smoke_15ep",
            "skip_eval": True,
            "skip_if": {
                "summary": "reports/hsp/amp_on_smoke_15ep_summary.json",
                "weights_run_name": "amp_on_smoke_15ep",
            },
        }
        skip, reason = should_skip_job(job, repo_root=repo)
        self.assertTrue(skip)
        self.assertTrue(
            "weights exist" in reason or "summary complete" in reason,
            reason,
        )

    def test_should_skip_eval_only_when_summary_or_hsp_artifacts_complete(self) -> None:
        from harchoc.gpu_queue import should_skip_job

        repo = Path(__file__).resolve().parents[1]
        summary = repo / "reports/hsp/amp_on_smoke_15ep_summary.json"
        weights = repo / "runs/amp_on_smoke_15ep/weights/best.pt"
        if not weights.is_file():
            self.skipTest("amp_on_smoke_15ep weights missing")
        job = {
            "id": "amp_smoke_15ep_on_hsp_eval",
            "kind": "amp_smoke",
            "run_name": "amp_on_smoke_15ep",
            "eval_only": True,
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "eval_out_dir": "reports/hsp",
            "skip_if": {"summary": "reports/hsp/amp_on_smoke_15ep_summary.json"},
        }
        skip, reason = should_skip_job(job, repo_root=repo)
        if summary.is_file():
            self.assertTrue(skip, reason)
            self.assertTrue(
                "summary complete" in reason or "HSP eval artifacts complete" in reason,
                reason,
            )
        else:
            self.assertFalse(skip)

    def test_build_amp_smoke_stages_train_only_skips_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "amp_smoke_15ep_on",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "run_name": "amp_on_smoke_15ep",
            "eval_out_dir": "reports/hsp",
            "skip_eval": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertNotIn("eval_test", ids)
        self.assertIn("summary", ids)

    def test_build_amp_smoke_stages_includes_hsp_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "amp_smoke_15ep_on_hsp_eval",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "run_name": "amp_on_smoke_15ep",
            "eval_only": True,
            "eval_out_dir": "reports/hsp",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
        self.assertEqual(eval_stage.get("internal"), "smoke_hsp_eval")
        self.assertEqual(eval_stage["meta"]["out_dir"], "reports/hsp")

    def test_build_aug_smoke_s14_eval_only_max_det_300(self) -> None:
        from harchoc.gpu_queue import build_job_stages, expand_aug_smoke_jobs_from_index

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = self._write_pending_fixture_index(repo, Path(td), ("S14",))
            job = next(
                j
                for j in expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
                if j.get("smoke_id") == "S14"
            )
            self.assertTrue(job.get("eval_only"))
            self.assertEqual(job.get("max_det"), 300)
            self.assertEqual(job.get("weights_run_name"), "aug_smoke_close3")
            stages = build_job_stages(job, repo_root=repo)
            ids = [s["stage_id"] for s in stages]
            self.assertNotIn("train", ids)
            self.assertIn("eval_test", ids)
            eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
            self.assertEqual(eval_stage["meta"]["max_det"], 300)
            self.assertEqual(eval_stage["meta"]["weights_run_name"], "aug_smoke_close3")

    def test_run_job_aug_smoke_s14_dry_run_eval_max_det(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        weights = repo / "runs/aug_smoke_close3/weights/best.pt"
        if not weights.is_file():
            self.skipTest("S1 weights missing")
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "aug_smoke_S14",
                "kind": "aug_smoke",
                "smoke_id": "S14",
                "run_name": "aug_smoke_eval300",
                "weights_run_name": "aug_smoke_close3",
                "eval_only": True,
                "max_det": 300,
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            text = (log_root / "aug_smoke_S14" / "eval_test.log").read_text(encoding="utf-8")
            self.assertIn("--export-max-det", text)
            self.assertIn("300", text)
            self.assertIn("aug_smoke_close3", text)
            self.assertIn("aug_smoke_eval300_preds.json", text)

    def test_build_amp_smoke_eval_only_skips_train(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "amp_smoke_15ep_on_hsp_eval",
            "kind": "amp_smoke",
            "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
            "run_name": "amp_on_smoke_15ep",
            "eval_only": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)

    def test_run_job_amp_smoke_eval_only_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "amp_smoke_15ep_on_hsp_eval",
                "kind": "amp_smoke",
                "train_config": "configs/experiments/train_amp_on_15ep_smoke.json",
                "run_name": "amp_on_smoke_15ep",
                "eval_only": True,
                "eval_out_dir": "reports/hsp",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            eval_log = log_root / "amp_smoke_15ep_on_hsp_eval" / "eval_test.log"
            self.assertTrue(eval_log.is_file())
            text = eval_log.read_text(encoding="utf-8")
            self.assertIn("eval_export", text)
            self.assertIn("reports/hsp/amp_on_smoke_15ep_preds.json", text)

    def test_build_sg_smoke_stages_train_only_skips_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "sg_yolo_nas_s_smoke",
            "kind": "sg_smoke",
            "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
            "run_name": "sg_yolo_nas_s_smoke_15ep",
            "eval_out_dir": "reports/aug_smoke",
            "skip_eval": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertIn("train", ids)
        self.assertNotIn("eval_test", ids)

    def test_build_sg_smoke_stages_includes_hsp_eval(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "sg_yolo_nas_s_hsp_eval",
            "kind": "sg_smoke",
            "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
            "run_name": "sg_yolo_nas_s_smoke_15ep",
            "eval_only": True,
            "eval_out_dir": "reports/aug_smoke",
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)
        self.assertIn("summary", ids)
        eval_stage = next(s for s in stages if s["stage_id"] == "eval_test")
        self.assertEqual(eval_stage.get("internal"), "smoke_hsp_eval")

    def test_build_sg_smoke_eval_only_skips_train(self) -> None:
        from harchoc.gpu_queue import build_job_stages

        repo = Path(__file__).resolve().parents[1]
        job = {
            "id": "sg_yolo_nas_s_hsp_eval",
            "kind": "sg_smoke",
            "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
            "run_name": "sg_yolo_nas_s_smoke_15ep",
            "eval_only": True,
        }
        stages = build_job_stages(job, repo_root=repo)
        ids = [s["stage_id"] for s in stages]
        self.assertNotIn("train", ids)
        self.assertIn("eval_test", ids)

    def test_run_job_sg_smoke_eval_only_dry_run(self) -> None:
        from harchoc.gpu_queue import run_job

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            log_root = Path(td) / "logs"
            job = {
                "id": "sg_yolo_nas_s_hsp_eval",
                "kind": "sg_smoke",
                "train_config": "configs/experiments/train_sg_yolo_nas_s_smoke_15ep.json",
                "run_name": "sg_yolo_nas_s_smoke_15ep",
                "eval_only": True,
                "eval_out_dir": "reports/aug_smoke",
            }
            result = run_job(
                job,
                repo_root=repo,
                defaults={},
                dry_run=True,
                min_free_mib=5500,
                log_root=log_root,
            )
            self.assertEqual(result["status"], "dry_run_complete")
            eval_log = log_root / "sg_yolo_nas_s_hsp_eval" / "eval_test.log"
            self.assertTrue(eval_log.is_file())
            text = eval_log.read_text(encoding="utf-8")
            self.assertIn("error_analysis", text)
            self.assertIn("reports/aug_smoke/sg_yolo_nas_s_smoke_15ep_error.json", text)

    def test_infer_smoke_eval_backend(self) -> None:
        from harchoc.aug_smoke_runner import infer_smoke_eval_backend

        self.assertEqual(infer_smoke_eval_backend("runs/x/weights/best.pth"), "supergradients")
        self.assertEqual(infer_smoke_eval_backend("runs/x/weights/best.pt"), "ultralytics")

    def test_validate_missing_train_config_raises(self) -> None:
        from harchoc.gpu_queue import _validate_job_files

        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(FileNotFoundError):
            _validate_job_files(
                {
                    "id": "bad",
                    "kind": "train_compare",
                    "train_config": "configs/experiments/no_such_train.json",
                },
                repo,
            )


class GpuQueueDedupIntegrationTests(unittest.TestCase):
    """End-to-end manifest load: index expansion + recipe/preds dedup + audit-only tier."""

    def _repo(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _aug_pending_template(self, repo: Path) -> Path:
        return repo / "configs/experiments/gpu_queue_aug_pending.json"

    def _load_pending_manifest(
        self,
        repo: Path,
        tmp_dir: Path,
        pending_ids: tuple[str, ...],
        *,
        include_equivalence_classes: bool = True,
    ) -> dict:
        helper = GpuQueueManifestTests()
        index_rel = helper._write_pending_fixture_index(
            repo,
            tmp_dir,
            pending_ids,
            include_equivalence_classes=include_equivalence_classes,
        )
        return helper._load_manifest_with_index(
            repo,
            template=self._aug_pending_template(repo),
            index_rel=index_rel,
            tmp_dir=tmp_dir,
        )

    def _aug_job(self, manifest: dict, smoke_id: str) -> dict | None:
        sid = smoke_id.upper()
        return next((j for j in manifest["jobs"] if j.get("smoke_id") == sid), None)

    def test_integration_gpu_pending_s6_skipped_preds_duplicate(self) -> None:
        """gpu_pending S6 with complete S3 preds → skipped (preds duplicate)."""
        repo = self._repo()
        s3 = repo / "reports/aug_smoke/s3_summary.json"
        if not s3.is_file():
            self.skipTest("s3_summary.json missing")
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            manifest = self._load_pending_manifest(
                repo,
                Path(td),
                ("S6",),
                include_equivalence_classes=False,
            )
            job = self._aug_job(manifest, "S6")
            self.assertIsNotNone(job)
            self.assertTrue(job.get("skip"))
            reason = str(job.get("skip_reason") or "")
            self.assertIn("preds duplicate of complete smoke S3", reason)
            self.assertIn("41e79d28", reason)

    def test_integration_gpu_pending_s7_skipped(self) -> None:
        """gpu_pending S7 → skipped (preds duplicate of complete S3)."""
        repo = self._repo()
        s7 = repo / "reports/aug_smoke/s7_summary.json"
        if not s7.is_file():
            self.skipTest("s7_summary.json missing")
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            manifest = self._load_pending_manifest(
                repo,
                Path(td),
                ("S7",),
                include_equivalence_classes=False,
            )
            job = self._aug_job(manifest, "S7")
            self.assertIsNotNone(job)
            self.assertTrue(job.get("skip"))
            self.assertIn("preds duplicate of complete smoke S3", job.get("skip_reason") or "")

    def test_integration_gpu_pending_s0_skipped_recipe_duplicate(self) -> None:
        """gpu_pending S0 with complete S1 recipe → omitted (recipe duplicate)."""
        repo = self._repo()
        s1 = repo / "reports/aug_smoke/s1_summary.json"
        if not s1.is_file():
            self.skipTest("s1_summary.json missing")
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            manifest = self._load_pending_manifest(repo, Path(td), ("S0",))
            ids = {j.get("id") for j in manifest["jobs"]}
            self.assertNotIn("aug_smoke_S0", ids)

    def test_integration_gpu_pending_s8_not_skipped(self) -> None:
        """gpu_pending S8 unique preds → NOT skipped."""
        repo = self._repo()
        s8 = repo / "reports/aug_smoke/s8_summary.json"
        if not s8.is_file():
            self.skipTest("s8_summary.json missing")
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            manifest = self._load_pending_manifest(repo, Path(td), ("S8",))
            job = self._aug_job(manifest, "S8")
            self.assertIsNotNone(job)
            self.assertEqual(job.get("id"), "aug_smoke_S8")
            self.assertNotEqual(job.get("skip"), True)

    def test_integration_audit_only_expansion_skips_s6_s7(self) -> None:
        """equivalence_classes audit-only tier skips gpu_pending S6/S7 at expansion."""
        repo = self._repo()
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            manifest = self._load_pending_manifest(repo, Path(td), ("S6", "S7"))
            for sid in ("S6", "S7"):
                job = self._aug_job(manifest, sid)
                self.assertIsNotNone(job, msg=sid)
                self.assertTrue(job.get("skip"), msg=sid)
                reason = str(job.get("skip_reason") or "")
                self.assertIn("audit-only equivalence class", reason)
                self.assertIn("canonical S3", reason)
                self.assertIn("41e79d28", reason)

    def test_integration_zoo_core_dry_run_manifest_has_ten_train_rows(self) -> None:
        """zoo_core dry-run plan has exactly 10 train rows (no accidental duplicates)."""
        import os

        from scripts.benchmark_matrix import main as benchmark_main

        repo = self._repo()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            out_path = tdp / "matrix_plan.json"
            rc = benchmark_main(
                [
                    "--dry-run",
                    "--group",
                    "zoo_core",
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out_path.read_text(encoding="utf-8"))
            runs = obj.get("runs") or []
            self.assertEqual(len(runs), 10)
            basenames = [Path(r["config"]["path"]).name for r in runs]
            self.assertEqual(len(basenames), len(set(basenames)), msg=basenames)

    def test_integration_zoo_core_8gb_dry_run_has_eight_train_rows(self) -> None:
        """zoo_core_8gb: 4× YOLO *m + 4× external DETR (8 GiB path; no Ultralytics RT-DETR)."""
        import os

        from scripts.benchmark_matrix import main as benchmark_main

        repo = self._repo()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            dataset_root = tdp / "dataset_root"
            dataset_root.mkdir(parents=True, exist_ok=True)
            os.environ["DATASET_ROOT"] = str(dataset_root)
            os.environ.pop("YOLO_DATA_YAML", None)
            os.environ.pop("DATASET_NAME", None)

            out_path = tdp / "matrix_plan_8gb.json"
            rc = benchmark_main(
                [
                    "--dry-run",
                    "--group",
                    "zoo_core_8gb",
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out_path.read_text(encoding="utf-8"))
            runs = obj.get("runs") or []
            self.assertEqual(len(runs), 8)
            basenames = {Path(r["config"]["path"]).name for r in runs}
            self.assertNotIn("rtdetr_l_nq1024.yaml", basenames)
            self.assertNotIn("rtdetr_x_default.yaml", basenames)
            self.assertIn("yolov8m_default.yaml", basenames)
            self.assertIn("deim_dfine_l_default.yaml", basenames)


class ExperimentGpuQueueCliTests(unittest.TestCase):
    def _dry_run_gpu_queue_aug_pending(self, *, invoke) -> tuple[int, str]:
        repo = Path(__file__).resolve().parents[1]
        manifest = repo / "configs/experiments/gpu_queue_aug_pending.json"
        if not manifest.is_file():
            self.skipTest("gpu_queue_aug_pending.json missing")
        import io
        from contextlib import redirect_stdout

        with mock.patch("harchoc.gpu_queue._run_subprocess_stage", return_value=0):
            with mock.patch("harchoc.gpu_queue.wait_gpu_free") as wg:
                wg.return_value = {"status": "dry_run"}
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = invoke(manifest)
        return rc, buf.getvalue()

    def test_gpu_queue_dry_run_cli(self) -> None:
        from scripts.experiment import main

        rc, _ = self._dry_run_gpu_queue_aug_pending(
            invoke=lambda manifest: main(
                ["gpu-queue", "--manifest", str(manifest), "--dry-run"]
            )
        )
        self.assertEqual(rc, 0)

    def test_gpu_queue_dry_run_experiment_matches_run_gpu_queue(self) -> None:
        from scripts.experiment import main as experiment_main
        from scripts.run_gpu_queue import main as run_gpu_queue_main

        rc_exp, out_exp = self._dry_run_gpu_queue_aug_pending(
            invoke=lambda manifest: experiment_main(
                ["gpu-queue", "--manifest", str(manifest), "--dry-run"]
            )
        )
        rc_cli, out_cli = self._dry_run_gpu_queue_aug_pending(
            invoke=lambda manifest: run_gpu_queue_main(
                ["--manifest", str(manifest), "--dry-run"]
            )
        )
        self.assertEqual(rc_exp, rc_cli)
        self.assertEqual(out_exp, out_cli)


if __name__ == "__main__":
    unittest.main()
