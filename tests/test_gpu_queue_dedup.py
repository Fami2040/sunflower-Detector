"""Tests for harchoc.gpu_queue (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._gpu_queue_fixtures import (
    enrich_aug_smoke_job_from_index,
    index_entry_recipe_fingerprint,
    load_manifest_with_index,
    write_pending_fixture_index,
)


class GpuQueueDedupTests(unittest.TestCase):
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
            index_rel = write_pending_fixture_index(repo, Path(td), pending_train_ids)
            index = load_aug_smoke_index(repo / index_rel)
            owners = _complete_recipe_owners(repo_root=repo, index_path=index_rel)
            complete_fps = set(owners)

            pending_fps: dict[str, str] = {}
            for sid in pending_train_ids:
                entry = find_smoke_entry(index, sid)
                self.assertEqual(entry.get("status"), "gpu_pending", msg=sid)
                fp = index_entry_recipe_fingerprint(entry, repo=repo)
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
                enriched = enrich_aug_smoke_job_from_index(job, repo=repo)
                self.assertEqual(
                    _job_train_recipe_fingerprint(enriched, repo_root=repo),
                    pending_fps[sid],
                )

    def test_expand_aug_smoke_skips_audit_only_gpu_pending(self) -> None:
        from harchoc.gpu_queue import expand_aug_smoke_jobs_from_index

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "tests") as td:
            index_rel = write_pending_fixture_index(
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
            index_rel = write_pending_fixture_index(repo, Path(td), ("S4",))
            index = load_aug_smoke_index(repo / index_rel)
            job = next(
                j
                for j in expand_aug_smoke_jobs_from_index(repo_root=repo, index_path=index_rel)
                if j.get("smoke_id") == "S4"
            )
            self.assertNotIn("train_config", job)
            expected = index_entry_recipe_fingerprint(find_smoke_entry(index, "S4"), repo=repo)
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

        fp_s0 = index_entry_recipe_fingerprint(find_smoke_entry(index, "S0"), repo=repo)
        fp_s1 = index_entry_recipe_fingerprint(find_smoke_entry(index, "S1"), repo=repo)
        fp_s13 = index_entry_recipe_fingerprint(find_smoke_entry(index, "S13"), repo=repo)

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
        m = load_gpu_queue_manifest(repo / "configs/experiments/archive/gpu_queue_full.json", repo_root=repo)
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



if __name__ == "__main__":
    unittest.main()
