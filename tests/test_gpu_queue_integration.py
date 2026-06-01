"""Tests for harchoc.gpu_queue (CI-safe, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._gpu_queue_fixtures import load_manifest_with_index, write_pending_fixture_index

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
        index_rel = write_pending_fixture_index(
            repo,
            tmp_dir,
            pending_ids,
            include_equivalence_classes=include_equivalence_classes,
        )
        return load_manifest_with_index(
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


if __name__ == "__main__":
    unittest.main()
