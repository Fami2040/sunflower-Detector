from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestManuscriptRepro(unittest.TestCase):
    def test_build_chain_includes_core_steps(self) -> None:
        from harchoc.manuscript_repro import build_manuscript_repro_chain, load_manuscript_repro_bundle

        repo = Path(__file__).resolve().parents[1]
        bundle_path = repo / "configs/experiments/manuscript_repro_bundle.json"
        bundle = load_manuscript_repro_bundle(bundle_path)
        steps = build_manuscript_repro_chain(bundle, repo_root=repo, skip_gpu_check=True)
        ids = [s[0] for s in steps]
        self.assertEqual(
            ids,
            [
                "split_drift",
                "eval_val_export",
                "eval_test_export",
                "threshold_sweep_val",
                "threshold_sweep_test_locked",
                "error_analysis_val",
                "error_analysis_test",
                "dual_metric",
            ],
        )
        self.assertTrue(all(argv[0].endswith(".py") for _, argv in steps))

    def test_load_rejects_unknown_schema(self) -> None:
        from harchoc.manuscript_repro import load_manuscript_repro_bundle

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"schema_version": "nope"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manuscript_repro_bundle(p)

    def test_bundle_has_split_sha256(self) -> None:
        from harchoc.manuscript_repro import load_manuscript_repro_bundle
        from harchoc.run_metadata import collect_repo_split_files

        repo = Path(__file__).resolve().parents[1]
        bundle = load_manuscript_repro_bundle(repo / "configs/experiments/manuscript_repro_bundle.json")
        live = collect_repo_split_files(repo)
        for split in ("train", "val", "test"):
            self.assertEqual(
                bundle["repo_splits"]["files"][split]["sha256"],
                live["files"][split]["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
