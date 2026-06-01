import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class RunMetadataTests(unittest.TestCase):
    def test_collect_run_metadata_includes_file_hashes(self) -> None:
        from harchoc.run_metadata import collect_run_metadata

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            repo_root = tdp / "repo"
            repo_root.mkdir(parents=True, exist_ok=True)

            mf = repo_root / "data" / "manifest.json"
            mf.parent.mkdir(parents=True, exist_ok=True)
            mf.write_text('{"datasets":[{"name":"x","extracted_paths":["data/x"]}]}', "utf-8")

            w = repo_root / "models" / "best2.pt"
            w.parent.mkdir(parents=True, exist_ok=True)
            w.write_bytes(b"weights")

            meta = collect_run_metadata(
                repo_root=repo_root,
                dataset_manifest=mf,
                extra_files={"weights": w},
            )

            self.assertIn("python", meta)
            self.assertIn("platform", meta)
            self.assertIn("files", meta)
            self.assertEqual(meta["files"]["dataset_manifest"]["exists"], True)
            self.assertEqual(meta["files"]["weights"]["exists"], True)
            self.assertIsInstance(meta["files"]["dataset_manifest"].get("sha256"), str)
            self.assertIsInstance(meta["files"]["weights"].get("sha256"), str)

            # No .git in temp repo root => git should be None.
            self.assertIsNone(meta.get("git"))

    def test_collect_run_metadata_optional_repo_splits(self) -> None:
        from harchoc.run_metadata import collect_run_metadata

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            splits = repo_root / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "test.txt").write_text("images/test/x.jpg\n", "utf-8")

            meta = collect_run_metadata(repo_root=repo_root, include_repo_splits=True)
            self.assertIn("repo_splits", meta)
            self.assertTrue(meta["repo_splits"]["files"]["test"]["exists"])

    def test_try_git_info_strict_records_commit_failure(self) -> None:
        from harchoc.run_metadata import _try_git_info

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".git").mkdir()
            warnings: list[str] = []
            with mock.patch(
                "harchoc.run_metadata.subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, "git"),
            ):
                info = _try_git_info(repo, warnings=warnings)
            self.assertIsNone(info)
            self.assertEqual(len(warnings), 1)
            self.assertIn("git rev-parse HEAD", warnings[0])


if __name__ == "__main__":
    unittest.main()

