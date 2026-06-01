import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")


class ExternalReposTests(unittest.TestCase):
    def test_derive_specs_from_detector_sources(self) -> None:
        from harchoc.detector_sources import load_detector_sources
        from harchoc.external_repos import derive_external_repo_specs

        registry = load_detector_sources()
        specs = derive_external_repo_specs(registry=registry)
        self.assertEqual(set(specs.keys()), {"deim", "dfine", "rtdetrv2_pytorch"})
        self.assertEqual(set(specs["deim"].source_ids), {"deim_dfine_l", "deim_rtdetrv2_l"})
        self.assertEqual(specs["dfine"].source_ids, ("dfine_l",))
        self.assertEqual(specs["rtdetrv2_pytorch"].subdir, "rtdetrv2_pytorch")
        self.assertTrue(specs["dfine"].url.endswith(".git"))

    def test_stacks_required_by_bench(self) -> None:
        from harchoc.external_repos import stacks_required_by_bench

        repo = Path(__file__).resolve().parents[1]
        bench_paths = [
            repo / "configs" / "bench" / "dfine_l_default.yaml",
            repo / "configs" / "bench" / "deim_dfine_l_default.yaml",
        ]
        stacks = stacks_required_by_bench(
            bench_dir=repo / "configs" / "bench",
            bench_config_paths=bench_paths,
        )
        self.assertEqual(stacks, {"dfine", "deim"})

    def test_validate_repo_layout_checks_config_paths(self) -> None:
        from harchoc.detector_sources import load_detector_sources
        from harchoc.external_repos import ExternalRepoSpec, validate_repo_layout

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / "DEIM"
            work.mkdir()
            (work / "train.py").write_text("# stub\n", encoding="utf-8")
            registry = load_detector_sources()
            entry = registry["deim_dfine_l"]
            cfg = work / entry.config_relpath
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text("task: detection\n", encoding="utf-8")
            spec = ExternalRepoSpec(
                key="deim",
                train_stack="deim",
                url="https://example.com/DEIM.git",
                ref="main",
                cache_dirname="DEIM",
                train_script="train.py",
                subdir=None,
                source_ids=("deim_dfine_l",),
                commit_pin=None,
            )
            issues = validate_repo_layout(
                spec, registry=registry, external_root=root
            )
            self.assertEqual(issues, [])

    def test_ensure_external_repo_clones_when_missing(self) -> None:
        from harchoc.external_repos import ExternalRepoSpec, ensure_external_repo

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            def fake_clone(cmd: list[str], **kwargs: object) -> object:
                dest = Path(cmd[-1])
                dest.mkdir(parents=True)
                (dest / "train.py").write_text("# t\n", encoding="utf-8")
                (dest / ".git").mkdir()
                proc = mock.MagicMock()
                proc.returncode = 0
                proc.stdout = "abc123\n"
                proc.stderr = ""
                return proc

            spec = ExternalRepoSpec(
                key="dfine",
                train_stack="dfine",
                url="https://example.com/D-FINE.git",
                ref="master",
                cache_dirname="D-FINE",
                train_script="train.py",
                subdir=None,
                source_ids=(),
                commit_pin=None,
            )
            with patch("subprocess.run", side_effect=fake_clone):
                with patch(
                    "harchoc.external_repos._git_head",
                    return_value="abc123",
                ):
                    with patch(
                        "harchoc.external_repos.validate_repo_layout",
                        return_value=[],
                    ):
                        row = ensure_external_repo(
                            spec, download=True, external_root=root
                        )
            self.assertTrue(row.get("valid"))


if __name__ == "__main__":
    unittest.main()
