import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")


class DetectorSourcesTests(unittest.TestCase):
    def test_registry_loads_four_entries(self) -> None:
        from harchoc.detector_sources import (
            default_detector_sources_path,
            load_detector_sources,
            load_train_stack_metadata,
        )

        path = default_detector_sources_path()
        entries = load_detector_sources(path)
        stacks = load_train_stack_metadata(path)
        self.assertIn("rtdetrv2_l", entries)
        self.assertIn("dfine_l", entries)
        self.assertIn("deim_rtdetrv2_l", entries)
        self.assertIn("deim_dfine_l", entries)
        self.assertIn("deim", stacks)
        self.assertIn("dfine", stacks)
        self.assertIn("rtdetrv2_pytorch", stacks)
        self.assertTrue(entries["dfine_l"].checkpoint_url.startswith("https://github.com/"))

    def test_default_external_weights_dir_uses_repo_root(self) -> None:
        from harchoc.detector_sources import _repo_root, default_external_weights_dir

        old = os.environ.pop("EXTERNAL_WEIGHTS_CACHE_DIR", None)
        try:
            p = default_external_weights_dir()
            self.assertEqual(p, (_repo_root() / "data/weights/external").resolve())
        finally:
            if old is not None:
                os.environ["EXTERNAL_WEIGHTS_CACHE_DIR"] = old

    def test_external_entry_provenance(self) -> None:
        from harchoc.detector_sources import external_entry_provenance, load_detector_sources

        entry = load_detector_sources()["dfine_l"]
        with tempfile.TemporaryDirectory() as td:
            with patch(
                "harchoc.detector_sources.default_external_weights_dir",
                return_value=Path(td),
            ):
                prov = external_entry_provenance(entry)
            self.assertEqual(prov["source_id"], "dfine_l")
            self.assertEqual(prov["train_stack"], "dfine")
            self.assertFalse(prov["exists"])

    def test_bench_external_provenance(self) -> None:
        from harchoc.bench_config import bench_external_provenance, load_bench_config

        repo = Path(__file__).resolve().parents[1]
        cfg = load_bench_config(repo / "configs" / "bench" / "dfine_l_default.yaml")
        with tempfile.TemporaryDirectory() as td:
            with patch(
                "harchoc.detector_sources.default_external_weights_dir",
                return_value=Path(td),
            ):
                prov = bench_external_provenance(cfg)
            self.assertIsNotNone(prov)
            assert prov is not None
            self.assertEqual(prov["source_id"], "dfine_l")
            self.assertIn("checkpoint_url", prov)

    def test_download_http_checkpoint(self) -> None:
        from harchoc.detector_sources import DetectorSourceEntry, download_external_checkpoint

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "dfine_l_coco.pth"

            def fake_http(url: str, path: Path) -> None:
                self.assertIn("dfine_l_coco.pth", url)
                path.write_bytes(b"fake")

            entry = DetectorSourceEntry(
                source_id="dfine_l",
                label="D-FINE-L",
                family="dfine",
                train_stack="dfine",
                repos={"primary": "https://github.com/Peterande/D-FINE"},
                config_relpath="configs/dfine/dfine_hgnetv2_l_coco.yml",
                checkpoint_url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth",
                checkpoint_cache_name="dfine_l_coco.pth",
                gdown_id=None,
                baseline_source_id=None,
            )
            with patch(
                "harchoc.detector_sources.default_external_weights_dir",
                return_value=Path(td),
            ):
                with patch("harchoc.detector_sources._download_http", side_effect=fake_http):
                    info = download_external_checkpoint(entry)
            self.assertTrue(dest.is_file())
            self.assertTrue(info.get("downloaded"))

    def test_external_availability_requires_repo(self) -> None:
        from harchoc.external_detector_train import external_bench_availability

        with tempfile.TemporaryDirectory() as td:
            weights = Path(td) / "weights"
            weights.mkdir()
            partial_repo = Path(td) / "partial_dfine"
            partial_repo.mkdir()

            with patch(
                "harchoc.detector_sources.default_external_weights_dir",
                return_value=weights,
            ):
                (weights / "dfine_l_coco.pth").write_bytes(b"ok")
                with patch(
                    "harchoc.external_detector_train.resolve_external_repo_path",
                    return_value=None,
                ):
                    ok, reason = external_bench_availability(
                        model_id="dfine_l",
                        source_id="dfine_l",
                    )
                self.assertFalse(ok)
                self.assertIn("missing_repo", reason or "")
                self.assertIn("check_weights_cache", reason or "")

                with patch(
                    "harchoc.external_detector_train.resolve_external_repo_path",
                    return_value=partial_repo,
                ):
                    ok, reason = external_bench_availability(
                        model_id="dfine_l",
                        source_id="dfine_l",
                    )
                self.assertFalse(ok)
                self.assertIn("invalid_repo", reason or "")

                (weights / "dfine_l_coco.pth").unlink()
                with patch(
                    "harchoc.external_detector_train.resolve_external_repo_path",
                    return_value=partial_repo,
                ):
                    ok, reason = external_bench_availability(
                        model_id="dfine_l",
                        source_id="dfine_l",
                    )
                self.assertFalse(ok)
                self.assertIn("checkpoint_not_cached", reason or "")

    def test_bench_external_validation_unknown_source(self) -> None:
        from harchoc.bench_config import load_bench_config

        repo = Path(__file__).resolve().parents[1]
        good = repo / "configs" / "bench" / "dfine_l_default.yaml"
        load_bench_config(good)

        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.yaml"
            bad.write_text(
                "name: bad\nbackend: external\nsource_id: not_a_real_id\nmodel_id: not_a_real_id\n"
                "groups: zoo_core\ninfer:\n  imgsz: 1280\nepochs: 100\npatience: 50\nseed: 0\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                load_bench_config(bad)


if __name__ == "__main__":
    unittest.main()
