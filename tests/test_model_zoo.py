import os
import tempfile
import unittest
from pathlib import Path


class ModelZooTests(unittest.TestCase):
    def test_backend_availability_external_requires_source(self) -> None:
        from harchoc.model_zoo import backend_availability

        ok, reason = backend_availability("external")
        self.assertFalse(ok)
        self.assertEqual(reason, "external_requires_source_id")

    def test_ultralytics_identifier_resolves_to_cache_path(self) -> None:
        from harchoc.model_zoo import resolve_weights_ref

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "weights_cache"
            old = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(cache_dir)
                res = resolve_weights_ref(backend="ultralytics", model="yolov8n.pt")
            finally:
                if old is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old

            self.assertEqual(res.kind, "ultralytics_id")
            self.assertIsNotNone(res.cache_path)
            assert res.cache_path is not None
            self.assertTrue(str(res.cache_path).endswith("weights_cache/ultralytics/yolov8n.pt"))
            self.assertFalse(res.exists)
            self.assertEqual(res.resolution, "not_cached")

    def test_ultralytics_file_path_resolves_without_cache(self) -> None:
        from harchoc.model_zoo import resolve_weights_ref

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "w.pt"
            p.write_bytes(b"fake")
            res = resolve_weights_ref(backend="ultralytics", model=str(p))
            self.assertEqual(res.kind, "file_path")
            self.assertIsNone(res.cache_path)
            self.assertIsNotNone(res.resolved_path)
            assert res.resolved_path is not None
            self.assertEqual(res.resolved_path, p.resolve())
            self.assertTrue(res.exists)
            self.assertEqual(res.resolution, "existing_file")


    def test_download_hints_use_assets_url(self) -> None:
        from harchoc.model_zoo import download_hints_for_cache_path, ultralytics_assets_url

        url = ultralytics_assets_url("yolov8n.pt")
        self.assertIn("github.com/ultralytics/assets", url)
        self.assertIn("yolov8n.pt", url)
        self.assertIn("v8.4.0", url)
        self.assertIn(
            "v8.4.0",
            ultralytics_assets_url("yolo26n.pt", assets_tag="v8.4.0"),
        )
        hints = download_hints_for_cache_path(
            cache_path=Path("data/weights/ultralytics/yolov8n.pt"),
            identifier="yolov8n.pt",
        )
        self.assertIn("wget", hints)
        self.assertEqual(hints["url"], url)

    def test_load_weights_manifest_missing_file(self) -> None:
        from harchoc.model_zoo import WEIGHTS_MANIFEST_SCHEMA, load_weights_manifest

        with tempfile.TemporaryDirectory() as td:
            mf = Path(td) / "weights_manifest.json"
            manifest = load_weights_manifest(mf)
            self.assertEqual(manifest["schema_version"], WEIGHTS_MANIFEST_SCHEMA)
            self.assertEqual(manifest["entries"], {})

    def test_sync_and_verify_weights_manifest_sha(self) -> None:
        from harchoc.model_zoo import (
            file_sha256,
            load_weights_manifest,
            sync_weights_manifest_from_report_entries,
            verify_weights_manifest,
            write_weights_manifest,
        )

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "weights" / "ultralytics"
            cache.mkdir(parents=True)
            weight = cache / "yolov8n.pt"
            weight.write_bytes(b"fake-wt")
            sha = file_sha256(weight)
            mf = Path(td) / "weights_manifest.json"

            sync_weights_manifest_from_report_entries(
                mf,
                entries=[
                    {
                        "identifier": "yolov8n.pt",
                        "exists": True,
                        "cache_path": str(weight),
                        "sha256": sha,
                        "size_bytes": weight.stat().st_size,
                        "bench_configs": ["one.yaml"],
                    }
                ],
            )
            loaded = load_weights_manifest(mf)
            entry = loaded["entries"]["yolov8n.pt"]  # type: ignore[index]
            self.assertEqual(entry["sha256"], sha)
            self.assertEqual(verify_weights_manifest(mf), [])

            entry["sha256"] = "0" * 64
            write_weights_manifest(mf, loaded)
            issues = verify_weights_manifest(mf)
            self.assertTrue(any("sha256 mismatch" in i for i in issues))

    def test_verify_weights_manifest_missing_cache_file(self) -> None:
        from harchoc.model_zoo import verify_weights_manifest, write_weights_manifest

        with tempfile.TemporaryDirectory() as td:
            mf = Path(td) / "weights_manifest.json"
            write_weights_manifest(
                mf,
                {
                    "schema_version": "weights_manifest.v1",
                    "entries": {
                        "yolov8n.pt": {
                            "cache_path": str(Path(td) / "missing.pt"),
                            "sha256": "a" * 64,
                        }
                    },
                },
            )
            issues = verify_weights_manifest(mf)
            self.assertTrue(any("cache file missing" in i for i in issues))


if __name__ == "__main__":
    unittest.main()

