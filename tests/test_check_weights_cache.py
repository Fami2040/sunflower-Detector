import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harchoc.bench_assets import build_report
from scripts.check_weights_cache import main as check_main


class CheckWeightsCacheTests(unittest.TestCase):
    def test_lists_ultralytics_ids_from_bench_dir(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        report = build_report(bench_dir=repo_root / "configs" / "bench", pattern="*.yaml")
        ids = {e["identifier"] for e in report["identifiers"]}  # type: ignore[index]
        self.assertIn("yolov8n.pt", ids)
        self.assertIn("yolo11s.pt", ids)
        self.assertNotIn("yolo_nas_s", ids)

    def test_missing_cache_includes_download_hints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench"
            bench.mkdir()
            (bench / "one.yaml").write_text(
                "\n".join(
                    [
                        "name: one",
                        "backend: ultralytics",
                        "model: yolov8n.pt",
                        "epochs: 1",
                        "",
                    ]
                ),
                "utf-8",
            )
            old = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(Path(td) / "weights")
                report = build_report(bench_dir=bench, pattern="*.yaml")
            finally:
                if old is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old

            self.assertEqual(report["summary"]["missing"], 1)  # type: ignore[index]
            entry = report["identifiers"][0]  # type: ignore[index]
            self.assertFalse(entry["exists"])
            self.assertIn("download_hints", entry)
            self.assertIn("wget", entry["download_hints"])
            self.assertIn("yolov8n.pt", entry["download_hints"]["url"])

    def test_cached_identifier_reports_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench"
            bench.mkdir()
            (bench / "one.yaml").write_text(
                "name: one\nbackend: ultralytics\nmodel: yolov8n.pt\nepochs: 1\n",
                "utf-8",
            )
            cache = Path(td) / "weights" / "ultralytics"
            cache.mkdir(parents=True)
            (cache / "yolov8n.pt").write_bytes(b"fake")

            old = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(Path(td) / "weights")
                report = build_report(bench_dir=bench, pattern="*.yaml")
            finally:
                if old is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old

            self.assertEqual(report["summary"]["missing"], 0)  # type: ignore[index]
            self.assertTrue(report["identifiers"][0]["exists"])  # type: ignore[index]

    def test_ci_does_not_fail_on_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench"
            bench.mkdir()
            (bench / "one.yaml").write_text(
                "name: one\nbackend: ultralytics\nmodel: yolov8n.pt\nepochs: 1\n",
                "utf-8",
            )
            old_cache = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(Path(td) / "weights")
                with patch.dict(os.environ, {"CI": "1"}, clear=False):
                    rc = check_main(["--bench-dir", str(bench), "--warn-only"])
                self.assertEqual(rc, 0)
            finally:
                if old_cache is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old_cache

    def test_strict_fails_when_cached_but_not_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench"
            bench.mkdir()
            (bench / "one.yaml").write_text(
                "name: one\nbackend: ultralytics\nmodel: yolov8n.pt\nepochs: 1\n",
                "utf-8",
            )
            weights = Path(td) / "weights"
            cache = weights / "ultralytics"
            cache.mkdir(parents=True)
            (cache / "yolov8n.pt").write_bytes(b"fake")
            (weights / "weights_manifest.json").write_text(
                json.dumps({"schema_version": "weights_manifest.v1", "entries": {}}),
                "utf-8",
            )

            old_cache = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(weights)
                os.environ.pop("CI", None)
                rc = check_main(["--bench-dir", str(bench), "--strict"])
                self.assertEqual(rc, 1)
                report = build_report(
                    bench_dir=bench,
                    pattern="*.yaml",
                    manifest_path=weights / "weights_manifest.json",
                    check_manifest=True,
                )
                self.assertEqual(report["summary"]["missing"], 0)  # type: ignore[index]
                self.assertEqual(report["summary"]["missing_from_manifest"], 1)  # type: ignore[index]
            finally:
                if old_cache is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old_cache

    def test_strict_exits_nonzero_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench"
            bench.mkdir()
            (bench / "one.yaml").write_text(
                "name: one\nbackend: ultralytics\nmodel: yolov8n.pt\nepochs: 1\n",
                "utf-8",
            )
            old_cache = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(Path(td) / "weights")
                os.environ.pop("CI", None)
                rc = check_main(["--bench-dir", str(bench), "--strict"])
                self.assertEqual(rc, 1)
            finally:
                if old_cache is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old_cache


    def test_download_populates_cache(self) -> None:
        from harchoc.model_zoo import download_ultralytics_weight

        with tempfile.TemporaryDirectory() as td:
            bench = Path(td) / "bench"
            bench.mkdir()
            (bench / "one.yaml").write_text(
                "name: one\nbackend: ultralytics\nmodel: yolov8n.pt\nepochs: 1\n",
                "utf-8",
            )
            cache = Path(td) / "weights" / "ultralytics"
            cache.mkdir(parents=True)
            old_cache = os.environ.get("WEIGHTS_CACHE_DIR")
            try:
                os.environ["WEIGHTS_CACHE_DIR"] = str(Path(td) / "weights")
                manifest = Path(td) / "weights" / "weights_manifest.json"
                def _fake_dl(*, identifier: str, cache_path: Path, **_: object) -> dict[str, object]:
                    from harchoc.model_zoo import file_sha256

                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(b"fake-wt")
                    return {
                        "identifier": identifier,
                        "downloaded": True,
                        "cache_path": str(cache_path),
                        "sha256": file_sha256(cache_path),
                        "size_bytes": cache_path.stat().st_size,
                    }

                with patch(
                    "harchoc.bench_assets.download_ultralytics_weight",
                    side_effect=_fake_dl,
                ):
                    report = build_report(
                        bench_dir=bench,
                        pattern="*.yaml",
                        download=True,
                        manifest_path=manifest,
                    )
                self.assertEqual(report["summary"]["missing"], 0)  # type: ignore[index]
                self.assertTrue((cache / "yolov8n.pt").is_file())
                self.assertTrue(manifest.is_file())
                mf = json.loads(manifest.read_text("utf-8"))
                self.assertIn("yolov8n.pt", mf["entries"])
                self.assertEqual(
                    mf["entries"]["yolov8n.pt"]["sha256"],
                    report["identifiers"][0]["sha256"],  # type: ignore[index]
                )
                info = download_ultralytics_weight(
                    identifier="yolov8n.pt", cache_path=cache / "yolov8n.pt"
                )
                self.assertFalse(info["downloaded"])
            finally:
                if old_cache is None:
                    os.environ.pop("WEIGHTS_CACHE_DIR", None)
                else:
                    os.environ["WEIGHTS_CACHE_DIR"] = old_cache


if __name__ == "__main__":
    unittest.main()
