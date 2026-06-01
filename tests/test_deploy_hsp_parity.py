"""Tests for deploy vs HSP parity report."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class DeployHspParityTests(unittest.TestCase):
    def test_build_payload_schema(self) -> None:
        from harchoc.deploy_hsp_parity import build_deploy_hsp_parity_payload

        with tempfile.TemporaryDirectory() as td:
            thresh = Path(td) / "threshold_val.json"
            thresh.write_text(
                json.dumps(
                    {
                        "schema_version": "threshold_sweep_run.v1",
                        "selected": {"row": {"conf_thr": 0.15}},
                    }
                ),
                encoding="utf-8",
            )
            old = os.environ.get("HARCHOC_LOCKED_CONF")
            os.environ.pop("HARCHOC_LOCKED_CONF", None)
            os.environ.pop("HARCHOC_LOCKED_CONF_JSON", None)
            try:
                payload = build_deploy_hsp_parity_payload(locked_conf_from=str(thresh))
            finally:
                if old is not None:
                    os.environ["HARCHOC_LOCKED_CONF"] = old
            self.assertEqual(payload["schema_version"], "deploy_hsp_parity.v1")
            self.assertIn("deploy_conf", payload)
            self.assertIn("sahi_slice", payload)
            self.assertEqual(payload["hsp_locked_conf"]["conf"], 0.15)

    def test_build_payload_with_per_image(self) -> None:
        from harchoc.deploy_hsp_parity import build_deploy_hsp_parity_payload

        per_image = [
            {
                "image_path": "/tmp/a.jpg",
                "locked_conf": 0.15,
                "sahi_count": {"developed": 10, "aborted": 2, "total": 12},
                "hsp_fullframe_locked": {"developed": 8, "aborted": 1, "total": 9},
                "delta_total": 3,
            }
        ]
        payload = build_deploy_hsp_parity_payload(
            locked_conf_from="reports/hsp/threshold_val.json",
            image_paths=["/tmp/a.jpg"],
            per_image=per_image,
        )
        self.assertEqual(len(payload["per_image"]), 1)
        self.assertEqual(payload["image_sample_summary"]["n_compared"], 1)
        self.assertEqual(payload["per_image"][0]["delta_total"], 3)

    def test_build_per_image_parity_rows_mocked(self) -> None:
        from harchoc.deploy_hsp_parity import build_per_image_parity_rows

        def sahi(_path: str) -> dict[str, int]:
            return {"developed": 5, "aborted": 1, "total": 6}

        def hsp(_path: str) -> dict[str, int]:
            return {"developed": 4, "aborted": 0, "total": 4}

        rows = build_per_image_parity_rows(
            ["/fake/img.jpg"],
            locked_conf=0.15,
            weights="models/best2.pt",
            sahi_count_fn=sahi,
            fullframe_count_fn=hsp,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sahi_count"]["total"], 6)
        self.assertEqual(rows[0]["hsp_fullframe_locked"]["total"], 4)
        self.assertEqual(rows[0]["delta_total"], 2)

    def test_resolve_sample_skips_without_torch(self) -> None:
        import builtins

        from harchoc.deploy_hsp_parity import resolve_parity_image_sample

        real_import = builtins.__import__

        def mock_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "torch":
                raise ImportError("mocked no torch")
            return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            img = root / "images" / "test" / "a.jpg"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            split = Path(td) / "test.txt"
            split.write_text("images/test/a.jpg\n", encoding="utf-8")
            thresh = Path(td) / "threshold_val.json"
            thresh.write_text(
                json.dumps(
                    {
                        "schema_version": "threshold_sweep_run.v1",
                        "selected": {"row": {"conf_thr": 0.15}},
                    }
                ),
                encoding="utf-8",
            )
            weights = Path(td) / "best.pt"
            weights.write_bytes(b"stub")

            with mock.patch("builtins.__import__", side_effect=mock_import):
                paths, per_image, note = resolve_parity_image_sample(
                    sample_images=1,
                    split_file=split,
                    dataset_root=root,
                    locked_conf_from=str(thresh),
                    weights=weights,
                )

        self.assertEqual(len(paths), 1)
        self.assertIsNone(per_image)
        self.assertIn("torch", note or "")

    def test_resolve_sample_uses_injected_counters(self) -> None:
        from harchoc.deploy_hsp_parity import resolve_parity_image_sample

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            img = root / "images" / "test" / "a.jpg"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            split = Path(td) / "test.txt"
            split.write_text("images/test/a.jpg\n", encoding="utf-8")
            thresh = Path(td) / "threshold_val.json"
            thresh.write_text(
                json.dumps(
                    {
                        "schema_version": "threshold_sweep_run.v1",
                        "selected": {"row": {"conf_thr": 0.15}},
                    }
                ),
                encoding="utf-8",
            )
            weights = Path(td) / "best.pt"
            weights.write_bytes(b"stub")

            paths, per_image, note = resolve_parity_image_sample(
                sample_images=1,
                split_file=split,
                dataset_root=root,
                locked_conf_from=str(thresh),
                weights=weights,
                sahi_count_fn=lambda _p: {"developed": 3, "aborted": 1, "total": 4},
                fullframe_count_fn=lambda _p: {"developed": 2, "aborted": 1, "total": 3},
            )

        self.assertIsNone(note)
        self.assertEqual(len(paths), 1)
        self.assertIsNotNone(per_image)
        assert per_image is not None
        self.assertEqual(per_image[0]["delta_total"], 1)

    def test_experiment_argv_deploy_parity(self) -> None:
        from harchoc.experiment_argv import argv_for_deploy_parity

        argv = argv_for_deploy_parity(
            {
                "locked_conf_from": "reports/hsp/threshold_val.json",
                "out": "reports/hsp/x.json",
                "sample_images": 3,
            }
        )
        self.assertIn("--locked-conf-from", argv)
        self.assertIn("reports/hsp/x.json", argv)
        self.assertIn("--sample-images", argv)
        self.assertIn("3", argv)
        self.assertIn("--split-file", argv)
        self.assertIn("data/splits/test.txt", argv)


if __name__ == "__main__":
    unittest.main()
