import os
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


class ExternalDetectorTrainOverlayTests(unittest.TestCase):
    def test_overlay_dfine_uses_epochs_not_epoches(self) -> None:
        from harchoc.external_detector_train import write_train_overlay_yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "dfine_hgnetv2_l_coco.yml"
            upstream.write_text("epochs: 80\n", encoding="utf-8")
            coco = root / "coco"
            for split in ("train", "val"):
                (coco / "images" / split).mkdir(parents=True)
                (coco / "annotations").mkdir(parents=True, exist_ok=True)
            (coco / "annotations" / "instances_train.json").write_text("{}", encoding="utf-8")
            (coco / "annotations" / "instances_val.json").write_text("{}", encoding="utf-8")
            out = root / "overlay.yml"
            write_train_overlay_yaml(
                out_path=out,
                upstream_config=upstream,
                coco_root=coco,
                output_dir=root / "out",
                epochs=100,
                imgsz=1280,
                train_stack="dfine",
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("epochs: 100", text)
            self.assertNotIn("epoches:", text)
            self.assertIn("eval_spatial_size: [1280, 1280]", text)
            self.assertIn("base_size: 1280", text)
            self.assertIn("stop_epoch: 90", text)

    def test_overlay_deim_scales_schedule_to_target_epochs(self) -> None:
        from harchoc.external_detector_train import write_train_overlay_yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "deim_hgnetv2_l_coco.yml"
            upstream.write_text("epoches: 58\n", encoding="utf-8")
            coco = root / "coco"
            for split in ("train", "val"):
                (coco / "images" / split).mkdir(parents=True)
                (coco / "annotations").mkdir(parents=True, exist_ok=True)
            (coco / "annotations" / "instances_train.json").write_text("{}", encoding="utf-8")
            (coco / "annotations" / "instances_val.json").write_text("{}", encoding="utf-8")
            out = root / "overlay.yml"
            write_train_overlay_yaml(
                out_path=out,
                upstream_config=upstream,
                coco_root=coco,
                output_dir=root / "out",
                epochs=100,
                imgsz=1280,
                train_stack="deim",
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("epoches: 100", text)
            self.assertIn("flat_epoch: 50", text)
            self.assertIn("stop_epoch: 86", text)
            self.assertIn("mixup_epochs: [7, 50]", text)
            self.assertIn("mosaic_prob: 0", text)
            self.assertNotIn("Mosaic", text)
            self.assertIn("epoch: 86", text)
            self.assertNotIn("epoch: [", text)

    def test_overlay_rtdetrv2_uses_scales_not_base_size(self) -> None:
        from harchoc.external_detector_train import write_train_overlay_yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "rtdetrv2_r50vd_6x_coco.yml"
            upstream.write_text("epoches: 72\n", encoding="utf-8")
            coco = root / "coco"
            for split in ("train", "val"):
                (coco / "images" / split).mkdir(parents=True)
                (coco / "annotations").mkdir(parents=True, exist_ok=True)
            (coco / "annotations" / "instances_train.json").write_text("{}", encoding="utf-8")
            (coco / "annotations" / "instances_val.json").write_text("{}", encoding="utf-8")
            out = root / "overlay.yml"
            write_train_overlay_yaml(
                out_path=out,
                upstream_config=upstream,
                coco_root=coco,
                output_dir=root / "out",
                epochs=100,
                imgsz=1280,
                train_stack="rtdetrv2_pytorch",
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("scales:", text)
            self.assertNotIn("base_size:", text)
            self.assertIn("Resize, size: [1280, 1280]", text)

    def test_cli_updates_use_stack_epoch_field(self) -> None:
        from harchoc.external_detector_train import build_train_cli_updates

        dfine = build_train_cli_updates(
            train_stack="dfine",
            upstream_config=Path("configs/dfine/dfine_hgnetv2_l_coco.yml"),
            epochs=100,
            imgsz=1280,
        )
        self.assertTrue(any(u.startswith("epochs=100") for u in dfine))
        self.assertFalse(any(u.startswith("epoches=") for u in dfine))

        deim = build_train_cli_updates(
            train_stack="deim",
            upstream_config=Path("configs/deim_dfine/deim_hgnetv2_l_coco.yml"),
            epochs=100,
            imgsz=1280,
        )
        self.assertTrue(any(u.startswith("epoches=100") for u in deim))

    def test_rtdetrv2_collate_scales_double_640_defaults(self) -> None:
        from harchoc.external_detector_train import _rtdetrv2_collate_scales

        scales = _rtdetrv2_collate_scales(1280)
        self.assertEqual(scales[5], 1280)
        self.assertTrue(all(s % 32 == 0 for s in scales))


class DeimDataloaderSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("HARCHOC_DEIM_DATALOADER_SMOKE") == "1",
        "set HARCHOC_DEIM_DATALOADER_SMOKE=1 with DEIM repo + COCO export",
    )
    def test_deim_overlay_transforms_one_sample(self) -> None:
        from harchoc.external_detector_train import verify_external_train_smoke
        from harchoc.external_repos import resolve_external_repo_path

        repo = resolve_external_repo_path("deim")
        if repo is None:
            self.skipTest("DEIM repo not present")
        overlay = Path(
            os.environ.get(
                "HARCHOC_DEIM_OVERLAY",
                _REPO_ROOT
                / "runs/hsp_zoo/deim_dfine_l_e100_s0/harchoc_train_overlay.yml",
            )
        )
        if not overlay.is_file():
            self.skipTest(f"overlay missing: {overlay}")
        ok, reason = verify_external_train_smoke(
            overlay_path=overlay, repo=repo, train_stack="deim", epoch=0
        )
        self.assertTrue(ok, reason)


class ExternalTrainSmokeAllStacksTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("HARCHOC_EXTERNAL_DATALOADER_SMOKE") == "1",
        "set HARCHOC_EXTERNAL_DATALOADER_SMOKE=1 with repos + COCO export",
    )
    def test_all_zoo_external_stacks_smoke(self) -> None:
        from harchoc.detector_sources import entry_for_bench
        from harchoc.external_detector_train import (
            verify_external_train_smoke,
            write_train_overlay_yaml,
        )
        from harchoc.external_repos import resolve_external_repo_path

        coco = Path(os.environ["HARCHOC_EXTERNAL_COCO_ROOT"])
        cases = (
            ("deim_dfine_l", "deim"),
            ("deim_rtdetrv2_l", "deim"),
            ("dfine_l", "dfine"),
            ("rtdetrv2_l", "rtdetrv2_pytorch"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for source_id, stack in cases:
                entry = entry_for_bench(model_id=source_id, source_id=source_id)
                self.assertIsNotNone(entry)
                assert entry is not None
                repo = resolve_external_repo_path(stack)
                self.assertIsNotNone(repo)
                assert repo is not None
                upstream = (repo / entry.config_relpath).resolve()
                overlay = Path(tmp) / f"{source_id}.yml"
                write_train_overlay_yaml(
                    out_path=overlay,
                    upstream_config=upstream,
                    coco_root=coco,
                    output_dir=Path(tmp) / f"{source_id}_out",
                    epochs=100,
                    imgsz=1280,
                    train_stack=stack,
                    batch=1,
                )
                ok, reason = verify_external_train_smoke(
                    overlay_path=overlay,
                    repo=repo,
                    train_stack=stack,
                    epoch=0,
                )
                self.assertTrue(ok, f"{source_id}: {reason}")


if __name__ == "__main__":
    unittest.main()
