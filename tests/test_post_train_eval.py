from __future__ import annotations

import os
import unittest
from unittest import mock


class PostTrainEvalPolicyTests(unittest.TestCase):
    def test_skip_from_eval_section(self) -> None:
        from harchoc.post_train_eval import post_train_eval_skipped

        self.assertTrue(post_train_eval_skipped(cli_skip=False, eval_section={"skip": True}))
        self.assertFalse(post_train_eval_skipped(cli_skip=False, eval_section={"device": "cpu"}))
        self.assertTrue(post_train_eval_skipped(cli_skip=True, eval_section=None))

    def test_device_from_config(self) -> None:
        from harchoc.post_train_eval import resolve_post_train_eval_device

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_post_train_eval_device({"device": "cpu"}),
                "cpu",
            )

    def test_device_env_post_train_over_export(self) -> None:
        from harchoc.post_train_eval import resolve_post_train_eval_device

        with mock.patch.dict(
            os.environ,
            {"HARCHOC_POST_TRAIN_EVAL_DEVICE": "cpu", "HARCHOC_EXPORT_DEVICE": "cuda"},
            clear=True,
        ):
            self.assertEqual(resolve_post_train_eval_device(None), "cpu")

    def test_build_argv_forwards_device_and_max_det(self) -> None:
        from harchoc.post_train_eval import build_post_train_eval_argv

        argv = build_post_train_eval_argv(
            recorded_weights="/w/best.pt",
            eval_out="/tmp/eval.json",
            manifest="data/manifest.json",
            default_dataset_name="sunflower-cvat-1093",
            dataset_name=None,
            dataset_root=None,
            yolo_data_yaml=None,
            split_file="data/splits/test.txt",
            eval_section={"max_det": 3000, "device": "cpu"},
            train_imgsz=1280,
        )
        self.assertIn("--device", argv)
        self.assertIn("cpu", argv)
        self.assertIn("--export-device", argv)
        self.assertIn("--max-det", argv)
        self.assertIn("3000", argv)
        self.assertIn("--imgsz", argv)
    def test_restore_cuda_visible_after_ultralytics_cpu_clears_mask(self) -> None:
        from harchoc.post_train_eval import restore_cuda_visible_devices_after_ultralytics_cpu

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            restore_cuda_visible_devices_after_ultralytics_cpu(None)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            restore_cuda_visible_devices_after_ultralytics_cpu("0")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0")

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            restore_cuda_visible_devices_after_ultralytics_cpu("0")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0")


if __name__ == "__main__":
    unittest.main()
