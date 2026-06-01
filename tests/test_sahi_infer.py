from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from harchoc.sahi_infer import (
    SahiSliceConfig,
    load_ultralytics_detection_model,
    model_confidence_min_from_env,
    run_sliced_prediction,
)


class SahiInferTests(unittest.TestCase):
    def test_slice_config_from_env(self) -> None:
        env = {"SLICE_SIZE": "640", "OVERLAP": "0.28", "NMS_IOU": "0.45"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = SahiSliceConfig.from_env()
        self.assertEqual(cfg.slice_size, 640)
        self.assertAlmostEqual(cfg.overlap, 0.28)
        self.assertAlmostEqual(cfg.nms_iou, 0.45)

    def test_model_confidence_min_from_env(self) -> None:
        env = {
            "CONF_THR": "",
            "CONF_THR_FERTILIZED": "0.06",
            "CONF_THR_UNFERTILIZED": "0.04",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertAlmostEqual(model_confidence_min_from_env(), 0.04)

    def test_run_sliced_prediction_forwards_kwargs(self) -> None:
        cfg = SahiSliceConfig(slice_size=512, overlap=0.32, nms_iou=0.58)
        model = object()
        gsp = mock.MagicMock(return_value="result")
        fake_predict = mock.MagicMock(get_sliced_prediction=gsp)
        fake_sahi = mock.MagicMock(predict=fake_predict)
        with mock.patch.dict(
            sys.modules,
            {"sahi": fake_sahi, "sahi.predict": fake_predict},
        ):
            out = run_sliced_prediction("/tmp/img.jpg", model, cfg)
        self.assertEqual(out, "result")
        gsp.assert_called_once_with(
            image="/tmp/img.jpg",
            detection_model=model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.32,
            overlap_width_ratio=0.32,
            postprocess_type="NMS",
            postprocess_match_threshold=0.58,
        )

    def test_load_ultralytics_detection_model(self) -> None:
        fake_model = object()
        adm = mock.MagicMock()
        adm.from_pretrained.return_value = fake_model
        fake_sahi = mock.MagicMock(AutoDetectionModel=adm)
        with mock.patch.dict(sys.modules, {"sahi": fake_sahi}):
            out = load_ultralytics_detection_model(
                "/models/best2.pt",
                device="cpu",
                confidence_threshold=0.05,
            )
        self.assertIs(out, fake_model)
        adm.from_pretrained.assert_called_once_with(
            model_type="ultralytics",
            model_path="/models/best2.pt",
            confidence_threshold=0.05,
            device="cpu",
        )


    def test_module_importable(self) -> None:
        import harchoc.sahi_infer as si

        self.assertTrue(callable(si.run_sliced_prediction))


if __name__ == "__main__":
    unittest.main()
