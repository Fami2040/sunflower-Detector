import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harchoc.strict_ml import StrictWarnings


class EvalExportStrictTests(unittest.TestCase):
    def test_read_image_size_ok(self) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            raise unittest.SkipTest("PIL not installed")

        from harchoc.eval_export import read_image_size

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.png"
            Image.new("RGB", (40, 30), color=(0, 0, 0)).save(p)
            w, h = read_image_size(p)
        self.assertEqual((w, h), (40, 30))

    def test_read_image_size_fallback_records_warning(self) -> None:
        from harchoc.eval_export import read_image_size

        sw = StrictWarnings()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HARCHOC_STRICT_ML", None)
            w, h = read_image_size(
                Path("/no/such/sunflower-eval-export-test.png"),
                strict_warnings=sw,
            )
        self.assertEqual((w, h), (1, 1))
        self.assertEqual(len(sw.items), 1)
        self.assertEqual(sw.items[0]["code"], "pil_image_open_failed")

    def test_read_image_size_strict_raises(self) -> None:
        from harchoc.eval_export import read_image_size

        sw = StrictWarnings()
        with mock.patch.dict(os.environ, {"HARCHOC_STRICT_ML": "1"}):
            with self.assertRaises(RuntimeError):
                read_image_size(
                    Path("/no/such/sunflower-eval-export-test.png"),
                    strict_warnings=sw,
                )

    def test_ultralytics_box_parse_skips_invalid_row(self) -> None:
        from harchoc.eval_export import ultralytics_results_to_detections

        class _Boxes:
            xyxy = ["not-a-tensor", [1.0, 2.0, 3.0, 4.0]]
            cls = None
            conf = None

        class _Result:
            boxes = _Boxes()

        sw = StrictWarnings()
        dets = ultralytics_results_to_detections(_Result(), strict_warnings=sw)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sw.items[0]["code"], "ultralytics_box_parse")

    def test_build_gt_export_uses_read_image_size(self) -> None:
        from harchoc.eval_export import build_gt_export

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split = root / "split.txt"
            split.write_text("images/test/a.png\n", encoding="utf-8")
            img_dir = root / "images" / "test"
            img_dir.mkdir(parents=True)
            (img_dir / "a.png").write_bytes(b"not-a-png")
            sw = StrictWarnings()
            with mock.patch(
                "harchoc.eval_export.read_image_size",
                return_value=(100, 80),
            ) as ris:
                out = build_gt_export(
                    split_file=split,
                    dataset_root=root,
                    strict_warnings=sw,
                )
            ris.assert_called_once()
        self.assertEqual(len(out["images"]), 1)


class EvalScriptStrictTests(unittest.TestCase):
    def test_eval_dry_run_includes_strict_warnings(self) -> None:
        from scripts.eval import main

        with mock.patch("scripts.eval.write_json") as wj:
            wj.return_value = "out.json"
            rc = main(
                [
                    "--dry-run",
                    "--export-gt-json",
                    "reports/hsp/gt_test.json",
                    "--export-preds-json",
                    "reports/hsp/preds_test.json",
                ]
            )
        self.assertEqual(rc, 0)
        payload = wj.call_args[0][1]
        self.assertEqual(payload.get("strict_warnings"), [])

    def test_extract_ultralytics_metrics_missing_box_warns(self) -> None:
        from scripts.eval import _extract_ultralytics_metrics

        sw = StrictWarnings()
        m50, m95, per = _extract_ultralytics_metrics(object(), strict_warnings=sw)
        self.assertIsNone(m50)
        self.assertIsNone(m95)
        self.assertIsNone(per)
        self.assertEqual(sw.items[0]["code"], "ultralytics_metrics_missing_box")

    def test_extract_ultralytics_metrics_map50_parse_warns(self) -> None:
        from scripts.eval import _extract_ultralytics_metrics

        class _Box:
            map50 = "not-a-float"
            map = 0.5
            maps = None

        class _Res:
            box = _Box()

        sw = StrictWarnings()
        m50, m95, _ = _extract_ultralytics_metrics(_Res(), strict_warnings=sw)
        self.assertIsNone(m50)
        self.assertIsNotNone(m95)
        assert m95 is not None
        self.assertAlmostEqual(m95, 0.5)
        codes = [w["code"] for w in sw.items]
        self.assertIn("ultralytics_metrics_map50", codes)


class StrictWarningsTests(unittest.TestCase):
    def test_warn_accumulates(self) -> None:
        sw = StrictWarnings()
        sw.warn("code_a", "msg", raise_if_strict=False)
        self.assertEqual(len(sw.as_list()), 1)

    def test_warn_raises_when_strict(self) -> None:
        sw = StrictWarnings()
        with mock.patch.dict(os.environ, {"HARCHOC_STRICT_ML": "1"}):
            with self.assertRaises(RuntimeError):
                sw.warn("pil_image_open_failed", "bad", raise_if_strict=True)


if __name__ == "__main__":
    unittest.main()
