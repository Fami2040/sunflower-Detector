import io
import os
import unittest
from unittest import mock

from harchoc import strict_ml


class StrictMlEnvTests(unittest.TestCase):
    def test_strict_ml_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HARCHOC_STRICT_ML", None)
            self.assertFalse(strict_ml.strict_ml_enabled())

    def test_strict_ml_enabled_values(self) -> None:
        for val in ("1", "true", "yes", "TRUE"):
            with self.subTest(val=val):
                with mock.patch.dict(os.environ, {"HARCHOC_STRICT_ML": val}):
                    self.assertTrue(strict_ml.strict_ml_enabled())


class RequireTorchTests(unittest.TestCase):
    def test_require_torch_returns_module(self) -> None:
        sentinel = object()
        with mock.patch(
            "harchoc.gpu_probe.try_import_torch",
            return_value=(sentinel, None, None),
        ):
            self.assertIs(strict_ml.require_torch(), sentinel)

    def test_require_torch_raises_with_mamba_hint(self) -> None:
        with mock.patch(
            "harchoc.gpu_probe.try_import_torch",
            return_value=(None, None, "Failed to import torch: no module"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                strict_ml.require_torch()
        msg = str(ctx.exception)
        self.assertIn("PyTorch required", msg)
        self.assertIn("mamba run -n harchoc python", msg)


class RequireCudaTests(unittest.TestCase):
    def test_require_cuda_returns_payload_when_available(self) -> None:
        payload = {"cuda_available": True, "device_name": "mock"}
        with mock.patch("harchoc.strict_ml.require_torch", return_value=object()):
            with mock.patch("harchoc.gpu_probe.torch_cuda_payload", return_value=payload):
                self.assertEqual(strict_ml.require_cuda(strict=False), payload)

    def test_require_cuda_strict_raises(self) -> None:
        payload = {"cuda_available": False}
        with mock.patch("harchoc.strict_ml.require_torch", return_value=object()):
            with mock.patch("harchoc.gpu_probe.torch_cuda_payload", return_value=payload):
                with self.assertRaises(RuntimeError):
                    strict_ml.require_cuda(strict=True)


class FailOrWarnTests(unittest.TestCase):
    def test_fail_or_warn_strict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            strict_ml.fail_or_warn("boom", strict=True)

    def test_fail_or_warn_non_strict_prints_stderr(self) -> None:
        buf = io.StringIO()
        with mock.patch.object(strict_ml.sys, "stderr", buf):
            strict_ml.fail_or_warn("heads up", strict=False)
        self.assertIn("heads up", buf.getvalue())

    def test_fail_or_warn_uses_env_when_strict_none(self) -> None:
        with mock.patch.dict(os.environ, {"HARCHOC_STRICT_ML": "1"}):
            with self.assertRaises(RuntimeError):
                strict_ml.fail_or_warn("strict env")


class MlWarningsSinkTests(unittest.TestCase):
    def test_ml_warnings_sink_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HARCHOC_STRICT_ML", None)
            self.assertIsNone(strict_ml.ml_warnings_sink())

    def test_ml_warnings_sink_returns_list_when_strict(self) -> None:
        with mock.patch.dict(os.environ, {"HARCHOC_STRICT_ML": "1"}):
            sink = strict_ml.ml_warnings_sink()
            self.assertIsNotNone(sink)
            assert sink is not None
            self.assertEqual(sink, [])


class AppendCaptureWarningTests(unittest.TestCase):
    def test_append_capture_warning_noop_when_sink_none(self) -> None:
        cap = strict_ml.FailureCapture(context="x", exc_type="RuntimeError", exc_msg="nope")
        strict_ml.append_capture_warning(None, cap)

    def test_append_capture_warning_formats_message(self) -> None:
        warnings: list[str] = []
        cap = strict_ml.FailureCapture(context="import torch", exc_type="ImportError", exc_msg="no module")
        strict_ml.append_capture_warning(warnings, cap)
        self.assertEqual(warnings, ["import torch: ImportError: no module"])


class CaptureFailureTests(unittest.TestCase):
    def test_capture_failure_no_exception(self) -> None:
        with strict_ml.capture_failure("ok") as cap:
            pass
        self.assertFalse(cap.failed)
        self.assertIsNone(cap.exc_type)

    def test_capture_failure_records_exception(self) -> None:
        with strict_ml.capture_failure("probe") as cap:
            raise ValueError("bad tensor")
        self.assertTrue(cap.failed)
        self.assertEqual(cap.exc_type, "ValueError")
        self.assertEqual(cap.exc_msg, "bad tensor")
        self.assertEqual(cap.context, "probe")


class StrictWarningsTests(unittest.TestCase):
    def test_as_list_is_copy(self) -> None:
        sw = strict_ml.StrictWarnings()
        sw.warn("x", "y", raise_if_strict=False)
        lst = sw.as_list()
        lst.clear()
        self.assertEqual(len(sw.items), 1)


class ImportTests(unittest.TestCase):
    def test_module_importable_without_torch(self) -> None:
        self.assertTrue(callable(strict_ml.strict_ml_enabled))


if __name__ == "__main__":
    unittest.main()
