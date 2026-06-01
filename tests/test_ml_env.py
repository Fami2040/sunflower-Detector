import subprocess
import unittest
from unittest import mock

from harchoc.hsp_export_protocol import (
    DEFAULT_SPLIT_FILE,
    EXPORT_CONF,
    EXPORT_DEVICE,
    EXPORT_IOU,
    eval_export_cli_flags,
)
from harchoc.ml_env import (
    allow_base_python,
    default_mamba_env,
    mamba_run_argv,
    probe_torch_via_mamba,
    repo_python_cmd,
    run_repo_python,
)


class MlEnvTests(unittest.TestCase):
    def test_default_env(self) -> None:
        self.assertEqual(default_mamba_env(), "harchoc")

    def test_mamba_run_argv(self) -> None:
        self.assertEqual(
            mamba_run_argv("scripts/check_gpu.py", "--dry-run"),
            ["mamba", "run", "-n", "harchoc", "python", "scripts/check_gpu.py", "--dry-run"],
        )

    def test_probe_torch_via_mamba(self) -> None:
        probe = probe_torch_via_mamba()
        if not probe.get("ok"):
            self.skipTest(probe.get("error", "mamba/torch unavailable"))
        self.assertTrue(probe.get("cuda_available"))

    def test_allow_base_python(self) -> None:
        with mock.patch.dict("os.environ", {"HARCHOC_ALLOW_BASE_PYTHON": "1"}, clear=False):
            self.assertTrue(allow_base_python())
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(allow_base_python())

    def test_repo_python_cmd_mamba(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                repo_python_cmd(["scripts/check_gpu.py"]),
                mamba_run_argv("scripts/check_gpu.py"),
            )

    def test_repo_python_cmd_base_when_allowed(self) -> None:
        import sys

        with mock.patch.dict("os.environ", {"HARCHOC_ALLOW_BASE_PYTHON": "1"}, clear=False):
            self.assertEqual(
                repo_python_cmd(["scripts/check_gpu.py"]),
                [sys.executable, "scripts/check_gpu.py"],
            )

    def test_run_repo_python_invokes_subprocess(self) -> None:
        with mock.patch("harchoc.ml_env.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            with mock.patch.dict("os.environ", {"HARCHOC_ALLOW_BASE_PYTHON": "1"}, clear=False):
                proc = run_repo_python(["scripts/check_gpu.py"], repo_root="/tmp/repo")
            self.assertEqual(proc.returncode, 0)
            run.assert_called_once()
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["cwd"], "/tmp/repo")


class HspExportProtocolTests(unittest.TestCase):
    def test_defaults(self) -> None:
        self.assertEqual(EXPORT_CONF, 0.001)
        self.assertEqual(EXPORT_IOU, 0.3)
        self.assertEqual(DEFAULT_SPLIT_FILE, "data/splits/test.txt")
        self.assertEqual(EXPORT_DEVICE, "cpu")

    def test_eval_export_cli_flags(self) -> None:
        flags = eval_export_cli_flags(max_det=3000, device="cpu")
        self.assertIn("--export-conf", flags)
        self.assertIn("0.001", flags)
        self.assertIn("--export-iou", flags)
        self.assertIn("0.3", flags)


if __name__ == "__main__":
    unittest.main()
