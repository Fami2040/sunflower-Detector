import unittest

from harchoc.ml_env import default_mamba_env, mamba_run_argv, probe_torch_via_mamba


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


if __name__ == "__main__":
    unittest.main()
