import os
import subprocess
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class SmokeTests(unittest.TestCase):
    def test_run_infer_once_usage(self) -> None:
        p = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "run_infer_once.py")],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Usage: python run_infer_once.py <image>", p.stdout)


if __name__ == "__main__":
    unittest.main()

