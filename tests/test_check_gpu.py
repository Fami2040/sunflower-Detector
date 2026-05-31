import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CheckGpuScriptTests(unittest.TestCase):
    def test_check_gpu_dry_run_exits_zero(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "check_gpu.py"
        self.assertTrue(script.exists(), f"Missing script: {script}")

        proc = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # In CI / lightweight envs torch may be absent; script should still exit 0 and print guidance.
        if "Failed to import torch" in out:
            self.assertIn("Install PyTorch first", out)
        else:
            self.assertIn("torch:", out)
            self.assertIn("cuda_available:", out)

    def test_check_gpu_json_out_writes_schema(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "check_gpu.py"
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "gpu_check.json"
            proc = subprocess.run(
                [sys.executable, str(script), "--dry-run", "--json-out", str(out_path)],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")
            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("script"), "check_gpu")
            self.assertEqual(payload.get("schema_version"), "check_gpu.v1")
            self.assertTrue(payload.get("dry_run"))


if __name__ == "__main__":
    unittest.main()

