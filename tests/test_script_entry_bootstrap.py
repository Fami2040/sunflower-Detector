"""Import smoke for script bootstrap helper and migrated entrypoints."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# High-traffic scripts migrated to harchoc.script_entry bootstrap (keep in sync).
MIGRATED_SCRIPT_MODULES = (
    "scripts.train",
    "scripts.eval",
    "scripts.experiment",
    "scripts.benchmark_matrix",
    "scripts.threshold_sweep",
    "scripts.error_analysis",
    "scripts.finetune",
    "scripts.pre_train_gate",
    "scripts.check_gpu",
    "scripts.eval_domains",
    "scripts.make_figures",
    "scripts.cv_eval",
    "scripts.matrix_seed_stats",
)


class ScriptEntryBootstrapTests(unittest.TestCase):
    def test_bootstrap_repo_imports_idempotent(self) -> None:
        from harchoc.script_entry import bootstrap_repo_imports, ensure_repo_root_on_syspath

        ensure_repo_root_on_syspath()
        bootstrap_repo_imports()
        bootstrap_repo_imports()

        import harchoc

        self.assertTrue(str(REPO_ROOT) in sys.path or REPO_ROOT.samefile(Path(harchoc.__file__).parent.parent))

    def test_migrated_scripts_import_under_ci_pythonpath(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": ".",
            "HARCHOC_ALLOW_BASE_PYTHON": "1",
        }
        for mod_name in MIGRATED_SCRIPT_MODULES:
            with self.subTest(module=mod_name):
                proc = subprocess.run(
                    [sys.executable, "-c", f"import {mod_name}"],
                    cwd=REPO_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    proc.returncode,
                    0,
                    msg=f"{mod_name}: stderr={proc.stderr!r} stdout={proc.stdout!r}",
                )

    def test_migrated_scripts_importable_in_process(self) -> None:
        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        for mod_name in MIGRATED_SCRIPT_MODULES:
            with self.subTest(module=mod_name):
                importlib.import_module(mod_name)

    def test_train_script_imports_as_file_without_pythonpath(self) -> None:
        code = """
import importlib.util
from pathlib import Path
p = Path("scripts/train.py").resolve()
spec = importlib.util.spec_from_file_location("scripts.train", p)
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(mod)
assert hasattr(mod, "main")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_check_gpu_script_imports_as_file_without_pythonpath(self) -> None:
        code = """
import importlib.util
from pathlib import Path
p = Path("scripts/check_gpu.py").resolve()
spec = importlib.util.spec_from_file_location("scripts.check_gpu", p)
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(mod)
assert hasattr(mod, "main")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_gpu_shims_exit_2(self) -> None:
        for script in ("scripts/gpu_sanity.py", "scripts/gpu_smoke_ultralytics.py"):
            with self.subTest(script=script):
                proc = subprocess.run(
                    [sys.executable, script],
                    cwd=REPO_ROOT,
                    env={**os.environ, "PYTHONPATH": "."},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 2, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
