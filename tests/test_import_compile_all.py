import os
import py_compile
import unittest
from importlib import import_module
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ImportCompileAllTests(unittest.TestCase):
    def test_all_python_files_compile(self) -> None:
        for path in REPO_ROOT.rglob("*.py"):
            # Avoid accidental compilation of vendored/virtualenv dirs if present.
            parts = set(path.parts)
            if ".venv" in parts or "venv" in parts or "__pycache__" in parts:
                continue
            py_compile.compile(str(path), doraise=True)

    def test_scripts_package_imports(self) -> None:
        scripts_dir = REPO_ROOT / "scripts"
        self.assertTrue(scripts_dir.is_dir())

        for path in scripts_dir.glob("*.py"):
            name = path.stem
            if name.startswith("__"):
                continue
            import_module(f"scripts.{name}")


if __name__ == "__main__":
    unittest.main()

