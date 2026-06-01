"""
Mamba/conda env conventions for GPU training and torch probes.

All live train/eval/GPU checks use the project env (default ``harchoc``), not base Python.
CI sets ``HARCHOC_ALLOW_BASE_PYTHON=1`` for unittest only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any


def allow_base_python() -> bool:
    """True when CI/unittest may run repo scripts without ``mamba run``."""
    return os.getenv("HARCHOC_ALLOW_BASE_PYTHON", "").strip() in ("1", "true", "yes")


def default_mamba_env() -> str:
    return (os.getenv("HARCHOC_MAMBA_ENV") or "harchoc").strip() or "harchoc"


def mamba_available() -> bool:
    return shutil.which("mamba") is not None


def mamba_run_argv(*python_args: str, env_name: str | None = None) -> list[str]:
    """``mamba run -n <env> python ...`` argv prefix (caller supplies script args after python)."""
    name = env_name or default_mamba_env()
    return ["mamba", "run", "-n", name, "python", *python_args]


def mamba_run_shell_command(
    *python_args: str,
    env_name: str | None = None,
) -> str:
    """Single-line command for docs/backlog (repo-root relative)."""
    return " ".join(mamba_run_argv(*python_args, env_name=env_name))


def repo_python_cmd(
    argv: list[str],
    *,
    use_mamba: bool = True,
    env_name: str | None = None,
) -> list[str]:
    """Argv to run ``python <argv>`` in-repo, optionally via ``mamba run``."""
    if use_mamba and not allow_base_python():
        return list(mamba_run_argv(*argv, env_name=env_name))
    return [sys.executable, *argv]


def run_repo_python(
    argv: list[str],
    *,
    repo_root: str | os.PathLike[str],
    use_mamba: bool = True,
    env_name: str | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run repo-relative script argv under mamba (or base Python in CI)."""
    cmd = repo_python_cmd(argv, use_mamba=use_mamba, env_name=env_name)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=run_env,
        check=check,
        capture_output=False,
        text=True,
    )


def run_in_mamba_env(
    argv: list[str],
    *,
    env_name: str | None = None,
    cwd: str | os.PathLike[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    name = env_name or default_mamba_env()
    cmd = ["mamba", "run", "-n", name, *argv]
    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=run_env,
        check=False,
    )


def probe_torch_via_mamba(*, env_name: str | None = None) -> dict[str, Any]:
    """
    Import torch inside ``mamba run -n <env>`` and return cuda availability.

    Use when the current interpreter is base/CI Python without torch.
    """
    name = env_name or default_mamba_env()
    if not mamba_available():
        return {
            "ok": False,
            "mamba_env": name,
            "error": "mamba not on PATH",
        }
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'torch_version': getattr(torch,'__version__',None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    proc = run_in_mamba_env(["python", "-c", code], env_name=name)
    if proc.returncode != 0:
        return {
            "ok": False,
            "mamba_env": name,
            "error": (proc.stderr or proc.stdout or "mamba run failed").strip(),
            "returncode": proc.returncode,
        }
    try:
        payload = __import__("json").loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception as ex:
        return {
            "ok": False,
            "mamba_env": name,
            "error": f"parse failed: {ex}",
            "stdout": proc.stdout,
        }
    return {
        "ok": True,
        "mamba_env": name,
        **payload,
    }


def should_reexec_in_mamba_for_torch() -> bool:
    """True when current process lacks torch and we should delegate to mamba."""
    if os.getenv("HARCHOC_MAMBA_REEXEC", "").strip() in ("1", "true", "yes"):
        return False
    if os.getenv("HARCHOC_ALLOW_BASE_PYTHON", "").strip() in ("1", "true", "yes"):
        # CI unittest uses base python intentionally; GPU scripts should set reexec anyway
        pass
    from harchoc.gpu_probe import try_import_torch

    _torch, _tv, err = try_import_torch()
    if err is None:
        return False
    return mamba_available()


def reexec_script_in_mamba_env(argv: list[str] | None = None) -> None:
    """
    Replace current process with ``mamba run -n harchoc python <this_script> ...``.

    Sets ``HARCHOC_MAMBA_REEXEC=1`` on the child to avoid loops.
    """
    name = default_mamba_env()
    script = sys.argv[0]
    rest = list(argv if argv is not None else sys.argv[1:])
    child_env = os.environ.copy()
    child_env["HARCHOC_MAMBA_REEXEC"] = "1"
    os.execve(
        shutil.which("mamba") or "mamba",
        ["mamba", "run", "-n", name, "python", script, *rest],
        child_env,
    )
