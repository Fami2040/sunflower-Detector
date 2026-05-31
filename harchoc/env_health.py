from __future__ import annotations

import re
import subprocess
from typing import Any


# Known benign pip-check noise when SG stack is installed.
_SG_PIP_CHECK_ALLOW = (
    re.compile(r"super-gradients.*(requires|has requirement).*numpy", re.I),
    re.compile(r"data-gradients.*(requires|has requirement).*numpy", re.I),
    re.compile(r"opencv-python\s+\d+.*(requires|has requirement).*numpy", re.I),
)


def run_pip_check(*, env: str | None = None) -> tuple[int, str]:
    cmd = ["python", "-m", "pip", "check"]
    if env:
        cmd = ["mamba", "run", "-n", env, *cmd]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (p.stdout or "") + (p.stderr or "")
    return int(p.returncode), out.strip()


def classify_pip_check_output(text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    issues: list[str] = []
    ignored: list[str] = []
    for ln in lines:
        if any(pat.search(ln) for pat in _SG_PIP_CHECK_ALLOW):
            ignored.append(ln)
        else:
            issues.append(ln)
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "ignored_sg_numpy": ignored,
        "raw_line_count": len(lines),
    }


def verify_supergradients_import(*, env: str | None = None) -> dict[str, Any]:
    code = (
        "import super_gradients as sg; "
        "print(getattr(sg, '__version__', 'unknown'))"
    )
    cmd = ["python", "-c", code]
    if env:
        cmd = ["mamba", "run", "-n", env, *cmd]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return {
            "ok": False,
            "import_error": (p.stderr or p.stdout or "import failed").strip(),
        }
    return {"ok": True, "version": (p.stdout or "").strip()}


def env_health_report(*, env: str, with_super_gradients: bool) -> dict[str, Any]:
    rc, pip_out = run_pip_check(env=env)
    pip = classify_pip_check_output(pip_out)
    pip["returncode"] = rc
    if pip_out and not pip["issues"] and not pip["ignored_sg_numpy"]:
        pip["raw"] = pip_out

    sg: dict[str, Any] | None = None
    if with_super_gradients:
        sg = verify_supergradients_import(env=env)

    ok = pip["ok"] and (sg is None or sg.get("ok"))
    return {
        "status": "ok" if ok else "issues",
        "env": env,
        "pip_check": pip,
        "super_gradients": sg,
        "remediation": (
            "Re-run: python scripts/bootstrap_env.py --env {env} --with-super-gradients "
            "(re-pins numpy<2 after data-gradients)."
        ).format(env=env),
    }
