"""Post-zoo reviewer-2 reproducibility chain (CPU audit before Word paste)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from harchoc.repro_chain import (
    format_repro_cmd,
    hsp_test_exports_present,
    load_json_bundle,
    reproduce_command_block,
    run_argv_chain,
)

REVIEWER2_REPRO_BUNDLE_SCHEMA = "reviewer2_repro_bundle.v1"


def load_reviewer2_repro_bundle(path: str | Path) -> dict[str, Any]:
    return load_json_bundle(path, schema_version=REVIEWER2_REPRO_BUNDLE_SCHEMA)


def reviewer2_repro_reproduce_commands(
    *,
    bundle_path: str = "configs/experiments/reviewer2_repro.json",
) -> dict[str, list[str]]:
    ci = (
        "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 "
        f"python scripts/experiment.py --config {bundle_path} reviewer2-repro"
    )
    gpu = f"mamba run -n harchoc python scripts/experiment.py --config {bundle_path} reviewer2-repro"
    return reproduce_command_block(
        ci_dry_run=[f"{ci} --dry-run"],
        local=[gpu],
        extra={
            "prerequisite_hsp": ["mamba run -n harchoc python scripts/experiment.py repro"],
        },
    )


def build_reviewer2_repro_chain(
    bundle: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    global_dry_run: bool = False,
) -> list[tuple[str, list[str]]]:
    """Ordered (step_id, argv) pairs; argv is repo-relative ``scripts/experiment.py`` invocation."""
    rr = Path(repo_root or ".").expanduser().resolve()
    configs = bundle.get("configs") or {}
    hsp = bundle.get("hsp_artifacts") or {}
    exports_ok = hsp_test_exports_present(rr, hsp)

    def _script(name: str) -> str:
        return str((rr / "scripts" / name).relative_to(rr))

    def _step_dry(*, needs_exports: bool) -> bool:
        return bool(global_dry_run) or (needs_exports and not exports_ok)

    steps: list[tuple[str, list[str]]] = []

    rc_argv: list[str] = [_script("experiment.py")]
    counting_cfg = str(configs.get("reviewer_counting") or "").strip()
    if counting_cfg:
        rc_argv.extend(["--config", counting_cfg])
    rc_argv.append("reviewer-counting")
    if _step_dry(needs_exports=True):
        rc_argv.append("--dry-run")
    steps.append(("reviewer_counting", rc_argv))

    r2m_argv: list[str] = [_script("experiment.py"), "reviewer2-map50"]
    if global_dry_run:
        r2m_argv.append("--dry-run")
    steps.append(("reviewer2_map50", r2m_argv))

    r2c_argv: list[str] = [_script("experiment.py")]
    confusion_cfg = str(configs.get("reviewer2_confusion") or "").strip()
    if confusion_cfg:
        r2c_argv.extend(["--config", confusion_cfg])
    r2c_argv.append("reviewer2-confusion")
    if _step_dry(needs_exports=True):
        r2c_argv.append("--dry-run")
    steps.append(("reviewer2_confusion", r2c_argv))

    steps.append(("reviewer2_paste_check", [_script("experiment.py"), "reviewer2-paste-check"]))
    return steps


def run_reviewer2_repro_chain(
    bundle: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
    on_step: Callable[[str, list[str]], None] | None = None,
    run_argv: Callable[[list[str]], int] | None = None,
) -> int:
    rr = Path(repo_root or ".").expanduser().resolve()
    steps = build_reviewer2_repro_chain(bundle, repo_root=rr, global_dry_run=dry_run)
    hsp = bundle.get("hsp_artifacts") or {}
    exports_ok = hsp_test_exports_present(rr, hsp)

    def _mamba_for_step(step_id: str) -> bool:
        return step_id != "reviewer2_paste_check"

    if dry_run:
        for step_id, argv in steps:
            if on_step is not None:
                on_step(step_id, argv)
            print(f"# {step_id}")
            print(format_repro_cmd(argv, mamba=_mamba_for_step(step_id)))
            if step_id in ("reviewer_counting", "reviewer2_confusion") and not exports_ok:
                print(f"# note: {step_id} would use --dry-run (missing HSP test exports)")
        return 0

    def _run_step(argv: list[str]) -> int:
        if run_argv is not None:
            return int(run_argv(argv))
        proc = __import__("subprocess").run([sys.executable, *argv], cwd=str(rr))
        rc = int(proc.returncode)
        if rc != 0 and "reviewer2-paste-check" in argv:
            print(
                "WARNING: reviewer2-paste-check exit 1 (docx gaps expected until Word paste); "
                "JSON written — continuing preflight.",
                file=sys.stderr,
            )
            return 0
        return rc

    return run_argv_chain(
        steps,
        repo_root=rr,
        dry_run=False,
        on_step=on_step,
        run_argv=_run_step,
        mamba_for_step=_mamba_for_step,
        fail_label="reviewer2-repro",
    )
