"""Tray holdout before/after eval plan and execution for ``scripts/finetune.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.domain_eval import domain_split_file, tray_keys_from_catalog_blob
from harchoc.finetune_pipeline import (
    hsp_transfer_paths,
    metrics_from_eval_json,
)
from harchoc.hsp_eval_chain import (
    DEFAULT_LOCKED_CONF_FROM,
    build_error_analysis_argv,
    extract_count_mae,
)
from harchoc.json_io import load_json_dict
from harchoc.post_train_eval import build_post_train_eval_argv

TRAY_EVAL_ROLES = ("tray", "val", "test")


def _nested_catalog(raw: dict[str, Any]) -> dict[str, Any] | None:
    nested = raw.get("catalog")
    if isinstance(nested, dict):
        return nested
    if raw.get("domains"):
        return raw
    return None


def resolve_tray_holdout_keys(
    *,
    cli_tray_keys: list[str] | None,
    transfer: dict[str, Any],
    catalog_path: Path | None,
) -> list[str]:
    """Tray keys to evaluate; CLI and transfer YAML override catalog."""
    keys: list[str] = []
    if cli_tray_keys:
        keys.extend(k.strip() for k in cli_tray_keys if k and k.strip())
    if not keys:
        raw = transfer.get("tray_keys")
        if isinstance(raw, list):
            keys.extend(str(k).strip() for k in raw if str(k).strip())
        single = transfer.get("tray_key")
        if single is not None and str(single).strip():
            keys.append(str(single).strip())
    if not keys and catalog_path is not None and catalog_path.is_file():
        blob = _nested_catalog(load_json_dict(catalog_path))
        if blob is not None:
            keys = tray_keys_from_catalog_blob(blob)
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def split_file_for_role(
    *,
    role: str,
    tray_key: str | None,
    domains_dir: Path,
    splits_dir: Path,
) -> str:
    if role == "test":
        return str((splits_dir / "test.txt").resolve())
    if tray_key is None:
        raise ValueError(f"tray_key required for role {role!r}")
    split_name = "test" if role == "tray" else role
    return domain_split_file(tray_key=tray_key, split=split_name, domains_dir=domains_dir)


def tray_eval_out_path(
    *,
    reports_dir: Path,
    phase: str,
    role: str,
    tray_key: str | None = None,
) -> Path:
    stem = f"tray_eval_{phase}_{role}"
    if tray_key:
        stem = f"{stem}_{tray_key}"
    return reports_dir / f"{stem}.json"


def _append_hsp_export_argv(
    argv: list[str],
    *,
    paths: dict[str, Path],
    locked_conf_from: str,
) -> list[str]:
    out = list(argv)
    out += [
        "--export-gt-json",
        str(paths["gt"]),
        "--export-preds-json",
        str(paths["preds"]),
        "--locked-conf-from",
        locked_conf_from,
        "--export-only",
    ]
    return out


def _run_error_analysis(
    *,
    repo_root: Path,
    paths: dict[str, Path],
    locked_conf_from: str,
) -> int:
    from harchoc.ml_env import run_repo_python

    argv = build_error_analysis_argv(
        paths["gt"],
        paths["preds"],
        locked_conf_from,
        paths["error"],
        repo_root=repo_root,
    )
    proc = run_repo_python(argv, repo_root=repo_root)
    return int(proc.returncode)


def _record_role_metrics(
    *,
    eval_out: Path,
    hsp_paths: dict[str, Path],
) -> dict[str, Any]:
    mae, _ci = extract_count_mae(hsp_paths["error"])
    m = metrics_from_eval_json(eval_out)
    if mae is None:
        mae = m.get("count_mae")
    return {
        "eval_json": str(eval_out),
        "error_json": str(hsp_paths["error"]),
        "gt_json": str(hsp_paths["gt"]),
        "preds_json": str(hsp_paths["preds"]),
        "count_mae": mae,
        "mAP50": m.get("mAP50"),
        "mAP50_95": m.get("mAP50_95"),
    }


def build_tray_eval_commands(
    *,
    phase: str,
    weights: str,
    tray_keys: list[str],
    reports_dir: Path,
    domains_dir: Path,
    splits_dir: Path,
    manifest: str,
    default_dataset_name: str,
    dataset_name: str | None,
    dataset_root: str | None,
    yolo_data_yaml: str | None,
    eval_section: dict[str, Any] | None,
    train_imgsz: int | None,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    hsp_counting: bool = True,
) -> list[dict[str, Any]]:
    """Planned eval.py invocations for one phase (before or after fine-tune)."""
    section = eval_section if isinstance(eval_section, dict) else {}
    use_hsp = bool(hsp_counting and section.get("hsp_counting", True))
    locked = str(section.get("locked_conf_from") or locked_conf_from)

    commands: list[dict[str, Any]] = []
    for role in TRAY_EVAL_ROLES:
        if role == "test":
            split_file = split_file_for_role(
                role=role,
                tray_key=None,
                domains_dir=domains_dir,
                splits_dir=splits_dir,
            )
            out = tray_eval_out_path(reports_dir=reports_dir, phase=phase, role=role)
            hsp_paths = (
                hsp_transfer_paths(reports_dir, phase=phase, role=role, tray_key=None)
                if use_hsp
                else None
            )
            argv = build_post_train_eval_argv(
                recorded_weights=weights,
                eval_out=str(hsp_paths["eval"] if hsp_paths else out),
                manifest=manifest,
                default_dataset_name=default_dataset_name,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                yolo_data_yaml=yolo_data_yaml,
                split_file=split_file,
                eval_section=eval_section,
                train_imgsz=train_imgsz,
            )
            if use_hsp and hsp_paths is not None:
                argv = _append_hsp_export_argv(argv, paths=hsp_paths, locked_conf_from=locked)
            commands.append(
                {
                    "phase": phase,
                    "role": role,
                    "split_file": split_file,
                    "out": str(out),
                    "argv": argv,
                    "hsp_paths": {k: str(v) for k, v in hsp_paths.items()} if hsp_paths else None,
                }
            )
            continue

        for tray_key in tray_keys:
            split_file = split_file_for_role(
                role=role,
                tray_key=tray_key,
                domains_dir=domains_dir,
                splits_dir=splits_dir,
            )
            out = tray_eval_out_path(
                reports_dir=reports_dir,
                phase=phase,
                role=role,
                tray_key=tray_key,
            )
            hsp_paths = (
                hsp_transfer_paths(reports_dir, phase=phase, role=role, tray_key=tray_key)
                if use_hsp
                else None
            )
            argv = build_post_train_eval_argv(
                recorded_weights=weights,
                eval_out=str(hsp_paths["eval"] if hsp_paths else out),
                manifest=manifest,
                default_dataset_name=default_dataset_name,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                yolo_data_yaml=yolo_data_yaml,
                split_file=split_file,
                eval_section=eval_section,
                train_imgsz=train_imgsz,
            )
            if use_hsp and hsp_paths is not None:
                argv = _append_hsp_export_argv(argv, paths=hsp_paths, locked_conf_from=locked)
            commands.append(
                {
                    "phase": phase,
                    "role": role,
                    "tray_key": tray_key,
                    "split_file": split_file,
                    "out": str(out),
                    "argv": argv,
                    "hsp_paths": {k: str(v) for k, v in hsp_paths.items()} if hsp_paths else None,
                }
            )
    return commands


def paths_from_tray_eval_commands(commands: list[dict[str, Any]]) -> dict[str, Any]:
    """Map role / tray_key → eval JSON path for finetune metadata (dry-run)."""
    paths: dict[str, Any] = {}
    for cmd in commands:
        role = str(cmd["role"])
        out = str(cmd["out"])
        tray_key = cmd.get("tray_key")
        if tray_key:
            bucket = paths.setdefault(str(tray_key), {})
            assert isinstance(bucket, dict)
            bucket[role] = out
        else:
            paths[role] = out
    return paths


def build_tray_eval_plan(
    *,
    enabled: bool,
    tray_keys: list[str],
    base_weights: str,
    after_weights: str,
    reports_dir: Path,
    domains_dir: Path,
    splits_dir: Path,
    manifest: str,
    default_dataset_name: str,
    dataset_name: str | None,
    dataset_root: str | None,
    yolo_data_yaml: str | None,
    eval_section: dict[str, Any] | None,
    train_imgsz: int | None,
    locked_conf_from: str = DEFAULT_LOCKED_CONF_FROM,
    hsp_counting: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "tray_keys": tray_keys}
    kwargs = dict(
        tray_keys=tray_keys,
        reports_dir=reports_dir,
        domains_dir=domains_dir,
        splits_dir=splits_dir,
        manifest=manifest,
        default_dataset_name=default_dataset_name,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        yolo_data_yaml=yolo_data_yaml,
        eval_section=eval_section,
        train_imgsz=train_imgsz,
        locked_conf_from=locked_conf_from,
        hsp_counting=hsp_counting,
    )
    before_cmds = build_tray_eval_commands(phase="before", weights=base_weights, **kwargs)
    after_cmds = build_tray_eval_commands(phase="after", weights=after_weights, **kwargs)
    return {
        "enabled": True,
        "tray_keys": tray_keys,
        "hsp_counting": hsp_counting,
        "locked_conf_from": locked_conf_from,
        "domains_dir": str(domains_dir),
        "splits_dir": str(splits_dir),
        "before": {"weights": base_weights, "commands": before_cmds},
        "after": {"weights": after_weights, "commands": after_cmds},
    }


def run_tray_eval_commands(
    commands: list[dict[str, Any]],
    *,
    eval_main: Any,
    repo_root: Path | None = None,
    skip_missing_splits: bool = True,
) -> tuple[dict[str, Any], list[int], list[str]]:
    """
    Run eval.py (+ error_analysis when HSP paths set) for each planned command.

    Returns (paths dict with count_mae, return codes, warnings).
    """
    rr = (repo_root or Path.cwd()).resolve()
    paths: dict[str, Any] = {}
    rcs: list[int] = []
    warnings: list[str] = []

    for cmd in commands:
        split_file = Path(str(cmd["split_file"]))
        if skip_missing_splits and not split_file.is_file():
            msg = f"Skipping tray eval ({cmd.get('role')}): split file missing: {split_file}"
            warnings.append(msg)
            print(msg)
            continue

        argv = list(cmd["argv"])
        rc = int(eval_main(argv))
        rcs.append(rc)

        role = str(cmd["role"])
        tray_key = cmd.get("tray_key")
        eval_out = Path(str(cmd.get("hsp_paths", {}).get("eval") if cmd.get("hsp_paths") else cmd["out"]))
        if not eval_out.is_absolute():
            eval_out = (rr / eval_out).resolve()

        metrics: dict[str, Any] = {"eval_json": str(eval_out), "count_mae": None}
        hsp_raw = cmd.get("hsp_paths")
        if isinstance(hsp_raw, dict) and hsp_raw:
            hsp_paths = {k: Path(str(v)) for k, v in hsp_raw.items()}
            locked = ""
            for i, tok in enumerate(argv):
                if tok == "--locked-conf-from" and i + 1 < len(argv):
                    locked = argv[i + 1]
                    break
            if rc == 0 and hsp_paths.get("gt", Path()).is_file() and locked:
                err_rc = _run_error_analysis(
                    repo_root=rr,
                    paths=hsp_paths,
                    locked_conf_from=locked,
                )
                rcs.append(err_rc)
                if err_rc != 0:
                    warnings.append(f"error_analysis failed for {role} {tray_key or ''}: rc={err_rc}")
            metrics = _record_role_metrics(eval_out=eval_out, hsp_paths=hsp_paths)
        else:
            metrics.update(metrics_from_eval_json(eval_out))

        if tray_key:
            bucket = paths.setdefault(str(tray_key), {})
            assert isinstance(bucket, dict)
            bucket[role] = metrics
        else:
            paths[role] = metrics

    return paths, rcs, warnings
