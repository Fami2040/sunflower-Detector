"""Tray holdout before/after eval plan and execution for ``scripts/finetune.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harchoc.domain_eval import domain_split_file, tray_keys_from_catalog_blob
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
) -> list[dict[str, Any]]:
    """Planned eval.py invocations for one phase (before or after fine-tune)."""
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
            argv = build_post_train_eval_argv(
                recorded_weights=weights,
                eval_out=str(out),
                manifest=manifest,
                default_dataset_name=default_dataset_name,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                yolo_data_yaml=yolo_data_yaml,
                split_file=split_file,
                eval_section=eval_section,
                train_imgsz=train_imgsz,
            )
            commands.append(
                {
                    "phase": phase,
                    "role": role,
                    "split_file": split_file,
                    "out": str(out),
                    "argv": argv,
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
            argv = build_post_train_eval_argv(
                recorded_weights=weights,
                eval_out=str(out),
                manifest=manifest,
                default_dataset_name=default_dataset_name,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                yolo_data_yaml=yolo_data_yaml,
                split_file=split_file,
                eval_section=eval_section,
                train_imgsz=train_imgsz,
            )
            commands.append(
                {
                    "phase": phase,
                    "role": role,
                    "tray_key": tray_key,
                    "split_file": split_file,
                    "out": str(out),
                    "argv": argv,
                }
            )
    return commands


def paths_from_tray_eval_commands(commands: list[dict[str, Any]]) -> dict[str, Any]:
    """Map role / tray_key → eval JSON path for finetune metadata."""
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
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "tray_keys": tray_keys}
    before_cmds = build_tray_eval_commands(
        phase="before",
        weights=base_weights,
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
    )
    after_cmds = build_tray_eval_commands(
        phase="after",
        weights=after_weights,
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
    )
    return {
        "enabled": True,
        "tray_keys": tray_keys,
        "domains_dir": str(domains_dir),
        "splits_dir": str(splits_dir),
        "before": {"weights": base_weights, "commands": before_cmds},
        "after": {"weights": after_weights, "commands": after_cmds},
    }


def run_tray_eval_commands(
    commands: list[dict[str, Any]],
    *,
    eval_main: Any,
    skip_missing_splits: bool = True,
) -> tuple[dict[str, Any], list[int], list[str]]:
    """
    Run eval.py for each planned command.

    Returns (paths dict, return codes, warnings).
    """
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
        out = str(cmd["out"])
        role = str(cmd["role"])
        tray_key = cmd.get("tray_key")
        if tray_key:
            bucket = paths.setdefault(str(tray_key), {})
            assert isinstance(bucket, dict)
            bucket[role] = out
        else:
            paths[role] = out

    return paths, rcs, warnings
