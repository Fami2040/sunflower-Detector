from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()

from harchoc.finetune_tray_eval import (
    build_tray_eval_plan,
    paths_from_tray_eval_commands,
    resolve_tray_holdout_keys,
    run_tray_eval_commands,
)
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.json_io import load_json_dict
from harchoc.script_scaffold import build_versioned_dry_run_payload, resolve_dataset_args
from harchoc.schemas import with_schema_version
from scripts._common_cli import add_dataset_args, add_dry_run_arg, require_existing_dir, write_json


def _load_transfer_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _merge_finetune_train_config(
    *,
    repo_root: Path,
    base_weights: str,
    transfer: dict[str, Any],
    experiment_json: Path,
) -> dict[str, Any]:
    from harchoc.train_config import resolve_train_config_extends

    cfg = resolve_train_config_extends(
        load_json_dict(experiment_json),
        repo_root=repo_root,
        config_path=experiment_json,
    )
    cfg["model"] = str(base_weights)
    if transfer.get("epochs") is not None:
        cfg["epochs"] = int(transfer["epochs"])
    if transfer.get("lr") is not None:
        cfg["lr0"] = float(transfer["lr"])
    if transfer.get("seed") is not None:
        cfg["seed"] = int(transfer["seed"])
    if transfer.get("freeze") is not None:
        cfg["freeze"] = transfer["freeze"]
    if transfer.get("freeze_backbone") is not None:
        cfg["freeze_backbone"] = bool(transfer["freeze_backbone"])
    if transfer.get("unfreeze_epoch") is not None:
        cfg["unfreeze_epoch"] = int(transfer["unfreeze_epoch"])
    return cfg


def _resolve_catalog_path(raw: str, repo_root: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _tray_eval_context(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    transfer: dict[str, Any],
    train_doc: dict[str, Any] | None,
    dataset_root: str | None,
) -> dict[str, Any]:
    catalog_path = _resolve_catalog_path(str(args.tray_catalog), repo_root)
    tray_keys = resolve_tray_holdout_keys(
        cli_tray_keys=list(args.tray_key) if args.tray_key else None,
        transfer=transfer,
        catalog_path=catalog_path,
    )
    if not tray_keys and bool(args.tray_eval):
        tray_keys = ["_example"]

    eval_section = train_doc.get("eval") if isinstance(train_doc, dict) else None
    if not isinstance(eval_section, dict):
        eval_section = {}

    domains_dir = Path(str(args.domains_dir))
    if not domains_dir.is_absolute():
        domains_dir = (repo_root / domains_dir).resolve()
    splits_dir = Path(str(args.splits_dir))
    if not splits_dir.is_absolute():
        splits_dir = (repo_root / splits_dir).resolve()

    imgsz = train_doc.get("imgsz") if isinstance(train_doc, dict) else None
    reports_dir = Path(str(args.out)).expanduser().parent
    if not reports_dir.is_absolute():
        reports_dir = (repo_root / reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "tray_keys": tray_keys,
        "eval_section": eval_section,
        "domains_dir": domains_dir,
        "splits_dir": splits_dir,
        "train_imgsz": int(imgsz) if imgsz is not None else None,
        "reports_dir": reports_dir,
        "catalog_path": catalog_path,
        "dataset_root": dataset_root,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune a pretrained model on a target domain.")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--base-weights", default=HSP_DETECTION_WEIGHTS, help="Pretrained weights to start from.")
    p.add_argument(
        "--config",
        default="configs/experiments/finetune_tray.json",
        help="Train experiment JSON (extends baseline).",
    )
    p.add_argument(
        "--transfer-config",
        default="configs/transfer/finetune_minimal.yaml",
        help="Transfer policy YAML (freeze metadata).",
    )
    p.add_argument("--name", default="finetune_tray", help="Ultralytics run name under --out-dir.")
    p.add_argument("--out-dir", default="runs/transfer", help="Training output directory.")
    p.add_argument("--out", default="reports/transfer/finetune.json", help="Finetune run metadata JSON.")
    p.add_argument(
        "--tray-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run tray holdout eval before/after train via eval.py (default: on).",
    )
    p.add_argument(
        "--tray-key",
        action="append",
        default=[],
        dest="tray_key",
        metavar="TRAY_KEY",
        help="Tray holdout key(s); default from transfer YAML or --tray-catalog.",
    )
    p.add_argument(
        "--tray-catalog",
        default="reports/domains/catalog.json",
        help="Domain catalog JSON for tray_key discovery.",
    )
    p.add_argument(
        "--domains-dir",
        default="data/domains",
        help="Per-tray split lists (test_{tray_key}.txt, val_{tray_key}.txt).",
    )
    p.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Canonical split lists (test.txt for manuscript test eval).",
    )
    p.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        default=None,
        help="Staged unfreeze: 1=frozen backbone (finetune_tray_stage1), 2=full unfreeze (stage2 YAML).",
    )
    args = p.parse_args(argv)

    if args.stage == 1:
        args.config = "configs/experiments/finetune_tray_stage1.json"
        args.transfer_config = "configs/transfer/finetune_stage1.yaml"
        if args.name == "finetune_tray":
            args.name = "finetune_tray_s1"
    elif args.stage == 2:
        args.config = "configs/experiments/finetune_tray_stage2.json"
        args.transfer_config = "configs/transfer/finetune_stage2.yaml"
        if args.name == "finetune_tray":
            args.name = "finetune_tray_s2"

    repo_root = Path(__file__).resolve().parents[1]
    exp_path = Path(args.config).expanduser()
    if not exp_path.is_absolute():
        exp_path = (repo_root / exp_path).resolve()
    transfer_path = Path(args.transfer_config).expanduser()
    if not transfer_path.is_absolute():
        transfer_path = (repo_root / transfer_path).resolve()

    transfer_preview: dict[str, Any] = {}
    if transfer_path.is_file():
        transfer_preview = _load_transfer_yaml(transfer_path)

    train_doc_preview = _merge_finetune_train_config(
        repo_root=repo_root,
        base_weights=str(args.base_weights),
        transfer=transfer_preview,
        experiment_json=exp_path,
    )

    if args.dry_run:
        from harchoc.train_kwargs import resolve_freeze_policy

        _, freeze_policy, _ = resolve_freeze_policy(transfer_preview)
        ctx = _tray_eval_context(
            repo_root=repo_root,
            args=args,
            transfer=transfer_preview,
            train_doc=train_doc_preview,
            dataset_root=getattr(args, "dataset_root", None),
        )
        after_weights = str((Path(args.out_dir) / str(args.name) / "weights" / "best.pt").resolve())
        tray_plan = build_tray_eval_plan(
            enabled=bool(args.tray_eval),
            tray_keys=ctx["tray_keys"],
            base_weights=str(args.base_weights),
            after_weights=after_weights,
            reports_dir=ctx["reports_dir"],
            domains_dir=ctx["domains_dir"],
            splits_dir=ctx["splits_dir"],
            manifest=str(args.manifest),
            default_dataset_name=str(args.default_dataset_name),
            dataset_name=getattr(args, "dataset_name", None),
            dataset_root=ctx["dataset_root"],
            yolo_data_yaml=getattr(args, "yolo_data_yaml", None),
            eval_section=ctx["eval_section"],
            train_imgsz=ctx["train_imgsz"],
        )
        out_path = write_json(
            args.out,
            build_versioned_dry_run_payload(
                script="finetune",
                schema_version="finetune_run.v1",
                out=args.out,
                base_weights=args.base_weights,
                config=str(exp_path),
                transfer_config=str(transfer_path),
                transfer_policy=freeze_policy,
                tray_eval_plan=tray_plan,
                tray_eval_before=paths_from_tray_eval_commands(
                    tray_plan.get("before", {}).get("commands", [])
                )
                if tray_plan.get("enabled")
                else None,
                tray_eval_after=paths_from_tray_eval_commands(
                    tray_plan.get("after", {}).get("commands", [])
                )
                if tray_plan.get("enabled")
                else None,
            ),
        )
        print(f"Wrote {out_path}")
        return 0

    spec = resolve_dataset_args(args)
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
    transfer = transfer_preview if transfer_preview else _load_transfer_yaml(transfer_path)
    train_doc = _merge_finetune_train_config(
        repo_root=repo_root,
        base_weights=str(args.base_weights),
        transfer=transfer,
        experiment_json=exp_path,
    )

    ctx = _tray_eval_context(
        repo_root=repo_root,
        args=args,
        transfer=transfer,
        train_doc=train_doc,
        dataset_root=str(spec.root),
    )
    tray_keys_live = resolve_tray_holdout_keys(
        cli_tray_keys=list(args.tray_key) if args.tray_key else None,
        transfer=transfer,
        catalog_path=ctx["catalog_path"],
    )

    from scripts._common_cli import extend_dataset_argv
    from scripts.train import main as train_main

    tmp_path: str | None = None
    rc = 1
    eval_rcs: list[int] = []
    tray_eval_warnings: list[str] = []
    tray_eval_before: dict[str, Any] | None = None
    tray_eval_after: dict[str, Any] | None = None
    run_dir = Path(args.out_dir) / str(args.name)
    weights_out: str | None = None

    def _run_phase(phase: str, weights: str) -> dict[str, Any] | None:
        from harchoc.finetune_tray_eval import build_tray_eval_commands

        if not bool(args.tray_eval) or not weights.strip():
            return None
        from scripts.eval import main as eval_main

        commands = build_tray_eval_commands(
            phase=phase,
            weights=weights,
            tray_keys=tray_keys_live,
            reports_dir=ctx["reports_dir"],
            domains_dir=ctx["domains_dir"],
            splits_dir=ctx["splits_dir"],
            manifest=str(args.manifest),
            default_dataset_name=str(args.default_dataset_name),
            dataset_name=getattr(args, "dataset_name", None),
            dataset_root=ctx["dataset_root"],
            yolo_data_yaml=getattr(args, "yolo_data_yaml", None),
            eval_section=ctx["eval_section"],
            train_imgsz=ctx["train_imgsz"],
        )
        paths, phase_rcs, warns = run_tray_eval_commands(commands, eval_main=eval_main)
        eval_rcs.extend(phase_rcs)
        tray_eval_warnings.extend(warns)
        return paths or None

    try:
        if bool(args.tray_eval):
            tray_eval_before = _run_phase("before", str(args.base_weights))

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(train_doc, tmp)
            tmp.flush()
            tmp_path = tmp.name
        dataset_env: dict[str, str] = {}
        if getattr(args, "dataset_name", None):
            dataset_env["DATASET_NAME"] = str(args.dataset_name)
        if getattr(args, "dataset_root", None):
            dataset_env["DATASET_ROOT"] = str(args.dataset_root)
        if getattr(args, "yolo_data_yaml", None):
            dataset_env["YOLO_DATA_YAML"] = str(args.yolo_data_yaml)
        train_argv = extend_dataset_argv(
            [
                "--name",
                str(args.name),
                "--config",
                tmp_path,
                "--out-dir",
                str(args.out_dir),
                "--skip-eval",
            ],
            manifest=args.manifest,
            default_dataset_name=args.default_dataset_name,
            dataset_env=dataset_env or None,
        )
        rc = int(train_main(train_argv))
        best = run_dir / "weights" / "best.pt"
        if best.is_file():
            weights_out = str(best.resolve())

        if bool(args.tray_eval) and weights_out:
            tray_eval_after = _run_phase("after", weights_out)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    payload = with_schema_version(
        {
            "status": "ok" if rc == 0 and not any(r != 0 for r in eval_rcs) else "failed",
            "script": "finetune",
            "dataset": str(spec.root),
            "base_weights": str(args.base_weights),
            "config": str(exp_path),
            "transfer_config": str(transfer_path),
            "transfer_policy": {
                "freeze_backbone": transfer.get("freeze_backbone"),
                "unfreeze_epoch": transfer.get("unfreeze_epoch"),
                "freeze": transfer.get("freeze"),
            },
            "tray_eval_enabled": bool(args.tray_eval),
            "tray_keys": tray_keys_live,
            "tray_eval_before": tray_eval_before,
            "tray_eval_after": tray_eval_after,
            "tray_eval_warnings": tray_eval_warnings or None,
            "train_returncode": rc,
            "tray_eval_returncodes": eval_rcs or None,
            "run_dir": str(run_dir.resolve()),
            "weights": weights_out,
        },
        schema_version="finetune_run.v1",
    )
    out_path = write_json(args.out, payload)
    print(f"Wrote {out_path}")
    final_rc = rc
    if any(r != 0 for r in eval_rcs):
        final_rc = 1
    return 0 if final_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
