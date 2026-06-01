from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.head_roi_eval import dry_run_payload, run_head_roi_eval
from harchoc.hsp_eval_chain import build_ultralytics_export_argv
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.json_io import load_json, write_json
from harchoc.ml_env import run_repo_python
from scripts._common_cli import add_dataset_args, add_dry_run_arg, cli_print, require_conda_env


def _export_cache_paths(repo_root: Path, run_name: str) -> tuple[Path, Path]:
    prefix = repo_root / "reports/hsp" / run_name
    return prefix.with_name(prefix.name + "_gt.json"), prefix.with_name(prefix.name + "_preds.json")


def _run_export(
    *,
    repo_root: Path,
    weights: str,
    split_file: str,
    device: str,
    run_name: str,
    max_det: int,
) -> tuple[Path, Path]:
    gt_path, preds_path = _export_cache_paths(repo_root, run_name)
    if gt_path.is_file() and preds_path.is_file():
        return gt_path, preds_path
    argv = build_ultralytics_export_argv(
        repo_root=repo_root,
        run_name=run_name,
        weights=weights,
        out_dir="reports/hsp",
        split_file=split_file,
        max_det=max_det,
    )
    export_argv = list(argv)
    for i, tok in enumerate(export_argv):
        if tok == "--export-device":
            export_argv[i + 1] = device
            break
    else:
        export_argv.extend(["--export-device", device])
    proc = run_repo_python(export_argv, repo_root=repo_root)
    if proc.returncode != 0:
        raise SystemExit(f"eval export failed: exit {proc.returncode}")
    if not gt_path.is_file() or not preds_path.is_file():
        raise SystemExit(f"export missing artifacts: {gt_path} / {preds_path}")
    return gt_path, preds_path


def main(argv: list[str] | None = None) -> int:
    require_conda_env()
    p = argparse.ArgumentParser(
        description="Eval-only: apply GT-union head ROI mask to preds, then count @ locked conf."
    )
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--weights", default=HSP_DETECTION_WEIGHTS)
    p.add_argument("--split-file", default="data/splits/test.txt")
    p.add_argument("--device", default="cpu", help="Export inference device (cpu or cuda:0).")
    p.add_argument("--locked-conf-from", default="reports/hsp/threshold_val.json")
    p.add_argument("--out", default="reports/hsp/head_roi_eval_smoke.json")
    p.add_argument("--gt-json", default="", help="Reuse existing GT export (skip infer if set with preds).")
    p.add_argument("--preds-json", default="", help="Reuse existing preds export.")
    p.add_argument("--export-run-name", default="head_roi_eval_smoke_cache")
    p.add_argument("--max-det", type=int, default=3000)
    p.add_argument("--margin-frac", type=float, default=0.02)
    args = p.parse_args(argv)

    repo_root = _repo_root.resolve()
    out_path = Path(args.out).expanduser()
    weights = str(args.weights)
    split_file = str(args.split_file)
    device = str(args.device)
    locked = str(args.locked_conf_from)
    gt_arg = str(args.gt_json or "").strip()
    preds_arg = str(args.preds_json or "").strip()

    if args.dry_run:
        payload = dry_run_payload(
            locked_conf_from=locked,
            weights=weights,
            split_file=split_file,
            out_path=str(out_path),
            device=device,
            gt_json=gt_arg or None,
            preds_json=preds_arg or None,
        )
        gt_cache, pr_cache = _export_cache_paths(repo_root, str(args.export_run_name))
        export_argv = build_ultralytics_export_argv(
            repo_root=repo_root,
            run_name=str(args.export_run_name),
            weights=weights,
            out_dir="reports/hsp",
            split_file=split_file,
            max_det=int(args.max_det),
        )
        cli_print(f"# export: {' '.join(export_argv)}")
        cli_print(f"# cache: {gt_cache} {pr_cache}")
        cli_print(f"# out: {out_path}")
        write_json(out_path, payload)
        return 0

    if gt_arg and preds_arg:
        gt_path = Path(gt_arg).expanduser()
        preds_path = Path(preds_arg).expanduser()
    else:
        gt_path, preds_path = _run_export(
            repo_root=repo_root,
            weights=weights,
            split_file=split_file,
            device=device,
            run_name=str(args.export_run_name),
            max_det=int(args.max_det),
        )

    gt = load_json(gt_path)
    preds = load_json(preds_path)
    result = run_head_roi_eval(
        gt=gt,
        preds=preds,
        locked_conf_from=locked,
        weights=weights,
        split_file=split_file,
        margin_frac=float(args.margin_frac),
    )
    result["gt_json"] = str(gt_path.resolve())
    result["preds_json"] = str(preds_path.resolve())
    result["export_device"] = device
    write_json(out_path, result)
    cli_print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
