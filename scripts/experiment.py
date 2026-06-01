from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()

from harchoc.experiment_config import load_config_json, merge_experiment_config, script_section_from_config
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from scripts._common_cli import add_dataset_args, add_dry_run_arg, write_json


def _ns_to_dict(ns: argparse.Namespace) -> dict[str, Any]:
    return {k: v for k, v in vars(ns).items() if v is not argparse.SUPPRESS}


_SUBCOMMAND_RUN_KIND: dict[str, str] = {
    "benchmark": "benchmark_matrix",
    "cv-eval": "cv_eval",
}


def _merged_for_subcommand(
    *,
    config_obj: dict[str, Any],
    subcommand: str,
    cli_fields: dict[str, Any],
    force_dry_run: bool,
) -> dict[str, Any]:
    cfg = config_obj.get(subcommand)
    if not isinstance(cfg, dict):
        alt_key = _SUBCOMMAND_RUN_KIND.get(subcommand, subcommand.replace("-", "_"))
        if alt_key != subcommand:
            cfg = config_obj.get(alt_key)
    cfg_obj = cfg if isinstance(cfg, dict) else {}
    if not cfg_obj:
        section = _SUBCOMMAND_RUN_KIND.get(subcommand, subcommand.replace("-", "_"))
        cfg_obj = script_section_from_config(config_obj, section)
    merged = merge_experiment_config(config=cfg_obj, cli=cli_fields)
    if force_dry_run:
        merged["dry_run"] = True
    return merged


def _run_gradcam(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_gradcam
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_gradcam(fields)
    if bool(fields.get("dry_run")):
        print("# gradcam")
        print(_format_cmd(["scripts/make_figures.py", *argv], mamba=True))
        return 0
    from scripts.make_figures import main as legacy_main

    return legacy_main(argv)


def _run_deploy_parity(fields: dict[str, Any]) -> int:
    from harchoc.deploy_hsp_parity import (
        build_deploy_hsp_parity_payload,
        resolve_parity_image_sample,
        write_deploy_hsp_parity,
    )
    from harchoc.experiment_argv import argv_for_deploy_parity
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_deploy_parity(fields)
    out = str(fields.get("out") or "reports/hsp/deploy_hsp_parity.json")
    locked = fields.get("locked_conf_from") or "reports/hsp/threshold_val.json"
    sample_n = int(fields.get("sample_images") or 0)
    split_file = str(fields.get("split_file") or "data/splits/test.txt")
    weights = str(fields.get("weights") or HSP_DETECTION_WEIGHTS)
    if bool(fields.get("dry_run")):
        print("# deploy-parity")
        print(_format_cmd(["scripts/experiment.py", "deploy-parity", *argv], mamba=False))
        payload = build_deploy_hsp_parity_payload(locked_conf_from=str(locked))
        if sample_n > 0:
            print(f"# would sample up to {sample_n} images from {split_file}")
        print(f"# would write {out} status={payload.get('status')}")
        return 0

    image_paths: list[str] | None = None
    per_image = None
    notes: str | None = None
    if sample_n > 0:
        image_paths, per_image, skip_note = resolve_parity_image_sample(
            sample_images=sample_n,
            split_file=split_file,
            dataset_root=fields.get("dataset_root"),
            locked_conf_from=str(locked),
            weights=weights,
            manifest_path=fields.get("manifest"),
            dataset_name=fields.get("dataset_name"),
            yolo_data_yaml=fields.get("yolo_data_yaml"),
            default_dataset_name=fields.get("default_dataset_name"),
        )
        if skip_note:
            notes = skip_note

    write_deploy_hsp_parity(
        out,
        locked_conf_from=str(locked),
        image_paths=image_paths,
        per_image=per_image,
        notes=notes,
    )
    print(f"Wrote {out}")
    if per_image:
        print(f"Compared {len(per_image)} image(s) (SAHI vs full-frame @ locked conf)")
    elif sample_n > 0 and notes:
        print(f"Image sample skipped: {notes}")
    return 0


def _run_map_cpu(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_map_cpu
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_map_cpu(fields)
    if bool(fields.get("dry_run")):
        print("# map-cpu")
        print(_format_cmd(["scripts/eval.py", *argv], mamba=True))
        return 0
    from scripts.eval import main as legacy_main

    return legacy_main(argv)


def _run_tune_sahi(fields: dict[str, Any]) -> int:
    from harchoc.experiment_argv import argv_for_tune_sahi
    from harchoc.manuscript_repro import _format_cmd

    argv = argv_for_tune_sahi(fields)
    if bool(fields.get("dry_run")):
        print("# tune-sahi")
        print(_format_cmd(["scripts/experiment.py", "tune-sahi", *argv], mamba=True))
        return 0
    raise SystemExit(
        "tune-sahi live grid was removed with tune_sahi_params.py; "
        "use run_infer_once.py or experiment.py deploy-parity for deploy tuning."
    )


def _run_fp_budget_sweep(fields: dict[str, Any]) -> int:
    from harchoc.fp_budget_sweep import run_fp_budget_sweep
    from harchoc.manuscript_repro import _format_cmd
    from harchoc.threshold_protocol import enforce_tuning_guardrails, infer_split_role, resolve_dataset_root_for_splits
    from scripts._common_cli import cli_print, read_json, require_conda_env, resolve_light_gt_preds, write_json

    require_conda_env()
    out = str(fields.get("out") or "reports/hsp/fp_budget_sweep.json")
    if bool(fields.get("dry_run")):
        payload = run_fp_budget_sweep(
            out=out,
            gt_json="",
            preds_json="",
            gt={},
            preds={},
            fp_budget_grid=list(fields.get("fp_budget_grid") or []),
            dry_run=True,
        )
        write_json(out, payload)
        print("# fp-budget-sweep")
        print(_format_cmd(["scripts/experiment.py", "fp-budget-sweep", "--out", out], mamba=False))
        cli_print(f"Wrote {out}")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    gt_json = str(fields.get("gt_json") or "").strip()
    preds_json = str(fields.get("preds_json") or "").strip()
    if bool(fields.get("light")):
        gt_path, preds_path = resolve_light_gt_preds(repo_root=repo_root, gt_json=gt_json, preds_json=preds_json)
        gt_json, preds_json = str(gt_path), str(preds_path)

    gt_obj = read_json(gt_json) if gt_json else None
    preds_obj = read_json(preds_json) if preds_json else None
    if gt_obj is None or preds_obj is None:
        raise SystemExit("fp-budget-sweep requires --gt-json and --preds-json (or --light)")

    ds_root = resolve_dataset_root_for_splits(repo_root=repo_root, dataset_root=None)
    split_role, split_hints = infer_split_role(
        gt_json=gt_json,
        preds_json=preds_json,
        split_file=(str(fields.get("split_file") or "").strip() or None),
        gt=gt_obj,
        repo_root=repo_root,
        dataset_root=ds_root,
    )
    sweep_from = (str(fields.get("sweep_from") or "").strip() or None)
    locked_from = (str(fields.get("locked_conf_from") or "").strip() or None)
    enforce_tuning_guardrails(
        split_role,
        locked_conf_from=sweep_from or locked_from,
        iou_grid=None,
        calibrate="none",
    )

    grid_raw = fields.get("fp_budget_grid")
    fp_grid = list(grid_raw) if isinstance(grid_raw, list) and grid_raw else None
    iou_raw = fields.get("iou")
    iou_thr = float(iou_raw) if iou_raw is not None else None

    payload = run_fp_budget_sweep(
        out=out,
        gt_json=gt_json,
        preds_json=preds_json,
        gt=gt_obj,
        preds=preds_obj,
        sweep_from=(str(fields.get("sweep_from") or "").strip() or None),
        fp_budget_grid=fp_grid,
        iou_thr=iou_thr,
        split_role=split_role,
        split_hints=split_hints,
        locked_conf_from=(str(fields.get("locked_conf_from") or "").strip() or None),
        summary_out=(str(fields.get("summary_out") or "").strip() or None),
    )
    cli_print(f"Wrote {out}")
    summary_out = (str(fields.get("summary_out") or "").strip() or None)
    if summary_out:
        cli_print(f"Wrote {summary_out}")
    return 0


def _run_dual_metric(fields: dict[str, Any]) -> int:
    from harchoc.dual_metric_report import build_dry_run_report, merge_dual_metric_from_paths
    from harchoc.experiment_argv import argv_for_dual_metric_bundle

    fields = argv_for_dual_metric_bundle(fields)
    out_path = str(fields.get("out") or "reports/dual_metric/report.json")
    eval_paths = fields.get("eval")
    eval_list: list[str] = list(eval_paths) if isinstance(eval_paths, list) else []

    inputs = {
        "eval_val": str(fields.get("eval_val") or ""),
        "eval_test": str(fields.get("eval_test") or ""),
        "eval_val_map": str(fields.get("eval_val_map") or ""),
        "eval_test_map": str(fields.get("eval_test_map") or ""),
        "sweep_val": str(fields.get("sweep") or ""),
        "sweep_test": str(fields.get("sweep_test") or ""),
        "error_val": str(fields.get("error_val") or ""),
        "error_test": str(fields.get("error_test") or ""),
    }

    if bool(fields.get("dry_run")):
        payload = build_dry_run_report(out=out_path, inputs=inputs)
        write_json(out_path, payload)
        return 0

    sweep_val = str(fields.get("sweep") or "").strip()
    error_val = str(fields.get("error_val") or "").strip()
    error_test = str(fields.get("error_test") or "").strip()
    if not sweep_val or not error_val or not error_test:
        raise SystemExit(
            "dual-metric requires --sweep, --error-val, and --error-test "
            "(plus val/test eval via --eval-val/--eval-test or --eval ×2)"
        )

    payload = merge_dual_metric_from_paths(
        eval_val=(str(fields.get("eval_val")).strip() or None) if fields.get("eval_val") else None,
        eval_test=(str(fields.get("eval_test")).strip() or None) if fields.get("eval_test") else None,
        eval_val_map=(str(fields.get("eval_val_map")).strip() or None)
        if fields.get("eval_val_map")
        else None,
        eval_test_map=(str(fields.get("eval_test_map")).strip() or None)
        if fields.get("eval_test_map")
        else None,
        eval_paths=eval_list,
        sweep_val=sweep_val,
        sweep_test=(str(fields.get("sweep_test")).strip() or None) if fields.get("sweep_test") else None,
        error_val=error_val,
        error_test=error_test,
    )
    write_json(out_path, payload)
    return 0


def _run_hpo(fields: dict[str, Any]) -> int:
    """
    Budgeted HPO planner.

    Intentionally lightweight: generates trial train configs and writes a JSON report.
    Execution of training loops is intentionally out-of-scope here (use the generated
    train configs with `scripts/train.py` / `scripts/benchmark_matrix.py`).
    """
    from harchoc.hpo_search import plan_hpo_trials

    out_path = str(fields.get("out") or "reports/hpo/hpo_plan.json")
    base_train_config = str(fields.get("base_train_config") or "").strip()
    if not base_train_config:
        raise SystemExit("hpo requires --base-train-config (path to train config JSON)")

    trials = int(fields.get("trials") or 20)
    seed = int(fields.get("seed") or 0)
    space_raw = fields.get("space")
    if space_raw is None:
        raise SystemExit("hpo requires --space (JSON object or path to JSON file)")

    if isinstance(space_raw, str):
        import json
        from pathlib import Path

        s = space_raw.strip()
        p = Path(s).expanduser()
        space = json.loads(p.read_text("utf-8")) if p.is_file() else json.loads(s)
    elif isinstance(space_raw, dict):
        space = space_raw
    else:
        raise SystemExit("--space must be a JSON object or a JSON string/path")

    payload = plan_hpo_trials(
        base_train_config=base_train_config,
        space=space,
        trials=trials,
        seed=seed,
    )
    write_json(out_path, payload)
    return 0


_DATASET_INHERIT_KEYS = (
    "manifest",
    "default_dataset_name",
    "dataset_name",
    "dataset_root",
    "yolo_data_yaml",
)


def _inherit_dataset_fields(*, dataset_cfg: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fields)
    for key in _DATASET_INHERIT_KEYS:
        if key not in merged and dataset_cfg.get(key) is not None:
            merged[key] = dataset_cfg[key]
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Unified experiment entrypoint.")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Inline JSON object or path to JSON file. Can be repeated; later entries override earlier.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run for the selected subcommand.",
    )

    sp = p.add_subparsers(dest="cmd", required=True)

    # `splits` legacy args
    ps = sp.add_parser("splits", help="Create train/val/test split lists.")
    add_dataset_args(ps, suppress_defaults=True)
    add_dry_run_arg(ps, suppress_defaults=True)
    ps.add_argument("--mode", choices=["from-folders", "random"], default=argparse.SUPPRESS)
    ps.add_argument("--out-dir", default=argparse.SUPPRESS)
    ps.add_argument("--ext", action="append", default=argparse.SUPPRESS)
    ps.add_argument("--glob", default=argparse.SUPPRESS)
    ps.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    ps.add_argument("--val-frac", type=float, default=argparse.SUPPRESS)
    ps.add_argument("--test-frac", type=float, default=argparse.SUPPRESS)

    # `describe` legacy args
    pd = sp.add_parser("describe", help="Describe dataset split stats.")
    add_dataset_args(pd, suppress_defaults=True)
    add_dry_run_arg(pd, suppress_defaults=True)
    pd.add_argument("--split", choices=["train", "val", "test"], default=argparse.SUPPRESS)
    pd.add_argument("--split-file", action="append", default=argparse.SUPPRESS)
    pd.add_argument("--out", default=argparse.SUPPRESS)

    # `eval` legacy args
    pe = sp.add_parser("eval", help="Evaluate a trained model.")
    add_dataset_args(pe, suppress_defaults=True)
    add_dry_run_arg(pe, suppress_defaults=True)
    pe.add_argument("--weights", default=argparse.SUPPRESS)
    pe.add_argument("--split-file", default=argparse.SUPPRESS)
    pe.add_argument("--out", default=argparse.SUPPRESS)

    # `benchmark` legacy args
    pb = sp.add_parser("benchmark", help="Run benchmark matrix harness.")
    add_dataset_args(pb, suppress_defaults=True)
    # keep name aligned with legacy file: bench-config, bench-dir, ...
    pb.add_argument("--bench-config", action="append", default=argparse.SUPPRESS)
    pb.add_argument("--bench-dir", default=argparse.SUPPRESS)
    pb.add_argument("--pattern", default=argparse.SUPPRESS)
    pb.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    pb.add_argument("--out", default=argparse.SUPPRESS)
    pb.add_argument("--no-train", action="store_true", default=argparse.SUPPRESS)
    pb.add_argument("--no-eval", action="store_true", default=argparse.SUPPRESS)
    pb.add_argument("--eval-out", default=argparse.SUPPRESS)
    pb.add_argument("--no-dry-run", action="store_true", default=argparse.SUPPRESS)
    pb.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)

    # `train` args (new)
    pt = sp.add_parser("train", help="Train a detection model with reproducible outputs.")
    add_dataset_args(pt, suppress_defaults=True)
    add_dry_run_arg(pt, suppress_defaults=True)
    pt.add_argument("--config", default=argparse.SUPPRESS)
    pt.add_argument("--out-dir", default=argparse.SUPPRESS)
    pt.add_argument("--name", default=argparse.SUPPRESS)

    ph = sp.add_parser("hpo", help="Plan a budget-capped hyperparameter search.")
    ph.add_argument("--out", default=argparse.SUPPRESS)
    ph.add_argument("--base-train-config", default=argparse.SUPPRESS)
    ph.add_argument("--space", default=argparse.SUPPRESS)
    ph.add_argument("--trials", type=int, default=argparse.SUPPRESS)
    ph.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    pr = sp.add_parser(
        "repro",
        help="Run the HSP manuscript reproducibility chain from manuscript_repro_bundle.json.",
    )
    pr.add_argument(
        "--bundle",
        default=argparse.SUPPRESS,
        help="Path to manuscript_repro_bundle.v1 JSON (default: configs/experiments/manuscript_repro_bundle.json).",
    )
    add_dry_run_arg(pr, suppress_defaults=True)
    pr.add_argument(
        "--skip-gpu-check",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip scripts/check_gpu.py at chain start.",
    )
    pr.add_argument(
        "--include-test-map",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Append optional test mAP eval + dual-metric regen (GPU; may OOM on 8 GiB).",
    )

    pcv = sp.add_parser(
        "cv-eval",
        help="K-fold split planning and fold-metric aggregation (delegates to cv_eval.py).",
    )
    add_dataset_args(pcv, suppress_defaults=True)
    add_dry_run_arg(pcv, suppress_defaults=True)
    pcv.add_argument("--weights", default=argparse.SUPPRESS)
    pcv.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    pcv.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    pcv.add_argument("--splits-dir", default=argparse.SUPPRESS)
    pcv.add_argument("--fold-metrics", action="append", default=argparse.SUPPRESS)
    pcv.add_argument("--out", default=argparse.SUPPRESS)
    pcv.add_argument("--write-fold-splits", default=argparse.SUPPRESS)

    pgc = sp.add_parser(
        "gradcam",
        help="Grad-CAM FP crop panel (delegates to make_figures.py --figure fig_gradcam_panel).",
    )
    add_dry_run_arg(pgc, suppress_defaults=True)
    pgc.add_argument("--out-dir", default=argparse.SUPPRESS)
    pgc.add_argument("--meta-out", default=argparse.SUPPRESS)
    pgc.add_argument("--error-report", default=argparse.SUPPRESS)
    pgc.add_argument("--weights", default=argparse.SUPPRESS)
    pgc.add_argument("--panel-size", type=int, default=argparse.SUPPRESS)

    pdp = sp.add_parser(
        "deploy-parity",
        help="Deploy SAHI conf vs HSP locked conf side-by-side JSON (R-SCI-2).",
    )
    add_dataset_args(pdp, suppress_defaults=True)
    add_dry_run_arg(pdp, suppress_defaults=True)
    pdp.add_argument(
        "--locked-conf-from",
        default=argparse.SUPPRESS,
        help="Val threshold JSON for manuscript locked conf.",
    )
    pdp.add_argument("--out", default=argparse.SUPPRESS)
    pdp.add_argument(
        "--sample-images",
        type=int,
        default=argparse.SUPPRESS,
        help="Compare SAHI vs full-frame counts on N test-split images.",
    )
    pdp.add_argument("--split-file", default=argparse.SUPPRESS)
    pdp.add_argument("--weights", default=argparse.SUPPRESS)

    pcm = sp.add_parser(
        "map-cpu",
        help="Test-split mAP eval on CPU (SCI-MAP-CPU; delegates to eval.py).",
    )
    add_dataset_args(pcm, suppress_defaults=True)
    add_dry_run_arg(pcm, suppress_defaults=True)
    pcm.add_argument("--weights", default=argparse.SUPPRESS)
    pcm.add_argument("--split-file", default=argparse.SUPPRESS)
    pcm.add_argument("--imgsz", type=int, default=argparse.SUPPRESS)
    pcm.add_argument("--max-det", type=int, default=argparse.SUPPRESS)
    pcm.add_argument("--device", default=argparse.SUPPRESS)
    pcm.add_argument("--out", default=argparse.SUPPRESS)

    pts = sp.add_parser(
        "tune-sahi",
        help="Deploy SAHI slice/conf grid vs manual GT (dry-run argv only; live grid removed).",
    )
    add_dry_run_arg(pts, suppress_defaults=True)
    pts.add_argument(
        "--image",
        default=argparse.SUPPRESS,
        help="Reference head image (default: test_sunflower_tune.png).",
    )

    pdm = sp.add_parser(
        "dual-metric",
        help="Merge eval, threshold sweep, and error-analysis JSON into a manuscript table.",
    )
    add_dry_run_arg(pdm, suppress_defaults=True)
    pdm.add_argument("--out", default=argparse.SUPPRESS)
    pdm.add_argument("--eval-val", default=argparse.SUPPRESS)
    pdm.add_argument("--eval-test", default=argparse.SUPPRESS)
    pdm.add_argument(
        "--eval-val-map",
        default=argparse.SUPPRESS,
        help="Optional eval_run.v1 with mAP when --eval-val is export-only.",
    )
    pdm.add_argument(
        "--eval-test-map",
        default=argparse.SUPPRESS,
        help="Optional eval_run.v1 with mAP when --eval-test is export-only.",
    )
    pdm.add_argument(
        "--eval",
        action="append",
        default=argparse.SUPPRESS,
        help="eval_run.v1 path; split_role val|test (use twice or with --eval-val/--eval-test).",
    )
    pdm.add_argument("--sweep", default=argparse.SUPPRESS, help="Val threshold_sweep_run.v1 JSON.")
    pdm.add_argument("--sweep-test", default=argparse.SUPPRESS, help="Optional test sweep with locked block.")
    pdm.add_argument("--error-val", default=argparse.SUPPRESS)
    pdm.add_argument("--error-test", default=argparse.SUPPRESS)

    pfb = sp.add_parser(
        "fp-budget-sweep",
        help="Val constraint-sweep ablation: count-first vs F1-max vs max_fp_per_image grid (P1-FP-BUDGET).",
    )
    add_dataset_args(pfb, suppress_defaults=True)
    add_dry_run_arg(pfb, suppress_defaults=True)
    pfb.add_argument("--out", default=argparse.SUPPRESS)
    pfb.add_argument("--gt-json", default=argparse.SUPPRESS)
    pfb.add_argument("--preds-json", default=argparse.SUPPRESS)
    pfb.add_argument(
        "--sweep-from",
        default=argparse.SUPPRESS,
        help="Reuse rows from threshold_sweep_run.v1 (e.g. reports/hsp/threshold_val.json).",
    )
    pfb.add_argument("--split-file", default=argparse.SUPPRESS)
    pfb.add_argument("--iou", type=float, default=argparse.SUPPRESS)
    pfb.add_argument(
        "--fp-budget-grid",
        type=float,
        action="append",
        default=argparse.SUPPRESS,
        help="Repeatable max_fp_per_image caps (default grid in harchoc.fp_budget_sweep).",
    )
    pfb.add_argument("--light", action="store_true", default=argparse.SUPPRESS)

    pgq = sp.add_parser(
        "gpu-queue",
        help="Sequential GPU backlog queue from gpu_queue manifest (dry-run or --run).",
    )
    pgq.add_argument(
        "--manifest",
        required=True,
        help="Path to gpu_queue_manifest.v1 JSON (e.g. configs/experiments/gpu_queue_full.json).",
    )
    add_dry_run_arg(pgq, suppress_defaults=True)
    pgq.add_argument(
        "--run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Execute jobs (default: dry-run prints stage commands).",
    )
    pgq.add_argument(
        "--resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume from reports/gpu_queue/run_state.json.",
    )
    pgq.add_argument("--job", default=argparse.SUPPRESS, help="Run only this job id.")
    pgq.add_argument(
        "--state-path",
        default=argparse.SUPPRESS,
        help="Override run state JSON (default reports/gpu_queue/run_state.json).",
    )
    pgq.add_argument(
        "--min-free-mib",
        type=int,
        default=argparse.SUPPRESS,
        help="GPU wait threshold MiB free (default 5500).",
    )

    pvas = sp.add_parser(
        "validate-aug-smoke",
        help="Validate aug_smoke_index.json vs runtime train/aug configs (CI-safe, no ML deps).",
    )
    pvas.add_argument(
        "--index",
        default=argparse.SUPPRESS,
        help="aug_smoke_index.v1 JSON (default configs/experiments/aug_smoke_index.json).",
    )

    pal = sp.add_parser(
        "aug-leaderboard",
        help="Regenerate reports/aug_smoke/leaderboard.md from index + summaries (P2-AUG-RANK-REPORT).",
    )
    pal.add_argument(
        "--index",
        default=argparse.SUPPRESS,
        help="aug_smoke_index.v1 JSON (default configs/experiments/aug_smoke_index.json).",
    )
    pal.add_argument(
        "--out-dir",
        default=argparse.SUPPRESS,
        help="Output directory (default reports/aug_smoke).",
    )
    pal.add_argument(
        "--no-json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip leaderboard.json.",
    )

    args = p.parse_args(argv)

    # Merge all --config entries (left-to-right).
    config_obj: dict[str, Any] = {}
    for raw in args.config:
        cfg = load_config_json(raw)
        config_obj = merge_experiment_config(config=config_obj, cli=cfg)

    dataset_cfg = config_obj.get("dataset")
    dataset_cfg_obj = dataset_cfg if isinstance(dataset_cfg, dict) else {}

    cmd = str(args.cmd)
    cli_fields = _ns_to_dict(args)
    # Remove top-level keys not relevant to underlying scripts.
    cli_fields.pop("cmd", None)
    cli_fields.pop("config", None)
    force_dry_run = bool(args.dry_run)
    cli_fields.pop("dry_run", None)  # global dry-run handled below

    merged_fields = _merged_for_subcommand(
        config_obj=config_obj,
        subcommand=cmd,
        cli_fields=cli_fields,
        force_dry_run=force_dry_run,
    )
    merged_fields = _inherit_dataset_fields(dataset_cfg=dataset_cfg_obj, fields=merged_fields)

    if cmd == "splits":
        from harchoc.experiment_argv import argv_for_splits
        from scripts.make_splits import main as legacy_main

        return legacy_main(argv_for_splits(merged_fields))
    if cmd == "describe":
        from harchoc.experiment_argv import argv_for_describe
        from scripts.describe_split import main as legacy_main

        return legacy_main(argv_for_describe(merged_fields))
    if cmd == "eval":
        from harchoc.experiment_argv import argv_for_eval
        from scripts.eval import main as legacy_main

        return legacy_main(argv_for_eval(merged_fields))
    if cmd == "benchmark":
        from harchoc.experiment_argv import argv_for_benchmark
        from scripts.benchmark_matrix import main as legacy_main

        return legacy_main(argv_for_benchmark(merged_fields))
    if cmd == "train":
        from harchoc.experiment_argv import argv_for_train
        from scripts.train import main as legacy_main

        return legacy_main(argv_for_train(merged_fields))
    if cmd == "hpo":
        return _run_hpo(merged_fields)
    if cmd == "dual-metric":
        return _run_dual_metric(merged_fields)
    if cmd == "fp-budget-sweep":
        return _run_fp_budget_sweep(merged_fields)
    if cmd == "cv-eval":
        from harchoc.experiment_argv import argv_for_cv_eval
        from scripts.cv_eval import main as legacy_main

        return legacy_main(argv_for_cv_eval(merged_fields))
    if cmd == "gradcam":
        return _run_gradcam(merged_fields)
    if cmd == "deploy-parity":
        return _run_deploy_parity(merged_fields)
    if cmd == "map-cpu":
        return _run_map_cpu(merged_fields)
    if cmd == "tune-sahi":
        return _run_tune_sahi(merged_fields)
    if cmd == "repro":
        from harchoc.manuscript_repro import load_manuscript_repro_bundle, run_manuscript_repro_chain

        repo_root = Path(__file__).resolve().parents[1]
        bundle_path = str(
            merged_fields.get("bundle") or "configs/experiments/manuscript_repro_bundle.json"
        ).strip()
        bundle = load_manuscript_repro_bundle(bundle_path)
        return run_manuscript_repro_chain(
            bundle,
            repo_root=repo_root,
            dry_run=bool(merged_fields.get("dry_run")),
            skip_gpu_check=bool(merged_fields.get("skip_gpu_check")),
            include_test_map=bool(merged_fields.get("include_test_map")),
        )
    if cmd == "validate-aug-smoke":
        from harchoc.aug_smoke_train import validate_aug_smoke_configs

        repo_root = Path(__file__).resolve().parents[1]
        index_arg = merged_fields.get("index") or cli_fields.get("index")
        errors = validate_aug_smoke_configs(
            repo_root,
            index_path=str(index_arg) if index_arg else None,
        )
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print("aug_smoke config validation OK")
        return 0
    if cmd == "aug-leaderboard":
        from harchoc.aug_smoke_leaderboard import write_aug_smoke_leaderboard

        repo_root = Path(__file__).resolve().parents[1]
        paths = write_aug_smoke_leaderboard(
            repo_root=repo_root,
            index_path=str(
                merged_fields.get("index")
                or cli_fields.get("index")
                or "configs/experiments/aug_smoke_index.json"
            ),
            out_dir=str(
                merged_fields.get("out_dir")
                or cli_fields.get("out_dir")
                or "reports/aug_smoke"
            ),
            write_json=not bool(merged_fields.get("no_json") or cli_fields.get("no_json")),
        )
        for label, path in paths.items():
            print(f"Wrote {label}: {path}")
        return 0
    if cmd == "gpu-queue":
        from harchoc.experiment_argv import argv_for_gpu_queue
        from scripts.run_gpu_queue import main as run_gpu_queue_main

        return run_gpu_queue_main(argv_for_gpu_queue(merged_fields))
    raise SystemExit(f"Unknown subcommand: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
