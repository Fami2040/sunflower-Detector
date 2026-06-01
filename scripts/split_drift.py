from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()
from harchoc.config_coerce import as_dict, child_dict, split_lists_from_source
from harchoc.datasets import describe_dataset, resolve_dataset
from harchoc.strict_ml import capture_failure, fail_or_warn
from harchoc.schemas import with_schema_version
from harchoc.splits_io import read_split_list
from harchoc.split_drift_extended import build_extended_drift_block
from harchoc.split_drift_policy import DriftAcceptanceConfig, evaluate_acceptance
from harchoc.split_drift_plots import emit_split_drift_plots
from harchoc.split_leakage_audit import audit_split_leakage, splits_from_split_dir
from scripts._common_cli import add_dataset_args, add_dry_run_arg, cli_print, require_existing_dir, write_json
from scripts.describe_split import describe_split_stats


def _available_splits(*, dataset_root: Path, splits_dir: str) -> dict[str, Path]:
    d = dataset_root / splits_dir
    candidates = {"train": d / "train.txt", "val": d / "val.txt", "test": d / "test.txt"}
    return {k: v for k, v in candidates.items() if v.exists()}


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return float(a) / float(b)


def _as_float(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _class_probs(class_counts: dict[str, int]) -> dict[str, float]:
    total = sum(int(v) for v in class_counts.values())
    if total <= 0:
        return {k: 0.0 for k in class_counts}
    return {k: float(v) / float(total) for k, v in class_counts.items()}


def _dist_l1(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def _js_divergence(p: dict[str, float], q: dict[str, float], *, eps: float = 1e-12) -> float | None:
    """
    Jensen–Shannon divergence in nats.
    Returns None if both distributions are empty/degenerate.
    """
    import math

    keys = set(p) | set(q)
    if not keys:
        return None

    sp = sum(float(p.get(k, 0.0)) for k in keys)
    sq = sum(float(q.get(k, 0.0)) for k in keys)
    if sp <= 0 and sq <= 0:
        return None
    pn = {k: (float(p.get(k, 0.0)) / sp if sp > 0 else 0.0) for k in keys}
    qn = {k: (float(q.get(k, 0.0)) / sq if sq > 0 else 0.0) for k in keys}

    m = {k: 0.5 * (pn[k] + qn[k]) for k in keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            ak = float(a[k])
            bk = float(b[k])
            if ak <= 0:
                continue
            s += ak * math.log(max(ak, eps) / max(bk, eps))
        return s

    return 0.5 * _kl(pn, m) + 0.5 * _kl(qn, m)


def _pairwise_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_img = a.get("images", {})
    b_img = b.get("images", {})
    a_lbl = a.get("labels", {})
    b_lbl = b.get("labels", {})
    a_den = a.get("density", {})
    b_den = b.get("density", {})

    def _mean(d: dict[str, Any]) -> float | None:
        return _as_float(d.get("mean"))

    class_counts_a = dict(a_lbl.get("class_counts") or {})
    class_counts_b = dict(b_lbl.get("class_counts") or {})
    pa = _class_probs({k: int(v) for k, v in class_counts_a.items()})
    pb = _class_probs({k: int(v) for k, v in class_counts_b.items()})

    return {
        "images": {
            "count_ratio": _safe_div(_as_float(b_img.get("count")), _as_float(a_img.get("count"))),
            "width_mean_ratio": _safe_div(_mean(b_img.get("width", {})), _mean(a_img.get("width", {}))),
            "height_mean_ratio": _safe_div(_mean(b_img.get("height", {})), _mean(a_img.get("height", {}))),
            "file_size_mean_ratio": _safe_div(
                _mean(b_img.get("file_size_bytes", {})), _mean(a_img.get("file_size_bytes", {}))
            ),
        },
        "labels": {
            "boxes_per_image_mean_ratio": _safe_div(
                _mean(b_lbl.get("boxes_per_image", {})), _mean(a_lbl.get("boxes_per_image", {}))
            ),
            "class_dist_l1": _dist_l1(pa, pb),
            "class_jsd_nats": _js_divergence(pa, pb),
        },
        "density": {
            "boxes_per_megapixel_ratio": _safe_div(
                _as_float(b_den.get("boxes_per_megapixel")),
                _as_float(a_den.get("boxes_per_megapixel")),
            ),
        },
    }


def _try_ks_payload(a: list[float], b: list[float]) -> dict[str, Any]:
    """
    Two-sample KS test on proxy series (requires scipy from requirements.txt / harchoc env).
    """
    try:
        from scipy.stats import ks_2samp  # type: ignore
    except ImportError as ex:
        raise SystemExit(
            "scipy is required for --with-ks.\n"
            "Install the project env: mamba env create -f envs/mamba.yml && "
            "python scripts/bootstrap_env.py --env harchoc\n"
            f"Import error: {ex}"
        ) from ex
    if not a or not b:
        return {"available": True, "n_a": len(a), "n_b": len(b), "statistic": None, "pvalue": None}
    r = ks_2samp(a, b)
    return {
        "available": True,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "statistic": float(getattr(r, "statistic", None) or 0.0),
        "pvalue": float(getattr(r, "pvalue", None) or 0.0),
    }


def _collect_proxy_series(
    *, dataset_root: Path, split_list: list[str], limit: int
) -> tuple[list[float], list[float], list[float]]:
    from harchoc.eval_export import read_image_size
    from scripts.describe_split import _infer_label_path, _parse_yolo_label_file

    widths: list[float] = []
    heights: list[float] = []
    boxes: list[float] = []
    root = dataset_root.resolve()
    for rel in split_list[: max(0, int(limit))]:
        p = Path(rel)
        img = p if p.is_absolute() else (root / p)
        if not img.is_file():
            continue
        with capture_failure(f"read image size {img}") as cap:
            w, h = read_image_size(img)
        if cap.failed:
            fail_or_warn(f"{cap.context}: {cap.exc_type}: {cap.exc_msg}")
            continue
        widths.append(float(w))
        heights.append(float(h))
        lbl = _infer_label_path(dataset_root=root, image_path=img)
        boxes.append(float(len(_parse_yolo_label_file(lbl))))
    return widths, heights, boxes


def split_drift_report(*, dataset_root: Path, splits_dir: str, yolo_data_yaml: Path | None) -> dict[str, Any]:
    split_files = _available_splits(dataset_root=dataset_root, splits_dir=splits_dir)

    splits: dict[str, dict[str, Any]] = {}
    used_split_lists: dict[str, list[str] | None] = {}

    for split in ("train", "val", "test"):
        txt = split_files.get(split)
        split_list = read_split_list(txt, missing_ok=True) if txt is not None else []
        used_split_lists[split] = split_list if split_list else None

        split_files_arg: list[Path] | None
        if txt is not None and split_list:
            split_files_arg = [txt]
        else:
            split_files_arg = None

        splits[split] = describe_split_stats(
            dataset_root=dataset_root,
            split=split,
            split_files=split_files_arg,
            yolo_data_yaml=yolo_data_yaml,
        )

    comparisons: dict[str, Any] = {
        "train_vs_val": _pairwise_comparison(splits["train"], splits["val"]),
        "train_vs_test": _pairwise_comparison(splits["train"], splits["test"]),
        "val_vs_test": _pairwise_comparison(splits["val"], splits["test"]),
    }

    return {
        "status": "ok",
        "script": "split_drift",
        "source": {
            "dataset_root": str(dataset_root),
            "splits_dir": splits_dir,
            "split_lists": used_split_lists,
        },
        "splits": splits,
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare train/val/test splits via lightweight proxy statistics.")
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory (relative to dataset root) containing train.txt/val.txt/test.txt if present.",
    )
    p.add_argument(
        "--out",
        default="reports/hsp/split_drift_p0.json",
        help="Where to write JSON report (parent dir created as needed).",
    )
    p.add_argument(
        "--with-ks",
        action="store_true",
        help="Add KS tests on proxy series (requires scipy; installed via requirements.txt / bootstrap_env).",
    )
    p.add_argument(
        "--ks-limit",
        type=int,
        default=5000,
        help="Max images per split to sample for KS tests (only when --with-ks).",
    )
    p.add_argument(
        "--acceptance-config",
        default="",
        help="Optional JSON file with KS/JSD/L1 acceptance thresholds.",
    )
    p.add_argument(
        "--emit-plots",
        action="store_true",
        help="Write drift summary PNGs under --plots-dir (requires matplotlib).",
    )
    p.add_argument(
        "--plots-dir",
        default="figures/split_drift",
        help="Output directory for --emit-plots.",
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="Add extended block: per-class box means, bbox area quantiles, images-per-tray.",
    )
    p.add_argument(
        "--extended-limit",
        type=int,
        default=0,
        help="Max images per split for --extended scan (0 = no limit).",
    )
    p.add_argument(
        "--catalog",
        default="",
        help="Optional domain catalog JSON (e.g. reports/domains/catalog.json) for tray reference.",
    )
    args = p.parse_args(argv)

    if args.dry_run:
        out_path = write_json(
            args.out,
            with_schema_version(
                {
                    "status": "dry-run",
                    "script": "split_drift",
                    "out": str(Path(args.out)),
                },
                schema_version="split_drift_report.v1",
            ),
        )
        cli_print(f"Wrote {out_path}")
        return 0

    spec = resolve_dataset(
        manifest_path=args.manifest,
        default_dataset_name=args.default_dataset_name,
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        yolo_data_yaml=args.yolo_data_yaml,
    )
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
    root = Path(spec.root)

    payload = split_drift_report(dataset_root=root, splits_dir=str(args.splits_dir), yolo_data_yaml=spec.yolo_data_yaml)
    splits_dir = root / str(args.splits_dir)
    payload["leakage_audit"] = audit_split_leakage(splits_from_split_dir(splits_dir))
    if bool(args.extended):
        split_lists = split_lists_from_source(payload)
        catalog_path = Path(args.catalog).expanduser() if (args.catalog or "").strip() else None
        payload["extended"] = build_extended_drift_block(
            dataset_root=root,
            split_lists=split_lists,
            catalog_path=catalog_path,
            scan_limit=int(args.extended_limit),
        )
    if bool(args.with_ks):
        # Add best-effort KS tests without changing the base schema.
        split_lists = split_lists_from_source(payload)

        def _ls(name: str) -> list[str]:
            v = split_lists.get(name)
            return list(v) if isinstance(v, list) else []

        a_w, a_h, a_b = _collect_proxy_series(dataset_root=root, split_list=_ls("train"), limit=int(args.ks_limit))
        v_w, v_h, v_b = _collect_proxy_series(dataset_root=root, split_list=_ls("val"), limit=int(args.ks_limit))
        t_w, t_h, t_b = _collect_proxy_series(dataset_root=root, split_list=_ls("test"), limit=int(args.ks_limit))

        comps = as_dict(payload.get("comparisons"))
        def _add(dst: dict[str, Any], *, a: tuple[list[float], list[float], list[float]], b: tuple[list[float], list[float], list[float]]) -> None:
            img = child_dict(dst, "images")
            lbl = child_dict(dst, "labels")
            img["width_ks"] = _try_ks_payload(a[0], b[0])
            img["height_ks"] = _try_ks_payload(a[1], b[1])
            lbl["boxes_per_image_ks"] = _try_ks_payload(a[2], b[2])
            dst["images"] = img
            dst["labels"] = lbl

        train_vs_val = comps.get("train_vs_val")
        if isinstance(train_vs_val, dict):
            _add(train_vs_val, a=(a_w, a_h, a_b), b=(v_w, v_h, v_b))
        train_vs_test = comps.get("train_vs_test")
        if isinstance(train_vs_test, dict):
            _add(train_vs_test, a=(a_w, a_h, a_b), b=(t_w, t_h, t_b))
        val_vs_test = comps.get("val_vs_test")
        if isinstance(val_vs_test, dict):
            _add(val_vs_test, a=(v_w, v_h, v_b), b=(t_w, t_h, t_b))
        payload["comparisons"] = comps

    acc_cfg = DriftAcceptanceConfig()
    acc_path = (args.acceptance_config or "").strip()
    if acc_path:
        acc_cfg = DriftAcceptanceConfig.from_json_file(Path(acc_path))
    payload["acceptance"] = evaluate_acceptance(payload, cfg=acc_cfg)

    plots_manifest: dict[str, Any] | None = None
    if bool(args.emit_plots):
        plots_manifest = emit_split_drift_plots(payload, out_dir=Path(args.plots_dir))
        payload["plots"] = plots_manifest

    payload = with_schema_version(
        {"dataset": {"description": describe_dataset(spec)}, **payload},
        schema_version="split_drift_report.v1",
    )

    out_path = write_json(args.out, payload)
    cli_print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
