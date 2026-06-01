from __future__ import annotations

import argparse
import os
from pathlib import Path

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()
from harchoc.domain_eval import (
    CATALOG_RUN_SCHEMA,
    build_domain_eval_payload,
    catalog_record_domains,
    planned_tray_eval_entries,
    tray_keys_from_catalog_blob,
)
from harchoc.domain_tags import (
    attach_tray_tags_to_domains,
    catalog_domains_from_dataset,
    filter_split_entries_by_domains,
    merge_domain_metadata_tags,
)
from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS
from harchoc.json_io import load_json_dict
from harchoc.splits_io import resolve_splits_dir
from harchoc.schemas import with_schema_version
from harchoc.script_scaffold import resolve_dataset_args
from scripts._common_cli import add_dataset_args, add_dry_run_arg, cli_print, require_existing_dir, write_json

_DEFAULT_CATALOG = "reports/domains/catalog.json"
_DEFAULT_EVAL_OUT = "reports/domains/domain_eval.json"
_CANONICAL_TEST_SPLIT = "data/splits/test.txt"


def _write_domain_split_lists(
    *,
    catalog: dict[str, object],
    dataset_root: Path,
    out_dir: Path,
    splits_dir: str,
) -> list[str]:
    from harchoc.splits_io import read_split_list

    root = dataset_root.resolve()
    sd = resolve_splits_dir(dataset_root=root, splits_dir=splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    domains = catalog.get("domains")
    if not isinstance(domains, list):
        return written

    for split in ("train", "val", "test"):
        txt = sd / f"{split}.txt"
        if not txt.is_file():
            continue
        entries = read_split_list(txt, missing_ok=True)
        assert isinstance(entries, list)
        for dom_rec in domains:
            if not isinstance(dom_rec, dict):
                continue
            key = str(dom_rec.get("tray_key") or "")
            if not key:
                continue
            subset = filter_split_entries_by_domains(
                [str(x) for x in entries], dataset_root=root, domains={key}
            )
            if not subset:
                continue
            out_path = out_dir / f"{split}_{key}.txt"
            out_path.write_text(
                "\n".join(subset) + ("\n" if subset else ""),
                encoding="utf-8",
            )
            written.append(str(out_path))
    return written


def _resolve_catalog_path(args: argparse.Namespace, repo_root: Path) -> Path:
    p = Path(args.catalog).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _load_catalog_blob(catalog_path: Path) -> dict[str, object] | None:
    if not catalog_path.is_file():
        return None
    raw = load_json_dict(catalog_path)
    nested = raw.get("catalog")
    if isinstance(nested, dict):
        return nested
    if catalog_record_domains(raw):
        return raw
    return raw


def _load_catalog_run_metadata(catalog_path: Path) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Return (catalog blob, domain_metadata_tags) from eval_domains_run.v1 or legacy catalog."""
    if not catalog_path.is_file():
        return None, None
    raw = load_json_dict(catalog_path)
    blob = _load_catalog_blob(catalog_path)
    tags = raw.get("domain_metadata_tags")
    tags_out = tags if isinstance(tags, dict) else None
    return blob, tags_out


def _finalize_domain_eval_payload(
    payload: dict[str, object],
    *,
    domain_metadata_tags: dict[str, object] | None,
) -> dict[str, object]:
    if domain_metadata_tags is None:
        return payload
    payload["domain_metadata_tags"] = domain_metadata_tags
    domains = payload.get("domains")
    if isinstance(domains, list):
        payload["domains"] = attach_tray_tags_to_domains(
            [d for d in domains if isinstance(d, dict)],
            domain_metadata_tags,
        )
    return payload


def _optional_dataset_root_for_dry_run(args: argparse.Namespace) -> Path | None:
    """Dry-run: explicit --dataset-root or DATASET_ROOT only (no manifest fallback)."""
    if getattr(args, "dataset_root", None):
        p = Path(str(args.dataset_root)).expanduser().resolve()
        return p if p.is_dir() else None
    env = (os.environ.get("DATASET_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    return None


def _tray_keys_for_dry_run(
    *,
    catalog_path: Path,
    dataset_root: Path | None,
    splits_dir: str,
) -> tuple[list[str], dict[str, object] | None, dict[str, object] | None]:
    blob, tags = _load_catalog_run_metadata(catalog_path)
    if blob is not None:
        keys = tray_keys_from_catalog_blob(blob)
        if keys:
            return keys, blob, tags

    if dataset_root is not None and dataset_root.is_dir():
        built = catalog_domains_from_dataset(
            dataset_root=dataset_root,
            splits_dir=splits_dir,
        )
        keys = tray_keys_from_catalog_blob(built)
        if keys:
            return keys, built, tags

    return ["_example"], None, tags



def _merge_tray_count_mae(args: argparse.Namespace, repo_root: Path) -> int:
    from harchoc.domain_count_mae import (
        build_domain_count_mae_sidecar,
        merge_tray_count_mae_results,
        run_tray_count_mae_eval,
    )
    from harchoc.json_io import load_json_dict
    from harchoc.threshold_lock import load_locked_conf

    eval_path = Path(args.out).expanduser()
    if not eval_path.is_absolute():
        eval_path = (repo_root / eval_path).resolve()
    if not eval_path.is_file():
        raise SystemExit(f"domain_eval JSON not found: {eval_path}")

    spec = resolve_dataset_args(args)
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
    root = Path(spec.root)

    payload = load_json_dict(eval_path)
    domains = payload.get("domains")
    if not isinstance(domains, list):
        raise SystemExit("domain_eval.v1 missing domains list")

    locked_from = str(args.locked_conf_from or "reports/hsp/threshold_val.json")
    locked_path = Path(locked_from)
    if not locked_path.is_absolute():
        locked_path = (repo_root / locked_path).resolve()
    locked_conf = load_locked_conf(locked_path)

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (repo_root / reports_dir).resolve()
    domains_abs = Path(args.domains_dir)
    if not domains_abs.is_absolute():
        domains_abs = (repo_root / domains_abs).resolve()

    tray_keys = [
        str(d.get("tray_key"))
        for d in domains
        if isinstance(d, dict) and str(d.get("tray_key") or "").strip()
    ]
    results = []
    for key in tray_keys:
        _key, rc, block = run_tray_count_mae_eval(
            tray_key=key,
            weights=str(args.weights),
            domains_dir=domains_abs,
            reports_dir=reports_dir,
            locked_conf_from=str(locked_path),
            device=str(args.device),
            manifest=str(args.manifest),
            default_dataset_name=str(args.default_dataset_name),
            dataset_name=getattr(args, "dataset_name", None),
            dataset_root=str(root),
            yolo_data_yaml=getattr(args, "yolo_data_yaml", None),
        )
        results.append((key, rc, block))

    payload = merge_tray_count_mae_results(
        payload,
        results,
        locked_conf_from=str(locked_path),
        locked_conf=float(locked_conf),
    )
    sidecar_path = Path(args.count_mae_sidecar)
    if not sidecar_path.is_absolute():
        sidecar_path = (repo_root / sidecar_path).resolve()
    sidecar = build_domain_count_mae_sidecar(
        domain_eval=payload,
        summary=payload.get("count_mae_summary") or {},
        weights=str(args.weights),
    )
    write_json(eval_path, payload)
    write_json(sidecar_path, sidecar)
    cli_print(f"Merged count MAE into {eval_path}")
    cli_print(f"Wrote sidecar {sidecar_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Domain catalog from YOLO label class tags + optional per-domain split lists."
    )
    add_dataset_args(p)
    add_dry_run_arg(p)
    p.add_argument("--weights", default=HSP_DETECTION_WEIGHTS, help="HSP detection checkpoint for per-tray eval.")
    p.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Split lists under dataset root (train/val/test.txt).",
    )
    p.add_argument(
        "--domains-dir",
        default="data/domains",
        help="Where to write per-domain split lists when --write-domain-splits is set.",
    )
    p.add_argument(
        "--write-domain-splits",
        action="store_true",
        help="Write data/domains/{split}_{tray_key}.txt subset lists for future eval.",
    )
    p.add_argument(
        "--catalog",
        default=_DEFAULT_CATALOG,
        help="Domain catalog JSON (read for --dry-run; written on catalog build).",
    )
    p.add_argument(
        "--out",
        default=_DEFAULT_EVAL_OUT,
        help="Per-domain eval JSON (domain_eval.v1); catalog build uses --catalog.",
    )
    p.add_argument(
        "--run-tray-eval",
        action="store_true",
        help="After catalog/splits, run eval.py on one tray (CPU by default) and fill metrics.",
    )
    p.add_argument(
        "--run-all-trays",
        action="store_true",
        help="Run eval.py on every catalog tray_key (CPU default); merge metrics into domain_eval.v1.",
    )
    p.add_argument(
        "--tray-key",
        default="",
        help="Tray key for --run-tray-eval (default: first key in catalog).",
    )
    p.add_argument(
        "--locked-conf-from",
        default="reports/hsp/threshold_val.json",
        help="Val threshold JSON for locked conf on tray eval export.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for --run-tray-eval (cpu avoids post-train GPU contention).",
    )
    p.add_argument(
        "--reports-dir",
        default="reports/domains",
        help="Per-tray eval JSON output directory for --run-tray-eval.",
    )
    p.add_argument(
        "--merge-tray-count-mae",
        action="store_true",
        help="Post-process domain_eval.v1: export-only count MAE per tray at locked conf (CPU).",
    )
    p.add_argument(
        "--count-mae-sidecar",
        default="reports/domains/domain_count_mae.json",
        help="Sidecar path for domain_count_mae.v1 (default: reports/domains/domain_count_mae.json).",
    )
    p.add_argument(
        "--import-domain-tags",
        default="",
        metavar="CSV",
        help=(
            "Optional CSV (columns: tray_key, variety, maturity, lighting, site) "
            "merged into catalog domain_metadata_tags.per_tray."
        ),
    )
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    catalog_path = _resolve_catalog_path(args, repo_root)

    if bool(args.merge_tray_count_mae):
        return _merge_tray_count_mae(args, repo_root)

    if args.dry_run:
        tray_keys, catalog_blob, catalog_tags = _tray_keys_for_dry_run(
            catalog_path=catalog_path,
            dataset_root=_optional_dataset_root_for_dry_run(args),
            splits_dir=str(args.splits_dir),
        )
        notes = (
            "Dry-run scaffold only: per-domain mAP/count metrics require eval.py loop. "
            "Tray keys from --catalog, dataset catalog, or placeholder _example."
        )
        if bool(args.run_all_trays):
            notes = (
                "Dry-run: lists planned per-tray eval.py runs (--run-all-trays); "
                "no torch/GPU. Re-run without --dry-run when dataset root "
                "(DATASET_ROOT, --dataset-root, or data/manifest.json) and split lists exist."
            )
        payload = build_domain_eval_payload(
            status="dry-run",
            script="eval_domains",
            out=args.out,
            weights=str(args.weights),
            catalog_path=catalog_path if catalog_path.is_file() else None,
            domains_dir=args.domains_dir,
            canonical_split_file=_CANONICAL_TEST_SPLIT,
            tray_keys=tray_keys,
            catalog=catalog_blob,
            domain_metadata_tags=catalog_tags,
            notes=notes,
        )
        payload = _finalize_domain_eval_payload(payload, domain_metadata_tags=catalog_tags)
        if bool(args.run_all_trays):
            locked = str(args.locked_conf_from) if args.locked_conf_from else None
            payload["planned_tray_evals"] = planned_tray_eval_entries(
                tray_keys,
                domains_dir=args.domains_dir,
                weights=str(args.weights),
                device=str(args.device),
                locked_conf_from=locked,
            )
        out_path = write_json(args.out, payload)
        cli_print(f"Wrote {out_path}")
        return 0

    spec = resolve_dataset_args(args)
    require_existing_dir(spec.root, what="Dataset root", hint="Export DATASET_ROOT=/path/to/extracted/dataset")
    root = Path(spec.root)

    catalog = catalog_domains_from_dataset(
        dataset_root=root,
        splits_dir=str(args.splits_dir),
    )

    domain_lists: list[str] = []
    if bool(args.write_domain_splits):
        domain_lists = _write_domain_split_lists(
            catalog=catalog,
            dataset_root=root,
            out_dir=Path(args.domains_dir),
            splits_dir=str(args.splits_dir),
        )

    tray_keys = tray_keys_from_catalog_blob(catalog)
    n_trays = int(catalog.get("n_domains") or len(tray_keys))

    import_tags_path: Path | None = None
    import_tags_raw = str(getattr(args, "import_domain_tags", "") or "").strip()
    if import_tags_raw:
        import_tags_path = Path(import_tags_raw).expanduser()
        if not import_tags_path.is_absolute():
            import_tags_path = (repo_root / import_tags_path).resolve()

    domain_metadata_tags = merge_domain_metadata_tags(
        n_trays=n_trays,
        csv_path=import_tags_path,
        catalog_tray_keys=set(tray_keys),
    )

    tags_note = (
        "domain_metadata_tags merged from --import-domain-tags CSV (P1-DOMAIN-TAGS)."
        if import_tags_path is not None
        else "Optional variety/maturity/lighting/site tags: domain_metadata_tags (P1-DOMAIN-TAGS; TBD)."
    )
    catalog_payload = with_schema_version(
        {
            "status": "ok",
            "script": "eval_domains",
            "weights": str(args.weights),
            "dataset_root": str(root),
            "splits_dir": str(args.splits_dir),
            "catalog": catalog,
            "domain_metadata_tags": domain_metadata_tags,
            "domain_split_lists": domain_lists,
            "notes": (
                "Class ids in labels/*/*.txt: 0=developed, 1=aborted (see harchoc/sunflower_dataset.py). "
                "Tray keys are parsed from image stems (e.g. 349-10-2). "
                f"{tags_note} "
                "Per-domain mAP/count eval: domain_eval.v1 via --run-all-trays + --merge-tray-count-mae."
            ),
        },
        schema_version=CATALOG_RUN_SCHEMA,
    )
    catalog_out = write_json(catalog_path, catalog_payload)
    cli_print(f"Wrote catalog {catalog_out}")

    eval_notes = "Per-domain metrics null until --run-tray-eval or --run-all-trays."
    eval_status = "scaffold"

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (repo_root / reports_dir).resolve()
    domains_abs = Path(args.domains_dir)
    if not domains_abs.is_absolute():
        domains_abs = (repo_root / domains_abs).resolve()

    if bool(args.run_all_trays) and tray_keys:
        from harchoc.domain_eval_loop import (
            merge_tray_eval_results_into_domain_eval,
            run_all_tray_domain_evals,
        )

        if not domain_lists:
            domain_lists = _write_domain_split_lists(
                catalog=catalog,
                dataset_root=root,
                out_dir=domains_abs,
                splits_dir=str(args.splits_dir),
            )
            if domain_lists:
                cli_print(f"Wrote {len(domain_lists)} per-tray split list(s) for --run-all-trays.")

        results = run_all_tray_domain_evals(
            tray_keys,
            weights=str(args.weights),
            domains_dir=domains_abs,
            reports_dir=reports_dir,
            locked_conf_from=str(args.locked_conf_from) if args.locked_conf_from else None,
            device=str(args.device),
            manifest=str(args.manifest),
            default_dataset_name=str(args.default_dataset_name),
            dataset_name=getattr(args, "dataset_name", None),
            dataset_root=str(root),
            yolo_data_yaml=getattr(args, "yolo_data_yaml", None),
        )
        eval_payload = build_domain_eval_payload(
            status="scaffold",
            script="eval_domains",
            out=args.out,
            weights=str(args.weights),
            catalog_path=catalog_path,
            domains_dir=args.domains_dir,
            canonical_split_file=_CANONICAL_TEST_SPLIT,
            tray_keys=tray_keys,
            catalog=catalog,
            domain_metadata_tags=domain_metadata_tags,
            notes=(
                f"All-tray eval on device={args.device!r}; "
                f"locked conf from {args.locked_conf_from!r}."
            ),
        )
        eval_payload = merge_tray_eval_results_into_domain_eval(
            eval_payload, results, device=str(args.device)
        )
        eval_payload = _finalize_domain_eval_payload(
            eval_payload, domain_metadata_tags=domain_metadata_tags
        )
        eval_status = str(eval_payload.get("status", "scaffold"))
    elif bool(args.run_tray_eval) and tray_keys:
        from harchoc.domain_eval_loop import apply_tray_metrics_to_domain_eval, run_tray_domain_eval

        target_key = str(args.tray_key).strip() or tray_keys[0]
        if not bool(args.write_domain_splits):
            cli_print(
                "Note: --run-tray-eval expects per-tray split files; "
                "pass --write-domain-splits on first catalog build."
            )
        eval_rc, metrics = run_tray_domain_eval(
            tray_key=target_key,
            weights=str(args.weights),
            domains_dir=domains_abs,
            reports_dir=reports_dir,
            locked_conf_from=str(args.locked_conf_from) if args.locked_conf_from else None,
            device=str(args.device),
            manifest=str(args.manifest),
            default_dataset_name=str(args.default_dataset_name),
            dataset_name=getattr(args, "dataset_name", None),
            dataset_root=str(root),
            yolo_data_yaml=getattr(args, "yolo_data_yaml", None),
        )
        eval_status = "partial" if metrics else "failed"
        eval_notes = (
            f"Tray eval for {target_key!r}: rc={eval_rc}, device={args.device}. "
            "Additional trays: re-run with --tray-key."
        )
        eval_payload = build_domain_eval_payload(
            status=eval_status,
            script="eval_domains",
            out=args.out,
            weights=str(args.weights),
            catalog_path=catalog_path,
            domains_dir=args.domains_dir,
            canonical_split_file=_CANONICAL_TEST_SPLIT,
            tray_keys=tray_keys,
            catalog=catalog,
            domain_metadata_tags=domain_metadata_tags,
            notes=eval_notes,
        )
        eval_payload = apply_tray_metrics_to_domain_eval(
            eval_payload, tray_key=target_key, metrics=metrics, eval_rc=eval_rc
        )
        eval_payload = _finalize_domain_eval_payload(
            eval_payload, domain_metadata_tags=domain_metadata_tags
        )
    else:
        eval_payload = build_domain_eval_payload(
            status=eval_status,
            script="eval_domains",
            out=args.out,
            weights=str(args.weights),
            catalog_path=catalog_path,
            domains_dir=args.domains_dir,
            canonical_split_file=_CANONICAL_TEST_SPLIT,
            tray_keys=tray_keys,
            catalog=catalog,
            domain_metadata_tags=domain_metadata_tags,
            notes=eval_notes,
        )
        eval_payload = _finalize_domain_eval_payload(
            eval_payload, domain_metadata_tags=domain_metadata_tags
        )

    eval_out = write_json(args.out, eval_payload)
    cli_print(f"Wrote eval {eval_out}")
    return 0 if eval_status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
