from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from harchoc.script_entry import bootstrap_repo_imports

bootstrap_repo_imports()

from harchoc.bench_assets import build_weights_prep_report, default_weights_manifest_path
from harchoc.external_repos import write_external_repos_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Prep zoo assets: Ultralytics .pt cache, external DETR checkpoints, "
            "and upstream training repos (DEIM / D-FINE / RT-DETR)."
        )
    )
    p.add_argument("--bench-dir", default="configs/bench", help="Directory of bench YAML/JSON configs.")
    p.add_argument("--pattern", default="*.yaml", help="Glob pattern within --bench-dir.")
    p.add_argument(
        "--bench-config",
        action="append",
        default=[],
        help="Explicit bench config path (repeatable). If set, ignores --bench-dir glob.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Optional JSON report path (default: stdout only unless set).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download missing Ultralytics weights, external .pth checkpoints, "
            "and git-clone upstream repos into external/ (prep only)."
        ),
    )
    p.add_argument(
        "--manifest",
        default="",
        help="Write/update weights manifest JSON (default: data/weights/weights_manifest.json when --download).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 1 when any bench ultralytics identifier is missing from cache or "
            "weights_manifest.json (default unless CI=1)."
        ),
    )
    p.add_argument(
        "--warn-only",
        action="store_true",
        help="Always exit 0; still print missing entries and download hints.",
    )
    p.add_argument(
        "--sync-repos-manifest",
        action="store_true",
        help=(
            "Regenerate configs/external/external_repos.v1.json from "
            "configs/external/detector_sources.v1.json (no downloads)."
        ),
    )
    args = p.parse_args(argv)

    if args.sync_repos_manifest:
        out = write_external_repos_manifest()
        print(json.dumps({"synced": str(out.resolve())}, indent=2) + "\n")
        if not (args.download or args.strict or args.out or args.bench_config):
            return 0

    bench_dir = Path(args.bench_dir)
    bench_paths = [Path(x) for x in args.bench_config] if args.bench_config else None
    if bench_paths is None and not bench_dir.is_dir():
        raise SystemExit(f"Bench dir not found: {bench_dir}")

    manifest_path: Path | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.download or args.strict:
        manifest_path = default_weights_manifest_path()

    report = build_weights_prep_report(
        bench_dir=bench_dir,
        pattern=str(args.pattern),
        bench_config_paths=bench_paths,
        download=bool(args.download),
        manifest_path=manifest_path,
        check_manifest=bool(args.strict),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, "utf-8")
    else:
        print(text, end="")

    from harchoc.config_coerce import as_dict, child_dict

    summary = as_dict(report.get("summary"))
    missing = int(summary.get("missing", 0))
    missing_manifest = int(summary.get("missing_from_manifest", 0))
    ext_missing = 0
    ext_summary = child_dict(as_dict(report.get("external")), "summary")
    ext_missing = int(ext_summary.get("missing", 0))
    repos_missing = 0
    external_repos = report.get("external_repos")
    if isinstance(external_repos, dict):
        repos_summary = external_repos.get("summary")
        if isinstance(repos_summary, dict):
            repos_missing = int(repos_summary.get("missing", 0))
    if (
        (missing or missing_manifest or ext_missing or repos_missing)
        and not args.warn_only
        and (args.strict or not os.getenv("CI"))
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
