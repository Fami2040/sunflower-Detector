from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from harchoc.repro_chain import (
    format_repro_cmd,
    load_json_bundle,
    run_argv_chain,
)

MANUSCRIPT_REPRO_BUNDLE_SCHEMA = "manuscript_repro_bundle.v1"


def load_manuscript_repro_bundle(path: str | Path) -> dict[str, Any]:
    return load_json_bundle(path, schema_version=MANUSCRIPT_REPRO_BUNDLE_SCHEMA)


def _format_cmd(argv: list[str], *, mamba: bool) -> str:
    """Backward-compatible alias; prefer ``harchoc.repro_chain.format_repro_cmd``."""
    return format_repro_cmd(argv, mamba=mamba)


def build_manuscript_repro_chain(
    bundle: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    skip_gpu_check: bool = False,
    include_test_map: bool = False,
) -> list[tuple[str, list[str]]]:
    """Return ordered (step_id, argv) pairs; argv is repo-relative script invocation."""
    from harchoc.experiment_argv import argv_for_dual_metric, dual_metric_fields_from_bundle_art
    from harchoc.hsp_export_protocol import (
        DEFAULT_EXPORT_MAX_DET,
        DEFAULT_SPLIT_FILE,
        DEFAULT_VAL_SPLIT_FILE,
        EXPORT_CONF,
        EXPORT_IOU,
    )

    rr = Path(repo_root or ".").expanduser().resolve()
    w = str(bundle["weights"])
    exp = bundle.get("export_hyperparams") or {}
    cfg = bundle.get("configs") or {}
    art = bundle.get("artifacts") or {}

    def _script(name: str) -> str:
        return str((rr / "scripts" / name).relative_to(rr))

    steps: list[tuple[str, list[str]]] = []

    if not skip_gpu_check:
        steps.append(("check_gpu", [_script("check_gpu.py")]))

    steps.append(
        (
            "split_drift",
            [_script("split_drift.py"), "--with-ks", "--out", str(art["split_drift"])],
        )
    )

    common_export = [
        "--weights",
        w,
        "--imgsz",
        str(exp.get("imgsz", 1280)),
        "--export-only",
        "--export-conf",
        str(exp.get("conf", EXPORT_CONF)),
        "--export-iou",
        str(exp.get("iou", EXPORT_IOU)),
        "--export-max-det",
        str(exp.get("max_det", DEFAULT_EXPORT_MAX_DET)),
    ]
    dev = str(exp.get("export_device") or "").strip()
    if dev:
        common_export.extend(["--export-device", dev])

    steps.append(
        (
            "eval_val_export",
            [
                _script("eval.py"),
                *common_export,
                "--split-file",
                DEFAULT_VAL_SPLIT_FILE,
                "--export-gt-json",
                str(art["gt_val"]),
                "--export-preds-json",
                str(art["preds_val"]),
                "--out",
                str(art["eval_val"]),
            ],
        )
    )
    steps.append(
        (
            "eval_test_export",
            [
                _script("eval.py"),
                *common_export,
                "--split-file",
                DEFAULT_SPLIT_FILE,
                "--export-gt-json",
                str(art["gt_test"]),
                "--export-preds-json",
                str(art["preds_test"]),
                "--out",
                str(art["eval_test"]),
            ],
        )
    )

    steps.extend(
        [
            (
                "threshold_sweep_val",
                [_script("threshold_sweep.py"), "--config", str(cfg["threshold_sweep_val"])],
            ),
            (
                "threshold_sweep_test_locked",
                [_script("threshold_sweep.py"), "--config", str(cfg["threshold_sweep_test_locked"])],
            ),
            (
                "error_analysis_val",
                [_script("error_analysis.py"), "--config", str(cfg["error_analysis_val"])],
            ),
            (
                "error_analysis_test",
                [_script("error_analysis.py"), "--config", str(cfg["error_analysis_test"])],
            ),
            (
                "dual_metric",
                [
                    _script("experiment.py"),
                    *argv_for_dual_metric(dual_metric_fields_from_bundle_art(art)),
                ],
            ),
        ]
    )

    if include_test_map:
        from harchoc.experiment_argv import argv_for_map_cpu

        steps.append(
            (
                "eval_test_map",
                [
                    _script("experiment.py"),
                    "map-cpu",
                    *argv_for_map_cpu(
                        {
                            "weights": w,
                            "split_file": "data/splits/test.txt",
                            "imgsz": exp.get("imgsz", 1280),
                            "max_det": exp.get("max_det", 3000),
                            "device": "cpu",
                            "out": str(art["eval_test_map"]),
                        }
                    ),
                ],
            )
        )
        steps.append(
            (
                "dual_metric_with_map",
                [
                    _script("experiment.py"),
                    *argv_for_dual_metric(
                        dual_metric_fields_from_bundle_art(art, include_test_map=True)
                    ),
                ],
            )
        )

    return steps


def run_manuscript_repro_chain(
    bundle: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
    skip_gpu_check: bool = False,
    include_test_map: bool = False,
    on_step: Callable[[str, list[str]], None] | None = None,
    run_argv: Callable[[list[str]], int] | None = None,
) -> int:
    rr = Path(repo_root or ".").expanduser().resolve()
    from harchoc.experiment_argv import argv_for_repro_steps

    steps = argv_for_repro_steps(
        bundle,
        repo_root=rr,
        skip_gpu_check=skip_gpu_check,
        include_test_map=include_test_map,
    )
    return run_argv_chain(
        steps,
        repo_root=rr,
        dry_run=dry_run,
        on_step=on_step,
        run_argv=run_argv,
        mamba_for_step=lambda _sid: True,
        fail_label="repro",
    )
