"""Publication-ready markdown (and optional LaTeX) tables from HSP / aug / zoo artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_leaderboard import (
    BEST2_REFERENCE,
    _format_ci,
    _format_mae,
    build_leaderboard_payload,
)
from harchoc.dual_metric_report import (
    TEST_SPLIT_METRIC_LABEL,
    VAL_SPLIT_METRIC_LABEL,
    extract_detection_metrics,
)
from harchoc.json_io import load_json_dict
from harchoc.reviewer2_paste_check import load_matrix_rows, matrix_group_row_ids
from harchoc.schemas import with_schema_version

MANUSCRIPT_TABLES_MANIFEST_SCHEMA = "manuscript_tables_manifest.v1"
DEFAULT_OUT_DIR = "reports/manuscript/tables"
DEFAULT_DUAL_METRIC = "reports/hsp/dual_metric.json"
DEFAULT_MATRIX_TRAIN = "reports/hsp/matrix_train.json"
DEFAULT_MATRIX_ROWS = "configs/zoo/matrix_rows.v1.json"
DEFAULT_AUG_INDEX = "configs/experiments/aug_smoke_index.json"
DEFAULT_AUG_OUT = "reports/aug_smoke"
DEFAULT_MODEL_LABEL = "models/best2.pt"
DEFAULT_MATRIX_GROUP = "zoo_yolo_only"
# P0-5 quartet when matrix_rows.v1.json has no zoo_yolo_only tags (bench YAML only).
ZOO_YOLO_ONLY_ROW_IDS: tuple[str, ...] = ("yolov8m", "yolov10m", "yolo11m", "yolo26m")
DEFAULT_TOP_N = 10

FOOTNOTE_LOCKED_CONF = (
    "Operating confidence fixed on **val** (`min_count_mae` on `data/splits/val.txt`) "
    "and applied unchanged on **test** (`data/splits/test.txt`, *n*=109)."
)
def _manifest_path(path: Path, repo_root: Path) -> str:
    """Repo-relative path when under *repo_root*; else absolute string."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


FOOTNOTE_TEST_SPLIT = (
    "Primary manuscript metrics: **test** count MAE and ranking mAP50 at locked conf; "
    "val metrics are threshold-selection transparency only."
)


def fmt_conf(conf: float | None) -> str:
    if conf is None:
        return "—"
    return f"{float(conf):.2f}"


def fmt_mae(mae: float | None, *, digits: int = 1) -> str:
    if mae is None:
        return "—"
    return f"{float(mae):.{digits}f}"


def fmt_map(val: float | None, *, digits: int = 3) -> str:
    if val is None:
        return "—"
    return f"{float(val):.{digits}f}"


def fmt_ci_block(ci: dict[str, Any] | None) -> str:
    if not ci:
        return "—"
    lo = ci.get("low")
    hi = ci.get("high")
    if lo is None or hi is None:
        return "—"
    conf = ci.get("confidence")
    pct = int(round(float(conf) * 100)) if conf is not None else 95
    return f"{float(lo):.1f}–{float(hi):.1f} ({pct}%)"


def _row_by_split(dm: dict[str, Any], split: str) -> dict[str, Any] | None:
    for row in dm.get("rows") or []:
        if isinstance(row, dict) and str(row.get("split") or "") == split:
            return row
    return None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_headline_rows(
    *,
    dual_metric: dict[str, Any] | None,
    model_label: str = DEFAULT_MODEL_LABEL,
) -> list[dict[str, Any]]:
    """Headline table rows: production anchor + optional val transparency."""
    if dual_metric is None:
        ref = BEST2_REFERENCE
        return [
            {
                "model": model_label,
                "split": "test",
                "split_role": TEST_SPLIT_METRIC_LABEL,
                "count_mae": ref.get("test_count_mae"),
                "mae_ci": ref.get("test_count_mae_ci"),
                "map50": None,
                "locked_conf": 0.15,
                "status": "reference_only",
                "source": str(ref.get("source") or "aug_smoke_leaderboard.BEST2_REFERENCE"),
            }
        ]

    op = _mapping(dual_metric.get("operating_point"))
    locked_conf = op.get("locked_conf")
    rows_out: list[dict[str, Any]] = []
    inputs = _mapping(dual_metric.get("inputs"))

    test_row = _row_by_split(dual_metric, "test")
    if test_row:
        counting = _mapping(test_row.get("counting"))
        detection = _mapping(test_row.get("detection"))
        rows_out.append(
            {
                "model": model_label,
                "split": "test",
                "split_role": test_row.get("split_role_label") or TEST_SPLIT_METRIC_LABEL,
                "count_mae": counting.get("mae"),
                "mae_ci": counting.get("mae_ci"),
                "map50": detection.get("mAP50"),
                "map50_95": detection.get("mAP50_95"),
                "locked_conf": test_row.get("operating_conf") or locked_conf,
                "status": "ok",
                "source": inputs.get("eval_test") or inputs.get("error_test"),
            }
        )

    val_row = _row_by_split(dual_metric, "val")
    if val_row:
        counting = _mapping(val_row.get("counting"))
        raw_detection = val_row.get("detection")
        detection = (
            extract_detection_metrics(val_row)
            if not raw_detection
            else _mapping(raw_detection)
        )
        rows_out.append(
            {
                "model": model_label,
                "split": "val",
                "split_role": val_row.get("split_role_label") or VAL_SPLIT_METRIC_LABEL,
                "count_mae": counting.get("mae"),
                "mae_ci": counting.get("mae_ci"),
                "map50": detection.get("mAP50"),
                "map50_95": detection.get("mAP50_95"),
                "locked_conf": val_row.get("operating_conf") or op.get("selected_conf"),
                "status": "ok",
                "source": inputs.get("eval_val"),
            }
        )
    return rows_out


def render_headline_md(
    rows: list[dict[str, Any]],
    *,
    dual_metric_path: str | None,
    title: str = "Headline metrics (production anchor)",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Model:** `{rows[0].get('model') if rows else DEFAULT_MODEL_LABEL}` · "
        f"**Metric:** test count MAE @ val-locked conf · **Split:** `data/splits/test.txt`",
        "",
        "| Split | Count MAE | 95% CI | mAP50 | mAP50-95 | Conf | Status |",
        "|-------|----------:|--------|------:|---------:|-----:|--------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("split") or "—"),
                    fmt_mae(row.get("count_mae")),
                    fmt_ci_block(row.get("mae_ci") if isinstance(row.get("mae_ci"), dict) else None),
                    fmt_map(row.get("map50")),
                    fmt_map(row.get("map50_95")),
                    fmt_conf(row.get("locked_conf") if row.get("locked_conf") is not None else None),
                    str(row.get("status") or "—"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Footnotes", "", f"1. {FOOTNOTE_LOCKED_CONF}", f"2. {FOOTNOTE_TEST_SPLIT}"])
    if dual_metric_path:
        lines.append(f"3. Source: [`{dual_metric_path}`]({dual_metric_path}) (`dual_metric_report.v1`).")
    ref_status = rows[0].get("status") if rows else None
    if ref_status == "reference_only":
        lines.append(
            "4. `dual_metric.json` absent — values from leaderboard reference; re-run "
            "`experiment.py repro` or `dual-metric` for on-disk HSP merge."
        )
    return "\n".join(lines) + "\n"


def render_headline_tex(rows: list[dict[str, Any]], *, caption: str) -> str:
    body = []
    for row in rows:
        if str(row.get("split")) != "test":
            continue
        body.append(
            " & ".join(
                [
                    str(row.get("model") or DEFAULT_MODEL_LABEL).replace("_", r"\_"),
                    fmt_mae(row.get("count_mae")),
                    fmt_map(row.get("map50")),
                    fmt_conf(row.get("locked_conf") if row.get("locked_conf") is not None else None),
                ]
            )
            + r" \\"
        )
    if not body:
        body.append(r"\multicolumn{4}{c}{---} \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{" + caption.replace("_", r"\_") + r"}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Model & Count MAE & mAP$_{50}$ & Conf \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_aug_top_n_rows(
    *,
    repo_root: Path,
    index_path: str,
    out_dir: str,
    top_n: int,
    leaderboard_json: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if leaderboard_json and leaderboard_json.is_file():
        payload = load_json_dict(leaderboard_json)
    else:
        payload = build_leaderboard_payload(
            repo_root=repo_root,
            index_path=index_path,
            out_dir=out_dir,
        )
    ranked = list(payload.get("ranked_rows") or [])
    return ranked[: max(0, int(top_n))], payload


def render_aug_top_n_md(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    reference: dict[str, Any],
    title: str = "Augmentation smoke — top ranked runs",
) -> str:
    ref_mae = float(reference.get("test_count_mae") or 0)
    lines = [
        f"# {title}",
        "",
        f"**Primary metric:** test count MAE (15-ep smoke unless noted). "
        f"**Reference best2:** {fmt_mae(ref_mae)} @ 100 ep. "
        f"Showing top **{top_n}** ranked rows (equivalence duplicates excluded).",
        "",
        "| Rank | ID | Run | MAE | Δ vs best2 | 95% CI | Key knobs |",
        "|-----:|----|-----|----:|-----------:|--------|-----------|",
    ]
    for i, row in enumerate(rows, start=1):
        mae = row.get("test_count_mae")
        delta = (float(mae) - ref_mae) if mae is not None else None
        delta_s = f"+{delta:.1f}" if delta is not None and delta >= 0 else (fmt_mae(delta) if delta is not None else "—")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row.get("smoke_id") or "—"),
                    f"`{row.get('run_name')}`",
                    fmt_mae(mae),
                    delta_s,
                    _format_ci(row.get("test_count_mae_ci") if isinstance(row.get("test_count_mae_ci"), dict) else None),
                    str(row.get("key_overrides") or "")[:60],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Footnotes",
            "",
            f"1. {FOOTNOTE_LOCKED_CONF}",
            "2. Full grid: [`reports/aug_smoke/leaderboard.md`](../../aug_smoke/leaderboard.md); "
            "index [`configs/experiments/aug_smoke_index.json`](../../../configs/experiments/aug_smoke_index.json).",
        ]
    )
    return "\n".join(lines) + "\n"


def render_aug_top_n_tex(rows: list[dict[str, Any]], *, caption: str) -> str:
    body = []
    for i, row in enumerate(rows, start=1):
        sid = str(row.get("smoke_id") or "—").replace("_", r"\_")
        body.append(
            f"{i} & {sid} & {fmt_mae(row.get('test_count_mae'))} & "
            + (str(row.get("key_overrides") or "")[:40].replace("_", r"\_"))
            + r" \\"
        )
    if not body:
        body.append(r"\multicolumn{4}{c}{---} \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{" + caption.replace("_", r"\_") + r"}",
            r"\begin{tabular}{rlrl}",
            r"\toprule",
            r"Rank & ID & MAE & Notes \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def expected_matrix_group_row_ids(matrix_doc: dict[str, Any], matrix_group: str) -> list[str]:
    """Row ids for a matrix group; ``zoo_yolo_only`` falls back to the P0-5 YOLO quartet."""
    ids = matrix_group_row_ids(matrix_doc, matrix_group)
    if matrix_group != "zoo_yolo_only":
        return ids
    if not ids:
        return list(ZOO_YOLO_ONLY_ROW_IDS)
    order = {rid: i for i, rid in enumerate(ZOO_YOLO_ONLY_ROW_IDS)}
    return sorted(ids, key=lambda x: order.get(x, len(order)))


def _matrix_run_for_id(
    runs: list[dict[str, Any]],
    row_id: str,
    *,
    stem: str | None = None,
) -> dict[str, Any] | None:
    for run in runs:
        if not isinstance(run, dict):
            continue
        name = str(run.get("name") or "")
        run_name = str(run.get("run_name") or "")
        if name == row_id or run_name.startswith(f"{row_id}_"):
            return run
    if stem:
        for run in runs:
            if not isinstance(run, dict):
                continue
            if str(run.get("train_config_stem") or "") == stem:
                return run
    return None


def build_zoo_core_rows(
    *,
    repo_root: Path,
    matrix_rows_path: str,
    matrix_train_path: str | Path | None,
    matrix_group: str = DEFAULT_MATRIX_GROUP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_doc = load_matrix_rows(repo_root, matrix_rows_path)
    expected_ids = expected_matrix_group_row_ids(matrix_doc, matrix_group)
    meta: dict[str, Any] = {
        "matrix_group": matrix_group,
        "expected_row_ids": expected_ids,
        "matrix_train_present": False,
        "matrix_train_status": "missing",
    }

    train_doc: dict[str, Any] | None = None
    if matrix_train_path:
        p = (repo_root / str(matrix_train_path)).resolve()
        if p.is_file():
            meta["matrix_train_present"] = True
            try:
                train_doc = load_json_dict(p)
                meta["matrix_train_status"] = str(train_doc.get("status") or "ok")
            except (OSError, TypeError, ValueError) as exc:
                meta["matrix_train_status"] = f"read_error: {exc}"
        else:
            meta["matrix_train_path"] = str(matrix_train_path)

    runs = list((train_doc or {}).get("runs") or [])
    rows_out: list[dict[str, Any]] = []
    for row_id in expected_ids:
        manifest_row = next(
            (r for r in (matrix_doc.get("rows") or []) if isinstance(r, dict) and r.get("id") == row_id),
            None,
        )
        stem = str(manifest_row.get("train_config_stem") or "") if manifest_row else None
        run = _matrix_run_for_id(runs, row_id, stem=stem or None)
        if run is None:
            rows_out.append(
                {
                    "id": row_id,
                    "run_name": f"{row_id}_e100_s0",
                    "test_count_mae": None,
                    "map50": None,
                    "map50_95": None,
                    "status": "pending",
                    "weights": None,
                }
            )
            continue
        status = str(run.get("status") or "unknown")
        rows_out.append(
            {
                "id": row_id,
                "run_name": run.get("run_name") or f"{row_id}_e100_s0",
                "test_count_mae": run.get("test_count_mae"),
                "map50": run.get("mAP50"),
                "map50_95": run.get("mAP50_95"),
                "status": status if status == "ok" and run.get("test_count_mae") is not None else status,
                "weights": run.get("weights"),
            }
        )
    meta["n_complete"] = sum(1 for r in rows_out if r.get("test_count_mae") is not None)
    meta["n_expected"] = len(expected_ids)
    return rows_out, meta


def render_zoo_core_md(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    matrix_train_path: str | None,
    title: str | None = None,
) -> str:
    group = str(meta.get("matrix_group") or DEFAULT_MATRIX_GROUP)
    if title is None:
        label = "zoo_yolo_only (P0-5)" if group == "zoo_yolo_only" else group
        title = f"Model zoo ({label}) — test metrics @ locked conf"
    n_ok = meta.get("n_complete", 0)
    n_exp = meta.get("n_expected", len(rows))
    lines = [
        f"# {title}",
        "",
        f"**Group:** `{meta.get('matrix_group')}` · **Rows on disk:** {n_ok}/{n_exp} with test count MAE. "
        f"**Train aggregate:** `{matrix_train_path or 'reports/hsp/matrix_train.json'}` "
        f"({meta.get('matrix_train_status', 'missing')}).",
        "",
        "| Model ID | Run | Test MAE | mAP50 | mAP50-95 | Status |",
        "|----------|-----|--------:|------:|---------:|--------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('id')}`",
                    f"`{row.get('run_name')}`",
                    fmt_mae(row.get("test_count_mae")),
                    fmt_map(row.get("map50")),
                    fmt_map(row.get("map50_95")),
                    str(row.get("status") or "—"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Footnotes",
            "",
            f"1. {FOOTNOTE_LOCKED_CONF}",
            f"2. {FOOTNOTE_TEST_SPLIT}",
            "3. Empty MAE cells: train or HSP test eval not finished (P0-5 `zoo_matrix_train`); "
            f"re-run `benchmark_matrix.py` with `--matrix-group {group}` after queue resume.",
        ]
    )
    if n_ok < n_exp:
        lines.append(
            f"4. Partial aggregate ({n_ok}/{n_exp}) — table lists all `{group}` slots with graceful placeholders."
        )
    return "\n".join(lines) + "\n"


def render_zoo_core_tex(rows: list[dict[str, Any]], *, caption: str) -> str:
    body = []
    for row in rows:
        rid = str(row.get("id") or "—").replace("_", r"\_")
        body.append(
            f"{rid} & {fmt_mae(row.get('test_count_mae'))} & {fmt_map(row.get('map50'))} & "
            f"{str(row.get('status') or '—')} \\\\"
        )
    if not body:
        body.append(r"\multicolumn{4}{c}{---} \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{" + caption.replace("_", r"\_") + r"}",
            r"\begin{tabular}{lrrl}",
            r"\toprule",
            r"Model & Test MAE & mAP$_{50}$ & Status \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_tables_repro_bundle(
    *,
    repo_root: str | Path,
    dual_metric_path: str = DEFAULT_DUAL_METRIC,
    matrix_train_path: str = DEFAULT_MATRIX_TRAIN,
    matrix_rows_path: str = DEFAULT_MATRIX_ROWS,
    aug_index_path: str = DEFAULT_AUG_INDEX,
    aug_out_dir: str = DEFAULT_AUG_OUT,
    model_label: str = DEFAULT_MODEL_LABEL,
    matrix_group: str = DEFAULT_MATRIX_GROUP,
    top_n: int = DEFAULT_TOP_N,
    aug_leaderboard_json: str | None = None,
) -> dict[str, Any]:
    rr = Path(repo_root).resolve()
    warnings: list[str] = []

    dm: dict[str, Any] | None = None
    dm_p = rr / dual_metric_path
    if dm_p.is_file():
        try:
            dm = load_json_dict(dm_p)
            if dm.get("schema_version") != "dual_metric_report.v1":
                warnings.append(f"dual_metric schema unexpected: {dm.get('schema_version')!r}")
        except (OSError, TypeError, ValueError) as exc:
            warnings.append(f"dual_metric read failed: {exc}")
    else:
        warnings.append(f"missing dual_metric: {dual_metric_path}")

    headline_rows = build_headline_rows(dual_metric=dm, model_label=model_label)
    aug_rows, aug_payload = build_aug_top_n_rows(
        repo_root=rr,
        index_path=aug_index_path,
        out_dir=aug_out_dir,
        top_n=top_n,
        leaderboard_json=(rr / aug_leaderboard_json) if aug_leaderboard_json else (rr / aug_out_dir / "leaderboard.json"),
    )
    zoo_rows, zoo_meta = build_zoo_core_rows(
        repo_root=rr,
        matrix_rows_path=matrix_rows_path,
        matrix_train_path=matrix_train_path,
        matrix_group=matrix_group,
    )

    return {
        "headline_rows": headline_rows,
        "aug_top_n_rows": aug_rows,
        "aug_reference": aug_payload.get("reference") or BEST2_REFERENCE,
        "zoo_core_rows": zoo_rows,
        "zoo_meta": zoo_meta,
        "warnings": warnings,
        "paths": {
            "dual_metric": dual_metric_path,
            "matrix_train": matrix_train_path,
            "matrix_rows": matrix_rows_path,
            "aug_index": aug_index_path,
            "aug_out_dir": aug_out_dir,
        },
    }


def write_manuscript_tables(
    *,
    repo_root: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    write_latex: bool = False,
    bundle: dict[str, Any] | None = None,
    **build_kwargs: Any,
) -> dict[str, Path]:
    rr = Path(repo_root).resolve()
    od = (rr / out_dir).resolve()
    od.mkdir(parents=True, exist_ok=True)

    data = bundle if bundle is not None else build_tables_repro_bundle(repo_root=rr, **build_kwargs)
    paths_cfg = data.get("paths") or {}

    headline_md = render_headline_md(
        data["headline_rows"],
        dual_metric_path=paths_cfg.get("dual_metric"),
    )
    aug_md = render_aug_top_n_md(
        data["aug_top_n_rows"],
        top_n=len(data["aug_top_n_rows"]),
        reference=data["aug_reference"],
    )
    zoo_md = render_zoo_core_md(
        data["zoo_core_rows"],
        data["zoo_meta"],
        matrix_train_path=paths_cfg.get("matrix_train"),
    )

    written: dict[str, Path] = {}
    for name, content in (
        ("headline_metrics.md", headline_md),
        ("aug_smoke_top_n.md", aug_md),
        ("zoo_core.md", zoo_md),
    ):
        p = od / name
        p.write_text(content, encoding="utf-8")
        written[name] = p

    if write_latex:
        tex_map = {
            "headline_metrics.tex": render_headline_tex(
                data["headline_rows"],
                caption="Headline test metrics at val-locked confidence",
            ),
            "aug_smoke_top_n.tex": render_aug_top_n_tex(
                data["aug_top_n_rows"],
                caption="Top augmentation smoke runs by test count MAE",
            ),
            "zoo_core.tex": render_zoo_core_tex(
                data["zoo_core_rows"],
                caption="Zoo core models: test MAE and mAP50",
            ),
        }
        for name, content in tex_map.items():
            p = od / name
            p.write_text(content, encoding="utf-8")
            written[name] = p

    manifest = with_schema_version(
        {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "partial" if data.get("warnings") else "ok",
            "warnings": list(data.get("warnings") or []),
            "outputs": {k: _manifest_path(v, rr) for k, v in written.items()},
            "tables": {
                "headline_metrics": {
                    "markdown": _manifest_path(written["headline_metrics.md"], rr),
                    "rows": data["headline_rows"],
                },
                "aug_smoke_top_n": {
                    "markdown": _manifest_path(written["aug_smoke_top_n.md"], rr),
                    "n_rows": len(data["aug_top_n_rows"]),
                },
                "zoo_core": {
                    "markdown": _manifest_path(written["zoo_core.md"], rr),
                    "meta": data["zoo_meta"],
                },
            },
            "footnotes": {
                "locked_conf": FOOTNOTE_LOCKED_CONF,
                "test_split": FOOTNOTE_TEST_SPLIT,
            },
            "inputs": paths_cfg,
        },
        schema_version=MANUSCRIPT_TABLES_MANIFEST_SCHEMA,
    )
    manifest_path = od / "tables_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written["tables_manifest.json"] = manifest_path
    return written


def build_tables_repro_dry_run(
    *,
    repo_root: str | Path,
    out_dir: str = DEFAULT_OUT_DIR,
    **kwargs: Any,
) -> dict[str, Any]:
    rr = Path(repo_root).resolve()
    data = build_tables_repro_bundle(repo_root=rr, **kwargs)
    return with_schema_version(
        {
            "status": "dry-run",
            "script": "tables-repro",
            "out_dir": str((rr / out_dir).resolve()),
            "would_write": [
                f"{out_dir}/headline_metrics.md",
                f"{out_dir}/aug_smoke_top_n.md",
                f"{out_dir}/zoo_core.md",
                f"{out_dir}/tables_manifest.json",
            ],
            "warnings": data.get("warnings"),
            "zoo_meta": data.get("zoo_meta"),
            "n_aug_rows": len(data.get("aug_top_n_rows") or []),
            "inputs": data.get("paths"),
        },
        schema_version=MANUSCRIPT_TABLES_MANIFEST_SCHEMA,
    )
