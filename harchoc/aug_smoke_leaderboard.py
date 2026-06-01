"""Aug smoke + sweep leaderboard from index and summary JSON artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import load_aug_smoke_index
from harchoc.model_zoo import file_sha256

LEADERBOARD_SCHEMA = "aug_smoke_leaderboard.v1"
DEFAULT_INDEX = "configs/experiments/aug_smoke_index.json"
DEFAULT_OUT_DIR = "reports/aug_smoke"
BEST2_REFERENCE = {
    "smoke_id": "best2",
    "run_name": "models/best2.pt",
    "epochs": 100,
    "test_count_mae": 61.26605504587156,
    "test_count_mae_ci": {
        "low": 51.30183486238532,
        "high": 71.28715596330274,
        "confidence": 0.95,
    },
    "source": "reports/hsp/fp_budget_sweep_test.json (locked conf 0.15, test split)",
    "key_overrides": "production YOLOv8m 100 ep; HSP primary reference",
}

SWEEP_KNOB_HINTS: dict[str, str] = {
    "aug_sweep_close10_15ep": "close_mosaic=2 scaled from 10 (15-ep sweep)",
    "aug_sweep_close15_15ep": "production close15 → effective close_mosaic=3 @ 15 ep",
    "aug_sweep_close25_15ep": "close_mosaic=4 @ 15 ep (production 25; patience=11)",
}


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != "aug_smoke_summary.v1":
        return None
    return obj


def _format_ci(ci: dict[str, Any] | None) -> str:
    if not ci:
        return "—"
    lo = ci.get("low")
    hi = ci.get("high")
    conf = ci.get("confidence")
    if lo is None or hi is None:
        return "—"
    pct = int(round(float(conf) * 100)) if conf is not None else 95
    return f"{float(lo):.1f}–{float(hi):.1f} ({pct}%)"


def _format_mae(mae: float | None) -> str:
    if mae is None:
        return "—"
    return f"{float(mae):.1f}"


def _sweep_knobs(run_name: str, smoke_id: str) -> str:
    return SWEEP_KNOB_HINTS.get(run_name) or SWEEP_KNOB_HINTS.get(smoke_id) or f"sweep {run_name}"


def _row_from_summary(
    summary: dict[str, Any],
    *,
    key_overrides: str | None = None,
    kind: str = "smoke",
) -> dict[str, Any]:
    run_name = str(summary.get("run_name") or "")
    smoke_id = str(summary.get("smoke_id") or "")
    knobs = key_overrides
    if not knobs and kind == "sweep":
        knobs = _sweep_knobs(run_name, smoke_id)
    return {
        "kind": kind,
        "smoke_id": smoke_id,
        "run_name": run_name,
        "test_count_mae": summary.get("test_count_mae"),
        "test_count_mae_ci": summary.get("test_count_mae_ci"),
        "key_overrides": knobs or "",
        "summary": str(summary.get("_source_path") or ""),
        "status": summary.get("status"),
        "generated_at": summary.get("generated_at"),
    }


def collect_leaderboard_rows(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> list[dict[str, Any]]:
    rr = Path(repo_root).resolve()
    index = load_aug_smoke_index(rr / index_path)
    od = rr / out_dir
    seen_summaries: set[str] = set()
    rows: list[dict[str, Any]] = []

    index_by_summary: dict[str, dict[str, Any]] = {}
    for entry in index.get("smokes") or []:
        sp = str(entry.get("summary") or "")
        if sp:
            index_by_summary[sp] = entry

    for rel in sorted(index_by_summary):
        summary_path = rr / rel
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        summary = dict(summary)
        summary["_source_path"] = rel
        entry = index_by_summary[rel]
        row = _row_from_summary(
            summary,
            key_overrides=str(entry.get("key_overrides") or ""),
            kind="smoke",
        )
        if entry.get("negative_control"):
            row["negative_control"] = True
        rows.append(row)
        seen_summaries.add(rel)

    for summary_path in sorted(od.glob("sweep_*_summary.json")):
        rel = str(summary_path.relative_to(rr))
        if rel in seen_summaries:
            continue
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        summary = dict(summary)
        summary["_source_path"] = rel
        rows.append(_row_from_summary(summary, kind="sweep"))
        seen_summaries.add(rel)

    for summary_path in sorted(od.glob("s[0-9]*_summary.json")):
        rel = str(summary_path.relative_to(rr))
        if rel in seen_summaries:
            continue
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        summary = dict(summary)
        summary["_source_path"] = rel
        entry = index_by_summary.get(rel, {})
        rows.append(
            _row_from_summary(
                summary,
                key_overrides=str(entry.get("key_overrides") or ""),
                kind="smoke",
            )
        )

    rows.sort(
        key=lambda r: (
            r.get("test_count_mae") is None,
            float(r.get("test_count_mae") or 1e18),
        )
    )
    return rows


def parse_equivalence_classes(
    index: dict[str, Any],
) -> tuple[set[str], dict[float, str], dict[str, tuple[str, str]]]:
    """Return audit-only smoke_ids, preds SHA by MAE, and audit sid -> (canonical, preds_sha)."""
    equiv = index.get("equivalence_classes") or {}
    audit_only: set[str] = set()
    verified_preds: dict[float, str] = {}
    audit_skip: dict[str, tuple[str, str]] = {}

    for cls in equiv.get("classes") or []:
        smoke_ids = [str(x).upper() for x in (cls.get("smoke_ids") or [])]
        if not smoke_ids:
            continue
        canonical = str(cls.get("canonical_smoke_id") or "").upper()
        if not canonical:
            canonical = "S1" if "S1" in smoke_ids else smoke_ids[0]
        sha = str(cls.get("preds_sha256") or "") if cls.get("preds_sha256") else ""
        for sid in smoke_ids:
            if sid != canonical:
                audit_only.add(sid)
                if sha:
                    audit_skip[sid] = (canonical, sha)
        mae = cls.get("test_count_mae")
        if sha and mae is not None:
            verified_preds[round(float(mae), 12)] = sha

    return audit_only, verified_preds, audit_skip


def partition_leaderboard_rows(
    rows: list[dict[str, Any]],
    *,
    audit_only: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into ranked, audit-duplicate, and eval-control sections."""
    ranked: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    eval_controls: list[dict[str, Any]] = []
    sort_key = lambda r: (  # noqa: E731
        r.get("test_count_mae") is None,
        float(r.get("test_count_mae") or 1e18),
    )
    for row in rows:
        sid = str(row.get("smoke_id") or "")
        if row.get("negative_control"):
            eval_controls.append(row)
        elif sid in audit_only:
            audit.append(row)
        else:
            ranked.append(row)
    ranked.sort(key=sort_key)
    audit.sort(key=sort_key)
    eval_controls.sort(key=sort_key)
    return ranked, audit, eval_controls


def _preds_sha_from_summary_path(repo_root: Path, summary_rel: str) -> str | None:
    sp = repo_root / summary_rel
    summary = _load_summary(sp)
    if summary is None:
        return None
    arts = summary.get("artifacts") or {}
    preds = arts.get("preds_json") or {}
    sha = preds.get("sha256")
    if sha:
        return str(sha)
    run_name = str(summary.get("run_name") or "")
    if not run_name:
        return None
    preds_path = sp.parent / f"{run_name}_preds.json"
    if not preds_path.is_file():
        return None
    return file_sha256(preds_path)


def find_mae_clusters(
    rows: list[dict[str, Any]],
    *,
    min_size: int = 2,
    repo_root: Path | None = None,
    verified_preds_by_mae: dict[float, str] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mae = row.get("test_count_mae")
        if mae is None:
            continue
        buckets[round(float(mae), 12)].append(row)
    clusters: list[dict[str, Any]] = []
    for mae_key in sorted(buckets):
        members = buckets[mae_key]
        if len(members) < min_size:
            continue
        preds_shas: list[str | None] = []
        if repo_root is not None:
            for m in members:
                rel = str(m.get("summary") or "")
                preds_shas.append(_preds_sha_from_summary_path(repo_root, rel) if rel else None)
        index_verified_sha = (verified_preds_by_mae or {}).get(mae_key)
        summaries_verified = bool(
            repo_root is not None
            and len(preds_shas) == len(members)
            and all(s is not None for s in preds_shas)
        )
        all_members_verified = summaries_verified or index_verified_sha is not None
        unique_preds = {s for s in preds_shas if s}
        if index_verified_sha and (not unique_preds or unique_preds <= {index_verified_sha}):
            unique_preds = {index_verified_sha}
        verified_count = len(unique_preds)
        preds_identical = all_members_verified and verified_count == 1
        if preds_identical:
            interpretation = (
                "identical test preds export (config equivalence or converged inference)"
            )
        elif verified_count > 1:
            interpretation = "distinct preds; bit-identical MAE likely coincidence"
        elif not all_members_verified:
            interpretation = "MAE match; preds SHA unverified for one or more members"
        else:
            interpretation = None
        clusters.append(
            {
                "test_count_mae": mae_key,
                "count": len(members),
                "smoke_ids": [str(m.get("smoke_id")) for m in members],
                "run_names": [str(m.get("run_name")) for m in members],
                "preds_sha256": sorted(unique_preds)[0] if preds_identical else None,
                "preds_sha256_distinct": verified_count,
                "preds_sha256_verified_count": verified_count,
                "interpretation": interpretation,
            }
        )
    return clusters


def _strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def build_leaderboard_payload(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    rr = Path(repo_root).resolve()
    index = load_aug_smoke_index(rr / index_path)
    rows = collect_leaderboard_rows(
        repo_root=rr,
        index_path=index_path,
        out_dir=out_dir,
    )
    audit_only, verified_preds, _ = parse_equivalence_classes(index)
    ranked_rows, audit_rows, eval_control_rows = partition_leaderboard_rows(
        rows, audit_only=audit_only
    )
    best_mae = min(
        (float(r["test_count_mae"]) for r in ranked_rows if r.get("test_count_mae") is not None),
        default=None,
    )
    ref_mae = float(BEST2_REFERENCE["test_count_mae"])
    equiv = index.get("equivalence_classes") or {}
    return {
        "schema_version": LEADERBOARD_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_metric": "test_count_mae",
        "epochs_smoke": 15,
        "reference": BEST2_REFERENCE,
        "best_smoke_mae": best_mae,
        "delta_best_smoke_vs_best2": (best_mae - ref_mae) if best_mae is not None else None,
        "equivalence_classes": equiv,
        "mae_clusters": find_mae_clusters(
            rows,
            repo_root=rr,
            verified_preds_by_mae=verified_preds,
        ),
        "ranked_rows": [_strip_private(r) for r in ranked_rows],
        "audit_duplicate_rows": [_strip_private(r) for r in audit_rows],
        "eval_control_rows": [_strip_private(r) for r in eval_control_rows],
        "rows": [_strip_private(r) for r in rows],
    }


def _row_kind_suffix(row: dict[str, Any]) -> str:
    kind = row.get("kind") or ""
    tags: list[str] = []
    if kind == "sweep":
        tags.append("sweep")
    return f" *({', '.join(tags)})*" if tags else ""


def _append_ranked_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| rank | smoke_id | run_name | MAE | 95% CI | key aug knobs | summary |",
            "|------|----------|----------|-----|--------|---------------|---------|",
        ]
    )
    for i, row in enumerate(rows, start=1):
        sid = row.get("smoke_id") or "—"
        sid_cell = f"{sid}{_row_kind_suffix(row)}"
        lines.append(
            f"| {i} | {sid_cell} | `{row.get('run_name')}` | "
            f"{_format_mae(row.get('test_count_mae'))} | "
            f"{_format_ci(row.get('test_count_mae_ci'))} | "
            f"{row.get('key_overrides', '')} | "
            f"`{row.get('summary', '')}` |"
        )


def _append_unranked_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| smoke_id | run_name | MAE | 95% CI | key aug knobs | summary |",
            "|----------|----------|-----|--------|---------------|---------|",
        ]
    )
    for row in rows:
        sid = row.get("smoke_id") or "—"
        lines.append(
            f"| {sid} | `{row.get('run_name')}` | "
            f"{_format_mae(row.get('test_count_mae'))} | "
            f"{_format_ci(row.get('test_count_mae_ci'))} | "
            f"{row.get('key_overrides', '')} | "
            f"`{row.get('summary', '')}` |"
        )


def render_leaderboard_md(payload: dict[str, Any]) -> str:
    ref = payload.get("reference") or BEST2_REFERENCE
    ref_mae = float(ref.get("test_count_mae") or 0)
    ref_ci = ref.get("test_count_mae_ci") or {}
    ranked_rows = payload.get("ranked_rows")
    if ranked_rows is None:
        ranked_rows = [
            r for r in (payload.get("rows") or []) if not r.get("negative_control")
        ]
    audit_rows = payload.get("audit_duplicate_rows") or []
    eval_control_rows = payload.get("eval_control_rows") or []
    lines: list[str] = [
        "# Aug smoke + sweep leaderboard",
        "",
        f"Generated: `{payload.get('generated_at')}` · primary metric: **test count MAE** "
        f"(val-locked conf, test split, *n*=109).",
        "",
        "## Reference (production)",
        "",
        "| ID | run | epochs | test count MAE | 95% CI | notes |",
        "|----|-----|--------|----------------|--------|-------|",
        f"| **best2** | `{ref.get('run_name')}` | {ref.get('epochs', 100)} | "
        f"**{_format_mae(ref_mae)}** | {_format_ci(ref_ci)} | {ref.get('key_overrides', '')} |",
        "",
        f"Source: `{ref.get('source', 'reports/hsp/fp_budget_sweep_test.json')}`. "
        f"No 15-ep smoke beats best2; best smoke/sweep is **+{payload.get('delta_best_smoke_vs_best2', 0):.1f} MAE** "
        f"vs best2 ({_format_mae(ref_mae)}).",
        "",
        "## Rankings (15 ep)",
        "",
    ]
    _append_ranked_table(lines, ranked_rows)

    if audit_rows:
        lines.extend(
            [
                "",
                "## Audit / duplicate class",
                "",
                "Training-equivalent duplicates (canonical per cluster ranked above); not scored separately.",
                "",
            ]
        )
        _append_unranked_table(lines, audit_rows)

    if eval_control_rows:
        lines.extend(
            [
                "",
                "## Eval controls",
                "",
                "Eval-only negative controls; excluded from ranked best smoke.",
                "",
            ]
        )
        _append_unranked_table(lines, eval_control_rows)

    clusters = payload.get("mae_clusters") or []
    if clusters:
        lines.extend(["", "## Duplicate MAE clusters", ""])
        lines.append(
            "Bit-identical test count MAE across runs — check `artifacts.preds_json.sha256` in summaries "
            "(follow-up **P1-AUG-DUP-MAE**; eval wiring is per-run weights, not shared preds)."
        )
        lines.append("")
        for cl in clusters:
            mae = cl.get("test_count_mae")
            ids = ", ".join(f"**{x}**" for x in cl.get("smoke_ids") or [])
            runs = ", ".join(f"`{x}`" for x in cl.get("run_names") or [])
            note = cl.get("interpretation") or ""
            sha = cl.get("preds_sha256")
            sha_note = f" · preds `{sha[:12]}…`" if sha else ""
            lines.append(
                f"- **{_format_mae(float(mae))}** ×{cl.get('count')}: {ids} — {runs}{sha_note}"
                + (f" — {note}" if note else "")
            )

    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/experiment.py aug-leaderboard",
            "```",
            "",
            "Auto-refreshed after each aug smoke / sweep summary via `harchoc.aug_smoke_leaderboard.refresh_aug_smoke_leaderboard`.",
            "",
            "Index: [`configs/experiments/aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json).",
        ]
    )
    return "\n".join(lines) + "\n"


def write_aug_smoke_leaderboard(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    write_json: bool = True,
) -> dict[str, Path]:
    rr = Path(repo_root).resolve()
    od = rr / out_dir
    od.mkdir(parents=True, exist_ok=True)
    payload = build_leaderboard_payload(
        repo_root=rr,
        index_path=index_path,
        out_dir=out_dir,
    )
    md_path = od / "leaderboard.md"
    md_path.write_text(render_leaderboard_md(payload), encoding="utf-8")
    out: dict[str, Path] = {"markdown": md_path.resolve()}
    if write_json:
        json_path = od / "leaderboard.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        out["json"] = json_path.resolve()
    return out


def refresh_aug_smoke_leaderboard(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Path] | None:
    """Best-effort leaderboard refresh (non-fatal on failure)."""
    try:
        return write_aug_smoke_leaderboard(
            repo_root=repo_root,
            index_path=index_path,
            out_dir=out_dir,
        )
    except Exception as exc:
        print(f"# aug smoke leaderboard refresh skipped: {exc}")
        return None
