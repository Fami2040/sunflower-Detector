"""Aug smoke comparative analysis from index + summary artifacts (CPU-only, no retrain)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_leaderboard import (
    BEST2_REFERENCE,
    build_leaderboard_payload,
    collect_leaderboard_rows,
    partition_leaderboard_rows,
)
from harchoc.aug_smoke_runner import load_aug_smoke_index
from harchoc.equivalence_index import parse_equivalence_classes

COMPARATIVE_SCHEMA = "aug_comparative_analysis.v1"
DEFAULT_INDEX = "configs/experiments/aug_smoke_index.json"
DEFAULT_OUT_DIR = "reports/aug_smoke"
REJECTED_SMOKE_IDS = frozenset({"S2"})
REJECTED_SWEEP_ARM_IDS = frozenset({"close10"})
AUG_CONFIRM_SUMMARY = "reports/aug_smoke/aug_confirm_winner_100ep_summary.json"
AUG_CONFIRM_AUG_CONFIG = "configs/aug/robustness_minimal.yaml"


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != "aug_smoke_summary.v1":
        return None
    return obj


def _sweep_display_id(arm: dict[str, Any]) -> str:
    """Leaderboard-style sweep id (e.g. close10 → CLOSE10)."""
    arm_id = str(arm.get("id") or "").strip()
    if arm_id.lower().startswith("close"):
        return arm_id.upper()
    return arm_id.upper()


def _row_from_sources(
    *,
    smoke_id: str,
    run_name: str,
    summary: dict[str, Any] | None,
    index_entry: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    mae = None
    ci = None
    summary_rel = str(index_entry.get("summary") or "")
    if summary is not None:
        mae = summary.get("test_count_mae")
        ci = summary.get("test_count_mae_ci")
    if mae is None and index_entry.get("test_count_mae") is not None:
        mae = float(index_entry["test_count_mae"])
    return {
        "kind": kind,
        "smoke_id": smoke_id,
        "run_name": run_name or str(index_entry.get("run_name") or ""),
        "aug_config": index_entry.get("aug_config"),
        "train_config": index_entry.get("train_config"),
        "test_count_mae": mae,
        "test_count_mae_ci": ci,
        "key_overrides": str(index_entry.get("key_overrides") or ""),
        "summary": summary_rel,
        "status": str(index_entry.get("status") or summary.get("status") if summary else ""),
        "negative_control": bool(index_entry.get("negative_control")),
        "eval_only": bool(index_entry.get("eval_only")),
        "mae_source": "summary" if summary and summary.get("test_count_mae") is not None else "index",
    }


def collect_index_aug_arms(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
) -> list[dict[str, Any]]:
    """All smokes + 15-ep sweep arms with MAE from summary or index fallback."""
    rr = Path(repo_root).resolve()
    index = load_aug_smoke_index(rr / index_path)
    arms: list[dict[str, Any]] = []

    for entry in index.get("smokes") or []:
        sid = str(entry.get("id") or "").upper()
        summary_rel = str(entry.get("summary") or "")
        summary = _load_summary(rr / summary_rel) if summary_rel else None
        arms.append(
            _row_from_sources(
                smoke_id=sid,
                run_name=str(entry.get("run_name") or ""),
                summary=summary,
                index_entry=entry,
                kind="smoke",
            )
        )

    for arm in (index.get("sweeps_15ep") or {}).get("arms") or []:
        sid = _sweep_display_id(arm)
        summary_rel = str(arm.get("summary") or "")
        summary = _load_summary(rr / summary_rel) if summary_rel else None
        arms.append(
            _row_from_sources(
                smoke_id=sid,
                run_name=str(arm.get("run_name") or ""),
                summary=summary,
                index_entry=arm,
                kind="sweep",
            )
        )
    return arms


def _confirm_100ep_reference(repo_root: Path, index: dict[str, Any]) -> dict[str, Any]:
    sweeps = index.get("sweeps_100ep") or {}
    mae = sweeps.get("baseline_confirm_mae")
    summary_rel = sweeps.get("baseline_confirm_summary") or AUG_CONFIRM_SUMMARY
    summary = _load_summary(repo_root / summary_rel)
    if summary and summary.get("test_count_mae") is not None:
        mae = float(summary["test_count_mae"])
    ci = summary.get("test_count_mae_ci") if summary else None
    return {
        "id": "aug_confirm_100ep",
        "run_name": str(summary.get("run_name") if summary else "aug_confirm_winner_100ep"),
        "epochs": 100,
        "aug_config": AUG_CONFIRM_AUG_CONFIG,
        "test_count_mae": float(mae) if mae is not None else None,
        "test_count_mae_ci": ci,
        "summary": summary_rel,
        "note": "100-ep production confirm on robustness_minimal.yaml",
    }


def build_rejected_arms(
    arms: list[dict[str, Any]],
    *,
    index: dict[str, Any],
    canonical_mae: float | None,
) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    arch = index.get("arch_mosaic0_ab") or {}
    rejected_ids = set(REJECTED_SMOKE_IDS)
    for arm in (index.get("sweeps_15ep") or {}).get("arms") or []:
        if str(arm.get("id") or "").lower() in REJECTED_SWEEP_ARM_IDS:
            rejected_ids.add(_sweep_display_id(arm))
    for row in arms:
        sid = str(row.get("smoke_id") or "").upper()
        if sid not in rejected_ids:
            continue
        mae = row.get("test_count_mae")
        delta = (float(mae) - canonical_mae) if mae is not None and canonical_mae is not None else None
        reason = ""
        if sid == "S2":
            comp = arch.get("comparison") or {}
            delta_ab = comp.get("delta_mosaic0_minus_baseline")
            reason = str(arch.get("interpretation") or "").strip() or (
                f"mosaic=0 rejected (+{delta_ab:.1f} MAE vs S1 cluster)"
                if delta_ab is not None
                else "mosaic=0; large MAE regression vs close3 winner"
            )
        elif sid == "CLOSE10":
            reason = "close_mosaic=10 @ 15 ep underperforms S1; Phase B 100 ep gated off"
        entry = {
            "smoke_id": sid,
            "run_name": row.get("run_name"),
            "aug_config": row.get("aug_config"),
            "test_count_mae": mae,
            "test_count_mae_ci": row.get("test_count_mae_ci"),
            "delta_vs_canonical_s1": delta,
            "versus_canonical": "S1",
            "reason": reason,
            "summary": row.get("summary"),
        }
        rejected.append(entry)
    rejected.sort(key=lambda r: float(r.get("test_count_mae") or 1e18))
    return rejected


def _derive_narrative_bullets(payload: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    ranked = payload.get("rankings") or []
    winner = ranked[0] if ranked else None
    ref = payload.get("reference_best2") or BEST2_REFERENCE
    confirm = payload.get("reference_aug_confirm_100ep") or {}
    best2_mae = float(ref.get("test_count_mae") or 0)
    confirm_mae = confirm.get("test_count_mae")
    best_smoke = payload.get("best_smoke_mae")

    if winner:
        wid = winner.get("smoke_id")
        wmae = winner.get("test_count_mae")
        bullets.append(
            f"Best 15-ep smoke: **{wid}** (`{winner.get('run_name')}`) at "
            f"**{float(wmae):.1f}** test count MAE — {winner.get('key_overrides', '')}."
        )

    equiv = payload.get("equivalence_classes") or {}
    for cls in equiv.get("classes") or []:
        ids = cls.get("smoke_ids") or []
        canon = cls.get("canonical_smoke_id")
        if len(ids) >= 2:
            bullets.append(
                f"Equivalence @ {float(cls.get('test_count_mae', 0)):.1f} MAE: "
                f"{', '.join(ids)} (canonical **{canon}**); do not re-train duplicates."
            )

    for arm in payload.get("rejected_arms") or []:
        sid = arm.get("smoke_id")
        mae = arm.get("test_count_mae")
        bullets.append(
            f"Rejected **{sid}**: MAE **{float(mae):.1f}** — {arm.get('reason', '')}"
        )

    if best_smoke is not None:
        gap = float(best_smoke) - best2_mae
        bullets.append(
            f"No 15-ep arm beats production **best2** ({best2_mae:.1f} MAE); "
            f"best smoke is **+{gap:.1f}** MAE vs best2."
        )

    if confirm_mae is not None:
        gap100 = float(confirm_mae) - best2_mae
        bullets.append(
            f"100-ep confirm on [`robustness_minimal.yaml`](../../configs/aug/robustness_minimal.yaml): "
            f"**{float(confirm_mae):.1f}** MAE vs best2 **{best2_mae:.1f}** "
            f"(**+{gap100:.1f}**); production recipe retained."
        )

    return bullets


def build_comparative_payload(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    rr = Path(repo_root).resolve()
    index = load_aug_smoke_index(rr / index_path)
    leaderboard = build_leaderboard_payload(
        repo_root=rr,
        index_path=index_path,
        out_dir=out_dir,
    )
    audit_only, _, _ = parse_equivalence_classes(index)
    rows = collect_leaderboard_rows(repo_root=rr, index_path=index_path, out_dir=out_dir)
    index_arms = collect_index_aug_arms(repo_root=rr, index_path=index_path)

    # Merge index fallback MAE for arms missing from leaderboard rows (e.g. S2 without summary).
    row_by_id = {str(r.get("smoke_id") or "").upper(): r for r in rows}
    for arm in index_arms:
        sid = str(arm.get("smoke_id") or "").upper()
        if sid not in row_by_id and arm.get("test_count_mae") is not None:
            rows.append(dict(arm))
            row_by_id[sid] = arm

    ranked_rows, audit_rows, eval_rows = partition_leaderboard_rows(rows, audit_only=audit_only)
    ranked_rows.sort(
        key=lambda r: (
            r.get("test_count_mae") is None,
            float(r.get("test_count_mae") or 1e18),
        )
    )

    rankings: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_rows, start=1):
        rankings.append({"rank": rank, **_public_row(row)})

    canonical_mae = None
    for r in ranked_rows:
        if str(r.get("smoke_id") or "").upper() == "S1":
            canonical_mae = r.get("test_count_mae")
            break
    if canonical_mae is None and ranked_rows:
        canonical_mae = ranked_rows[0].get("test_count_mae")

    confirm_ref = _confirm_100ep_reference(rr, index)
    ref_mae = float(BEST2_REFERENCE["test_count_mae"])
    best_smoke = leaderboard.get("best_smoke_mae")

    payload: dict[str, Any] = {
        "schema_version": COMPARATIVE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_metric": "test_count_mae",
        "epochs_smoke": index.get("epochs", 15),
        "reference_best2": BEST2_REFERENCE,
        "reference_aug_confirm_100ep": confirm_ref,
        "best_smoke_mae": best_smoke,
        "delta_best_smoke_vs_best2": leaderboard.get("delta_best_smoke_vs_best2"),
        "delta_confirm_100ep_vs_best2": (
            float(confirm_ref["test_count_mae"]) - ref_mae
            if confirm_ref.get("test_count_mae") is not None
            else None
        ),
        "rankings": rankings,
        "audit_duplicate_rows": [_public_row(r) for r in audit_rows],
        "eval_control_rows": [_public_row(r) for r in eval_rows],
        "equivalence_classes": index.get("equivalence_classes") or {},
        "mae_clusters": leaderboard.get("mae_clusters") or [],
        "rejected_arms": build_rejected_arms(
            index_arms,
            index=index,
            canonical_mae=float(canonical_mae) if canonical_mae is not None else None,
        ),
    }
    payload["narrative_bullets"] = _derive_narrative_bullets(payload)
    return payload


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def render_comparative_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Aug comparative analysis",
        "",
        f"Generated: `{payload.get('generated_at')}` · schema `{payload.get('schema_version')}`",
        "",
        "CPU-only synthesis from [`aug_smoke_index.json`](../../configs/experiments/aug_smoke_index.json) "
        "and existing `*_summary.json` artifacts — no re-training.",
        "",
        "## Narrative",
        "",
    ]
    for bullet in payload.get("narrative_bullets") or []:
        lines.append(f"- {bullet}")
    lines.extend(
        [
            "",
            "## Rankings (15 ep, deduped)",
            "",
            "| rank | smoke_id | MAE | 95% CI | aug knobs |",
            "|------|----------|-----|--------|-----------|",
        ]
    )
    for row in payload.get("rankings") or []:
        ci = row.get("test_count_mae_ci") or {}
        ci_s = (
            f"{float(ci.get('low', 0)):.1f}–{float(ci.get('high', 0)):.1f}"
            if ci.get("low") is not None
            else "—"
        )
        lines.append(
            f"| {row.get('rank')} | {row.get('smoke_id')} | "
            f"{float(row.get('test_count_mae', 0)):.1f} | {ci_s} | {row.get('key_overrides', '')} |"
        )
    lines.extend(
        [
            "",
            "## Rejected arms",
            "",
            "| smoke_id | MAE | vs S1 Δ | reason |",
            "|----------|-----|---------|--------|",
        ]
    )
    for arm in payload.get("rejected_arms") or []:
        delta = arm.get("delta_vs_canonical_s1")
        delta_s = f"+{float(delta):.1f}" if delta is not None else "—"
        lines.append(
            f"| {arm.get('smoke_id')} | {float(arm.get('test_count_mae', 0)):.1f} | "
            f"{delta_s} | {arm.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Aug MAE comparison](fig_aug_mae_comparison.png)",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/experiment.py aug-compare",
            "```",
            "",
            "JSON: [`comparative_analysis.json`](comparative_analysis.json) · "
            "Leaderboard: [`leaderboard.md`](leaderboard.md).",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_aug_mae_comparison_figure(
    payload: dict[str, Any],
    *,
    out_path: Path,
    journal_style: bool = True,
    max_bars: int = 12,
) -> dict[str, Any]:
    try:
        from harchoc.figure_style import add_panel_label, panel_label, prepare_matplotlib, savefig_kwargs

        prepare_matplotlib(journal_style=journal_style)
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as ex:
        return {"status": "skipped", "reason": f"missing_dependency:matplotlib ({ex})"}

    ranked = list(payload.get("rankings") or [])
    rejected = list(payload.get("rejected_arms") or [])
    rejected_ids = {str(a.get("smoke_id") or "").upper() for a in rejected}

    plot_rows: list[dict[str, Any]] = []
    for row in ranked:
        sid = str(row.get("smoke_id") or "").upper()
        if sid in rejected_ids:
            continue
        if row.get("test_count_mae") is None:
            continue
        plot_rows.append({**row, "tier": "ranked"})
    for arm in rejected:
        if arm.get("test_count_mae") is None:
            continue
        plot_rows.append({**arm, "tier": "rejected"})

    plot_rows.sort(key=lambda r: float(r.get("test_count_mae") or 1e18))
    if len(plot_rows) > max_bars:
        # Keep all rejected + best ranked arms.
        rej = [r for r in plot_rows if r.get("tier") == "rejected"]
        rest = [r for r in plot_rows if r.get("tier") != "rejected"]
        plot_rows = rest[: max(1, max_bars - len(rej))] + rej
        plot_rows.sort(key=lambda r: float(r.get("test_count_mae") or 1e18))

    if not plot_rows:
        return {"status": "skipped", "reason": "no MAE rows to plot"}

    labels = [str(r.get("smoke_id") or "?") for r in plot_rows]
    maes = [float(r["test_count_mae"]) for r in plot_rows]
    colors = ["#c44e52" if r.get("tier") == "rejected" else "#4c72b0" for r in plot_rows]
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for r in plot_rows:
        ci = r.get("test_count_mae_ci") or {}
        pt = float(r.get("test_count_mae") or 0)
        lo = ci.get("low")
        hi = ci.get("high")
        yerr_lo.append(pt - float(lo) if lo is not None else 0.0)
        yerr_hi.append(float(hi) - pt if hi is not None else 0.0)

    ref = payload.get("reference_best2") or BEST2_REFERENCE
    confirm = payload.get("reference_aug_confirm_100ep") or {}
    best2_mae = float(ref.get("test_count_mae") or 0)
    confirm_mae = confirm.get("test_count_mae")

    fig_h = max(2.8, 0.28 * len(plot_rows) + 0.8)
    figsize = (3.6, fig_h) if journal_style else (7, fig_h)
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(plot_rows))
    ax.barh(
        list(y_pos),
        maes,
        xerr=[yerr_lo, yerr_hi],
        color=colors,
        height=0.65,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test count MAE (lower is better)")
    ax.set_title("Aug smoke / sweep (15 ep) vs production references")
    ax.axvline(best2_mae, color="#55a868", linestyle="--", linewidth=1, label=f"best2 {best2_mae:.1f}")
    if confirm_mae is not None:
        ax.axvline(
            float(confirm_mae),
            color="#8172b3",
            linestyle=":",
            linewidth=1,
            label=f"100 ep confirm {float(confirm_mae):.1f}",
        )
    ax.legend(loc="lower right", fontsize=7)
    ax.invert_yaxis()
    if journal_style:
        add_panel_label(ax, panel_label(0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **savefig_kwargs(journal_style=journal_style))
    plt.close(fig)
    return {"status": "ok", "path": str(out_path), "n_bars": len(plot_rows)}


def write_aug_comparative_analysis(
    *,
    repo_root: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    write_figure: bool = True,
) -> dict[str, Path]:
    rr = Path(repo_root).resolve()
    od = rr / out_dir
    od.mkdir(parents=True, exist_ok=True)
    payload = build_comparative_payload(
        repo_root=rr,
        index_path=index_path,
        out_dir=out_dir,
    )
    json_path = od / "comparative_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = od / "comparative_analysis.md"
    md_path.write_text(render_comparative_md(payload), encoding="utf-8")
    out: dict[str, Path] = {
        "json": json_path.resolve(),
        "markdown": md_path.resolve(),
    }
    if write_figure:
        fig_path = od / "fig_aug_mae_comparison.png"
        emit_aug_mae_comparison_figure(payload, out_path=fig_path)
        out["figure"] = fig_path.resolve()
    return out
