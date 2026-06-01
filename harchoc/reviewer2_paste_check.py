"""Reviewer 2 manuscript paste readiness + SOTA inventory vs zoo matrix (no ML deps)."""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Status = Literal["pass", "fail", "warn", "skip"]

_SCHEMA = "reviewer2_paste_check.v1"
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_DEFAULT_DOCX = "reports/plants-4336582.docx"
_DEFAULT_CHECKLIST = "reports/reviewer2_docx_paste_checklist.md"
_DEFAULT_INVENTORY_MD = "reports/reviewer2_sota_inventory.md"
_DEFAULT_INVENTORY_JSON = "reports/reviewer2_sota_inventory.json"
_DEFAULT_MATRIX_ROWS = "configs/zoo/matrix_rows.v1.json"
_DEFAULT_OUT = "reports/reviewer2_paste_check.json"

# Checklist rows: paste target + optional artifact gates (from reviewer2_docx_paste_checklist.md).
_PASTE_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "paste_abstract",
        "section": "Abstract",
        "source": "reports/reviewer2_abstract_draft.md",
    },
    {
        "id": "paste_related_work",
        "section": "§2 Related work",
        "source": "reports/reviewer2_related_work.md",
        "also": ["docs/manuscript/related_work_outline.md"],
    },
    {
        "id": "paste_val_test_map",
        "section": "§2.2 val vs test mAP",
        "source": "docs/manuscript/val_test_map_gap.md",
    },
    {
        "id": "paste_methods_counting",
        "section": "Methods counting / n=50",
        "source": "docs/manuscript/reviewer_comments_backlog_gap.md",
    },
    {
        "id": "paste_sota_table",
        "section": "Results SOTA table",
        "source": "reports/reviewer2_sota_inventory.md",
        "artifacts": ["reports/hsp/matrix_train.json"],
        "artifacts_optional": True,
    },
    {
        "id": "paste_discussion_gen",
        "section": "Discussion generalization",
        "source": "reports/reviewer2_related_work.md",
        "also": ["reports/domains/domain_eval.json"],
    },
    {
        "id": "paste_figures",
        "section": "Figures",
        "figures": [
            "reports/figures/fig_error_taxonomy.png",
            "reports/figures/fig_ambiguous_panel.png",
            "reports/figures/fig_concept.png",
        ],
    },
    {
        "id": "paste_telegram_deploy",
        "section": "Telegram deploy stats",
        "status_if_missing_repo": "warn",
        "note": "96.4% / latency not in repo JSON — export bot logs before citing",
    },
)

# Headline numbers inventory cites (rounded for comparison).
_INVENTORY_MAE: tuple[tuple[str, float, str], ...] = (
    ("best2", 61.3, "reports/reviewer2_counting_metrics_computed.json"),
    ("aug_confirm_100ep", 64.1, "reports/aug_smoke/aug_confirm_winner_100ep_summary.json"),
    ("aug_smoke_s1", 68.9, "reports/aug_smoke/s1_summary.json"),
    ("yolo26m_e100_s0", 95.3, "reports/hsp/yolo26m_e100_s0_error.json"),
)

_ZOO_RUN_SUFFIX = "_e100_s0"
_MATRIX_GROUP_ZOO_CORE = "zoo_core"


def extract_docx_text(path: Path) -> str:
    """Extract plain text from .docx via word/document.xml (stdlib only)."""
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    parts: list[str] = []
    for node in root.iter(f"{_W_NS}t"):
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    return "".join(parts)


def _rel(repo_root: Path, rel: str) -> Path:
    return (repo_root / rel).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _item(
    item_id: str,
    *,
    section: str,
    status: Status,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": item_id,
        "section": section,
        "status": status,
        "message": message,
    }
    row.update(extra)
    return row


def _near(a: float, b: float, *, tol: float = 0.25) -> bool:
    return abs(a - b) <= tol


def _pooled_mae_from_error(doc: dict[str, Any]) -> float | None:
    counting = doc.get("counting_metrics")
    if isinstance(counting, dict) and counting.get("mae") is not None:
        return float(counting["mae"])
    summary = doc.get("summary")
    if isinstance(summary, dict):
        pooled = summary.get("pooled") or summary.get("counting")
        if isinstance(pooled, dict) and pooled.get("mae") is not None:
            return float(pooled["mae"])
    for key in ("pooled", "counting", "test"):
        block = doc.get(key)
        if isinstance(block, dict) and block.get("mae") is not None:
            return float(block["mae"])
    metrics = doc.get("metrics")
    if isinstance(metrics, dict) and metrics.get("mae") is not None:
        return float(metrics["mae"])
    return None


def _mae_from_counting_computed(doc: dict[str, Any]) -> float | None:
    pooled = doc.get("pooled")
    if isinstance(pooled, dict) and pooled.get("mae") is not None:
        return float(pooled["mae"])
    return None


def load_matrix_rows(repo_root: Path, matrix_rows_path: str | Path) -> dict[str, Any]:
    path = _rel(repo_root, str(matrix_rows_path))
    if not path.is_file():
        raise FileNotFoundError(f"missing matrix rows manifest: {matrix_rows_path}")
    return _read_json(path)


def matrix_group_row_ids(doc: dict[str, Any], group: str) -> list[str]:
    rows = doc.get("rows") or []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        groups = row.get("groups") or []
        if group in groups:
            out.append(str(row.get("id") or ""))
    return [x for x in out if x]


def parse_sota_inventory_md(text: str) -> dict[str, Any]:
    """Light parse of reviewer2_sota_inventory.md for run names and cited MAEs."""
    run_names = sorted(
        {
            m.group(1).strip()
            for m in re.finditer(r"`([a-z0-9_]+_e100_s0)`", text, flags=re.I)
        }
    )
    smoke_ids = sorted(
        {m.group(1).strip().upper() for m in re.finditer(r"\b(S\d{1,2})\b", text)}
    )
    maes: dict[str, float] = {}
    for m in re.finditer(r"\*\*(\d+\.\d+)\*\*", text):
        pass  # too noisy; use table rows below
    for label, pat in (
        ("best2", r"best2\.pt`\s*\|\s*\*\*(\d+\.\d+)"),
        ("aug_confirm", r"100 ep.*?\|\s*\*\*(\d+\.\d+)"),
        ("aug_s1", r"\|\s*S1\s*\|\s*(\d+\.\d+)"),
        ("yolo26m", r"YOLO26m.*?\*\*(\d+\.\d+)"),
    ):
        hit = re.search(pat, text, flags=re.I | re.S)
        if hit:
            maes[label] = float(hit.group(1))
    return {
        "run_names_e100": run_names,
        "smoke_ids": smoke_ids,
        "cited_mae": maes,
        "mentions_matrix_train": "matrix_train.json" in text,
        "mentions_zoo_core": "zoo_core" in text,
    }


def load_sota_inventory(
    repo_root: Path,
    *,
    inventory_md: str | Path,
    inventory_json: str | Path,
) -> dict[str, Any]:
    jp = _rel(repo_root, str(inventory_json))
    if jp.is_file():
        doc = _read_json(jp)
        doc["_source"] = str(inventory_json)
        return doc
    mp = _rel(repo_root, str(inventory_md))
    if not mp.is_file():
        raise FileNotFoundError(f"missing SOTA inventory: {inventory_md}")
    parsed = parse_sota_inventory_md(mp.read_text(encoding="utf-8"))
    parsed["_source"] = str(inventory_md)
    parsed["_format"] = "markdown"
    return parsed


def check_paste_sources(repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for spec in _PASTE_SOURCES:
        iid = str(spec["id"])
        section = str(spec["section"])
        if spec.get("note") and not spec.get("source"):
            items.append(
                _item(
                    iid,
                    section=section,
                    status=str(spec.get("status_if_missing_repo") or "warn"),  # type: ignore[arg-type]
                    message=str(spec["note"]),
                )
            )
            continue
        missing: list[str] = []
        src = str(spec.get("source") or "")
        if src and not _rel(repo_root, src).is_file():
            missing.append(src)
        for also in spec.get("also") or []:
            if not _rel(repo_root, str(also)).is_file():
                missing.append(str(also))
        for fig in spec.get("figures") or []:
            if not _rel(repo_root, str(fig)).is_file():
                missing.append(str(fig))
        art_missing: list[str] = []
        for art in spec.get("artifacts") or []:
            if not _rel(repo_root, str(art)).is_file():
                art_missing.append(str(art))
        if missing:
            items.append(
                _item(
                    iid,
                    section=section,
                    status="fail",
                    message=f"missing paste source(s): {', '.join(missing)}",
                    missing=missing,
                )
            )
        elif art_missing and not spec.get("artifacts_optional"):
            items.append(
                _item(
                    iid,
                    section=section,
                    status="fail",
                    message=f"missing artifact(s): {', '.join(art_missing)}",
                    missing=art_missing,
                )
            )
        elif art_missing:
            items.append(
                _item(
                    iid,
                    section=section,
                    status="warn",
                    message=f"paste source OK; pending artifact(s): {', '.join(art_missing)}",
                    missing=art_missing,
                )
            )
        else:
            sources = [s for s in [src, *(spec.get("also") or []), *(spec.get("figures") or [])] if s]
            items.append(
                _item(
                    iid,
                    section=section,
                    status="pass",
                    message="paste source(s) present on disk",
                    sources=sources,
                )
            )
    return items


def check_inventory_mae(repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for label, expected, rel_path in _INVENTORY_MAE:
        path = _rel(repo_root, rel_path)
        if not path.is_file():
            items.append(
                _item(
                    f"sota_mae_{label}",
                    section="SOTA inventory metrics",
                    status="fail",
                    message=f"missing artifact for {label}: {rel_path}",
                )
            )
            continue
        actual: float | None = None
        if rel_path.endswith(".json"):
            doc = _read_json(path)
            if "test_count_mae" in doc:
                actual = float(doc["test_count_mae"])
            elif label == "best2":
                actual = _mae_from_counting_computed(doc)
            else:
                actual = _pooled_mae_from_error(doc)
        else:
            text = path.read_text(encoding="utf-8")
            hit = re.search(
                r"\|\s*S1\s*\|[^|]*\|\s*(\d+\.\d+)\s*\|",
                text,
            )
            if hit:
                actual = float(hit.group(1))
        if actual is None:
            items.append(
                _item(
                    f"sota_mae_{label}",
                    section="SOTA inventory metrics",
                    status="warn",
                    message=f"could not read MAE from {rel_path}",
                )
            )
        elif _near(actual, expected):
            items.append(
                _item(
                    f"sota_mae_{label}",
                    section="SOTA inventory metrics",
                    status="pass",
                    message=f"{label} MAE {actual:.1f} matches inventory ~{expected}",
                    expected=expected,
                    actual=round(actual, 2),
                    path=rel_path,
                )
            )
        else:
            items.append(
                _item(
                    f"sota_mae_{label}",
                    section="SOTA inventory metrics",
                    status="fail",
                    message=f"{label} MAE {actual:.1f} != inventory {expected}",
                    expected=expected,
                    actual=round(actual, 2),
                    path=rel_path,
                )
            )
    return items


def check_inventory_vs_matrix(
    repo_root: Path,
    inventory: dict[str, Any],
    matrix_doc: dict[str, Any],
) -> dict[str, Any]:
    zoo_core_ids = matrix_group_row_ids(matrix_doc, _MATRIX_GROUP_ZOO_CORE)
    run_names = list(inventory.get("run_names_e100") or [])
    if not run_names and isinstance(inventory.get("zoo_runs"), list):
        run_names = [str(x) for x in inventory["zoo_runs"]]

    mapped: list[dict[str, str]] = []
    unmapped: list[str] = []
    for run in run_names:
        stem = run.replace(_ZOO_RUN_SUFFIX, "") if run.endswith(_ZOO_RUN_SUFFIX) else run
        if stem in zoo_core_ids:
            mapped.append({"run_name": run, "matrix_id": stem})
        else:
            unmapped.append(run)

    runs_dir = _rel(repo_root, "runs/detect/runs/hsp_zoo")
    if not runs_dir.is_dir():
        alt = _rel(repo_root, "runs/hsp_zoo")
        runs_dir = alt if alt.is_dir() else runs_dir

    on_disk = sorted(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    missing_weights = [r for r in run_names if r not in on_disk]

    matrix_train = _rel(repo_root, "reports/hsp/matrix_train.json")
    p0_summary = _rel(repo_root, "reports/hsp/p0_summary.md")
    matrix_status: dict[str, Any] = {
        "matrix_rows_total": len(matrix_doc.get("rows") or []),
        "zoo_core_ids": zoo_core_ids,
        "inventory_run_names": run_names,
        "mapped_to_zoo_core": mapped,
        "unmapped_run_names": unmapped,
        "zoo_runs_on_disk": on_disk,
        "inventory_runs_missing_on_disk": missing_weights,
        "matrix_train_json": str(matrix_train.relative_to(repo_root))
        if matrix_train.is_file()
        else None,
        "p0_summary_md": str(p0_summary.relative_to(repo_root)) if p0_summary.is_file() else None,
    }

    verified = False
    verify_reason = ""
    if matrix_train.is_file():
        from harchoc.queue_skip_gates import matrix_train_verified

        verified, verify_reason = matrix_train_verified(
            repo_root, matrix_train, _MATRIX_GROUP_ZOO_CORE
        )
    matrix_status["matrix_train_verified"] = verified
    matrix_status["matrix_train_verify_reason"] = verify_reason

    items: list[dict[str, Any]] = []
    if unmapped:
        items.append(
            _item(
                "sota_matrix_row_map",
                section="SOTA vs matrix_rows",
                status="warn",
                message=f"inventory run(s) not in zoo_core: {', '.join(unmapped)}",
                unmapped=unmapped,
            )
        )
    else:
        items.append(
            _item(
                "sota_matrix_row_map",
                section="SOTA vs matrix_rows",
                status="pass",
                message="inventory zoo run names map to matrix_rows zoo_core ids",
                mapped=mapped,
            )
        )

    if missing_weights:
        items.append(
            _item(
                "sota_zoo_weights_on_disk",
                section="SOTA vs matrix_rows",
                status="warn",
                message=f"inventory cites runs missing under hsp_zoo: {', '.join(missing_weights)}",
            )
        )
    else:
        items.append(
            _item(
                "sota_zoo_weights_on_disk",
                section="SOTA vs matrix_rows",
                status="pass",
                message="inventory zoo runs have train dirs on disk",
            )
        )

    if not matrix_train.is_file():
        items.append(
            _item(
                "sota_matrix_train_aggregate",
                section="SOTA vs matrix_rows",
                status="warn",
                message="reports/hsp/matrix_train.json missing (P0-5 pending)",
            )
        )
    elif verified:
        items.append(
            _item(
                "sota_matrix_train_aggregate",
                section="SOTA vs matrix_rows",
                status="pass",
                message=verify_reason or "matrix_train.json verified for zoo_core",
            )
        )
    else:
        items.append(
            _item(
                "sota_matrix_train_aggregate",
                section="SOTA vs matrix_rows",
                status="warn",
                message="matrix_train.json present but not verified for zoo_core",
            )
        )

    return {"matrix_status": matrix_status, "items": items}


def check_docx_claim_gaps(
    repo_root: Path,
    docx_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare submitted docx numeric claims to on-disk HSP artifacts."""
    gaps: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    eval_map = _rel(repo_root, "reports/hsp/eval_test_map.json")
    counting = _rel(repo_root, "reports/reviewer2_counting_metrics_computed.json")
    repo_map50: float | None = None
    repo_mae: float | None = None
    repo_mean_rel: float | None = None
    repo_pct_below_2: float | None = None

    if eval_map.is_file():
        repo_map50 = float(_read_json(eval_map).get("mAP50") or 0)
    if counting.is_file():
        cdoc = _read_json(counting)
        repo_mae = _mae_from_counting_computed(cdoc)
        rel = cdoc.get("per_image_relative_error_pct") or {}
        if isinstance(rel, dict):
            mean_val = rel.get("mean")
            if mean_val is not None:
                repo_mean_rel = float(mean_val)
            pct_val = rel.get("pct_below_2")
            if pct_val is not None:
                repo_pct_below_2 = float(pct_val)

    docx_has = {
        "map50_0793": "0.793" in docx_text,
        "map50_079": bool(re.search(r"\b0\.79\b", docx_text)),
        "mean_rel_13_2": "13.2" in docx_text,
        "pct_below_2_80": bool(re.search(r"80\s*%", docx_text)),
        "telegram_96_4": "96.4" in docx_text,
        "yolov8_keyword": bool(re.search(r"YOLOv8", docx_text, re.I)),
        "harchoc": "HARCHOC" in docx_text,
    }

    if docx_has["map50_0793"] and repo_map50 is not None:
        if not _near(repo_map50, 0.793, tol=0.05):
            gaps.append(
                {
                    "claim": "test mAP50 0.793 (docx abstract)",
                    "docx": 0.793,
                    "repo": round(repo_map50, 3),
                    "repo_path": "reports/hsp/eval_test_map.json",
                    "action": "paste reviewer2_abstract_draft.md; see reviewer2_map50_investigation.md",
                }
            )
            items.append(
                _item(
                    "docx_gap_map50",
                    section="Docx vs repo",
                    status="fail",
                    message=f"docx cites mAP50 0.793; repo test mAP50≈{repo_map50:.3f}",
                )
            )
        else:
            items.append(
                _item(
                    "docx_gap_map50",
                    section="Docx vs repo",
                    status="pass",
                    message="docx mAP50 aligns with repo eval",
                )
            )

    if docx_has["mean_rel_13_2"] and repo_mean_rel is not None:
        if not _near(repo_mean_rel, 13.2, tol=0.5):
            gaps.append(
                {
                    "claim": "mean relative counting error 13.2% (docx)",
                    "docx": 13.2,
                    "repo": repo_mean_rel,
                    "repo_path": "reports/reviewer2_counting_metrics_computed.json",
                    "action": "use full test n=109 stats from abstract draft",
                }
            )
    if docx_has["pct_below_2_80"] and repo_pct_below_2 is not None:
        if not _near(repo_pct_below_2, 80.0, tol=2.0):
            gaps.append(
                {
                    "claim": "80% images relative error <2% (docx)",
                    "docx": 80.0,
                    "repo": repo_pct_below_2,
                    "repo_path": "reports/reviewer2_counting_metrics_computed.json",
                    "action": "separate n=50 blinded audit from full test n=109",
                }
            )

    if docx_has["telegram_96_4"]:
        gaps.append(
            {
                "claim": "Telegram 96.4% success / 15–30s latency (docx)",
                "docx": "96.4%",
                "repo": None,
                "repo_path": None,
                "action": "export deploy logs before citing; not in repo JSON",
            }
        )
        items.append(
            _item(
                "docx_gap_telegram",
                section="Docx vs repo",
                status="warn",
                message="docx cites Telegram trial stats; no repo JSON artifact",
            )
        )

    if docx_has["yolov8_keyword"]:
        items.append(
            _item(
                "docx_gap_yolo_only",
                section="Docx vs repo",
                status="warn",
                message="docx keywords still say YOLOv8; SOTA inventory documents broader zoo",
            )
        )

    if repo_mae is not None and "61.3" not in docx_text:
        gaps.append(
            {
                "claim": "test count MAE 61.3 (repo headline)",
                "docx": "not found in docx extract",
                "repo": round(repo_mae, 1),
                "repo_path": "reports/reviewer2_counting_metrics_computed.json",
                "action": "paste revised abstract Results",
            }
        )

    if not items:
        items.append(
            _item(
                "docx_claims",
                section="Docx vs repo",
                status="pass",
                message="no tracked docx/repo metric conflicts",
            )
        )

    return items, gaps


def run_reviewer2_paste_check(
    repo_root: Path | None = None,
    *,
    docx_path: str | Path = _DEFAULT_DOCX,
    checklist_path: str | Path = _DEFAULT_CHECKLIST,
    inventory_md: str | Path = _DEFAULT_INVENTORY_MD,
    inventory_json: str | Path = _DEFAULT_INVENTORY_JSON,
    matrix_rows_path: str | Path = _DEFAULT_MATRIX_ROWS,
    out_path: str | Path = _DEFAULT_OUT,
) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    items: list[dict[str, Any]] = []

    if not _rel(root, str(checklist_path)).is_file():
        raise FileNotFoundError(f"missing checklist: {checklist_path}")

    matrix_doc = load_matrix_rows(root, matrix_rows_path)
    inventory = load_sota_inventory(
        root, inventory_md=inventory_md, inventory_json=inventory_json
    )

    items.extend(check_paste_sources(root))
    items.extend(check_inventory_mae(root))
    matrix_block = check_inventory_vs_matrix(root, inventory, matrix_doc)
    items.extend(matrix_block["items"])

    docx_rel = _rel(root, str(docx_path))
    docx_text = ""
    docx_meta: dict[str, Any] = {"path": str(docx_path), "exists": docx_rel.is_file()}
    gaps: list[dict[str, Any]] = []
    if docx_rel.is_file():
        docx_text = extract_docx_text(docx_rel)
        docx_meta["char_count"] = len(docx_text)
        docx_meta["claims_found"] = {
            "map50_0793": "0.793" in docx_text,
            "telegram_96_4": "96.4" in docx_text,
            "mean_rel_13_2": "13.2" in docx_text,
            "pct_below_2_80": bool(re.search(r"80\s*%", docx_text)),
        }
        doc_items, gaps = check_docx_claim_gaps(root, docx_text)
        items.extend(doc_items)
    else:
        items.append(
            _item(
                "docx_readable",
                section="Manuscript docx",
                status="warn",
                message=f"docx not found: {docx_path}",
            )
        )

    fail_n = sum(1 for it in items if it.get("status") == "fail")
    warn_n = sum(1 for it in items if it.get("status") == "warn")
    overall: Status = "fail" if fail_n else ("warn" if warn_n else "pass")

    report: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "status": overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "checklist": str(checklist_path),
        "docx": docx_meta,
        "docx_gaps": gaps,
        "sota_inventory_source": inventory.get("_source"),
        "sota_inventory_parsed": {
            k: v for k, v in inventory.items() if not str(k).startswith("_")
        },
        "matrix_rows": {
            "path": str(matrix_rows_path),
            "schema_version": matrix_doc.get("schema_version"),
            "row_count": len(matrix_doc.get("rows") or []),
            "zoo_core_ids": matrix_group_row_ids(matrix_doc, _MATRIX_GROUP_ZOO_CORE),
        },
        "matrix_alignment": matrix_block["matrix_status"],
        "items": items,
        "summary": {
            "pass": sum(1 for it in items if it.get("status") == "pass"),
            "warn": warn_n,
            "fail": fail_n,
        },
    }

    out = _rel(root, str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["out"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Reviewer 2 docx paste readiness + SOTA/matrix alignment (no GPU)."
    )
    parser.add_argument("--docx", default=_DEFAULT_DOCX)
    parser.add_argument("--checklist", default=_DEFAULT_CHECKLIST)
    parser.add_argument("--inventory-md", default=_DEFAULT_INVENTORY_MD)
    parser.add_argument("--inventory-json", default=_DEFAULT_INVENTORY_JSON)
    parser.add_argument("--matrix-rows", default=_DEFAULT_MATRIX_ROWS)
    parser.add_argument("--out", default=_DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        report = run_reviewer2_paste_check(
            docx_path=args.docx,
            checklist_path=args.checklist,
            inventory_md=args.inventory_md,
            inventory_json=args.inventory_json,
            matrix_rows_path=args.matrix_rows,
            out_path=args.out,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 1
    summary = report["summary"]
    print(
        f"reviewer2 paste check: {report['status']} "
        f"(pass={summary['pass']} warn={summary['warn']} fail={summary['fail']})"
    )
    print(f"Wrote {report['out']}")
    if report.get("docx_gaps"):
        print(f"docx gaps: {len(report['docx_gaps'])}")
    return 1 if summary.get("fail") else 0
