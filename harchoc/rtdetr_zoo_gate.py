"""15-ep RT-DETR smoke gate before zoo_core 100-ep Ultralytics DETR matrix rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harchoc.aug_smoke_runner import extract_count_mae

# zoo_core Ultralytics RT-DETR rows (100 ep) require verified 15-ep queue smokes first.
ZOO_CORE_RTDETR_15EP_GATES: tuple[dict[str, str], ...] = (
    {
        "matrix_row_id": "rtdetr_l_nq1024",
        "smoke_job_id": "rtdetr_queries_smoke",
        "summary": "reports/gpu_queue/summaries/rtdetr_queries_smoke.json",
        "eval_error_json": "reports/gpu_queue/eval/rtdetr_queries_smoke_15ep_error.json",
        "notes": "P1-RTDETR-Q: nq=1024 15-ep smoke before 100-ep zoo_core train.",
    },
    {
        "matrix_row_id": "rtdetr_x",
        "smoke_job_id": "rtdetr_imgsz1280",
        "summary": "reports/gpu_queue/summaries/rtdetr_imgsz1280.json",
        "eval_error_json": "reports/gpu_queue/eval/rtdetr_imgsz1280_smoke_15ep_error.json",
        "notes": "RT-DETR-L @ 1280 (300 queries) validates train/eval path before rtdetr-x 100 ep.",
    },
)


def _summary_has_count_mae(obj: dict[str, Any], repo_root: Path) -> bool:
    mae = obj.get("test_count_mae")
    if mae is not None:
        return True
    err = (obj.get("test_eval") or {}).get("error_json")
    if err:
        path = Path(err)
        if not path.is_absolute():
            path = repo_root / err
        mae_val, _ = extract_count_mae(path)
        return mae_val is not None
    return False


def _eval_error_has_count_mae(repo_root: Path, rel_path: str) -> bool:
    p = repo_root / rel_path
    if not p.is_file():
        return False
    mae, _ = extract_count_mae(p)
    return mae is not None


def rtdetr_15ep_gate_status(
    *,
    repo_root: Path,
    gate: dict[str, str],
) -> dict[str, Any]:
    """Return per-gate pass/fail with paths (no GPU)."""
    summary_rel = str(gate["summary"])
    err_rel = str(gate.get("eval_error_json") or "")
    sp = repo_root / summary_rel
    ok = False
    detail = "summary missing"
    if sp.is_file():
        try:
            obj = json.loads(sp.read_text(encoding="utf-8"))
            if obj.get("status") == "complete" and _summary_has_count_mae(obj, repo_root):
                if err_rel and not _eval_error_has_count_mae(repo_root, err_rel):
                    detail = f"eval_error_json missing count MAE: {err_rel}"
                else:
                    ok = True
                    detail = "verified"
            elif obj.get("status") == "complete":
                detail = "summary complete but missing test_count_mae (stale transcript?)"
            else:
                detail = f"summary status={obj.get('status')!r}"
        except Exception as ex:
            detail = f"summary unreadable: {ex}"
    return {
        "matrix_row_id": gate["matrix_row_id"],
        "smoke_job_id": gate["smoke_job_id"],
        "summary": summary_rel,
        "eval_error_json": err_rel or None,
        "passed": ok,
        "detail": detail,
        "notes": gate.get("notes"),
    }


def check_zoo_core_rtdetr_15ep_gates(*, repo_root: Path) -> list[dict[str, Any]]:
    return [rtdetr_15ep_gate_status(repo_root=repo_root, gate=g) for g in ZOO_CORE_RTDETR_15EP_GATES]


def zoo_core_rtdetr_gates_passed(*, repo_root: Path) -> bool:
    return all(s["passed"] for s in check_zoo_core_rtdetr_15ep_gates(repo_root=repo_root))


def bench_matrix_row_id(*, bench_path: Path) -> str:
    """Map ``configs/bench/rtdetr_l_nq1024.yaml`` → ``rtdetr_l_nq1024``."""
    stem = bench_path.stem
    if stem.endswith("_default"):
        return stem[: -len("_default")]
    return stem


def zoo_core_rtdetr_gate_skip_reason(
    *,
    repo_root: Path,
    bench_path: Path,
    groups: tuple[str, ...] | list[str],
    model: str | None,
) -> str | None:
    """Non-None when a zoo_core Ultralytics RT-DETR row must not train yet."""
    if "zoo_core" not in groups:
        return None
    from harchoc.rtdetr_limits import is_rtdetr_model

    if not is_rtdetr_model(model):
        return None
    row_id = bench_matrix_row_id(bench_path=bench_path)
    for gate in ZOO_CORE_RTDETR_15EP_GATES:
        if gate["matrix_row_id"] != row_id:
            continue
        st = rtdetr_15ep_gate_status(repo_root=repo_root, gate=gate)
        if not st["passed"]:
            return str(st["detail"])
    return None


def format_zoo_core_rtdetr_gate_blockers(repo_root: Path) -> str:
    statuses = check_zoo_core_rtdetr_15ep_gates(repo_root=repo_root)
    lines = [
        "zoo_core RT-DETR 100-ep rows require verified 15-ep GPU queue smokes (test count MAE):",
    ]
    for s in statuses:
        mark = "ok" if s["passed"] else "BLOCKED"
        lines.append(
            f"  [{mark}] {s['matrix_row_id']}: smoke {s['smoke_job_id']} — {s['detail']} ({s['summary']})"
        )
    lines.append(
        "Run gpu_queue_full jobs rtdetr_queries_smoke / rtdetr_imgsz1280 first (P1-RTDETR-COUNT-REFRESH)."
    )
    return "\n".join(lines)
