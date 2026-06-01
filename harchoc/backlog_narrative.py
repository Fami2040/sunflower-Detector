"""Parse backlog.md into structured manuscript narrative (no LLM)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harchoc.ml_env import mamba_run_shell_command

BACKLOG_NARRATIVE_SCHEMA = "backlog_narrative.v1"
_ID_TOKEN = r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*"
_ID_RE = re.compile(rf"\b({_ID_TOKEN})\b")
_BOLD_ID_RE = re.compile(rf"\*\*({_ID_TOKEN})\*\*")
_ID_FULL_RE = re.compile(rf"^{_ID_TOKEN}$")
_STATUS_WORDS = frozenset({"done", "next", "partial", "blocked", "ready", "closed", "defer"})


def _strip_md_bold(cell: str) -> str:
    return _BOLD_ID_RE.sub(r"\1", cell.strip()).strip()


def _id_from_cell(cell: str) -> str | None:
    m = _BOLD_ID_RE.search(cell)
    if m:
        return m.group(1)
    plain = _strip_md_bold(cell)
    if _ID_FULL_RE.match(plain):
        return plain
    return None


def _normalize_status(raw: str) -> str:
    t = _strip_md_bold(raw).strip()
    for w in _STATUS_WORDS:
        if w in t.lower():
            return w.capitalize() if w != "defer" else "Defer"
    return t or "Unknown"


def _parse_pipe_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            break
        if re.match(r"^\|[\s\-:|]+\|$", s):
            continue
        parts = [c.strip() for c in s.strip("|").split("|")]
        if parts:
            rows.append(parts)
    return rows


def _section_slices(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.+)$", line)
        if m:
            if current is not None:
                out[current] = "\n".join(buf)
            level = len(m.group(1))
            title = m.group(2).strip()
            key = title.lower()
            if level <= 2:
                current = key
                buf = []
            elif current is not None:
                buf.append(line)
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf)
    return out


def _subsection_label(line: str) -> str | None:
    m = re.match(r"^###\s+(.+)$", line.strip())
    return m.group(1).strip().lower() if m else None


def _queue_rows_from_section(body: str, *, default_section: str) -> list[dict[str, Any]]:
    lines = body.splitlines()
    items: list[dict[str, Any]] = []
    section = default_section
    i = 0
    while i < len(lines):
        line = lines[i]
        sub = _subsection_label(line)
        if sub:
            section = sub
            i += 1
            continue
        if line.strip().startswith("| ID"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            parsed = _parse_pipe_table(table_lines)
            for row in parsed[1:]:
                if len(row) < 5:
                    continue
                item_id = _id_from_cell(row[0])
                if not item_id:
                    continue
                items.append(
                    {
                        "id": item_id,
                        "priority": _strip_md_bold(row[1]),
                        "status": _normalize_status(row[2]),
                        "blocker": _strip_md_bold(row[3]) if row[3] not in ("—", "-", "") else None,
                        "next": _strip_md_bold(row[4]),
                        "section": section,
                    }
                )
            continue
        i += 1
    return items


def _model_stack_rows(body: str) -> list[dict[str, Any]]:
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("| Step"):
            parsed = _parse_pipe_table(lines[i : i + 20])
            out: list[dict[str, Any]] = []
            for row in parsed[1:]:
                if len(row) < 3:
                    continue
                out.append(
                    {
                        "step": _strip_md_bold(row[0]),
                        "focus": _strip_md_bold(row[1]),
                        "status": _normalize_status(row[2]),
                    }
                )
            return out
    return []


def _anchor_metrics(lines: list[str]) -> dict[str, str]:
    for i, line in enumerate(lines):
        if line.strip() == "| Anchor | Value |":
            parsed = _parse_pipe_table(lines[i : i + 12])
            return {
                r[0].replace("**", "").strip(): r[1].replace("**", "").strip()
                for r in parsed[1:]
                if len(r) >= 2 and r[0]
            }
    return {}


def _collect_archive_ids(archive_body: str) -> list[str]:
    found: set[str] = set()
    for m in _ID_RE.finditer(archive_body):
        tok = m.group(1)
        if tok.startswith(("HTTP", "YAML")):
            continue
        found.add(tok)
    return sorted(found)


def _refs_in_text(text: str) -> list[str]:
    if not text or text.strip() in ("—", "-", ""):
        return []
    found: set[str] = set()
    for m in _BOLD_ID_RE.finditer(text):
        found.add(m.group(1))
    plain = _strip_md_bold(text).strip()
    if _ID_FULL_RE.match(plain):
        found.add(plain)
    elif len(plain) <= 48:
        for m in _ID_RE.finditer(plain):
            found.add(m.group(1))
    return sorted(found)


def _header_meta(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines[:20]:
        if line.startswith("**Updated:**"):
            meta["updated"] = line.split("**Updated:**", 1)[-1].strip().strip("·").strip()
        if "**Branch:**" in line:
            m = re.search(r"\*\*Branch:\*\*\s*`([^`]+)`", line)
            if m:
                meta["branch"] = m.group(1)
    return meta


def experiment_subcommands(experiment_py: Path) -> set[str]:
    text = experiment_py.read_text(encoding="utf-8")
    return set(re.findall(r'add_parser\(\s*["\']([^"\']+)["\']', text))


def repro_commands_for_subcommands(subcmds: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    base = mamba_run_shell_command("scripts/experiment.py")
    ci = "PYTHONPATH=. HARCHOC_ALLOW_BASE_PYTHON=1 python scripts/experiment.py"

    if "reviewer2-repro" in subcmds:
        out["reviewer2_repro"] = [
            f"{base} reviewer2-repro",
            f"{ci} reviewer2-repro --dry-run",
            f"{base} repro --stage post-zoo",
        ]
    if "repro" in subcmds:
        out["manuscript_hsp"] = [f"{base} repro", f"{ci} repro --dry-run"]
        out["manuscript_full"] = [
            f"{base} repro --stage full",
            f"{ci} repro --stage full --dry-run",
        ]
        out["manuscript_preflight"] = [
            f"{base} repro --stage preflight",
            f"{ci} repro --stage preflight --dry-run",
        ]
    if "manuscript-preflight" in subcmds:
        out["manuscript_preflight"] = [
            f"{base} manuscript-preflight",
            f"{ci} manuscript-preflight --dry-run",
            f"{base} repro --stage preflight",
        ]
    if "aug-compare" in subcmds:
        out["aug_compare"] = [f"{base} aug-compare", f"{ci} aug-compare --dry-run"]
    if "backlog-narrative" in subcmds:
        out["backlog_narrative"] = [
            f"{base} backlog-narrative",
            f"{ci} backlog-narrative --dry-run",
        ]
    if "figures-repro" in subcmds:
        out["figures_repro"] = [
            f"{base} figures-repro",
            f"{ci} figures-repro --dry-run",
        ]
    if "tables-repro" in subcmds:
        out["tables_repro"] = [
            f"{base} tables-repro",
            f"{ci} tables-repro --dry-run",
        ]
    elif "dual-metric" in subcmds and "tables-repro" not in subcmds:
        out["tables_dual_metric"] = [f"{base} dual-metric --dry-run"]
    return out


def parse_backlog_md(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    sections = _section_slices(text)

    active: list[dict[str, Any]] = []
    now_body = sections.get("now", "")
    active.extend(_queue_rows_from_section(now_body, default_section="now"))

    archive_body = sections.get("archive", "")
    archive_ids = _collect_archive_ids(archive_body)
    archive_set = set(archive_ids)

    for item in active:
        blockers = _refs_in_text(item.get("blocker") or "")
        next_refs = _refs_in_text(item.get("next") or "")
        item["blocker_ids"] = blockers
        item["next_ids"] = next_refs
        item["blocker_in_archive"] = [b for b in blockers if b in archive_set]
        item["blocker_open"] = [b for b in blockers if b not in archive_set]
        item["next_done_in_archive"] = [n for n in next_refs if n in archive_set]
        item["next_still_open"] = [n for n in next_refs if n not in archive_set and n != item["id"]]

    by_status: dict[str, list[str]] = {}
    for item in active:
        by_status.setdefault(item["status"], []).append(item["id"])

    return {
        "source": str(p),
        "header": _header_meta(lines),
        "anchor": _anchor_metrics(lines),
        "model_stack": _model_stack_rows(sections.get("model stack (reference)", "")),
        "active_queue": active,
        "archive_ids": archive_ids,
        "status_index": by_status,
        "gap_doc": "docs/manuscript/reviewer_comments_backlog_gap.md",
        "reviewer2_gaps": "reports/reviewer2_programmatic_gaps.md",
    }


def build_backlog_narrative_payload(
    parsed: dict[str, Any],
    *,
    repo_root: Path,
    experiment_py: Path | None = None,
) -> dict[str, Any]:
    exp = experiment_py or (repo_root / "scripts" / "experiment.py")
    subcmds = experiment_subcommands(exp)
    repro = repro_commands_for_subcommands(subcmds)
    missing_subcmds = [
        s
        for s in ("manuscript-preflight", "figures-repro", "tables-repro", "aug-compare")
        if s not in subcmds
    ]

    done_archive = [i for i in parsed["archive_ids"] if i.startswith("MS-") or i.startswith("P")]
    open_next = parsed["status_index"].get("Next", []) + parsed["status_index"].get("Blocked", [])

    return {
        "schema_version": BACKLOG_NARRATIVE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": parsed["header"],
        "anchor": parsed["anchor"],
        "model_stack": parsed["model_stack"],
        "active_queue": parsed["active_queue"],
        "archive_id_count": len(parsed["archive_ids"]),
        "archive_sample_ids": parsed["archive_ids"][:40],
        "cross_links": {
            "open_ids": open_next,
            "archive_ms_and_p_count": len(done_archive),
            "items_with_blocker_in_archive": [
                i["id"] for i in parsed["active_queue"] if i.get("blocker_in_archive")
            ],
        },
        "repro_commands": repro,
        "repro_subcommands_missing": missing_subcmds,
        "sources": {
            "backlog": parsed["source"],
            "gap_map": parsed["gap_doc"],
            "reviewer2_gaps": parsed["reviewer2_gaps"],
        },
    }


def render_narrative_md(payload: dict[str, Any]) -> str:
    anchor = payload.get("anchor") or {}
    active = payload.get("active_queue") or []
    stack = payload.get("model_stack") or []
    repro = payload.get("repro_commands") or {}
    cross = payload.get("cross_links") or {}

    lines: list[str] = [
        "# Manuscript narrative (from backlog)",
        "",
        f"*Generated from backlog · {payload.get('generated_at', '')[:10]}*",
        "",
        "## Methods status",
        "",
    ]

    if anchor:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in list(anchor.items())[:6]:
            short = v.replace("\n", " ")[:120]
            lines.append(f"| {k} | {short} |")
        lines.append("")

    if stack:
        lines.append("**Model stack (reference):**")
        lines.append("")
        lines.append("| Step | Focus | Status |")
        lines.append("|------|-------|--------|")
        for row in stack:
            lines.append(f"| {row['step']} | {row['focus'][:60]} | {row['status']} |")
        lines.append("")

    open_items = [i for i in active if i["status"] in ("Next", "Partial", "Blocked", "Ready")]
    if open_items:
        lines.append("**Active queue (Now):**")
        lines.append("")
        lines.append("| ID | Status | Blocker → archive? | Next (open refs) |")
        lines.append("|----|--------|-------------------|------------------|")
        for item in open_items[:18]:
            b_arch = ", ".join(item.get("blocker_in_archive") or []) or "—"
            n_open = ", ".join(item.get("next_still_open") or []) or "—"
            lines.append(
                f"| {item['id']} | {item['status']} | {b_arch} | {n_open} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Results available",
            "",
            "- Headline test count MAE and locked conf: see anchor **best2** and "
            "`reports/hsp/dual_metric.json`, `reports/hsp/p0_summary.md`.",
            "- Error / TIDE / confusion: `reports/hsp/error_test_report.json`, "
            "`eval.py --confusion-matrix-only`.",
            "- Domain trays: `reports/domains/domain_eval.json`.",
            "- Aug closed: `reports/aug_smoke/leaderboard.md` (production `robustness_minimal`).",
            "- Manuscript drafts (repo): `docs/manuscript/reviewer_comments_backlog_gap.md` "
            "(MS-* Done sections).",
            "",
            "## Limitations / open",
            "",
        ]
    )

    blocked = [i for i in active if i["status"] == "Blocked"]
    partial = [i for i in active if i["status"] == "Partial"]
    if blocked or partial:
        for label, subset in (("Blocked", blocked), ("Partial", partial)):
            if subset:
                lines.append(f"- **{label}:** " + ", ".join(i["id"] for i in subset))
    lines.append(
        f"- **Open Next/Blocked IDs:** {', '.join(cross.get('open_ids') or []) or 'none'}"
    )
    lines.append(
        f"- Archive holds **{payload.get('archive_id_count', 0)}** ticket tokens; "
        f"cross-link sample: {', '.join((cross.get('items_with_blocker_in_archive') or [])[:8]) or 'n/a'}."
    )
    missing = payload.get("repro_subcommands_missing") or []
    if missing:
        lines.append(f"- Subcommands not in `experiment.py`: {', '.join(missing)}.")
    lines.append("")

    lines.extend(["## Repro commands", ""])
    for key, cmds in repro.items():
        lines.append(f"**{key.replace('_', ' ')}:**")
        for c in cmds:
            lines.append(f"```bash\n{c}\n```")
        lines.append("")

    lines.append(
        "**Gap index:** "
        f"[{payload['sources']['reviewer2_gaps']}]({payload['sources']['reviewer2_gaps']}) · "
        f"[{payload['sources']['gap_map']}]({payload['sources']['gap_map']})"
    )
    lines.append("")

    md = "\n".join(lines)
    if md.count("\n") >= 200:
        md = "\n".join(md.splitlines()[:199]) + "\n\n*(truncated to 200 lines)*\n"
    return md


def write_backlog_narrative(
    repo_root: str | Path,
    *,
    backlog_path: str | Path = "backlog.md",
    out_md: str | Path = "reports/manuscript/narrative_from_backlog.md",
    out_json: str | Path = "reports/manuscript/backlog_narrative.json",
) -> dict[str, Any]:
    rr = Path(repo_root).expanduser().resolve()
    parsed = parse_backlog_md(rr / backlog_path)
    payload = build_backlog_narrative_payload(parsed, repo_root=rr)
    md_path = (rr / out_md).resolve()
    json_path = (rr / out_json).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_narrative_md(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"md": str(md_path), "json": str(json_path), "payload": payload}


def run_backlog_narrative(repo_root: str | Path, fields: dict[str, Any]) -> int:
    if bool(fields.get("dry_run")):
        from harchoc.manuscript_repro import _format_cmd

        print("# backlog-narrative")
        print(_format_cmd(["scripts/experiment.py", "backlog-narrative"], mamba=False))
        return 0

    result = write_backlog_narrative(
        repo_root,
        backlog_path=str(fields.get("backlog") or "backlog.md"),
        out_md=str(fields.get("out_md") or "reports/manuscript/narrative_from_backlog.md"),
        out_json=str(fields.get("out_json") or "reports/manuscript/backlog_narrative.json"),
    )
    print(f"Wrote {result['md']}")
    print(f"Wrote {result['json']}")
    n_open = len(result["payload"]["cross_links"].get("open_ids") or [])
    print(f"backlog-narrative: {n_open} open Next/Blocked IDs indexed")
    return 0
