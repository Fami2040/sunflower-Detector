from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import sys; from pathlib import Path; _r = Path(__file__).resolve().parent.parent; (str(_r) not in sys.path) and sys.path.insert(0, str(_r)); from harchoc.script_entry import bootstrap_repo_imports; bootstrap_repo_imports()

def _parse_scalar(raw: str) -> str | int | float | bool | None:
    s = raw.strip()
    if not s:
        return ""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none"):
        return None
    if s and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _is_mapping_value(v: object) -> bool:
    return isinstance(v, dict)


@dataclass
class ParseReport:
    dropped_lines_total: int = 0
    dropped_by_reason: dict[str, int] | None = None
    dropped_samples: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.dropped_by_reason is None:
            self.dropped_by_reason = {}
        if self.dropped_samples is None:
            self.dropped_samples = []

    def drop(self, *, reason: str, path: Path, line_no: int, line: str) -> None:
        self.dropped_lines_total += 1
        assert self.dropped_by_reason is not None
        self.dropped_by_reason[reason] = int(self.dropped_by_reason.get(reason, 0)) + 1
        assert self.dropped_samples is not None
        if len(self.dropped_samples) < 25:
            self.dropped_samples.append(
                {"path": str(path), "line_no": int(line_no), "reason": reason, "line": line}
            )


def parse_minimal_yaml_with_report(path: Path) -> tuple[dict[str, object], ParseReport]:
    """
    Minimal YAML subset parser (no PyYAML dependency), with a best-effort report.

    Supported:
    - top-level "key: value"
    - top-level "key:" starting a one-level nested mapping
    - one-level nested mappings (2-space or tab indented)
    - block scalars: "key: >" or "key: |" with indented lines

    Everything else is ignored and recorded as "dropped" lines.
    """
    lines = path.read_text("utf-8").splitlines()
    out: dict[str, object] = {}
    current_map_key: str | None = None
    rep = ParseReport()

    i = 0
    while i < len(lines):
        line = lines[i]
        line_no = i + 1

        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue

        if line.lstrip().startswith("-"):
            rep.drop(reason="sequence_item", path=path, line_no=line_no, line=line)
            i += 1
            continue

        if line.startswith(("  ", "\t")):
            if current_map_key is None:
                rep.drop(reason="indented_without_parent", path=path, line_no=line_no, line=line)
                i += 1
                continue
            inner = line.lstrip()
            if ":" not in inner:
                rep.drop(reason="indented_no_colon", path=path, line_no=line_no, line=line)
                i += 1
                continue
            k, rest = inner.split(":", 1)
            k = k.strip()
            rest = rest.strip()
            if not k:
                rep.drop(reason="empty_key", path=path, line_no=line_no, line=line)
                i += 1
                continue
            m = out.get(current_map_key)
            if not _is_mapping_value(m):
                m = {}
                out[current_map_key] = m
            assert isinstance(m, dict)
            m[k] = _parse_scalar(rest) if rest else ""
            i += 1
            continue

        if ":" not in line:
            rep.drop(reason="top_level_no_colon", path=path, line_no=line_no, line=line)
            i += 1
            continue

        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        current_map_key = None

        if not key:
            rep.drop(reason="empty_key", path=path, line_no=line_no, line=line)
            i += 1
            continue

        if rest in (">", "|"):
            i += 1
            block: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith(("  ", "\t")):
                    block.append(nxt.lstrip())
                    i += 1
                elif not nxt.strip():
                    block.append("")
                    i += 1
                else:
                    break
            out[key] = (
                " ".join([b for b in block if b != ""]).strip()
                if rest == ">"
                else "\n".join(block).rstrip()
            )
            continue

        if rest == "":
            out[key] = {}
            current_map_key = key
            i += 1
            continue

        out[key] = _parse_scalar(rest)
        i += 1

    return out, rep


def _iter_yaml_files(dirs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        if d.is_file() and d.suffix.lower() in (".yaml", ".yml"):
            out.append(d.resolve())
            continue
        if d.is_dir():
            out.extend(sorted([p.resolve() for p in d.glob("*.y*ml") if p.is_file()]))
    # de-dupe, keep stable ordering
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def migrate_one(*, src: Path, out_dir: Path) -> tuple[Path, ParseReport, dict[str, Any]]:
    obj, rep = parse_minimal_yaml_with_report(src)
    payload: dict[str, Any] = {"source": str(src), "config": obj}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{src.stem}.json").resolve()
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")

    file_report: dict[str, Any] = {
        "source": str(src),
        "output": str(out_path),
        "dropped_lines_total": rep.dropped_lines_total,
        "dropped_by_reason": rep.dropped_by_reason,
        "dropped_samples": rep.dropped_samples,
    }
    return out_path, rep, file_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Migrate legacy YAML configs to JSON experiments.")
    p.add_argument(
        "--input-dir",
        action="append",
        default=["configs/bench"],
        help="Config directory (or file) containing *.yaml/*.yml. May be repeated.",
    )
    p.add_argument(
        "--output-dir",
        default="configs/experiments/legacy_migrated",
        help="Output directory for migrated JSON configs.",
    )
    p.add_argument(
        "--report",
        default="configs/experiments/legacy_migrated/migration_report.json",
        help="Where to write the migration report JSON.",
    )
    ns = p.parse_args(argv)

    input_dirs = [Path(x).expanduser() for x in ns.input_dir]
    out_dir = Path(ns.output_dir).expanduser()
    report_path = Path(ns.report).expanduser()

    files = _iter_yaml_files(input_dirs)
    report: dict[str, Any] = {
        "inputs": [str(p) for p in input_dirs],
        "output_dir": str(out_dir),
        "files_total": len(files),
        "files_migrated": 0,
        "errors": 0,
        "dropped_lines_total": 0,
        "dropped_by_reason": {},
        "files": [],
    }

    for src in files:
        try:
            _, rep, file_rep = migrate_one(src=src, out_dir=out_dir)
        except Exception as e:
            report["errors"] = int(report["errors"]) + 1
            report.setdefault("error_details", []).append({"source": str(src), "error": repr(e)})
            continue

        report["files_migrated"] = int(report["files_migrated"]) + 1
        report["files"].append(file_rep)
        report["dropped_lines_total"] = int(report["dropped_lines_total"]) + int(rep.dropped_lines_total)

        for k, v in (rep.dropped_by_reason or {}).items():
            dbr = report["dropped_by_reason"]
            dbr[k] = int(dbr.get(k, 0)) + int(v)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return 0 if int(report["errors"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

