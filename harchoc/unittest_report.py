"""Single-shot unittest discover runner with summary + failure index."""

from __future__ import annotations

import os
import sys
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class UnittestProblem:
    outcome: str  # FAIL | ERROR
    test_id: str
    message: str
    traceback: str


@dataclass
class UnittestReport:
    tests_run: int
    failures: int
    errors: int
    skipped: int
    successful: int
    was_successful: bool
    elapsed_s: float
    problems: list[UnittestProblem] = field(default_factory=list)
    other_lines: list[str] = field(default_factory=list)  # e.g. OK: checked=3


def discover_suite(
    *,
    repo_root: Path,
    start_dir: str = "tests",
    pattern: str = "test_*.py",
) -> unittest.TestSuite:
    top = str(repo_root.resolve())
    if top not in sys.path:
        sys.path.insert(0, top)
    loader = unittest.TestLoader()
    return loader.discover(start_dir=start_dir, pattern=pattern, top_level_dir=top)


class _TeeStdout:
    """Pass writes through; collect ``OK: …`` lines from tests for the summary block."""

    def __init__(self, underlying: TextIO, bucket: list[str]) -> None:
        self._underlying = underlying
        self._bucket = bucket
        self._pending = ""

    def write(self, s: str) -> int:
        self._underlying.write(s)
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            stripped = line.strip()
            if stripped.startswith("OK:"):
                self._bucket.append(stripped)
        return len(s)

    def flush(self) -> None:
        self._underlying.flush()
        if self._pending.strip().startswith("OK:"):
            self._bucket.append(self._pending.strip())
        self._pending = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


def run_unittest_discover(
    *,
    repo_root: Path,
    start_dir: str = "tests",
    pattern: str = "test_*.py",
    verbosity: int = 1,
    stream: TextIO | None = None,
    failfast: bool = False,
) -> UnittestReport:
    """Run unittest discover; collect failures/errors for one-shot reporting."""
    suite = discover_suite(repo_root=repo_root, start_dir=start_dir, pattern=pattern)
    out = stream or sys.stderr
    other_lines: list[str] = []
    stdout_prev = sys.stdout
    sys.stdout = _TeeStdout(stdout_prev, other_lines)  # type: ignore[assignment]
    try:
        runner = unittest.TextTestRunner(
            stream=out,
            verbosity=verbosity,
            failfast=failfast,
            buffer=True,
        )
        t0 = time.perf_counter()
        result = runner.run(suite)
        elapsed = time.perf_counter() - t0
    finally:
        sys.stdout.flush()
        sys.stdout = stdout_prev

    problems: list[UnittestProblem] = []
    for test, tb in result.failures:
        problems.append(
            UnittestProblem(
                outcome="FAIL",
                test_id=test.id(),
                message=_first_line(tb),
                traceback=tb,
            )
        )
    for test, tb in result.errors:
        problems.append(
            UnittestProblem(
                outcome="ERROR",
                test_id=test.id(),
                message=_first_line(tb),
                traceback=tb,
            )
        )

    return UnittestReport(
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        successful=result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        was_successful=result.wasSuccessful(),
        elapsed_s=elapsed,
        problems=problems,
        other_lines=other_lines,
    )


def _first_line(traceback: str) -> str:
    for line in traceback.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return "(no message)"


def format_unittest_report(
    report: UnittestReport,
    *,
    show_tracebacks: bool = False,
    include_other_lines: bool = True,
) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("TEST SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Ran {report.tests_run} tests in {report.elapsed_s:.3f}s")
    if report.was_successful:
        tail = f"OK"
        if report.skipped:
            tail += f" (skipped={report.skipped})"
        lines.append(tail)
    else:
        lines.append(
            f"FAILED (failures={report.failures}, errors={report.errors}, skipped={report.skipped})"
        )

    if include_other_lines and report.other_lines:
        lines.append("")
        lines.extend(report.other_lines)

    if report.problems:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"PROBLEMS ({len(report.problems)})")
        lines.append("=" * 70)
        for p in report.problems:
            lines.append(f"{p.outcome}: {p.test_id}")
            if p.message:
                lines.append(f"  → {p.message}")
            if show_tracebacks and p.traceback:
                lines.append(p.traceback.rstrip())
                lines.append("-" * 70)

    return "\n".join(lines) + "\n"


def ci_test_env(repo_root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(repo_root.resolve()),
        "HARCHOC_ALLOW_BASE_PYTHON": "1",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run unittest discover once; print summary + all FAIL/ERROR ids."
    )
    parser.add_argument("-s", "--start-dir", default="tests", help="Discover start dir")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Test file pattern")
    parser.add_argument("-q", "--quiet", action="store_true", help="Progress: failures only (verbosity 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include full tracebacks in report block")
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    for key, val in ci_test_env(repo_root).items():
        os.environ[key] = val

    verbosity = 0 if args.quiet else 1
    report = run_unittest_discover(
        repo_root=repo_root,
        start_dir=args.start_dir,
        pattern=args.pattern,
        verbosity=verbosity,
        failfast=args.failfast,
    )
    # Re-print captured import-time stdout lines that are not part of unittest (e.g. OK: checked=)
    # by running discover didn't capture them — they appear inline during the run on stderr/stdout.
    block = format_unittest_report(report, show_tracebacks=args.verbose)
    sys.stdout.write(block)
    sys.stdout.flush()
    return 0 if report.was_successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
