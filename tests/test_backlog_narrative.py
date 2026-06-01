from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harchoc.backlog_narrative import (
    BACKLOG_NARRATIVE_SCHEMA,
    build_backlog_narrative_payload,
    parse_backlog_md,
    render_narrative_md,
    write_backlog_narrative,
)

FIXTURE_BACKLOG = """\
# Backlog (fixture)

| Anchor | Value |
|--------|--------|
| **best2** | **61.3** MAE |

## Model stack (reference)

| Step | Focus | Status |
|------|--------|--------|
| 1 | Train parity | Done |
| 5 | Model zoo | **Next** |

## Now

### Science · training · eval

| ID | Pri | Status | Blocker | Next |
|----|-----|--------|---------|------|
| **P0-5** | P0 | **Next** | — | `matrix_train.json` |
| **P1-ZOO-PROV** | P1 | Partial | P0-5 | provenance rows |

### Manuscript (repo draft vs LaTeX)

| ID | Pri | Status | Blocker | Next |
|----|-----|--------|---------|------|
| **MS-SOTA** | P1 | Blocked | P0-5 | SOTA table |
| **MS-GEN** | P1 | Partial | DATA-ACQ-GEN | tray numbers |

## Archive

| ID | Evidence |
|----|----------|
| P0-0–P0-4 | split drift |
| MS-REPRO | experiment repro |
"""


class BacklogNarrativeTests(unittest.TestCase):
    def test_parse_fixture_tables(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "backlog.md"
            path.write_text(FIXTURE_BACKLOG, encoding="utf-8")
            parsed = parse_backlog_md(path)
        self.assertEqual(parsed["anchor"].get("best2"), "61.3 MAE")
        self.assertEqual(len(parsed["model_stack"]), 2)
        self.assertEqual(parsed["model_stack"][1]["status"], "Next")
        ids = {i["id"] for i in parsed["active_queue"]}
        self.assertEqual(ids, {"P0-5", "P1-ZOO-PROV", "MS-SOTA", "MS-GEN"})
        p05 = next(i for i in parsed["active_queue"] if i["id"] == "P0-5")
        self.assertEqual(p05["status"], "Next")
        prov = next(i for i in parsed["active_queue"] if i["id"] == "P1-ZOO-PROV")
        self.assertEqual(prov["blocker_ids"], ["P0-5"])
        self.assertEqual(prov["blocker_open"], ["P0-5"])
        self.assertIn("MS-REPRO", parsed["archive_ids"])

    def test_build_and_render_under_200_lines(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        parsed = parse_backlog_md(repo / "backlog.md")
        payload = build_backlog_narrative_payload(parsed, repo_root=repo)
        self.assertEqual(payload["schema_version"], BACKLOG_NARRATIVE_SCHEMA)
        self.assertIn("reviewer2_repro", payload["repro_commands"])
        repro = payload["repro_commands"]
        self.assertIn("reviewer2_repro", repro)
        if "figures-repro" in {
            s for s in repro
        } or "figures_repro" in repro:
            pass
        else:
            self.assertTrue(
                "figures_repro" in repro or "figures_gradcam" in repro,
                msg=f"expected figure repro entry, got {list(repro)}",
            )
        md = render_narrative_md(payload)
        self.assertLess(md.count("\n"), 200)
        self.assertIn("## Methods status", md)

    def test_write_backlog_narrative(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out_md = Path(td) / "narrative.md"
            out_json = Path(td) / "narrative.json"
            result = write_backlog_narrative(
                repo,
                backlog_path=repo / "backlog.md",
                out_md=out_md,
                out_json=out_json,
            )
            self.assertTrue(out_md.is_file())
            obj = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(obj["schema_version"], BACKLOG_NARRATIVE_SCHEMA)
            self.assertEqual(result["md"], str(out_md))


if __name__ == "__main__":
    unittest.main()
