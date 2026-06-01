from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path


class Reviewer2PasteCheckTests(unittest.TestCase):
    def test_extract_docx_text_minimal(self) -> None:
        from harchoc.reviewer2_paste_check import extract_docx_text

        with tempfile.TemporaryDirectory() as td:
            docx = Path(td) / "tiny.docx"
            # Minimal valid docx zip with one paragraph.
            inner = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>Hello mAP 0.793</w:t></w:r></w:p></w:body></w:document>"
            )
            with zipfile.ZipFile(docx, "w") as zf:
                zf.writestr("word/document.xml", inner)
            text = extract_docx_text(docx)
            self.assertIn("0.793", text)

    def test_run_reviewer2_paste_check_writes_json(self) -> None:
        from harchoc.reviewer2_paste_check import run_reviewer2_paste_check

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "paste_check.json"
            report = run_reviewer2_paste_check(repo, out_path=out)
            self.assertEqual(report["schema_version"], "reviewer2_paste_check.v1")
            self.assertTrue(out.is_file())
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("items", loaded)
            self.assertIn("matrix_rows", loaded)
            self.assertGreater(len(loaded["items"]), 0)

    def test_parse_sota_inventory_md_run_names(self) -> None:
        from harchoc.reviewer2_paste_check import parse_sota_inventory_md

        text = Path(__file__).resolve().parents[1].joinpath(
            "reports/_llm/sota_inventory.md"
        ).read_text(encoding="utf-8")
        parsed = parse_sota_inventory_md(text)
        self.assertIn("yolo26m_e100_s0", parsed["run_names_e100"])

    def test_paste_check_default_docx_drift_is_warn_not_fail(self) -> None:
        from harchoc.reviewer2_paste_check import extract_docx_text, run_reviewer2_paste_check

        repo = Path(__file__).resolve().parents[1]
        docx = repo / "reports/plants-4336582.docx"
        if not docx.is_file():
            self.skipTest("plants-4336582.docx not present")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "paste_check.json"
            report = run_reviewer2_paste_check(
                repo,
                out_path=out,
                strict_docx=False,
                write_drift_md=False,
            )
            map50_items = [it for it in report["items"] if it.get("id") == "docx_gap_map50"]
            if map50_items:
                self.assertEqual(map50_items[0]["status"], "warn")
            self.assertEqual(report["summary"]["fail"], 0)

    def test_experiment_paste_check_dry_run_exits_zero(self) -> None:
        import os

        from scripts.experiment import main

        os.environ.setdefault("HARCHOC_ALLOW_BASE_PYTHON", "1")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "paste.json"
            rc = main(["--dry-run", "reviewer2-paste-check", "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
