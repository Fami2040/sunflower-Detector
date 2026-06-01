"""CI-safe tests for publication table reproduction (fixture JSON only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ManuscriptTablesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]
        self.fixtures = self.repo / "tests/fixtures/manuscript_tables"
        self.aug_fixture = self.repo / "tests/fixtures/aug_smoke_leaderboard"

    def test_headline_from_dual_metric_fixture(self) -> None:
        from harchoc.json_io import load_json_dict
        from harchoc.manuscript_tables import build_headline_rows, render_headline_md

        dm = load_json_dict(self.fixtures / "dual_metric.json")
        rows = build_headline_rows(dual_metric=dm, model_label="models/best2.pt")
        self.assertEqual(len(rows), 2)
        test = next(r for r in rows if r["split"] == "test")
        self.assertAlmostEqual(float(test["count_mae"]), 61.266, places=2)
        self.assertAlmostEqual(float(test["map50"]), 0.180, places=3)
        self.assertAlmostEqual(float(test["locked_conf"]), 0.15, places=2)
        md = render_headline_md(rows, dual_metric_path="reports/hsp/dual_metric.json")
        self.assertIn("61.3", md)
        self.assertIn("val-locked conf", md)

    def test_zoo_core_partial_matrix_train(self) -> None:
        from harchoc.manuscript_tables import build_zoo_core_rows, render_zoo_core_md

        rows, meta = build_zoo_core_rows(
            repo_root=self.repo,
            matrix_rows_path="configs/zoo/matrix_rows.v1.json",
            matrix_train_path=str(self.fixtures / "matrix_train_partial.json"),
            matrix_group="zoo_core",
        )
        self.assertGreater(meta["n_expected"], 1)
        self.assertEqual(meta["n_complete"], 1)
        y8m = next(r for r in rows if r["id"] == "yolov8m")
        self.assertAlmostEqual(float(y8m["test_count_mae"]), 72.5, places=1)
        pending = [r for r in rows if r["status"] == "pending"]
        self.assertGreater(len(pending), 0)
        md = render_zoo_core_md(rows, meta, matrix_train_path="tests/fixtures/.../matrix_train_partial.json")
        self.assertIn("pending", md)
        self.assertIn("Partial aggregate", md)

    def test_zoo_yolo_only_expects_four_rows(self) -> None:
        from harchoc.manuscript_tables import (
            ZOO_YOLO_ONLY_ROW_IDS,
            build_zoo_core_rows,
            render_zoo_core_md,
        )

        rows, meta = build_zoo_core_rows(
            repo_root=self.repo,
            matrix_rows_path="configs/zoo/matrix_rows.v1.json",
            matrix_train_path=str(self.fixtures / "matrix_train_partial.json"),
            matrix_group="zoo_yolo_only",
        )
        self.assertEqual(meta["matrix_group"], "zoo_yolo_only")
        self.assertEqual(meta["n_expected"], len(ZOO_YOLO_ONLY_ROW_IDS))
        self.assertEqual(len(rows), len(ZOO_YOLO_ONLY_ROW_IDS))
        self.assertEqual({r["id"] for r in rows}, set(ZOO_YOLO_ONLY_ROW_IDS))
        md = render_zoo_core_md(rows, meta, matrix_train_path="tests/fixtures/.../partial.json")
        self.assertIn("zoo_yolo_only", md)
        self.assertNotIn("rtdetr_l", md)

    def test_aug_top_n_from_fixture_leaderboard(self) -> None:
        from harchoc.manuscript_tables import build_aug_top_n_rows, render_aug_top_n_md

        rows, payload = build_aug_top_n_rows(
            repo_root=self.aug_fixture,
            index_path="index.json",
            out_dir="reports/aug_smoke",
            top_n=3,
            leaderboard_json=None,
        )
        self.assertLessEqual(len(rows), 3)
        self.assertGreater(len(rows), 0)
        md = render_aug_top_n_md(rows, top_n=3, reference=payload.get("reference") or {})
        self.assertIn("S1", md)
        self.assertIn("best2", md.lower())

    def test_write_tables_repro_fixture_tree(self) -> None:
        from harchoc.manuscript_tables import write_manuscript_tables

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "reports/hsp").mkdir(parents=True)
            (tdp / "configs/zoo").mkdir(parents=True)
            import shutil

            shutil.copy(self.fixtures / "dual_metric.json", tdp / "reports/hsp/dual_metric.json")
            shutil.copy(
                self.fixtures / "matrix_train_partial.json",
                tdp / "reports/hsp/matrix_train.json",
            )
            shutil.copy(
                self.repo / "configs/zoo/matrix_rows.v1.json",
                tdp / "configs/zoo/matrix_rows.v1.json",
            )
            (tdp / "configs/experiments").mkdir(parents=True, exist_ok=True)
            shutil.copy(
                self.aug_fixture / "index.json",
                tdp / "configs/experiments/aug_smoke_index.json",
            )
            aug_src = self.aug_fixture / "reports/aug_smoke"
            aug_dst = tdp / "reports/aug_smoke"
            shutil.copytree(aug_src, aug_dst)

            written = write_manuscript_tables(
                repo_root=tdp,
                out_dir="reports/manuscript/tables",
                dual_metric_path="reports/hsp/dual_metric.json",
                matrix_train_path="reports/hsp/matrix_train.json",
                matrix_rows_path="configs/zoo/matrix_rows.v1.json",
                aug_index_path="configs/experiments/aug_smoke_index.json",
                aug_out_dir="reports/aug_smoke",
                top_n=5,
                write_latex=True,
            )
            self.assertTrue(written["tables_manifest.json"].is_file())
            manifest = json.loads(written["tables_manifest.json"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "manuscript_tables_manifest.v1")
            self.assertIn("headline_metrics.md", manifest["outputs"])
            self.assertTrue((tdp / "reports/manuscript/tables/headline_metrics.tex").is_file())

    def test_write_tables_absolute_out_dir(self) -> None:
        from harchoc.manuscript_tables import write_manuscript_tables

        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as out_td:
            tdp = Path(td)
            (tdp / "reports/hsp").mkdir(parents=True)
            (tdp / "configs/zoo").mkdir(parents=True)
            import shutil

            shutil.copy(self.fixtures / "dual_metric.json", tdp / "reports/hsp/dual_metric.json")
            shutil.copy(
                self.fixtures / "matrix_train_partial.json",
                tdp / "reports/hsp/matrix_train.json",
            )
            shutil.copy(
                self.repo / "configs/zoo/matrix_rows.v1.json",
                tdp / "configs/zoo/matrix_rows.v1.json",
            )
            (tdp / "configs/experiments").mkdir(parents=True, exist_ok=True)
            shutil.copy(
                self.aug_fixture / "index.json",
                tdp / "configs/experiments/aug_smoke_index.json",
            )
            shutil.copytree(self.aug_fixture / "reports/aug_smoke", tdp / "reports/aug_smoke")
            out_abs = Path(out_td) / "tables"
            written = write_manuscript_tables(
                repo_root=tdp,
                out_dir=str(out_abs),
                dual_metric_path="reports/hsp/dual_metric.json",
                matrix_train_path="reports/hsp/matrix_train.json",
                matrix_rows_path="configs/zoo/matrix_rows.v1.json",
                aug_index_path="configs/experiments/aug_smoke_index.json",
                aug_out_dir="reports/aug_smoke",
            )
            manifest = json.loads(written["tables_manifest.json"].read_text(encoding="utf-8"))
            headline_path = manifest["outputs"]["headline_metrics.md"]
            self.assertTrue(
                headline_path.startswith(str(out_abs)),
                msg=f"expected absolute path, got {headline_path!r}",
            )

    def test_dry_run_manifest(self) -> None:
        from harchoc.manuscript_tables import build_tables_repro_dry_run

        payload = build_tables_repro_dry_run(
            repo_root=self.aug_fixture,
            aug_index_path="index.json",
            aug_out_dir="reports/aug_smoke",
            matrix_rows_path=str(self.repo / "configs/zoo/matrix_rows.v1.json"),
            matrix_train_path="reports/hsp/matrix_train.json",
            dual_metric_path="reports/hsp/dual_metric.json",
        )
        self.assertEqual(payload["status"], "dry-run")
        self.assertIn("would_write", payload)


if __name__ == "__main__":
    unittest.main()
