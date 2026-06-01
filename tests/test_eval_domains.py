import json
import os
import tempfile
import unittest
from pathlib import Path


class EvalDomainsTests(unittest.TestCase):
    def test_eval_domains_catalog(self) -> None:
        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img = root / "images" / "test" / "200-3-1___aug2.jpg"
            lbl = root / "labels" / "test" / "200-3-1___aug2.txt"
            img.parent.mkdir(parents=True)
            lbl.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            lbl.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            splits = root / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "test.txt").write_text("images/test/200-3-1___aug2.jpg\n", encoding="utf-8")
            catalog_out = Path(td) / "catalog.json"
            eval_out = Path(td) / "domain_eval.json"
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                rc = main(
                    [
                        "--catalog",
                        str(catalog_out),
                        "--out",
                        str(eval_out),
                        "--splits-dir",
                        str(root / "data" / "splits"),
                    ]
                )
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old
            self.assertEqual(rc, 0)
            cat = json.loads(catalog_out.read_text("utf-8"))
            self.assertEqual(cat["status"], "ok")
            self.assertEqual(cat["schema_version"], "eval_domains_run.v1")
            self.assertGreaterEqual(cat["catalog"]["n_domains"], 1)

            ev = json.loads(eval_out.read_text("utf-8"))
            self.assertEqual(ev["schema_version"], "domain_eval.v1")
            self.assertEqual(ev["status"], "scaffold")
            self.assertGreaterEqual(len(ev["domains"]), 1)
            self.assertEqual(ev["domains"][0]["tray_key"], "200-3-1")

    def test_eval_domains_import_domain_tags(self) -> None:
        from scripts.eval_domains import main

        fixture = Path(__file__).resolve().parent / "fixtures" / "domain_tags_sample.csv"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img = root / "images" / "test" / "200-3-1___aug2.jpg"
            lbl = root / "labels" / "test" / "200-3-1___aug2.txt"
            img.parent.mkdir(parents=True)
            lbl.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            lbl.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            splits = root / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "test.txt").write_text("images/test/200-3-1___aug2.jpg\n", encoding="utf-8")
            catalog_out = Path(td) / "catalog.json"
            eval_out = Path(td) / "domain_eval.json"
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                rc = main(
                    [
                        "--catalog",
                        str(catalog_out),
                        "--out",
                        str(eval_out),
                        "--splits-dir",
                        str(root / "data" / "splits"),
                        "--import-domain-tags",
                        str(fixture),
                    ]
                )
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old
            self.assertEqual(rc, 0)
            cat = json.loads(catalog_out.read_text("utf-8"))
            tags = cat["domain_metadata_tags"]
            self.assertEqual(tags["status"], "partial")
            self.assertEqual(tags["per_tray"]["200-3-1"]["variety"], "PV545")
            self.assertEqual(tags["import"]["n_catalog_trays_tagged"], 1)

            ev = json.loads(eval_out.read_text("utf-8"))
            self.assertEqual(ev["domain_metadata_tags"]["per_tray"]["200-3-1"]["variety"], "PV545")
            self.assertEqual(ev["domains"][0]["tags"]["variety"], "PV545")

    def test_dry_run_schema_from_catalog_file(self) -> None:
        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            catalog_path = Path(td) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "schema_version": "eval_domains_run.v1",
                        "catalog": {
                            "domains": [
                                {"tray_key": "349-10-2", "n_images": 3},
                                {"tray_key": "3a2-2", "n_images": 1},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = Path(td) / "domain_eval.json"
            rc = main(
                [
                    "--dry-run",
                    "--catalog",
                    str(catalog_path),
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "domain_eval.v1")
            self.assertEqual(obj["status"], "dry-run")
            self.assertEqual(obj["script"], "eval_domains")
            keys = [d["tray_key"] for d in obj["domains"]]
            self.assertEqual(keys, ["349-10-2", "3a2-2"])
            for dom in obj["domains"]:
                self.assertIn("split_file", dom)
                self.assertIsNone(dom["metrics"])
                self.assertIsNone(dom["delta_vs_canonical"])
            self.assertIsNone(obj["canonical_test"]["metrics"])

    def test_dry_run_schema_without_gpu_or_dataset(self) -> None:
        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "domain_eval.json"
            catalog_path = Path(td) / "missing_catalog.json"
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ.pop("DATASET_ROOT", None)
                rc = main(
                    [
                        "--dry-run",
                        "--catalog",
                        str(catalog_path),
                        "--out",
                        str(out),
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "domain_eval.v1")
            self.assertEqual(obj["status"], "dry-run")
            self.assertEqual(len(obj["domains"]), 1)
            self.assertEqual(obj["domains"][0]["tray_key"], "_example")

    def test_run_all_trays_dry_run_lists_planned_evals(self) -> None:
        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            catalog_path = Path(td) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "schema_version": "eval_domains_run.v1",
                        "catalog": {
                            "domains": [
                                {"tray_key": "349-10-2", "n_images": 3},
                                {"tray_key": "3a2-2", "n_images": 1},
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            out = Path(td) / "domain_eval.json"
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ.pop("DATASET_ROOT", None)
                rc = main(
                    [
                        "--dry-run",
                        "--run-all-trays",
                        "--catalog",
                        str(catalog_path),
                        "--out",
                        str(out),
                        "--device",
                        "cpu",
                        "--locked-conf-from",
                        "reports/hsp/threshold_val.json",
                    ]
                )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["schema_version"], "domain_eval.v1")
            self.assertEqual(obj["status"], "dry-run")
            planned = obj.get("planned_tray_evals")
            self.assertIsInstance(planned, list)
            self.assertEqual(len(planned), 2)
            self.assertEqual(planned[0]["tray_key"], "349-10-2")
            self.assertEqual(planned[0]["device"], "cpu")
            self.assertEqual(planned[0]["locked_conf_from"], "reports/hsp/threshold_val.json")
            self.assertIn("test_349-10-2.txt", planned[0]["split_file"])

    def test_run_all_trays_exits_when_dataset_unresolvable(self) -> None:
        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "data" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "missing",
                                "extracted_paths": ["datasets/missing"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            catalog_path = Path(td) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    {
                        "catalog": {
                            "domains": [{"tray_key": "200-3-1", "n_images": 1}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            out = Path(td) / "domain_eval.json"
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ.pop("DATASET_ROOT", None)
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "--run-all-trays",
                            "--manifest",
                            str(manifest_path),
                            "--default-dataset-name",
                            "missing",
                            "--catalog",
                            str(catalog_path),
                            "--out",
                            str(out),
                        ]
                    )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

    def test_run_all_trays_resolves_dataset_from_manifest(self) -> None:
        import unittest.mock as mock

        from scripts.eval_domains import main

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "datasets" / "sunflower"
            img = root / "images" / "test" / "200-3-1___aug2.jpg"
            lbl = root / "labels" / "test" / "200-3-1___aug2.txt"
            img.parent.mkdir(parents=True)
            lbl.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            lbl.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            splits = root / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "test.txt").write_text("images/test/200-3-1___aug2.jpg\n", encoding="utf-8")

            manifest_path = base / "data" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "sunflower-cvat-1093",
                                "extracted_paths": ["datasets/sunflower"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            catalog_out = base / "catalog.json"
            eval_out = base / "domain_eval.json"
            old_root = os.environ.get("DATASET_ROOT")
            try:
                os.environ.pop("DATASET_ROOT", None)
                with mock.patch(
                    "harchoc.domain_eval_loop.run_all_tray_domain_evals",
                    return_value=[("200-3-1", 0, {"mAP50": 0.5, "mAP50_95": 0.3})],
                ):
                    rc = main(
                        [
                            "--run-all-trays",
                            "--manifest",
                            str(manifest_path),
                            "--default-dataset-name",
                            "sunflower-cvat-1093",
                            "--catalog",
                            str(catalog_out),
                            "--out",
                            str(eval_out),
                            "--splits-dir",
                            str(splits),
                            "--device",
                            "cpu",
                        ]
                    )
            finally:
                if old_root is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_root

            self.assertEqual(rc, 0)
            obj = json.loads(eval_out.read_text("utf-8"))
            self.assertNotEqual(obj["status"], "blocked")
            self.assertNotIn("blocker", obj)
            self.assertEqual(obj["domains"][0]["tray_key"], "200-3-1")
            self.assertAlmostEqual(obj["domains"][0]["metrics"]["mAP50"], 0.5)

    def test_run_tray_eval_applies_metrics_mocked(self) -> None:
        from harchoc.domain_eval_loop import apply_tray_metrics_to_domain_eval, run_tray_domain_eval

        with tempfile.TemporaryDirectory() as td:
            domains = Path(td) / "data" / "domains"
            domains.mkdir(parents=True)
            (domains / "test_200-3-1.txt").write_text("images/test/x.jpg\n", encoding="utf-8")

            def fake_eval(argv: list[str] | None) -> int:
                assert argv is not None
                out = Path(argv[argv.index("--out") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps({"mAP50": 0.5, "mAP50_95": 0.3}),
                    encoding="utf-8",
                )
                return 0

            rc, metrics = run_tray_domain_eval(
                tray_key="200-3-1",
                weights="models/best2.pt",
                domains_dir=domains,
                reports_dir=Path(td) / "reports",
                dataset_root="/tmp/unused",
                eval_main=fake_eval,
            )
            self.assertEqual(rc, 0)
            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(metrics["mAP50"], 0.5)

            payload = apply_tray_metrics_to_domain_eval(
                {
                    "domains": [
                        {"tray_key": "200-3-1", "metrics": None},
                        {"tray_key": "other", "metrics": None},
                    ]
                },
                tray_key="200-3-1",
                metrics=metrics,
                eval_rc=rc,
            )
            self.assertEqual(payload["domains"][0]["metrics"]["mAP50"], 0.5)
            self.assertIsNone(payload["domains"][1]["metrics"])


    def test_summarize_tray_count_mae(self) -> None:
        from harchoc.domain_count_mae import summarize_tray_count_mae

        domains = [
            {"tray_key": "a", "metrics": {"count_mae": 10.0, "mAP50": 0.5}},
            {"tray_key": "b", "metrics": {"count_mae": 20.0}},
            {"tray_key": "c", "metrics": {"mAP50": 0.3}},
        ]
        s = summarize_tray_count_mae(domains, locked_conf_from="t.json", locked_conf=0.15)
        self.assertEqual(s["n_trays_with_count_mae"], 2)
        self.assertAlmostEqual(s["count_mae_mean"], 15.0)

    def test_merge_tray_count_mae_mocked(self) -> None:
        import json
        import os
        import tempfile
        import unittest.mock as mock
        from pathlib import Path

        from scripts.eval_domains import _merge_tray_count_mae

        def fake_run(**_kw: object) -> tuple[str, int, dict[str, object] | None]:
            return (
                "200-3-1",
                0,
                {
                    "count_mae": 12.5,
                    "count_n_images": 1,
                    "locked_conf": 0.15,
                    "counting_metrics": {"mae": 12.5, "n_images": 1},
                },
            )

        with tempfile.TemporaryDirectory() as td:
            eval_out = Path(td) / "domain_eval.json"
            eval_out.write_text(
                json.dumps(
                    {
                        "schema_version": "domain_eval.v1",
                        "domains": [{"tray_key": "200-3-1", "metrics": {"mAP50": 0.4}}],
                    }
                ),
                encoding="utf-8",
            )
            root = Path(td) / "dataset"
            root.mkdir(parents=True)
            threshold = Path(td) / "threshold.json"
            threshold.write_text(
                json.dumps({"selected": {"row": {"conf_thr": 0.15}}, "match": {"iou": 0.3}}),
                encoding="utf-8",
            )

            from argparse import Namespace

            args = Namespace(
                out=str(eval_out),
                dataset_root=str(root),
                locked_conf_from=str(threshold),
                weights="models/best2.pt",
                device="cpu",
                reports_dir=str(Path(td) / "reports"),
                domains_dir=str(Path(td) / "data" / "domains"),
                manifest="data/manifest.json",
                default_dataset_name="sunflower",
                dataset_name=None,
                yolo_data_yaml=None,
                count_mae_sidecar=str(Path(td) / "domain_count_mae.json"),
            )

            old_env = os.environ.get("DATASET_ROOT")
            os.environ["DATASET_ROOT"] = str(root)
            try:
                with mock.patch(
                    "harchoc.domain_count_mae.run_tray_count_mae_eval",
                    side_effect=fake_run,
                ):
                    rc = _merge_tray_count_mae(args, Path(td))
            finally:
                if old_env is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old_env

            self.assertEqual(rc, 0)
            merged = json.loads(eval_out.read_text("utf-8"))
            self.assertAlmostEqual(merged["domains"][0]["metrics"]["count_mae"], 12.5)
            self.assertIn("count_mae_summary", merged)


if __name__ == "__main__":
    unittest.main()
