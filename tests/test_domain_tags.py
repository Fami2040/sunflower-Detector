import os
import tempfile
import unittest
from pathlib import Path


class DomainTagsTests(unittest.TestCase):
    def test_tray_key_from_stem(self) -> None:
        from harchoc.domain_tags import tray_key_from_stem

        self.assertEqual(tray_key_from_stem("349-10-2__________aug0"), "349-10-2")
        self.assertEqual(tray_key_from_stem("3a2-2___"), "3a2-2")

    def test_catalog_domains_from_labels(self) -> None:
        from harchoc.domain_tags import catalog_domains_from_dataset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            img = root / "images" / "val" / "349-10-2___aug0.jpg"
            lbl = root / "labels" / "val" / "349-10-2___aug0.txt"
            img.parent.mkdir(parents=True)
            lbl.parent.mkdir(parents=True)
            img.write_bytes(b"x")
            lbl.write_text("0 0.5 0.5 0.1 0.1\n1 0.6 0.6 0.1 0.1\n", encoding="utf-8")
            splits = root / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "val.txt").write_text("images/val/349-10-2___aug0.jpg\n", encoding="utf-8")

            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                cat = catalog_domains_from_dataset(
                    dataset_root=root,
                    splits_dir=root / "data" / "splits",
                )
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old

            self.assertEqual(cat["n_domains"], 1)
            dom = cat["domains"][0]
            self.assertEqual(dom["tray_key"], "349-10-2")
            self.assertEqual(dom["class_counts"]["0"], 1)
            self.assertEqual(dom["class_counts"]["1"], 1)


    def test_domain_metadata_tags_scaffold(self) -> None:
        from harchoc.domain_tags import domain_metadata_tags_scaffold

        sc = domain_metadata_tags_scaffold(n_trays=52)
        self.assertEqual(sc["schema_version"], "domain_metadata_tags.v0")
        self.assertEqual(sc["status"], "scaffold")
        self.assertEqual(sc["backlog_id"], "P1-DOMAIN-TAGS")
        self.assertIsNone(sc["tag_axes"]["variety"])
        self.assertEqual(sc["n_trays_in_catalog"], 52)
        self.assertEqual(sc["per_tray"], {})

    def test_load_domain_tags_csv_fixture(self) -> None:
        from harchoc.domain_tags import load_domain_tags_csv

        fixture = Path(__file__).resolve().parent / "fixtures" / "domain_tags_sample.csv"
        per_tray, summary = load_domain_tags_csv(fixture)
        self.assertEqual(summary["n_rows"], 3)
        self.assertEqual(summary["n_trays"], 3)
        self.assertEqual(
            per_tray["200-3-1"],
            {"variety": "PV545", "maturity": "ripe", "lighting": "benchtop", "site": "lab-a"},
        )
        self.assertEqual(per_tray["349-10-2"]["lighting"], "LED")

    def test_merge_domain_metadata_tags_from_csv(self) -> None:
        from harchoc.domain_tags import merge_domain_metadata_tags

        fixture = Path(__file__).resolve().parent / "fixtures" / "domain_tags_sample.csv"
        merged = merge_domain_metadata_tags(
            csv_path=fixture,
            catalog_tray_keys={"200-3-1", "349-10-2"},
            n_trays=2,
        )
        self.assertEqual(merged["status"], "partial")
        self.assertEqual(merged["n_trays_in_catalog"], 2)
        self.assertEqual(merged["per_tray"]["200-3-1"]["variety"], "PV545")
        self.assertEqual(merged["tag_axes"]["variety"], ["Other", "PV545"])
        imp = merged["import"]
        self.assertEqual(imp["n_catalog_trays_tagged"], 2)
        self.assertEqual(imp["n_catalog_trays_untagged"], 0)
        self.assertEqual(imp["unknown_tray_keys_in_csv"], ["unknown-tray"])

    def test_attach_tray_tags_to_domains(self) -> None:
        from harchoc.domain_tags import attach_tray_tags_to_domains

        domains = [
            {"tray_key": "200-3-1", "metrics": None},
            {"tray_key": "349-10-2", "metrics": None},
            {"tray_key": "untagged", "metrics": None},
        ]
        tags = {
            "per_tray": {
                "200-3-1": {"variety": "PV545", "maturity": "ripe", "lighting": "benchtop", "site": "lab-a"},
            }
        }
        out = attach_tray_tags_to_domains(domains, tags)
        self.assertEqual(out[0]["tags"]["variety"], "PV545")
        self.assertNotIn("tags", out[2])


if __name__ == "__main__":
    unittest.main()
