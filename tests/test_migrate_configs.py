import json
import tempfile
import unittest
from pathlib import Path


class MigrateConfigsTests(unittest.TestCase):
    def test_migrate_one_converts_yaml_to_json_and_reports_drops(self) -> None:
        from scripts.migrate_configs import migrate_one

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "tiny.yaml"
            src.write_text(
                "\n".join(
                    [
                        "name: bench_tiny",
                        "epochs: 3",
                        "infer:",
                        "  imgsz: 640",
                        "notes: >",
                        "  hello",
                        "  world",
                        "list:",
                        "  - a",
                        "  - b",
                        "weird_line_without_colon",
                        "",
                    ]
                ),
                "utf-8",
            )

            out_dir = base / "out"
            out_path, rep, file_report = migrate_one(src=src, out_dir=out_dir)

            self.assertTrue(out_path.exists())
            obj = json.loads(out_path.read_text("utf-8"))
            self.assertEqual(obj["source"], str(src))
            self.assertEqual(obj["config"]["name"], "bench_tiny")
            self.assertEqual(obj["config"]["epochs"], 3)
            self.assertEqual(obj["config"]["infer"]["imgsz"], 640)
            self.assertEqual(obj["config"]["notes"], "hello world")

            self.assertGreater(rep.dropped_lines_total, 0)
            self.assertEqual(file_report["source"], str(src))
            self.assertEqual(file_report["output"], str(out_path))
            self.assertEqual(file_report["dropped_lines_total"], rep.dropped_lines_total)
            self.assertIn("sequence_item", file_report["dropped_by_reason"])


if __name__ == "__main__":
    unittest.main()

