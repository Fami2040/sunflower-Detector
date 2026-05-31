from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harchoc.json_io import load_json, load_json_dict


class JsonIoTests(unittest.TestCase):
    def test_load_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            p.write_text(json.dumps({"a": 1}), encoding="utf-8")
            self.assertEqual(load_json(p), {"a": 1})

    def test_load_json_dict_requires_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "arr.json"
            p.write_text("[1]", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_json_dict(p)

    def test_load_json_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_json("/nonexistent/path.json")

    def test_common_cli_read_json_delegates(self) -> None:
        from scripts._common_cli import read_json, read_json_dict

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.json"
            p.write_text(json.dumps({"k": "v"}), encoding="utf-8")
            self.assertEqual(read_json(p), {"k": "v"})
            self.assertEqual(read_json_dict(p), {"k": "v"})


if __name__ == "__main__":
    unittest.main()
