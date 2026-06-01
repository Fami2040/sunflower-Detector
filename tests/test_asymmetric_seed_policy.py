from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestAsymmetricSeedPolicy(unittest.TestCase):
    def test_committed_policy_validates(self) -> None:
        from harchoc.asymmetric_seed_policy import load_asymmetric_seed_policy

        repo = Path(__file__).resolve().parents[1]
        policy_path = repo / "configs/eval/asymmetric_seed_policy.json"
        obj = load_asymmetric_seed_policy(policy_path)
        self.assertEqual(obj["schema_version"], "asymmetric_seed_policy.v1")
        self.assertEqual(obj["eval_policy"]["primary_split"], "test")
        self.assertEqual(obj["eval_policy"]["split_file"], "data/splits/test.txt")
        test_row = obj["prevalence"]["splits"]["test"]
        self.assertEqual(test_row["n_images"], 109)
        self.assertAlmostEqual(test_row["developed_fraction"], 0.5541, places=4)

    def test_rejects_bad_schema(self) -> None:
        from harchoc.asymmetric_seed_policy import load_asymmetric_seed_policy

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"schema_version": "nope"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_asymmetric_seed_policy(p)

    def test_rejects_inconsistent_fractions(self) -> None:
        from harchoc.asymmetric_seed_policy import validate_asymmetric_seed_policy

        repo = Path(__file__).resolve().parents[1]
        base = json.loads(
            (repo / "configs/eval/asymmetric_seed_policy.json").read_text(encoding="utf-8")
        )
        base["prevalence"]["splits"]["test"]["developed_fraction"] = 0.99
        with self.assertRaises(ValueError):
            validate_asymmetric_seed_policy(base)


if __name__ == "__main__":
    unittest.main()
