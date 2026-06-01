import json
import os
import tempfile
import unittest
from pathlib import Path


class HpoSearchTests(unittest.TestCase):
    def test_plan_hpo_trials_respects_budget_caps(self) -> None:
        from harchoc.hpo_search import plan_hpo_trials

        with tempfile.TemporaryDirectory() as td:
            repo = Path(__file__).resolve().parents[1]
            base = Path(td) / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "model": "yolov8n.pt",
                        "epochs": 100,
                        "imgsz": 1280,
                        "batch": 1,
                    }
                ),
                "utf-8",
            )

            old = os.environ.get("HARCHOC_MAX_EPOCHS")
            try:
                os.environ["HARCHOC_MAX_EPOCHS"] = "5"
                with self.assertRaises(SystemExit):
                    plan_hpo_trials(
                        base_train_config=str(base),
                        space={"params": {"epochs": {"kind": "int", "low": 10, "high": 12}}},
                        trials=1,
                        seed=0,
                    )
            finally:
                if old is None:
                    os.environ.pop("HARCHOC_MAX_EPOCHS", None)
                else:
                    os.environ["HARCHOC_MAX_EPOCHS"] = old

    def test_experiment_hpo_writes_report(self) -> None:
        from scripts.experiment import main as exp_main

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "model": "yolov8n.pt",
                        "epochs": 1,
                        "imgsz": 1280,
                        "batch": 1,
                    }
                ),
                "utf-8",
            )
            space = Path(td) / "space.json"
            space.write_text(
                json.dumps(
                    {
                        "params": {
                            "lr0": {"kind": "float", "low": 1e-4, "high": 1e-3, "log": True},
                            "optimizer": {"kind": "categorical", "choices": ["AdamW", "SGD"]},
                        }
                    }
                ),
                "utf-8",
            )

            out = Path(td) / "out.json"
            rc = int(
                exp_main(
                    [
                        "hpo",
                        "--base-train-config",
                        str(base),
                        "--space",
                        str(space),
                        "--trials",
                        "3",
                        "--seed",
                        "123",
                        "--out",
                        str(out),
                    ]
                )
            )
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj.get("schema_version"), "hpo_search.v1")
            self.assertEqual(len(obj.get("trials") or []), 3)


if __name__ == "__main__":
    unittest.main()

