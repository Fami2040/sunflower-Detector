import json
import os
import tempfile
import unittest
from pathlib import Path


class CvEvalTests(unittest.TestCase):
    def test_kfold_assign_and_aggregate(self) -> None:
        from harchoc.cv_eval_core import aggregate_fold_metrics, kfold_assign

        items = [f"img{i}.jpg" for i in range(10)]
        folds = kfold_assign(items, folds=5, seed=0)
        self.assertEqual(sum(len(f) for f in folds), 10)

        docs = [
            {"mAP50": 0.8, "mAP50_95": 0.4, "counting_metrics": {"mae": 2.0}},
            {"mAP50": 0.82, "mAP50_95": 0.42, "counting_metrics": {"mae": 2.1}},
        ]
        agg = aggregate_fold_metrics(docs)
        self.assertEqual(agg["n_folds"], 2)
        self.assertIn("mAP50", agg)
        self.assertIsNotNone(agg["mAP50"]["ci"])

    def test_cv_eval_main_with_fold_metrics(self) -> None:
        from scripts.cv_eval import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "dataset"
            splits = root / "data" / "splits"
            splits.mkdir(parents=True)
            (splits / "train.txt").write_text("images/a.jpg\nimages/b.jpg\n", "utf-8")
            m1 = Path(td) / "f0.json"
            m2 = Path(td) / "f1.json"
            m1.write_text(json.dumps({"mAP50": 0.7}), encoding="utf-8")
            m2.write_text(json.dumps({"mAP50": 0.75}), encoding="utf-8")
            out = Path(td) / "summary.json"
            old = os.environ.get("DATASET_ROOT")
            try:
                os.environ["DATASET_ROOT"] = str(root)
                rc = main(
                    [
                        "--out",
                        str(out),
                        "--fold-metrics",
                        str(m1),
                        "--fold-metrics",
                        str(m2),
                        "--folds",
                        "2",
                    ]
                )
            finally:
                if old is None:
                    os.environ.pop("DATASET_ROOT", None)
                else:
                    os.environ["DATASET_ROOT"] = old
            self.assertEqual(rc, 0)
            obj = json.loads(out.read_text("utf-8"))
            self.assertEqual(obj["status"], "ok")
            self.assertIsNotNone(obj.get("aggregation"))


if __name__ == "__main__":
    unittest.main()
