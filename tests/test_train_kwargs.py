import unittest
from typing import Any


class TrainKwargsTests(unittest.TestCase):
    def test_allowed_includes_runtime_forwarding_keys(self) -> None:
        from harchoc.train_kwargs import ALLOWED_TRAIN_KWARGS

        for key in ("amp", "nbs", "cache"):
            self.assertIn(key, ALLOWED_TRAIN_KWARGS)
        from harchoc.train_kwargs import TRAIN_POLICY_ONLY_KEYS
        self.assertIn("grad_clip", TRAIN_POLICY_ONLY_KEYS)

    def test_num_queries_not_forwarded_to_ultralytics(self) -> None:
        from harchoc.train_kwargs import TRAIN_POLICY_ONLY_KEYS, ultralytics_train_kwargs

        self.assertIn("num_queries", TRAIN_POLICY_ONLY_KEYS)
        kw = ultralytics_train_kwargs(
            {"epochs": 1, "num_queries": 300},
            data_yaml="/tmp/data.yaml",
            run_name="t",
        )
        self.assertNotIn("num_queries", kw)

    def test_ultralytics_train_kwargs_forwards_whitelisted_only(self) -> None:
        from harchoc.train_kwargs import ultralytics_train_kwargs

        cfg: dict[str, Any] = {
            "epochs": 1,
            "amp": False,
            "cache": False,
            "model": "yolov8n.pt",
            "eval": {"max_det": 300},
            "notes": "ignored",
        }
        kw = ultralytics_train_kwargs(cfg, data_yaml="/tmp/data.yaml", run_name="t")
        self.assertEqual(kw["epochs"], 1)
        self.assertFalse(kw["amp"])
        self.assertFalse(kw["cache"])
        self.assertNotIn("model", kw)
        self.assertNotIn("eval", kw)
        self.assertNotIn("notes", kw)

    def test_forwarded_train_keys_excludes_data_and_name(self) -> None:
        from harchoc.train_kwargs import forwarded_train_keys

        keys = forwarded_train_keys(
            {"data": "d.yaml", "name": "run", "epochs": 2, "project": "runs", "exist_ok": True}
        )
        self.assertEqual(keys, ["epochs"])


if __name__ == "__main__":
    unittest.main()
