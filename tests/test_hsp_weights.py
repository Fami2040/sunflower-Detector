import os
import unittest
from pathlib import Path

from harchoc.hsp_weights import HSP_DETECTION_WEIGHTS, resolve_detection_weights


class HspWeightsTests(unittest.TestCase):
    def test_default_is_best2(self) -> None:
        old = os.environ.pop("DETECTION_MODEL", None)
        try:
            self.assertEqual(str(resolve_detection_weights()), HSP_DETECTION_WEIGHTS)
            self.assertEqual(HSP_DETECTION_WEIGHTS, "models/best2.pt")
        finally:
            if old is not None:
                os.environ["DETECTION_MODEL"] = old

    def test_explicit_raw_overrides_env(self) -> None:
        old = os.environ.get("DETECTION_MODEL")
        try:
            os.environ["DETECTION_MODEL"] = "models/other.pt"
            self.assertEqual(str(resolve_detection_weights("models/best2.pt")), "models/best2.pt")
        finally:
            if old is None:
                os.environ.pop("DETECTION_MODEL", None)
            else:
                os.environ["DETECTION_MODEL"] = old

    def test_detection_model_env(self) -> None:
        old = os.environ.get("DETECTION_MODEL")
        try:
            os.environ["DETECTION_MODEL"] = "models/custom.pt"
            self.assertEqual(str(resolve_detection_weights()), "models/custom.pt")
        finally:
            if old is None:
                os.environ.pop("DETECTION_MODEL", None)
            else:
                os.environ["DETECTION_MODEL"] = old

    def test_repo_best2_exists_in_dev(self) -> None:
        p = Path(__file__).resolve().parents[1] / HSP_DETECTION_WEIGHTS
        if p.is_file():
            self.assertGreater(p.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
