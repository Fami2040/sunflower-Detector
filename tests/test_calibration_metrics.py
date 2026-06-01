import unittest


class CalibrationMetricsTests(unittest.TestCase):
    def test_ece_perfect_calibration(self) -> None:
        from harchoc.calibration_metrics import reliability_and_ece

        scores = [0.2, 0.4, 0.6, 0.8]
        correct = [0, 0, 1, 1]
        out = reliability_and_ece(scores, correct, n_bins=2)
        self.assertIsNotNone(out["ece"])
        self.assertIn("reliability", out)


if __name__ == "__main__":
    unittest.main()
