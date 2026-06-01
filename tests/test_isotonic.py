import unittest


class IsotonicTests(unittest.TestCase):
    def test_fit_isotonic_pava_is_monotone(self) -> None:
        from harchoc.isotonic import fit_isotonic_pava

        # Non-monotone targets over increasing scores -> PAVA should smooth.
        scores = [0.1, 0.2, 0.3, 0.4]
        targets = [0.0, 1.0, 0.2, 0.9]
        m = fit_isotonic_pava(scores=scores, targets=targets)
        self.assertGreaterEqual(len(m.x), 1)
        # y must be non-decreasing
        for a, b in zip(m.y, m.y[1:], strict=False):
            self.assertLessEqual(a, b)

    def test_predict_clamps_and_steps(self) -> None:
        from harchoc.isotonic import IsotonicModel

        m = IsotonicModel(x=(0.2, 0.5, 0.8), y=(0.1, 0.2, 0.9))
        self.assertAlmostEqual(m.predict_one(0.0), 0.1)
        self.assertAlmostEqual(m.predict_one(1.0), 0.9)
        self.assertAlmostEqual(m.predict_one(0.2), 0.1)
        self.assertAlmostEqual(m.predict_one(0.49), 0.1)
        self.assertAlmostEqual(m.predict_one(0.5), 0.2)
        self.assertAlmostEqual(m.predict_one(0.79), 0.2)


if __name__ == "__main__":
    unittest.main()

