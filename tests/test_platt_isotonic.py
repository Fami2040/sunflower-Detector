from __future__ import annotations

import unittest

from harchoc.platt import _fit_isotonic_scores


class PlattIsotonicTests(unittest.TestCase):
    def test_fit_isotonic_monotone(self) -> None:
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        out, meta = _fit_isotonic_scores(scores=scores, targets=labels)
        self.assertEqual(len(out), 5)
        self.assertIn(meta["calibrator"], ("sklearn_isotonic", "isotonic_pava"))
        self.assertLessEqual(out[0], out[-1])


if __name__ == "__main__":
    unittest.main()
