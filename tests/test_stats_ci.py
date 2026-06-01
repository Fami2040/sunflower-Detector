import unittest


class StatsCiTests(unittest.TestCase):
    def test_bootstrap_ci_mean_contains_sample_mean(self) -> None:
        from harchoc.stats_ci import bootstrap_ci

        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        r = bootstrap_ci(xs, stat="mean", n_resamples=200, random_state=0)
        self.assertEqual(r.method, "bootstrap_percentile")
        self.assertAlmostEqual(r.point, 4.5, places=5)
        self.assertLessEqual(r.low, r.point)
        self.assertGreaterEqual(r.high, r.point)

    def test_percentile_ci_tiny_n(self) -> None:
        from harchoc.stats_ci import percentile_ci

        r = percentile_ci([3.0], stat="mean")
        self.assertIsNone(r)

    def test_ci_for_values_single(self) -> None:
        from harchoc.stats_ci import ci_for_values

        r = ci_for_values([42.0])
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.method, "single_sample")
        self.assertEqual(r.point, 42.0)

    def test_bin_reliability_and_ece(self) -> None:
        from harchoc.stats_ci import bin_reliability, expected_calibration_error

        scores = [0.1, 0.2, 0.8, 0.9]
        correct = [0, 0, 1, 1]
        bins = bin_reliability(scores, correct, n_bins=2)
        self.assertEqual(len(bins.bin_counts), 2)
        ece = expected_calibration_error(bins)
        self.assertIsNotNone(ece)
        assert ece is not None
        self.assertGreaterEqual(ece, 0.0)

    def test_to_json_roundtrip_keys(self) -> None:
        from harchoc.stats_ci import bootstrap_ci

        r = bootstrap_ci([1.0, 2.0, 3.0], n_resamples=50, random_state=1)
        j = r.to_json()
        self.assertIn("low", j)
        self.assertIn("high", j)
        self.assertEqual(j["n"], 3)


if __name__ == "__main__":
    unittest.main()
