import unittest


class EnvHealthTests(unittest.TestCase):
    def test_classify_pip_check_suppresses_sg_numpy_noise(self) -> None:
        from harchoc.env_health import classify_pip_check_output

        raw = (
            "super-gradients 3.7.1 has requirement numpy<2.0,>=1.24.2, but you have numpy 2.1.0.\n"
            "some-other-package 1.0 has requirement foo, but you have bar 2.0.\n"
        )
        r = classify_pip_check_output(raw)
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["ignored_sg_numpy"]), 1)
        self.assertEqual(len(r["issues"]), 1)


if __name__ == "__main__":
    unittest.main()
