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

    def test_classify_pip_check_ignores_conda_wrapper_noise(self) -> None:
        from harchoc.env_health import classify_pip_check_output

        raw = (
            "ERROR conda.cli.main_run:execute(125): `conda run python -m pip check` failed.\n"
            "super-gradients 3.7.1 has requirement numpy<=1.23, but you have numpy 2.4.6.\n"
        )
        r = classify_pip_check_output(raw)
        self.assertTrue(r["ok"])
        self.assertEqual(r["issues"], [])

    def test_external_detr_python_modules_cover_stack_imports(self) -> None:
        from harchoc.external_detector_train import (
            _external_python_modules,
            external_detr_python_modules,
        )

        union = set(external_detr_python_modules())
        for stack in ("deim", "dfine", "rtdetrv2_pytorch"):
            self.assertTrue(
                set(_external_python_modules(stack)).issubset(union),
                stack,
            )


if __name__ == "__main__":
    unittest.main()
