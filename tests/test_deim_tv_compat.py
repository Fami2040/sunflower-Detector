import unittest


class DeimTvCompatTests(unittest.TestCase):
    def test_apply_patches_transform_methods(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed (use mamba env harchoc)")
        from harchoc.external_repos import resolve_external_repo_path

        repo = resolve_external_repo_path("deim")
        if repo is None:
            self.skipTest("DEIM repo not present")
        import sys

        repo_s = str(repo.resolve())
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
        from engine.data.transforms import _transforms as deim_transforms

        # Restore pristine classes if a prior test already patched.
        for name in ("ConvertPILImage", "ConvertBoxes"):
            cls = getattr(deim_transforms, name)
            if getattr(cls, "_harchoc_tv21_transform", False):
                del cls.transform
                del cls._harchoc_tv21_transform

        from harchoc.deim_tv_compat import apply_deim_torchvision_compat

        apply_deim_torchvision_compat()
        for name in ("ConvertPILImage", "ConvertBoxes"):
            cls = getattr(deim_transforms, name)
            self.assertTrue(getattr(cls, "_harchoc_tv21_transform", False))
            from PIL import Image as PILImage

            if name == "ConvertPILImage":
                out = cls()(PILImage.new("RGB", (8, 8)))
                self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
