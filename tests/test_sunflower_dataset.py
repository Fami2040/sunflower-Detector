import unittest

from harchoc.sunflower_dataset import (
    CLASS_ID_ABORTED,
    CLASS_ID_DEVELOPED,
    CLASS_NAMES,
    CLASS_NAMES_DICT,
)


class SunflowerDatasetTests(unittest.TestCase):
    def test_class_constants(self) -> None:
        self.assertEqual(CLASS_NAMES, ("developed", "aborted"))
        self.assertEqual(CLASS_NAMES_DICT[CLASS_ID_DEVELOPED], "developed")
        self.assertEqual(CLASS_NAMES_DICT[CLASS_ID_ABORTED], "aborted")


if __name__ == "__main__":
    unittest.main()
