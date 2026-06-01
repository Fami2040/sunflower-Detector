import unittest


class ResourceSnapshotTests(unittest.TestCase):
    def test_snapshot_skips_torch_when_disabled(self) -> None:
        from harchoc.resource_snapshot import snapshot_from_train_cfg

        snap = snapshot_from_train_cfg({"batch": 1, "imgsz": 1280}, include_torch=False)
        self.assertEqual(snap["batch"], 1)
        self.assertEqual(snap["imgsz"], 1280)
        self.assertEqual(snap.get("torch", {}).get("skipped"), True)


if __name__ == "__main__":
    unittest.main()

