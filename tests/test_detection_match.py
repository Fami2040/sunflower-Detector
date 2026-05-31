import unittest

from harchoc.detection_match import (
    _as_xyxy,
    _extract_boxes,
    _iou_xyxy,
    match_counts_for_threshold,
)


class DetectionMatchTests(unittest.TestCase):
    def test_iou_xyxy_identical(self) -> None:
        box = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(_iou_xyxy(box, box), 1.0)

    def test_iou_xyxy_disjoint(self) -> None:
        self.assertEqual(_iou_xyxy((0, 0, 1, 1), (5, 5, 6, 6)), 0.0)

    def test_as_xyxy_xywh(self) -> None:
        self.assertEqual(_as_xyxy([0, 0, 10, 10]), (0.0, 0.0, 10.0, 10.0))

    def test_extract_boxes_from_record(self) -> None:
        rec = {"annotations": [{"bbox": [0, 0, 10, 10], "category_id": 2, "score": 0.5}]}
        boxes = _extract_boxes(rec, key="annotations")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["bbox"], (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(boxes[0]["category_id"], 2)
        self.assertAlmostEqual(float(boxes[0]["score"]), 0.5)

    def test_match_counts_for_threshold(self) -> None:
        gt = {"images": [{"image_id": "a", "annotations": [{"bbox": [0, 0, 10, 10], "category_id": 0}]}]}
        preds = {
            "images": [
                {
                    "image_id": "a",
                    "detections": [
                        {"bbox": [0, 0, 10, 10], "category_id": 0, "score": 0.9},
                        {"bbox": [20, 20, 30, 30], "category_id": 0, "score": 0.8},
                    ],
                }
            ]
        }
        c = match_counts_for_threshold(gt=gt, preds=preds, conf_thr=0.5, iou_thr=0.5)
        self.assertEqual(c, {"tp": 1, "fp": 1, "fn": 0})


if __name__ == "__main__":
    unittest.main()
