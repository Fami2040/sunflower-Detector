from __future__ import annotations

import unittest
from dataclasses import dataclass

from harchoc.deploy_filters import DeployFilterConfig, filter_object_predictions


@dataclass
class _BBox:
    minx: float
    miny: float
    maxx: float
    maxy: float


@dataclass
class _Cat:
    id: int


@dataclass
class _Score:
    value: float


@dataclass
class _Pred:
    bbox: _BBox
    category: _Cat
    score: _Score


class DeployFiltersTests(unittest.TestCase):
    def test_conf_threshold_per_class(self) -> None:
        cfg = DeployFilterConfig(conf_thr_fertilized=0.5, conf_thr_unfertilized=0.3)
        preds = [
            _Pred(_BBox(0, 0, 10, 10), _Cat(0), _Score(0.6)),
            _Pred(_BBox(20, 20, 30, 30), _Cat(1), _Score(0.2)),
            _Pred(_BBox(40, 40, 50, 50), _Cat(1), _Score(0.4)),
        ]
        out = filter_object_predictions(preds, cfg)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out[0].category.id), 0)
        self.assertEqual(int(out[1].category.id), 1)

    def test_unfert_dedup_by_center(self) -> None:
        cfg = DeployFilterConfig(
            conf_thr_fertilized=0.0,
            conf_thr_unfertilized=0.0,
            unfert_vs_fert_suppress=False,
            unfert_dedup=True,
            unfert_dedup_center_ratio=2.0,
            unfert_dedup_min_pix=1.0,
        )
        preds = [
            _Pred(_BBox(0, 0, 10, 10), _Cat(1), _Score(0.9)),
            _Pred(_BBox(1, 1, 11, 11), _Cat(1), _Score(0.8)),
        ]
        out = filter_object_predictions(preds, cfg)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
