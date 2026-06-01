"""
Sunflower seed detection dataset (sunflower-cvat-1093).

Canonical class terminology for training, eval, and reports — use only these names.
"""

from __future__ import annotations

CLASS_ID_DEVELOPED = 0
CLASS_ID_ABORTED = 1

CLASS_NAMES: tuple[str, str] = ("developed", "aborted")

CLASS_NAMES_DICT: dict[int, str] = {
    CLASS_ID_DEVELOPED: "developed",
    CLASS_ID_ABORTED: "aborted",
}

# Ultralytics YOLO label line: ``<class_id> <cx> <cy> <w> <h>`` (normalized 0–1).
YOLO_LABEL_FORMAT = "class_id cx cy w h"
