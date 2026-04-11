from __future__ import annotations

from typing import Any

from .base import ModePlugin


def _boxes_count(boxes: Any) -> int:
    if boxes is None:
        return 0
    try:
        return int(len(boxes))
    except TypeError:
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            return 0
        return int(len(xyxy))


class StatusMode(ModePlugin):
    def __init__(self, *, reverse_order: bool = False) -> None:
        self.reverse_order = reverse_order

    def parse_observation(self, result: Any, frame_shape: tuple[int, int]) -> list[str]:
        del frame_shape
        boxes = getattr(result, "boxes", None)
        if boxes is None or _boxes_count(boxes) == 0:
            return []

        names: dict[int, str] = getattr(result, "names", {}) or {}
        has_cls = getattr(boxes, "cls", None) is not None
        count = _boxes_count(boxes)
        entries: list[tuple[float, str]] = []

        for i in range(count):
            if has_cls:
                cls_idx = int(boxes.cls[i])
                label = str(names.get(cls_idx, cls_idx))
            else:
                label = "unknown"

            x1, _, x2, _ = boxes.xyxy[i].tolist()
            entries.append((((x1 + x2) / 2.0), label))

        entries.sort(key=lambda item: item[0], reverse=self.reverse_order)
        return [label for _, label in entries]
