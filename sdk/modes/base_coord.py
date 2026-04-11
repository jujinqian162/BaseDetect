from __future__ import annotations

from typing import Any

from basedetect.coord3d import (
    CameraIntrinsics,
    TargetSpec,
    load_camera_config,
    pixel_to_3d,
    scale_intrinsics,
)

from ..types import Target3D

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


class BaseCoordMode(ModePlugin):
    def __init__(self, *, camera_yaml: str | None, coord3d: bool) -> None:
        self.coord3d = coord3d
        self._intrinsics: CameraIntrinsics | None = None
        self._target: TargetSpec | None = None

        if self.coord3d and camera_yaml is not None:
            intrinsics, target = load_camera_config(camera_yaml)
            self._intrinsics = intrinsics
            self._target = target

    def parse_observation(self, result: Any, frame_shape: tuple[int, int]) -> list[Target3D]:
        boxes = getattr(result, "boxes", None)
        count = _boxes_count(boxes)
        if boxes is None or count == 0:
            return []

        names: dict[int, str] = getattr(result, "names", {}) or {}
        frame_h, frame_w = frame_shape

        intrinsics = self._intrinsics
        target = self._target
        if self.coord3d and intrinsics is not None and target is not None:
            intrinsics = scale_intrinsics(intrinsics, frame_w, frame_h)

        targets: list[Target3D] = []
        has_cls = getattr(boxes, "cls", None) is not None

        for i in range(count):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_cx = (x1 + x2) / 2.0
            bbox_cy = (y1 + y2) / 2.0

            if has_cls:
                cls_idx = int(boxes.cls[i])
                label = str(names.get(cls_idx, cls_idx))
            else:
                label = "base"

            conf = float(boxes.conf[i]) if getattr(boxes, "conf", None) is not None else 0.0
            track_id = int(boxes.id[i]) if getattr(boxes, "id", None) is not None else None

            if self.coord3d and intrinsics is not None and target is not None:
                pos = pixel_to_3d(
                    intrinsics,
                    target,
                    bbox_cx,
                    bbox_cy,
                    bbox_w,
                    bbox_h,
                )
                x, y, z = pos.x, pos.y, pos.z
            else:
                x, y, z = 0.0, 0.0, 0.0

            targets.append(
                Target3D(
                    id=track_id,
                    label=label,
                    conf=conf,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                )
            )

        return targets
