from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import ProfileSettings, SDKSettings, load_settings, resolve_profile
from .modes import create_mode
from .temporal import CoordStabilizer, FrameWindow, StatusStabilizer
from .types import BaseCoordTarget, StatusTarget, Target3D


LOGGER = logging.getLogger("sdk.detector")


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


class Detector:
    """Simple SDK-style detector with profile switching and frame debouncing."""

    def __init__(self, *, config: str | Path, profile: str | None = None) -> None:
        self._settings: SDKSettings = load_settings(config)
        self._runtime = self._settings.runtime

        self._window = FrameWindow(size=self._runtime.queue_size)
        self._ready = False

        self._model: YOLO | None = None
        self._mode_plugin = None
        self._profile: ProfileSettings | None = None
        self._status_stabilizer: StatusStabilizer | None = None
        self._coord_stabilizer: CoordStabilizer | None = None

        self._debug_enabled = self._runtime.debug
        self._last_overlay: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._last_status_targets: list[StatusTarget] = []
        self._last_base_coord_targets: list[BaseCoordTarget] = []
        self._frame_index = 0

        self.switch_profile(profile)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def mode(self) -> str:
        assert self._profile is not None
        return self._profile.mode

    @property
    def profile_name(self) -> str:
        assert self._profile is not None
        return self._profile.name

    def switch_profile(self, profile: str | None) -> None:
        selected = resolve_profile(self._settings, profile)
        self._profile = selected

        self._mode_plugin = create_mode(selected)

        self._status_stabilizer = StatusStabilizer(
            vote_threshold=selected.vote_threshold,
        )
        self._coord_stabilizer = CoordStabilizer(
            min_votes=selected.min_votes,
            smoothing=selected.smoothing,
            ema_alpha=selected.ema_alpha,
        )

        self._window.clear()
        self._coord_stabilizer.reset()
        self._ready = False
        self._last_status_targets = []
        self._last_base_coord_targets = []
        self._frame_index = 0

        self._model = YOLO(selected.weights)
        names = getattr(self._model, "names", {})
        class_preview = []
        class_count = 0
        if isinstance(names, dict):
            class_count = len(names)
            class_preview = [
                str(v) for _, v in sorted(names.items(), key=lambda kv: kv[0])[:8]
            ]

        LOGGER.info(
            "Loaded profile=%s mode=%s weights=%s conf=%.2f classes=%d preview=%s",
            selected.name,
            selected.mode,
            selected.weights,
            selected.conf,
            class_count,
            class_preview,
        )
        if selected.mode == "status" and class_count > 0 and class_count < 2:
            LOGGER.warning(
                "Status mode is using a model with %d class (%s). "
                "This often means you loaded a base-only model, so status output may stay empty.",
                class_count,
                class_preview,
            )

    def detect(self, frame: np.ndarray) -> list[str] | list[Target3D]:
        if self._profile is None or self._mode_plugin is None or self._model is None:
            raise RuntimeError("Detector not initialized correctly.")

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        start = time.perf_counter()
        source = self._preprocess(frame)
        device = self._resolve_device(self._runtime.device)
        results = self._model.track(
            source, persist=True, device=device, conf=self._profile.conf
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        detections_count = _boxes_count(boxes)
        observation = self._mode_plugin.parse_observation(result, frame.shape[:2])
        if self._profile.mode == "status":
            self._last_status_targets = self._extract_status_targets(result)
            self._last_base_coord_targets = []
        else:
            self._last_status_targets = []
            self._last_base_coord_targets = self._extract_base_coord_targets(
                result=result,
                observation=observation,
            )

        self._window.push(observation)
        self._ready = self._window.ready_count >= self._runtime.warmup_frames

        empty_reason = ""

        if not self._ready:
            output = self._empty_output()
            empty_reason = f"warmup: queue={self._window.ready_count}/{self._runtime.warmup_frames}"
        else:
            history = self._window.snapshot()
            if self._profile.mode == "status":
                assert self._status_stabilizer is not None
                output = self._status_stabilizer.stabilize(history)
                if not output:
                    empty_reason = self._explain_status_empty(history)
            else:
                assert self._coord_stabilizer is not None
                output = self._coord_stabilizer.stabilize(history)
                if not output:
                    empty_reason = self._explain_coord_empty(history)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._refresh_debug(
            frame=frame,
            result=result,
            output=output,
            elapsed_ms=elapsed_ms,
            raw_observation=observation,
            detections_count=detections_count,
            empty_reason=empty_reason,
        )

        if self._debug_enabled:
            if self._profile.mode == "status":
                LOGGER.info(
                    "frame=%d profile=%s mode=status det=%d ready=%s raw=%s stable=%s%s",
                    self._frame_index,
                    self.profile_name,
                    detections_count,
                    self.ready,
                    observation,
                    output,
                    f" reason={empty_reason}" if empty_reason else "",
                )
            else:
                LOGGER.info(
                    "frame=%d profile=%s mode=base_coord det=%d ready=%s raw_n=%d stable_n=%d%s",
                    self._frame_index,
                    self.profile_name,
                    detections_count,
                    self.ready,
                    len(observation),
                    len(output),
                    f" reason={empty_reason}" if empty_reason else "",
                )

        self._frame_index += 1
        return output

    def debug_overlay(self) -> np.ndarray | None:
        if not self._debug_enabled:
            return None
        if self._last_overlay is None:
            return None
        return self._last_overlay.copy()

    def debug_info(self) -> dict[str, Any]:
        if not self._debug_enabled:
            return {}
        return dict(self._last_info)

    def latest_status_targets(self) -> list[StatusTarget]:
        return list(self._last_status_targets)

    def latest_base_coord_targets(self) -> list[BaseCoordTarget]:
        return list(self._last_base_coord_targets)

    def _extract_status_targets(self, result: Any) -> list[StatusTarget]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or _boxes_count(boxes) == 0:
            return []

        names: dict[int, str] = getattr(result, "names", {}) or {}
        has_cls = getattr(boxes, "cls", None) is not None
        has_conf = getattr(boxes, "conf", None) is not None
        has_id = getattr(boxes, "id", None) is not None
        count = _boxes_count(boxes)

        targets: list[StatusTarget] = []
        for i in range(count):
            if has_cls:
                cls_idx = int(boxes.cls[i])
                label = str(names.get(cls_idx, cls_idx))
            else:
                label = "unknown"

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            targets.append(
                StatusTarget(
                    id=int(boxes.id[i]) if has_id else None,
                    label=label,
                    conf=float(boxes.conf[i]) if has_conf else 0.0,
                    cx=float((x1 + x2) / 2.0),
                    cy=float((y1 + y2) / 2.0),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                )
            )

        targets.sort(key=lambda item: item.cx)
        return targets

    def _extract_base_coord_targets(
        self,
        *,
        result: Any,
        observation: list[Target3D],
    ) -> list[BaseCoordTarget]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or _boxes_count(boxes) == 0:
            return []

        count = _boxes_count(boxes)
        items: list[BaseCoordTarget] = []
        for i in range(min(count, len(observation))):
            target = observation[i]
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            items.append(
                BaseCoordTarget(
                    id=target.id,
                    label=target.label,
                    conf=target.conf,
                    cx=float((x1 + x2) / 2.0),
                    cy=float((y1 + y2) / 2.0),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    x=target.x,
                    y=target.y,
                    z=target.z,
                )
            )

        items.sort(key=lambda item: item.cx)
        return items

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if not self._runtime.grayscale_input:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _resolve_device(self, configured: str) -> str:
        if configured != "auto":
            return configured
        return "0" if torch.cuda.is_available() else "cpu"

    def _empty_output(self) -> list[str] | list[Target3D]:
        return []

    def _refresh_debug(
        self,
        *,
        frame: np.ndarray,
        result: Any,
        output: list[str] | list[Target3D],
        elapsed_ms: float,
        raw_observation: list[str] | list[Target3D],
        detections_count: int,
        empty_reason: str,
    ) -> None:
        if not self._debug_enabled:
            return

        overlay = result.plot()
        header = f"profile={self.profile_name} mode={self.mode} ready={self.ready}"
        cv2.putText(
            overlay,
            header,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            header,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if self.mode == "status":
            text = f"status={output}"
        else:
            text = f"targets={len(output)}"

        cv2.putText(
            overlay,
            text,
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self._last_overlay = overlay
        self._last_info = {
            "frame_index": self._frame_index,
            "profile": self.profile_name,
            "mode": self.mode,
            "ready": self.ready,
            "queue_size": self._runtime.queue_size,
            "warmup_frames": self._runtime.warmup_frames,
            "latency_ms": round(elapsed_ms, 3),
            "frame_shape": tuple(int(v) for v in frame.shape),
            "detections_count": detections_count,
            "raw_observation": raw_observation,
            "output_size": len(output),
            "empty_reason": empty_reason,
        }

    def _explain_status_empty(self, history: list[list[str]]) -> str:
        total_labels = sum(len(frame) for frame in history)
        if total_labels == 0:
            return "no detections in queue window"

        max_slots = max((len(frame) for frame in history), default=0)
        parts: list[str] = []
        assert self._status_stabilizer is not None
        threshold = self._status_stabilizer.vote_threshold

        for idx in range(max_slots):
            labels = [frame[idx] for frame in history if len(frame) > idx]
            if not labels:
                continue
            label, count = Counter(labels).most_common(1)[0]
            parts.append(f"slot{idx}:{label} {count}/{len(labels)}<thr{threshold}")

        if not parts:
            return "no valid slot observations"
        return "; ".join(parts)

    def _explain_coord_empty(self, history: list[list[Target3D]]) -> str:
        total_targets = sum(len(frame) for frame in history)
        if total_targets == 0:
            return "no detections in queue window"

        counts = Counter(
            (target.label, target.id) for frame in history for target in frame
        )
        assert self._coord_stabilizer is not None
        threshold = self._coord_stabilizer.min_votes
        preview = []
        for (label, track_id), votes in counts.most_common(4):
            key = f"{label}#{track_id}" if track_id is not None else f"{label}#none"
            preview.append(f"{key}:{votes}<thr{threshold}")

        return "not enough votes " + ", ".join(preview)
