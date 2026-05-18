from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from statistics import median
from typing import Any

from .types import Target3D


@dataclass
class FrameWindow:
    """Fixed-size frame history queue."""

    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("FrameWindow size must be positive.")
        self._frames: deque[Any] = deque(maxlen=self.size)

    def clear(self) -> None:
        self._frames.clear()

    def push(self, item: Any) -> None:
        self._frames.append(item)

    @property
    def ready_count(self) -> int:
        return len(self._frames)

    def snapshot(self) -> list[Any]:
        return list(self._frames)


class StatusStabilizer:
    """Stabilize status labels by slot-wise voting across recent frames."""

    def __init__(self, *, vote_threshold: int) -> None:
        if vote_threshold <= 0:
            raise ValueError("vote_threshold must be positive.")
        self.vote_threshold = vote_threshold

    def stabilize(self, history: list[list[str]]) -> list[str]:
        if not history:
            return []

        max_slots = max((len(frame) for frame in history), default=0)
        stable: list[str] = []

        for idx in range(max_slots):
            slot_labels = [frame[idx] for frame in history if len(frame) > idx]
            if not slot_labels:
                continue
            label, count = Counter(slot_labels).most_common(1)[0]
            if count >= self.vote_threshold:
                stable.append(label)

        return stable


class CoordStabilizer:
    """Stabilize 3D coordinates by label-wise temporal aggregation."""

    def __init__(
        self,
        *,
        min_votes: int,
        smoothing: str,
        ema_alpha: float,
        max_jump: float,
    ) -> None:
        if min_votes <= 0:
            raise ValueError("min_votes must be positive.")
        self.min_votes = min_votes
        self.smoothing = smoothing
        self.ema_alpha = ema_alpha
        self.max_jump = max_jump
        self._last_output: dict[str, tuple[float, float, float]] = {}

    def reset(self) -> None:
        self._last_output.clear()

    def stabilize(self, history: list[list[Target3D]]) -> list[Target3D]:
        if not history:
            return []

        grouped: dict[tuple[str, int | None], list[Target3D]] = {}
        for frame_targets in history:
            for target in frame_targets:
                key = (target.label, target.id)
                grouped.setdefault(key, []).append(target)

        active_keys: set[str] = set()
        stable: list[Target3D] = []
        for key, items in grouped.items():
            if len(items) < self.min_votes:
                continue

            label, track_id = key
            state_key = f"{label}:{track_id}"
            active_keys.add(state_key)

            if self.smoothing == "median":
                x = float(median(t.x for t in items))
                y = float(median(t.y for t in items))
                z = float(median(t.z for t in items))
            else:
                x, y, z = self._windowed_ema(samples=items)

            x, y, z = self._clamp_jump(state_key, x, y, z)

            latest = max(items, key=lambda t: t.conf)
            stable.append(
                Target3D(
                    id=track_id if track_id is not None else latest.id,
                    label=label,
                    conf=latest.conf,
                    x=x,
                    y=y,
                    z=z,
                )
            )

        stale = [k for k in self._last_output if k not in active_keys]
        for k in stale:
            del self._last_output[k]

        stable.sort(key=lambda target: (target.label, -1 if target.id is None else target.id))
        return stable

    def _windowed_ema(self, samples: list[Target3D]) -> tuple[float, float, float]:
        x, y, z = samples[0].x, samples[0].y, samples[0].z
        for s in samples[1:]:
            x = self.ema_alpha * s.x + (1.0 - self.ema_alpha) * x
            y = self.ema_alpha * s.y + (1.0 - self.ema_alpha) * y
            z = self.ema_alpha * s.z + (1.0 - self.ema_alpha) * z
        return x, y, z

    def _clamp_jump(
        self, state_key: str, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        prev = self._last_output.get(state_key)
        if prev is None or self.max_jump <= 0.0:
            self._last_output[state_key] = (x, y, z)
            return x, y, z

        dx, dy, dz = x - prev[0], y - prev[1], z - prev[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist <= self.max_jump:
            self._last_output[state_key] = (x, y, z)
            return x, y, z

        scale = self.max_jump / dist
        clamped = (prev[0] + dx * scale, prev[1] + dy * scale, prev[2] + dz * scale)
        self._last_output[state_key] = clamped
        return clamped
