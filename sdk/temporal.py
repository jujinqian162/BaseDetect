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
    ) -> None:
        if min_votes <= 0:
            raise ValueError("min_votes must be positive.")
        self.min_votes = min_votes
        self.smoothing = smoothing
        self.ema_alpha = ema_alpha
        self._ema_state: dict[str, tuple[float, float, float]] = {}

    def reset(self) -> None:
        self._ema_state.clear()

    def stabilize(self, history: list[list[Target3D]]) -> list[Target3D]:
        if not history:
            return []

        grouped: dict[tuple[str, int | None], list[Target3D]] = {}
        for frame_targets in history:
            for target in frame_targets:
                key = (target.label, target.id)
                grouped.setdefault(key, []).append(target)

        stable: list[Target3D] = []
        for key, items in grouped.items():
            if len(items) < self.min_votes:
                continue

            label, track_id = key

            if self.smoothing == "median":
                x = float(median(t.x for t in items))
                y = float(median(t.y for t in items))
                z = float(median(t.z for t in items))
            else:
                x, y, z = self._ema_label(label=label, track_id=track_id, samples=items)

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

        stable.sort(key=lambda target: (target.label, -1 if target.id is None else target.id))
        return stable

    def _ema_label(
        self,
        *,
        label: str,
        track_id: int | None,
        samples: list[Target3D],
    ) -> tuple[float, float, float]:
        sample = samples[-1]
        current = (sample.x, sample.y, sample.z)
        state_key = f"{label}:{track_id}"
        previous = self._ema_state.get(state_key)

        if previous is None:
            self._ema_state[state_key] = current
            return current

        ax = self.ema_alpha * current[0] + (1.0 - self.ema_alpha) * previous[0]
        ay = self.ema_alpha * current[1] + (1.0 - self.ema_alpha) * previous[1]
        az = self.ema_alpha * current[2] + (1.0 - self.ema_alpha) * previous[2]
        updated = (ax, ay, az)
        self._ema_state[state_key] = updated
        return updated
