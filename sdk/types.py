from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target3D:
    """Stable 3D target output for base-coordinate mode."""

    id: int | None
    label: str
    conf: float
    x: float
    y: float
    z: float
