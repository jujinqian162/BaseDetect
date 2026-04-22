from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatusTarget:
    """Per-frame status-mode detection with pixel-space box info."""

    id: int | None
    label: str
    conf: float
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class BaseCoordTarget:
    """Per-frame base-coord detection with both pixel and 3D info."""

    id: int | None
    label: str
    conf: float
    cx: float
    cy: float
    width: float
    height: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Target3D:
    """Stable 3D target output for base-coordinate mode."""

    id: int | None
    label: str
    conf: float
    x: float
    y: float
    z: float
