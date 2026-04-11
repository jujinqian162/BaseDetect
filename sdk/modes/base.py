from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModePlugin(ABC):
    """Per-frame mode processor interface."""

    @abstractmethod
    def parse_observation(self, result: Any, frame_shape: tuple[int, int]) -> Any:
        """Convert one YOLO result into mode-specific observation."""
