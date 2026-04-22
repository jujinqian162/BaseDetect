"""Public SDK API for BaseDetect."""

from .detector import Detector
from .types import BaseCoordTarget, StatusTarget, Target3D

__all__ = ["BaseCoordTarget", "Detector", "StatusTarget", "Target3D"]
