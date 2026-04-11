from __future__ import annotations

from ..config import ProfileSettings

from .base import ModePlugin
from .base_coord import BaseCoordMode
from .status import StatusMode


def create_mode(profile: ProfileSettings) -> ModePlugin:
    if profile.mode == "status":
        reverse = profile.order == "right_to_left"
        return StatusMode(reverse_order=reverse)

    return BaseCoordMode(
        camera_yaml=str(profile.camera_yaml) if profile.camera_yaml is not None else None,
        coord3d=profile.coord3d,
    )
