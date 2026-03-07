"""3D coordinate estimation via similar-triangle (pinhole camera model).

Given a detected bounding box and the known physical size of the target,
estimate the target's 3D position relative to the camera.

Output coordinate frame (user convention):
    X — right
    Y — forward (depth)
    Z — up

Relationship to the standard OpenCV camera coordinate frame:
    X_user =  X_cam
    Y_user =  Z_cam
    Z_user = -Y_cam
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters (pixel-level)."""

    fx: float  # focal length in x (pixels)
    fy: float  # focal length in y (pixels)
    cx: float  # principal point x (pixels)
    cy: float  # principal point y (pixels)
    image_width: int
    image_height: int


@dataclass(frozen=True)
class TargetSpec:
    """Physical dimensions and estimation strategy for the target object."""

    base_width_m: float  # real width in meters
    base_height_m: float  # real height in meters
    distance_method: str  # "width" | "height" | "average"


@dataclass(frozen=True)
class Position3D:
    """A 3D position in the user coordinate frame (X-right, Y-forward, Z-up)."""

    x: float  # meters, positive = right
    y: float  # meters, positive = forward (depth)
    z: float  # meters, positive = up

    def format(self, precision: int = 3) -> str:
        """Human-readable string for logging."""
        return (
            f"X={self.x:+.{precision}f}m  "
            f"Y={self.y:.{precision}f}m  "
            f"Z={self.z:+.{precision}f}m"
        )


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def _cm_to_m(value_cm: float) -> float:
    """Convert centimeters to meters."""
    return value_cm / 100.0


def _fov_to_focal_length(image_size_px: int, fov_deg: float) -> float:
    """Derive focal length (pixels) from field-of-view angle and image size.

    f = (image_size / 2) / tan(fov / 2)
    """
    half_fov_rad = math.radians(fov_deg / 2.0)
    return (image_size_px / 2.0) / math.tan(half_fov_rad)


def load_camera_config(config_path: str | Path) -> tuple[CameraIntrinsics, TargetSpec]:
    """Parse *camera.yaml* and return intrinsics + target spec.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    (CameraIntrinsics, TargetSpec)

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If required fields are missing or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    # ---- camera section ---------------------------------------------------
    cam: dict[str, Any] = raw.get("camera", {})
    image_width: int = int(cam["image_width"])
    image_height: int = int(cam["image_height"])

    # Prefer explicit fx/fy if provided; otherwise derive from FOV.
    if "fx" in cam and "fy" in cam:
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        LOGGER.info("Using calibrated focal lengths: fx=%.1f fy=%.1f", fx, fy)
    else:
        hfov = float(cam["horizontal_fov"])
        fx = _fov_to_focal_length(image_width, hfov)
        # Derive vertical FOV from horizontal FOV + aspect ratio so that
        # fy accounts for non-square pixels / aspect ratio correctly.
        vfov = 2.0 * math.degrees(
            math.atan(math.tan(math.radians(hfov / 2.0)) * image_height / image_width)
        )
        fy = _fov_to_focal_length(image_height, vfov)
        LOGGER.info(
            "Derived focal lengths from FOV (h=%.1f° v=%.1f°): fx=%.1f fy=%.1f",
            hfov,
            vfov,
            fx,
            fy,
        )

    cx = float(cam.get("cx", image_width / 2.0))
    cy = float(cam.get("cy", image_height / 2.0))

    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_width=image_width,
        image_height=image_height,
    )

    # ---- target section ---------------------------------------------------
    tgt: dict[str, Any] = raw.get("target", {})
    base_width_cm = float(tgt["base_width"])
    base_height_cm = float(tgt["base_height"])
    distance_method = str(tgt.get("distance_method", "average")).lower()

    if distance_method not in ("width", "height", "average"):
        raise ValueError(
            f"Invalid distance_method '{distance_method}'. "
            "Expected 'width', 'height', or 'average'."
        )

    target = TargetSpec(
        base_width_m=_cm_to_m(base_width_cm),
        base_height_m=_cm_to_m(base_height_cm),
        distance_method=distance_method,
    )

    LOGGER.info(
        "Target spec: width=%.1fcm height=%.1fcm method=%s",
        base_width_cm,
        base_height_cm,
        distance_method,
    )

    return intrinsics, target


# ---------------------------------------------------------------------------
# 3D estimation
# ---------------------------------------------------------------------------


def estimate_distance(
    intrinsics: CameraIntrinsics,
    target: TargetSpec,
    bbox_width_px: float,
    bbox_height_px: float,
) -> float:
    """Estimate distance (meters) to the target using similar triangles.

    D = (real_size * focal_length) / pixel_size
    """
    estimates: list[float] = []

    if target.distance_method in ("width", "average"):
        if bbox_width_px > 0:
            d_w = (target.base_width_m * intrinsics.fx) / bbox_width_px
            estimates.append(d_w)
        else:
            LOGGER.warning("bbox_width_px is zero; skipping width-based estimate")

    if target.distance_method in ("height", "average"):
        if bbox_height_px > 0:
            d_h = (target.base_height_m * intrinsics.fy) / bbox_height_px
            estimates.append(d_h)
        else:
            LOGGER.warning("bbox_height_px is zero; skipping height-based estimate")

    if not estimates:
        LOGGER.error("Cannot estimate distance: no valid bbox dimension")
        return 0.0

    return sum(estimates) / len(estimates)


def pixel_to_3d(
    intrinsics: CameraIntrinsics,
    target: TargetSpec,
    bbox_cx_px: float,
    bbox_cy_px: float,
    bbox_width_px: float,
    bbox_height_px: float,
) -> Position3D:
    """Convert a bounding-box centre + size to a 3D position.

    Steps
    -----
    1. Estimate depth *D* via :func:`estimate_distance`.
    2. Back-project the pixel centre to OpenCV camera coordinates::

           X_cam = (cx_px - cx) * D / fx
           Y_cam = (cy_px - cy) * D / fy
           Z_cam = D

    3. Convert to user coordinate frame (X-right, Y-forward, Z-up)::

           X =  X_cam
           Y =  Z_cam
           Z = -Y_cam

    Parameters
    ----------
    intrinsics : CameraIntrinsics
    target : TargetSpec
    bbox_cx_px, bbox_cy_px : float
        Bounding-box centre in pixel coordinates.
    bbox_width_px, bbox_height_px : float
        Bounding-box width and height in pixels.

    Returns
    -------
    Position3D
    """
    distance = estimate_distance(intrinsics, target, bbox_width_px, bbox_height_px)

    # Back-project to OpenCV camera frame
    x_cam = (bbox_cx_px - intrinsics.cx) * distance / intrinsics.fx
    y_cam = (bbox_cy_px - intrinsics.cy) * distance / intrinsics.fy
    z_cam = distance

    # OpenCV camera frame → user frame (X-right, Y-forward, Z-up)
    return Position3D(x=x_cam, y=z_cam, z=-y_cam)
