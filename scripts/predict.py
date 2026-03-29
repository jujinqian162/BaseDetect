from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import logging
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch
import yaml
from ultralytics import YOLO

from basedetect.paths import (
    camera_config,
    ensure_runtime_dirs,
    outputs_dir,
    pretrained_dir,
    project_root,
    runs_dir,
)
from basedetect.coord3d import (
    load_apriltag_config,
    load_camera_config,
    pixel_to_3d,
    scale_intrinsics,
)


LOGGER = logging.getLogger(__name__)
YELLOW = "\033[33m"
RESET = "\033[0m"

DEFAULT_MODE = "base coord"
DEFAULT_OPTIONS: dict[str, Any] = {
    "type": DEFAULT_MODE,
    "weights": "auto",
    "source": str(project_root() / "test" / "test3.mp4"),
    "output": str(outputs_dir() / "output.avi"),
    "device": "auto",
    "conf": 0.25,
    "save": True,
    "show": True,
    "camera_config": str(camera_config()),
    "coord3d": True,
    "apriltag": False,
    "apriltag_family": None,
    "apriltag_size": None,
}


def _is_pretrained(path_or_name: Path | str) -> bool:
    if isinstance(path_or_name, Path):
        try:
            return path_or_name.resolve().is_relative_to(pretrained_dir().resolve())
        except AttributeError:
            try:
                path_or_name.resolve().relative_to(pretrained_dir().resolve())
                return True
            except ValueError:
                return False
    candidate = Path(str(path_or_name)).name
    return candidate.startswith("yolov")


def _warn_fallback(target: str) -> None:
    message = (
        f"{YELLOW}⚠️ 警告：未找到训练权重，已回退到预训练模型 {target}。\n"
        f"⚠️ Warning: Trained weights unavailable. Falling back to pretrained model {target}.{RESET}"
    )
    LOGGER.warning(message)


def _warn_pretrained(target: str) -> None:
    message = (
        f"{YELLOW}⚠️ 警告：正在使用预训练模型 {target}。\n"
        f"⚠️ Warning: Using pretrained model {target}.{RESET}"
    )
    LOGGER.warning(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BaseDetect tracking on a video source."
    )
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "predict.yaml"),
        help="Top-level YAML runtime config. Missing file is ignored.",
    )
    parser.add_argument(
        "--type",
        default=None,
        help="Pipeline mode: 'status' or 'base coord'.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to trained weights, Ultralytics model name, or 'auto' to pick the latest run.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the annotated video.",
    )
    parser.add_argument(
        "--device", default=None, help="Device passed to Ultralytics inference."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for detections.",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Enable writing the annotated video to disk.",
    )
    save_group.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Skip writing the annotated video to disk.",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Enable on-screen display of annotated frames.",
    )
    show_group.add_argument(
        "--unshow",
        dest="show",
        action="store_false",
        help="Disable on-screen display of annotated frames.",
    )
    parser.add_argument(
        "--camera-config",
        default=None,
        help="Path to camera parameter YAML for 3D coordinate estimation.",
    )
    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument(
        "--coord3d",
        dest="coord3d",
        action="store_true",
        help="Enable 3D coordinate estimation.",
    )
    coord_group.add_argument(
        "--no-coord3d",
        dest="coord3d",
        action="store_false",
        help="Disable 3D coordinate estimation.",
    )
    apriltag_group = parser.add_mutually_exclusive_group()
    apriltag_group.add_argument(
        "--apriltag",
        dest="apriltag",
        action="store_true",
        help="Use AprilTag detection + pose estimation instead of YOLO bbox 3D estimation.",
    )
    apriltag_group.add_argument(
        "--no-apriltag",
        dest="apriltag",
        action="store_false",
        help="Disable AprilTag pose estimation.",
    )
    parser.add_argument(
        "--apriltag-family",
        default=None,
        help="AprilTag family used when --apriltag is enabled.",
    )
    parser.add_argument(
        "--apriltag-size",
        type=float,
        default=None,
        help="Physical AprilTag edge length in meters when --apriltag is enabled.",
    )
    parser.set_defaults(save=None, show=None, coord3d=None, apriltag=None)
    return parser.parse_args()


def resolve_weights(weights: str) -> str:
    if weights != "auto":
        path = Path(weights).expanduser()
        if path.exists():
            if _is_pretrained(path):
                _warn_pretrained(str(path))
            return str(path)
        if _is_pretrained(weights):
            _warn_pretrained(weights)
        return weights

    candidates = sorted(
        runs_dir().glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    local_default = pretrained_dir() / "yolov8n.pt"
    if local_default.exists():
        _warn_fallback(str(local_default))
        return str(local_default)
    _warn_fallback("yolov8n.pt")
    return "yolov8n.pt"


def resolve_source(source: str) -> str | int:
    path = Path(source).expanduser()
    if path.exists():
        return str(path)
    if source.isdigit():
        return int(source)
    return source


def _normalize_mode(raw_mode: str) -> str:
    normalized = str(raw_mode).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized == "status":
        return "status"
    if normalized in {"base coord", "base", "coord", "coordinate", "base coordinate"}:
        return "base coord"
    raise ValueError("Invalid type. Expected 'status' or 'base coord'.")


def _load_runtime_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Runtime config must be a mapping: {path}")

    payload = loaded.get("predict", loaded)
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config 'predict' section must be a mapping: {path}")

    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key).replace("-", "_")] = value
    return normalized


def _normalized_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        normalized[str(key).replace("-", "_")] = value
    return normalized


def _mode_section_keys(mode: str) -> tuple[str, ...]:
    if mode == "status":
        return ("status",)
    return ("base_coord", "base coord", "base")


def _resolve_option(
    cli_value: Any,
    config_value: Any,
    *,
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _resolve_runtime_options(args: argparse.Namespace) -> dict[str, Any]:
    config_data = _load_runtime_config(args.config)

    selected_type = _normalize_mode(
        _resolve_option(args.type, config_data.get("type"), default=DEFAULT_OPTIONS["type"])
    )

    merged_config: dict[str, Any] = {}
    common_section = config_data.get("common")
    if isinstance(common_section, dict):
        merged_config.update(_normalized_mapping(common_section))

    for section_key in _mode_section_keys(selected_type):
        mode_section = config_data.get(section_key)
        if isinstance(mode_section, dict):
            merged_config.update(_normalized_mapping(mode_section))
            break

    # Backward compatibility for legacy flat config keys.
    for key, value in config_data.items():
        if key in {"type", "common", "status", "base_coord", "base coord", "base"}:
            continue
        merged_config[key] = value

    options = {
        "type": selected_type,
        "weights": _resolve_option(args.weights, merged_config.get("weights"), default=DEFAULT_OPTIONS["weights"]),
        "source": _resolve_option(args.source, merged_config.get("source"), default=DEFAULT_OPTIONS["source"]),
        "output": _resolve_option(args.output, merged_config.get("output"), default=DEFAULT_OPTIONS["output"]),
        "device": _resolve_option(args.device, merged_config.get("device"), default=DEFAULT_OPTIONS["device"]),
        "conf": _resolve_option(args.conf, merged_config.get("conf"), default=DEFAULT_OPTIONS["conf"]),
        "save": bool(
            _resolve_option(args.save, merged_config.get("save"), default=DEFAULT_OPTIONS["save"])
        ),
        "show": bool(
            _resolve_option(args.show, merged_config.get("show"), default=DEFAULT_OPTIONS["show"])
        ),
        "camera_config": _resolve_option(
            args.camera_config,
            merged_config.get("camera_config"),
            default=DEFAULT_OPTIONS["camera_config"],
        ),
        "coord3d": bool(
            _resolve_option(args.coord3d, merged_config.get("coord3d"), default=DEFAULT_OPTIONS["coord3d"])
        ),
        "apriltag": bool(
            _resolve_option(args.apriltag, merged_config.get("apriltag"), default=DEFAULT_OPTIONS["apriltag"])
        ),
        "apriltag_family": _resolve_option(
            args.apriltag_family,
            merged_config.get("apriltag_family"),
            default=DEFAULT_OPTIONS["apriltag_family"],
        ),
        "apriltag_size": _resolve_option(
            args.apriltag_size,
            merged_config.get("apriltag_size"),
            default=DEFAULT_OPTIONS["apriltag_size"],
        ),
    }

    if isinstance(options["source"], str):
        source_candidate = Path(options["source"]).expanduser()
        if not source_candidate.is_absolute() and not str(options["source"]).isdigit():
            options["source"] = str(project_root() / source_candidate)

    if isinstance(options["output"], str):
        output_candidate = Path(options["output"]).expanduser()
        if not output_candidate.is_absolute():
            options["output"] = str(project_root() / output_candidate)

    if isinstance(options["camera_config"], str):
        camera_candidate = Path(options["camera_config"]).expanduser()
        if not camera_candidate.is_absolute():
            options["camera_config"] = str(project_root() / camera_candidate)

    if isinstance(options["weights"], str) and options["weights"] != "auto":
        weights_candidate = Path(options["weights"]).expanduser()
        if not weights_candidate.is_absolute() and weights_candidate.exists():
            options["weights"] = str(weights_candidate.resolve())

    if options["type"] == "status":
        options["coord3d"] = False
        options["apriltag"] = False
    return options


def _extract_status_labels(result: Any) -> list[str]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or _boxes_count(boxes) == 0:
        return []

    names: dict[int, str] = getattr(result, "names", {}) or {}
    entries: list[tuple[float, str]] = []
    has_cls = getattr(boxes, "cls", None) is not None
    count = _boxes_count(boxes)
    for i in range(count):
        if has_cls:
            cls_idx = int(boxes.cls[i])
            label = str(names.get(cls_idx, cls_idx))
        else:
            label = "unknown"
        x1, _, x2, _ = boxes.xyxy[i].tolist()
        entries.append((((x1 + x2) / 2.0), label))

    entries.sort(key=lambda item: item[0])
    return [label for _, label in entries]


def _draw_status_overlay(frame: Any, statuses: list[str]) -> None:
    status_text = f"status: {statuses}"
    cv2.putText(
        frame,
        status_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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


def _create_apriltag_detector(family: str) -> Any:
    try:
        apriltag_module = importlib.import_module("apriltag")
    except ImportError as exc:
        raise RuntimeError(
            "AprilTag mode requested, but module 'apriltag' is not available in the active environment."
        ) from exc

    detector_factory = getattr(apriltag_module, "apriltag", None)
    if detector_factory is None:
        raise RuntimeError("Loaded apriltag module does not expose apriltag().")

    return detector_factory(family, threads=max(1, os.cpu_count() or 1))


def _camera_to_user_frame(x_cam: float, y_cam: float, z_cam: float) -> tuple[float, float, float]:
    return x_cam, z_cam, -y_cam


def main() -> None:
    args = parse_args()
    runtime = _resolve_runtime_options(args)
    ensure_runtime_dirs()

    # ---- logging ----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    LOGGER.info(
        "Mode=%s save=%s show=%s coord3d=%s apriltag=%s",
        runtime["type"],
        runtime["save"],
        runtime["show"],
        runtime["coord3d"],
        runtime["apriltag"],
    )

    weights = resolve_weights(str(runtime["weights"]))
    model = YOLO(weights)

    source = resolve_source(str(runtime["source"]))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source {source!r}.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640

    # ---- coordinate estimation setup --------------------------------------
    intrinsics = None
    target_spec = None
    apriltag_detector = None
    apriltag_mirror_input = False

    if runtime["coord3d"] or runtime["apriltag"]:
        try:
            intrinsics, target_spec = load_camera_config(str(runtime["camera_config"]))
            if width != intrinsics.image_width or height != intrinsics.image_height:
                LOGGER.warning(
                    "Video resolution (%dx%d) differs from camera config (%dx%d). "
                    "Scaling intrinsics to match runtime frames.",
                    width,
                    height,
                    intrinsics.image_width,
                    intrinsics.image_height,
                )
                try:
                    intrinsics = scale_intrinsics(intrinsics, width, height)
                except Exception:
                    LOGGER.warning(
                        "Failed to scale intrinsics dynamically; keeping original calibration intrinsics."
                    )
        except Exception:
            LOGGER.exception("Failed to load camera config; coordinate estimation disabled")
            intrinsics = None
            target_spec = None

    if runtime["coord3d"] and intrinsics is not None and target_spec is not None:
        LOGGER.info("3D coordinate estimation enabled")

    if runtime["apriltag"]:
        try:
            apriltag_spec = load_apriltag_config(str(runtime["camera_config"]))
            if runtime["apriltag_family"] is None:
                runtime["apriltag_family"] = apriltag_spec.family
            if runtime["apriltag_size"] is None:
                runtime["apriltag_size"] = apriltag_spec.tag_size_m
            apriltag_mirror_input = apriltag_spec.mirror_input
            if float(runtime["apriltag_size"]) <= 0:
                raise ValueError("--apriltag-size must be positive.")
            apriltag_detector = _create_apriltag_detector(str(runtime["apriltag_family"]))
            LOGGER.info(
                "AprilTag pose estimation enabled (family=%s size=%.3fm mirror_input=%s)",
                runtime["apriltag_family"],
                float(runtime["apriltag_size"]),
                apriltag_mirror_input,
            )
        except Exception as exc:
            raise RuntimeError("Failed to initialize AprilTag pose estimation.") from exc

    if runtime["apriltag"] and (apriltag_detector is None or intrinsics is None):
        LOGGER.warning("AprilTag mode requested but detector/intrinsics unavailable; disabling AprilTag rendering.")
        runtime["apriltag"] = False

    device = str(runtime["device"])
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    writer = None
    if runtime["save"]:
        output_path = Path(str(runtime["output"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps or 25.0, (width, height))

    window_name = "BaseDetect Tracking"
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            results = model.track(
                gray_rgb,
                persist=True,
                device=device,
                conf=float(runtime["conf"]),
            )

            annotated = results[0].plot()

            if runtime["type"] == "status":
                statuses = _extract_status_labels(results[0])
                _draw_status_overlay(annotated, statuses)
                LOGGER.info("frame=%d status=%s", frame_idx, statuses)

            # ---- base bbox 3D coordinates -----------------------------------
            if runtime["coord3d"] and intrinsics is not None and target_spec is not None:
                boxes = getattr(results[0], "boxes", None)
                count = _boxes_count(boxes)
                if boxes is not None and count > 0:
                    for i in range(count):
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        bbox_cx = (x1 + x2) / 2.0
                        bbox_cy = (y1 + y2) / 2.0

                        pos = pixel_to_3d(
                            intrinsics,
                            target_spec,
                            bbox_cx,
                            bbox_cy,
                            bbox_w,
                            bbox_h,
                        )

                        track_id = int(boxes.id[i]) if boxes.id is not None else -1
                        conf = float(boxes.conf[i])

                        LOGGER.info(
                            "frame=%d track=#%d conf=%.2f "
                            "bbox=(%.0f,%.0f,%.0fx%.0f) %s",
                            frame_idx,
                            track_id,
                            conf,
                            bbox_cx,
                            bbox_cy,
                            bbox_w,
                            bbox_h,
                            pos.format(),
                        )

                        label_3d = f"({pos.x:+.2f},{pos.y:.2f},{pos.z:+.2f})"
                        text_x = int(x2) + 6
                        text_y = int(y1) + 22
                        cv2.putText(
                            annotated,
                            label_3d,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

            # ---- AprilTag pose coordinates ----------------------------------
            if runtime["apriltag"]:
                if apriltag_detector is None or intrinsics is None:
                    continue
                apriltag_input = cv2.flip(gray, 1) if apriltag_mirror_input else gray
                detections = apriltag_detector.detect(apriltag_input)
                for detection in detections:
                    pose = apriltag_detector.estimate_tag_pose(
                        detection,
                        float(runtime["apriltag_size"]),
                        intrinsics.fx,
                        intrinsics.fy,
                        intrinsics.cx,
                        intrinsics.cy,
                    )

                    center = detection["center"]
                    corners = detection["lb-rb-rt-lt"]
                    if apriltag_mirror_input:
                        center = center.copy()
                        center[0] = width - 1 - center[0]
                        corners = corners.copy()
                        corners[:, 0] = width - 1 - corners[:, 0]
                    tag_id = int(detection["id"])
                    margin = float(detection.get("margin", 0.0))
                    reproj_error = float(pose.get("error", 0.0))
                    translation = pose["t"].reshape(-1)
                    x_user, y_user, z_user = _camera_to_user_frame(
                        float(translation[0]),
                        float(translation[1]),
                        float(translation[2]),
                    )

                    LOGGER.info(
                        "frame=%d tag_id=%d center=(%.1f,%.1f) margin=%.2f reproj=%.4f "
                        "X=%+.3fm Y=%.3fm Z=%+.3fm",
                        frame_idx,
                        tag_id,
                        float(center[0]),
                        float(center[1]),
                        margin,
                        reproj_error,
                        x_user,
                        y_user,
                        z_user,
                    )

                    poly = corners.astype(int).reshape((-1, 1, 2))
                    center_xy = tuple(center.astype(int))
                    cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
                    cv2.circle(annotated, center_xy, 4, (0, 0, 255), -1)
                    cv2.putText(
                        annotated,
                        f"tag#{tag_id} ({x_user:+.2f},{y_user:.2f},{z_user:+.2f})",
                        (center_xy[0] + 8, center_xy[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            if writer is not None:
                writer.write(annotated)

            if runtime["show"]:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if runtime["show"]:
            cv2.destroyAllWindows()
        LOGGER.info("Processed %d frames", frame_idx)


if __name__ == "__main__":
    main()
