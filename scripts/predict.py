from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch
from ultralytics import YOLO

from basedetect.paths import (
    camera_config,
    ensure_runtime_dirs,
    outputs_dir,
    pretrained_dir,
    project_root,
    runs_dir,
)
from basedetect.coord3d import load_camera_config, pixel_to_3d, scale_intrinsics


LOGGER = logging.getLogger(__name__)
YELLOW = "\033[33m"
RESET = "\033[0m"


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
        "--weights",
        default="auto",
        help="Path to trained weights, Ultralytics model name, or 'auto' to pick the latest run.",
    )
    parser.add_argument(
        "--source",
        default=str(project_root() / "test" / "test3.mp4"),
        help="Video file path or camera index.",
    )
    parser.add_argument(
        "--output",
        default=str(outputs_dir() / "output.avi"),
        help="Path to save the annotated video.",
    )
    parser.add_argument(
        "--device", default="auto", help="Device passed to Ultralytics inference."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Skip writing the annotated video to disk.",
    )
    parser.add_argument(
        "--unshow",
        dest="show",
        action="store_false",
        help="Disable on-screen display of annotated frames.",
    )
    parser.add_argument(
        "--camera-config",
        default=str(camera_config()),
        help="Path to camera parameter YAML for 3D coordinate estimation.",
    )
    parser.add_argument(
        "--no-coord3d",
        dest="coord3d",
        action="store_false",
        help="Disable 3D coordinate estimation.",
    )
    parser.set_defaults(save=True, show=True, coord3d=True)
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


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    # ---- logging ----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    weights = resolve_weights(args.weights)
    model = YOLO(weights)

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source {source!r}.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640

    # ---- 3D coordinate estimation setup -----------------------------------
    intrinsics = None
    target_spec = None
    if args.coord3d:
        try:
            intrinsics, target_spec = load_camera_config(args.camera_config)
            if width != intrinsics.image_width or height != intrinsics.image_height:
                LOGGER.warning(
                    "Video resolution (%dx%d) differs from camera config (%dx%d). "
                    "Scaling intrinsics to match runtime frames.",
                    width,
                    height,
                    intrinsics.image_width,
                    intrinsics.image_height,
                )
                intrinsics = scale_intrinsics(intrinsics, width, height)
            LOGGER.info("3D coordinate estimation enabled")
        except Exception:
            LOGGER.exception("Failed to load camera config; 3D estimation disabled")
            intrinsics = None
            target_spec = None

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    writer = None
    if args.save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
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
                conf=args.conf,
            )

            annotated = results[0].plot()

            # ---- 3D coordinate estimation per detection -------------------
            if intrinsics is not None and target_spec is not None:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
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

                        # Draw 3D coordinates next to the bounding box
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

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        LOGGER.info("Processed %d frames", frame_idx)


if __name__ == "__main__":
    main()
