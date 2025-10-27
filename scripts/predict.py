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

from basedetect.paths import ensure_runtime_dirs, outputs_dir, pretrained_dir, project_root, runs_dir


LOGGER = logging.getLogger(__name__)
YELLOW = "\033[33m"
RESET = "\033[0m"


def _warn_fallback(target: str) -> None:
    message = (
        f"{YELLOW}⚠️ 警告：未找到训练权重，已回退到预训练模型 {target}。\n"
        f"⚠️ Warning: Trained weights unavailable. Falling back to pretrained model {target}.{RESET}"
    )
    LOGGER.warning(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BaseDetect tracking on a video source.")
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
    parser.add_argument("--device", default="auto", help="Device passed to Ultralytics inference.")
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
    parser.set_defaults(save=True)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames with OpenCV (may require a GUI session).",
    )
    return parser.parse_args()


def resolve_weights(weights: str) -> str:
    if weights != "auto":
        path = Path(weights).expanduser()
        if path.exists():
            return str(path)
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

    weights = resolve_weights(args.weights)
    model = YOLO(weights)

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source {source!r}.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640

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

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
