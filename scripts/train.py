from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from basedetect.paths import ensure_runtime_dirs, pretrained_dir, project_root, runs_dir


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


def warn_pretrained_usage(source: Path | str) -> None:
    LOGGER.warning(
        "%s⚠️ 警告：正在使用预训练权重 %s。\n⚠️ Warning: Training with pretrained weights %s.%s",
        YELLOW,
        source,
        source,
        RESET,
    )


DEFAULT_CONFIG = "configs/data-initial.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model for BaseDetect.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the dataset YAML file.")
    parser.add_argument(
        "--model",
        default=str(pretrained_dir() / "yolov8n.pt"),
        help="Initial weights or Ultralytics model checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Images per batch.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", default="auto", help="Device selection passed to Ultralytics.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument(
        "--project",
        default=str(runs_dir()),
        help="Output directory for Ultralytics experiment artifacts.",
    )
    parser.add_argument("--name", default="basedetect", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent checkpoint.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    config_path = resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    model_path = resolve_path(args.model)
    if model_path.exists():
        model_source = str(model_path)
        using_pretrained = _is_pretrained(model_path)
    else:
        model_source = args.model
        using_pretrained = _is_pretrained(model_source)

    if using_pretrained:
        warn_pretrained_usage(model_source)

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_source)
    model.train(
        data=str(config_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(Path(args.project)),
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
