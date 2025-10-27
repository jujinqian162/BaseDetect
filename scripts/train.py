from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from basedetect.datasets import ensure_demo_dataset
from basedetect.paths import ensure_runtime_dirs, pretrained_dir, project_root, runs_dir


DEFAULT_CONFIG = "configs/data-initial.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model for BaseDetect.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the dataset YAML (pass 'auto' to generate a synthetic demo dataset).",
    )
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

    if args.config == "auto":
        dataset_root = ensure_demo_dataset()
        config_path = dataset_root / "data.yaml"
    else:
        config_path = resolve_path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")

    model_path = resolve_path(args.model)
    model_source = str(model_path) if model_path.exists() else args.model

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
