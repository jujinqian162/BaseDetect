"""Dataset bootstrap utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .paths import datasets_dir


DEMO_DATASET_NAME = "demo"
_SPLIT_IMAGE_COUNTS: Dict[str, int] = {"train": 20, "valid": 5, "test": 5}
_IMAGE_SIZE = (640, 640)  # (width, height)


def ensure_demo_dataset(root: Path | None = None, overwrite: bool = False) -> Path:
    """Create a lightweight synthetic dataset so training works out-of-the-box."""
    dataset_root = root or datasets_dir() / DEMO_DATASET_NAME
    success_marker = dataset_root / ".generated"

    if not overwrite and success_marker.exists():
        _ensure_demo_config(dataset_root)
        return dataset_root

    if overwrite and dataset_root.exists():
        # Avoid removing user data accidentally by requiring the caller to ask explicitly.
        for path in dataset_root.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(dataset_root.glob("**/*"), reverse=True):
            if path.is_dir():
                path.rmdir()

    for split in _SPLIT_IMAGE_COUNTS:
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for split, count in _SPLIT_IMAGE_COUNTS.items():
        for idx in range(count):
            image, bbox = _generate_sample_image(rng)

            image_name = f"{split}_{idx:03d}.jpg"
            image_path = dataset_root / split / "images" / image_name
            label_path = dataset_root / split / "labels" / image_name.replace(".jpg", ".txt")

            cv2.imwrite(str(image_path), image)
            label_path.write_text(
                f"0 {bbox['x_center']:.6f} {bbox['y_center']:.6f} "
                f"{bbox['width']:.6f} {bbox['height']:.6f}\n"
            )

    readme = dataset_root / "README.demo.txt"
    readme.write_text(
        "Synthetic demo dataset generated automatically for smoke tests.\n"
        "Images contain rectangles representing machine bases so training commands "
        "can run immediately after cloning the repository.\n"
    )

    success_marker.write_text("generated")
    _ensure_demo_config(dataset_root, force=True)
    return dataset_root


def _generate_sample_image(rng: np.random.Generator) -> tuple[np.ndarray, Dict[str, float]]:
    width, height = _IMAGE_SIZE
    image = (rng.normal(loc=127, scale=40, size=(height, width, 3))).clip(0, 255).astype(np.uint8)

    box_width = rng.integers(low=int(width * 0.15), high=int(width * 0.45))
    box_height = rng.integers(low=int(height * 0.15), high=int(height * 0.45))
    x1 = rng.integers(low=0, high=width - box_width)
    y1 = rng.integers(low=0, high=height - box_height)
    x2 = x1 + box_width
    y2 = y1 + box_height

    color = (0, 255, 0)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=3)

    bbox = {
        "x_center": (x1 + x2) / 2.0 / width,
        "y_center": (y1 + y2) / 2.0 / height,
        "width": box_width / width,
        "height": box_height / height,
    }

    return image, bbox


def _ensure_demo_config(dataset_root: Path, force: bool = False) -> Path:
    config_path = dataset_root / "data.yaml"
    if config_path.exists() and not force:
        return config_path

    def _as_posix(*parts: str) -> str:
        return str(dataset_root.joinpath(*parts).resolve()).replace("\\", "/")

    contents = (
        f"train: {_as_posix('train', 'images')}\n"
        f"val: {_as_posix('valid', 'images')}\n"
        f"test: {_as_posix('test', 'images')}\n\n"
        "nc: 1\n"
        "names: ['base']\n"
    )
    config_path.write_text(contents)
    return config_path
