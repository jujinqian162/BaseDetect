"""Utility helpers for resolving project paths at runtime."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[1]


def configs_dir() -> Path:
    return project_root() / "configs"


def datasets_dir() -> Path:
    return project_root() / "datasets"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def outputs_dir() -> Path:
    return artifacts_dir() / "outputs"


def runs_dir() -> Path:
    return artifacts_dir() / "runs"


def weights_dir() -> Path:
    return project_root() / "weights"


def pretrained_dir() -> Path:
    return weights_dir() / "pretrained"


def ensure_runtime_dirs() -> None:
    """Create directories produced during normal workflows."""
    outputs_dir().mkdir(parents=True, exist_ok=True)
    runs_dir().mkdir(parents=True, exist_ok=True)
    pretrained_dir().mkdir(parents=True, exist_ok=True)
