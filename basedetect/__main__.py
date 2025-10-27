from __future__ import annotations

from basedetect.datasets import ensure_demo_dataset
from basedetect.paths import ensure_runtime_dirs


def main() -> None:
    ensure_runtime_dirs()
    ensure_demo_dataset()
    print("BaseDetect environment ready. Try `uv run scripts/predict.py`.")


if __name__ == "__main__":
    main()
