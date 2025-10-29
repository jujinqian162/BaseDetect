from __future__ import annotations

from basedetect.paths import ensure_runtime_dirs, project_root


def main() -> None:
    ensure_runtime_dirs()
    default_config = project_root() / "configs" / "data-initial.yaml"
    print(
        "BaseDetect environment ready. Default dataset config: "
        f"{default_config}. Try `uv run scripts/train.py`."
    )


if __name__ == "__main__":
    main()
