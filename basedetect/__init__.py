"""BaseDetect package initialization."""

from __future__ import annotations


def main() -> None:
    """Entry point that defers importing __main__ to avoid runpy warnings."""
    from .__main__ import main as _main

    _main()


__all__ = ["main"]
