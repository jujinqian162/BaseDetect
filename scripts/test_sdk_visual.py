from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdk import Detector, Target3D


LOGGER = logging.getLogger("scripts.test_sdk_visual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual SDK smoke test with profile switching.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "basedetect_sdk.yaml"),
        help="SDK config path.",
    )
    parser.add_argument(
        "--source",
        default=str(REPO_ROOT / "test" / "test7.mp4"),
        help="Video source path or camera index.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Initial profile name. If omitted, use active_profile in config.",
    )
    parser.add_argument(
        "--switch-to",
        default="base_coord_competition",
        help="Profile to switch to during runtime.",
    )
    parser.add_argument(
        "--switch-frame",
        type=int,
        default=120,
        help="Frame index to trigger profile switch. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Stop after this many frames. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable OpenCV window display.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one log line every N frames.",
    )
    return parser.parse_args()


def resolve_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)

    path = Path(source).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def format_targets(targets: list[Target3D]) -> str:
    if not targets:
        return "[]"
    chunks: list[str] = []
    for target in targets:
        chunks.append(
            f"{target.label}(id={target.id},conf={target.conf:.2f},"
            f"x={target.x:+.2f},y={target.y:.2f},z={target.z:+.2f})"
        )
    return "[" + ", ".join(chunks) + "]"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    detector = Detector(config=args.config, profile=args.profile)
    source = resolve_source(str(args.source))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    LOGGER.info(
        "SDK visual test started: profile=%s mode=%s source=%s",
        detector.profile_name,
        detector.mode,
        source,
    )

    frame_idx = 0
    switched = False
    window_name = "BaseDetect SDK Visual Test"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.switch_frame >= 0 and (not switched) and frame_idx >= args.switch_frame:
                detector.switch_profile(args.switch_to)
                switched = True
                LOGGER.info(
                    "Switched profile at frame=%d -> profile=%s mode=%s",
                    frame_idx,
                    detector.profile_name,
                    detector.mode,
                )

            output = detector.detect(frame)

            if args.log_every > 0 and frame_idx % args.log_every == 0:
                if detector.mode == "status":
                    LOGGER.info(
                        "frame=%d ready=%s mode=status status=%s",
                        frame_idx,
                        detector.ready,
                        output,
                    )
                else:
                    LOGGER.info(
                        "frame=%d ready=%s mode=base_coord targets=%s",
                        frame_idx,
                        detector.ready,
                        format_targets(output),
                    )

            if not args.no_show:
                overlay = detector.debug_overlay()
                show_frame = overlay if overlay is not None else frame
                cv2.imshow(window_name, show_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames >= 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    info = detector.debug_info()
    LOGGER.info("SDK visual test finished. frames=%d final_debug=%s", frame_idx, info)


if __name__ == "__main__":
    main()
