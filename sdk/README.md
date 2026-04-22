# BaseDetect SDK

Lightweight function-call SDK designed for robotics loops (including ROS users) without requiring topic-based integration.

## Goals

- Keep API small: one `Detector` object.
- Support profile switching at runtime.
- Reuse one temporal queue/debouncing abstraction for status and 3D coordinate modes.

## Quick Start

```python
from sdk import Detector

det = Detector(config="configs/basedetect_sdk.yaml", profile="status_competition")

status = det.detect(frame)  # list[str] in status mode
status_targets = det.latest_status_targets()  # list[StatusTarget], per-frame raw boxes
if det.ready:
    # call your control logic when stable output is ready
    pass

det.switch_profile("base_coord_competition")
targets = det.detect(frame)  # list[Target3D] in base_coord mode
base_coord_targets = det.latest_base_coord_targets()  # list[BaseCoordTarget], with cx/cy + xyz
```

## Public API

- `Detector(config: str | Path, profile: str | None = None)`
- `Detector.detect(frame)`
  - returns `list[str]` when mode=`status`
  - returns `list[Target3D]` when mode=`base_coord`
- `Detector.latest_status_targets()`
  - returns `list[StatusTarget]` from latest frame (ordered by `cx`, left to right)
  - valid in `status` mode; returns empty list in `base_coord`
- `Detector.latest_base_coord_targets()`
  - returns `list[BaseCoordTarget]` from latest frame (ordered by `cx`, left to right)
  - valid in `base_coord` mode; returns empty list in `status`
- `Detector.switch_profile(name)`
- `Detector.ready` (bool)
- `Detector.mode` ("status" or "base_coord")
- `Detector.profile_name`
- `Detector.debug_overlay()` (returns overlay frame or `None`)
- `Detector.debug_info()` (returns dict or `{}`)

### Output Types

- Status mode output: `list[str]`
  - Labels are ordered by horizontal position (`left_to_right` by default, configurable).
  - Temporal stabilization uses slot-wise voting in the recent frame window.

- Status target output: `list[StatusTarget]`
  - `StatusTarget` fields:
    - `id: int | None`
    - `label: str`
    - `conf: float`
    - `cx: float`
    - `cy: float`
    - `width: float`
    - `height: float`
  - Useful for control loops that need pixel-space error (e.g., PID on `cx - target_x`).

- Base-coordinate mode output: `list[Target3D]`
  - `Target3D` fields:
    - `id: int | None` (tracker id)
    - `label: str`
    - `conf: float`
    - `x: float`
    - `y: float`
    - `z: float`
  - If `coord3d=false`, targets are still returned, but `x/y/z` are `0.0`.

- Base-coordinate target output: `list[BaseCoordTarget]`
  - `BaseCoordTarget` fields:
    - `id: int | None`
    - `label: str`
    - `conf: float`
    - `cx: float`
    - `cy: float`
    - `width: float`
    - `height: float`
    - `x: float`
    - `y: float`
    - `z: float`
  - Useful for selecting one base by pixel-space rule (e.g., minimal `|cx - target_x|`) and then publishing 3D coordinates.

### Stability and `ready`

- The detector keeps a temporal queue (`runtime.queue_size`).
- `Detector.ready` remains `False` until `runtime.warmup_frames` observations are collected.
- Before `ready=True`, `detect()` returns empty output (`[]`) by design.
- After warmup:
  - Status mode emits stable labels when per-slot votes reach `vote_threshold`.
  - Base-coordinate mode emits stable targets when per-target votes reach `min_votes`.

### Switching Profiles at Runtime

- `switch_profile(name)` resets temporal history and `ready` state.
- The new profile loads its own model weights, confidence threshold, and mode-specific options.
- Typical pattern: run `status` for semantic state, then switch to `base_coord` for control coordinates.

## Config

Use `configs/basedetect_sdk.yaml` as the template. It includes comments for optional fields and defaults.

Important config fields:

- Top level:
  - `active_profile`: default profile used at startup.

- `runtime`:
  - `device`: `auto`, `cpu`, `0`, etc.
  - `queue_size`: temporal window length for stabilization.
  - `warmup_frames`: required observations before `ready=True`.
  - `debug`: enables `debug_overlay()` and `debug_info()`.
  - `grayscale_input`: convert BGR input to grayscale before YOLO inference.

- `profiles.<name>`:
  - Shared: `mode`, `weights`, `conf`.
  - Status mode: `vote_threshold`, `order` (`left_to_right` or `right_to_left`).
  - Base-coordinate mode: `coord3d`, `camera_yaml`, `smoothing` (`ema` or `median`), `ema_alpha`, `min_votes`.

## Visual Smoke Test

```bash
uv run scripts/test_sdk_visual.py --config configs/basedetect_sdk.yaml --source test/test7.mp4
```

Useful options:

- `--profile`: choose startup profile instead of `active_profile`.
- `--switch-to` + `--switch-frame`: auto-switch profile during playback.
- `--no-show`: run without OpenCV window.
