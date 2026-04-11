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
if det.ready:
    # call your control logic when stable output is ready
    pass

det.switch_profile("base_coord_competition")
targets = det.detect(frame)  # list[Target3D] in base_coord mode
```

## Public API

- `Detector(config: str | Path, profile: str | None = None)`
- `Detector.detect(frame)`
  - returns `list[str]` when mode=`status`
  - returns `list[Target3D]` when mode=`base_coord`
- `Detector.switch_profile(name)`
- `Detector.ready` (bool)
- `Detector.mode` ("status" or "base_coord")
- `Detector.profile_name`
- `Detector.debug_overlay()` (returns overlay frame or `None`)
- `Detector.debug_info()` (returns dict or `{}`)

## Config

Use `configs/basedetect_sdk.yaml` as the template. It includes comments for optional fields and defaults.
