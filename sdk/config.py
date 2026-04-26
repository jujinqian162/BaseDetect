from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _normalize_mode(raw_mode: Any) -> str:
    normalized = str(raw_mode).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized == "status":
        return "status"
    if normalized in {"base coord", "base", "coord", "coordinate", "base coordinate"}:
        return "base_coord"
    raise ValueError("Invalid mode. Expected 'status' or 'base_coord'.")


def _resolve_path(candidate: Any, *, base_dir: Path, required: bool = False) -> Path | None:
    if candidate is None:
        if required:
            raise ValueError("Missing required path field.")
        return None

    path = Path(str(candidate)).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        resolved = (base_dir / path).resolve()

    if required and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def _resolve_weights(candidate: Any, *, base_dir: Path, sdk_root: Path) -> str:
    if candidate is None:
        raise ValueError("Each profile requires a 'weights' field.")

    text = str(candidate)
    path = Path(text).expanduser()
    if path.is_absolute() and path.exists():
        return str(path)

    for root in (base_dir, sdk_root):
        relative = (root / path).resolve()
        if relative.exists():
            return str(relative)

    if path.exists():
        return str(path.resolve())

    is_path_like = (
        "/" in text
        or "\\" in text
        or str(path.parent) not in {"", "."}
    )
    if is_path_like:
        raise FileNotFoundError(
            f"Weights path not found: {text} (resolved from {base_dir} and {sdk_root})"
        )

    return text


@dataclass(frozen=True)
class RuntimeSettings:
    device: str
    queue_size: int
    warmup_frames: int
    debug: bool
    grayscale_input: bool


@dataclass(frozen=True)
class ProfileSettings:
    name: str
    mode: str
    weights: str
    conf: float
    vote_threshold: int
    order: str
    camera_yaml: Path | None
    coord3d: bool
    smoothing: str
    ema_alpha: float
    min_votes: int


@dataclass(frozen=True)
class SDKSettings:
    config_path: Path
    runtime: RuntimeSettings
    active_profile: str
    profiles: dict[str, ProfileSettings]


def _parse_runtime(raw_runtime: dict[str, Any]) -> RuntimeSettings:
    queue_size = int(raw_runtime.get("queue_size", 5))
    if queue_size <= 0:
        raise ValueError("runtime.queue_size must be positive.")

    warmup_frames = int(raw_runtime.get("warmup_frames", min(3, queue_size)))
    if warmup_frames <= 0:
        raise ValueError("runtime.warmup_frames must be positive.")
    if warmup_frames > queue_size:
        warmup_frames = queue_size

    return RuntimeSettings(
        device=str(raw_runtime.get("device", "auto")),
        queue_size=queue_size,
        warmup_frames=warmup_frames,
        debug=bool(raw_runtime.get("debug", False)),
        grayscale_input=bool(raw_runtime.get("grayscale_input", True)),
    )


def _majority_vote_threshold(size: int) -> int:
    return (size // 2) + 1


def _parse_profile(
    *,
    name: str,
    raw_profile: dict[str, Any],
    base_dir: Path,
    sdk_root: Path,
    runtime: RuntimeSettings,
) -> ProfileSettings:
    mode = _normalize_mode(raw_profile.get("mode"))
    weights = _resolve_weights(
        raw_profile.get("weights"),
        base_dir=base_dir,
        sdk_root=sdk_root,
    )

    conf = float(raw_profile.get("conf", 0.25))
    if conf < 0.0 or conf > 1.0:
        raise ValueError(f"profiles.{name}.conf must be in [0, 1].")

    default_votes = _majority_vote_threshold(runtime.queue_size)

    if mode == "status":
        vote_threshold = int(raw_profile.get("vote_threshold", default_votes))
        if vote_threshold <= 0 or vote_threshold > runtime.queue_size:
            raise ValueError(
                f"profiles.{name}.vote_threshold must be in [1, runtime.queue_size]."
            )
        order = str(raw_profile.get("order", "left_to_right")).strip().lower()
        if order not in {"left_to_right", "right_to_left"}:
            raise ValueError(
                f"profiles.{name}.order must be 'left_to_right' or 'right_to_left'."
            )

        return ProfileSettings(
            name=name,
            mode=mode,
            weights=weights,
            conf=conf,
            vote_threshold=vote_threshold,
            order=order,
            camera_yaml=None,
            coord3d=False,
            smoothing="ema",
            ema_alpha=0.35,
            min_votes=default_votes,
        )

    coord3d = bool(raw_profile.get("coord3d", True))
    camera_yaml: Path | None = None
    if coord3d:
        camera_yaml = _resolve_path(raw_profile.get("camera_yaml"), base_dir=base_dir, required=True)

    smoothing = str(raw_profile.get("smoothing", "ema")).strip().lower()
    if smoothing not in {"ema", "median"}:
        raise ValueError(f"profiles.{name}.smoothing must be 'ema' or 'median'.")

    ema_alpha = float(raw_profile.get("ema_alpha", 0.35))
    if ema_alpha <= 0.0 or ema_alpha > 1.0:
        raise ValueError(f"profiles.{name}.ema_alpha must be in (0, 1].")

    min_votes = int(raw_profile.get("min_votes", default_votes))
    if min_votes <= 0 or min_votes > runtime.queue_size:
        raise ValueError(
            f"profiles.{name}.min_votes must be in [1, runtime.queue_size]."
        )

    return ProfileSettings(
        name=name,
        mode=mode,
        weights=weights,
        conf=conf,
        vote_threshold=default_votes,
        order="left_to_right",
        camera_yaml=camera_yaml,
        coord3d=coord3d,
        smoothing=smoothing,
        ema_alpha=ema_alpha,
        min_votes=min_votes,
    )


def load_settings(config_path: str | Path) -> SDKSettings:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SDK config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    if not isinstance(payload, dict):
        raise ValueError("SDK config must be a YAML mapping.")

    runtime_raw = payload.get("runtime", {})
    if not isinstance(runtime_raw, dict):
        raise ValueError("runtime section must be a YAML mapping.")
    runtime = _parse_runtime(runtime_raw)

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("profiles section must be a non-empty YAML mapping.")

    profiles: dict[str, ProfileSettings] = {}
    sdk_root = path.parent.parent
    for profile_name, profile_payload in raw_profiles.items():
        if not isinstance(profile_payload, dict):
            raise ValueError(f"profiles.{profile_name} must be a YAML mapping.")
        profiles[str(profile_name)] = _parse_profile(
            name=str(profile_name),
            raw_profile=profile_payload,
            base_dir=path.parent,
            sdk_root=sdk_root,
            runtime=runtime,
        )

    active_profile = str(payload.get("active_profile", "")).strip()
    if not active_profile:
        active_profile = next(iter(profiles))

    if active_profile not in profiles:
        raise ValueError(f"active_profile '{active_profile}' was not found in profiles.")

    return SDKSettings(
        config_path=path,
        runtime=runtime,
        active_profile=active_profile,
        profiles=profiles,
    )


def resolve_profile(settings: SDKSettings, profile_name: str | None) -> ProfileSettings:
    selected = profile_name or settings.active_profile
    if selected not in settings.profiles:
        raise ValueError(f"Profile '{selected}' was not found in {settings.config_path}.")
    return settings.profiles[selected]
