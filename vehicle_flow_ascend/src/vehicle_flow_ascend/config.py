from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceConfig:
    path: str | None = None
    camera: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> "SourceConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(camera=value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return cls(camera=int(stripped))
            return cls(path=value)
        if isinstance(value, dict):
            path = _optional_str(value.get("path"))
            camera = _optional_int(value.get("camera"))
            if path is not None and camera is not None:
                raise ValueError(
                    "source mapping must specify only one of 'path' or 'camera'"
                )
            return cls(path=path, camera=camera)
        if value is None:
            return cls()
        raise TypeError("source must be a path string, camera integer, or mapping")

    def to_cli_value(self) -> str | int | None:
        if self.camera is not None:
            return self.camera
        return self.path


@dataclass(frozen=True)
class LineConfig:
    start: tuple[int, int] = (0, 0)
    end: tuple[int, int] = (0, 0)

    @classmethod
    def from_value(cls, value: Any) -> "LineConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(start=_point(value.get("start")), end=_point(value.get("end")))
        if isinstance(value, list | tuple) and len(value) == 2:
            return cls(start=_point(value[0]), end=_point(value[1]))
        raise TypeError("line must be a two-point list or mapping with start/end")

    def as_list(self) -> list[list[int]]:
        return [list(self.start), list(self.end)]


@dataclass(frozen=True)
class VehicleFlowConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    backend: str = "torch_yolov5"
    model_path: str | None = None
    yolov5_repo_path: str | None = None
    soc_version: str | None = None
    image_size: int = 640
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    line: LineConfig = field(default_factory=LineConfig)
    display: bool = False
    output_video: str | None = None
    max_frames: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "VehicleFlowConfig":
        data = data or {}
        return cls(
            source=SourceConfig.from_value(data.get("source")),
            backend=str(data.get("backend", cls.backend)),
            model_path=_optional_str(data.get("model_path")),
            yolov5_repo_path=_optional_str(data.get("yolov5_repo_path")),
            soc_version=_optional_str(data.get("soc_version")),
            image_size=int(data.get("image_size", cls.image_size)),
            confidence_threshold=float(
                data.get("confidence_threshold", cls.confidence_threshold)
            ),
            iou_threshold=float(data.get("iou_threshold", cls.iou_threshold)),
            line=LineConfig.from_value(data.get("line")),
            display=_bool(data.get("display", cls.display)),
            output_video=_optional_str(data.get("output_video")),
            max_frames=_optional_int(data.get("max_frames")),
        )

    def with_overrides(self, overrides: dict[str, Any]) -> "VehicleFlowConfig":
        updates: dict[str, Any] = {}
        if "source" in overrides and overrides["source"] is not None:
            updates["source"] = SourceConfig.from_value(overrides["source"])
        if "backend" in overrides and overrides["backend"] is not None:
            updates["backend"] = str(overrides["backend"])
        if "model_path" in overrides and overrides["model_path"] is not None:
            updates["model_path"] = _optional_str(overrides["model_path"])
        if "yolov5_repo_path" in overrides and overrides["yolov5_repo_path"] is not None:
            updates["yolov5_repo_path"] = _optional_str(overrides["yolov5_repo_path"])
        if "soc_version" in overrides and overrides["soc_version"] is not None:
            updates["soc_version"] = _optional_str(overrides["soc_version"])
        if "image_size" in overrides and overrides["image_size"] is not None:
            updates["image_size"] = int(overrides["image_size"])
        if "confidence_threshold" in overrides and overrides["confidence_threshold"] is not None:
            updates["confidence_threshold"] = float(overrides["confidence_threshold"])
        if "iou_threshold" in overrides and overrides["iou_threshold"] is not None:
            updates["iou_threshold"] = float(overrides["iou_threshold"])
        if "line" in overrides and overrides["line"] is not None:
            updates["line"] = LineConfig.from_value(overrides["line"])
        if "display" in overrides and overrides["display"] is not None:
            updates["display"] = _bool(overrides["display"])
        if "output_video" in overrides and overrides["output_video"] is not None:
            updates["output_video"] = _optional_str(overrides["output_video"])
        if "max_frames" in overrides and overrides["max_frames"] is not None:
            updates["max_frames"] = _optional_int(overrides["max_frames"])
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.to_cli_value()
        data["line"] = self.line.as_list()
        return data


def load_config(path: str | Path) -> VehicleFlowConfig:
    with Path(path).open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise TypeError("configuration root must be a mapping")
    return VehicleFlowConfig.from_mapping(data)


def load_config_with_overrides(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> VehicleFlowConfig:
    return load_config(path).with_overrides(overrides or {})


def _point(value: Any) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise TypeError("line point must contain exactly two numbers")
    return int(value[0]), int(value[1])


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"cannot parse boolean value: {value!r}")
