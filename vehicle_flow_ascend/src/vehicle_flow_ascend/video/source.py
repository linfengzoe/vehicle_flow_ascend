from __future__ import annotations

from dataclasses import dataclass

import cv2

from vehicle_flow_ascend.config import SourceConfig


@dataclass
class VideoSource:
    source: SourceConfig

    def __post_init__(self) -> None:
        source_value = self.source.camera if self.source.camera is not None else self.source.path
        if source_value is None:
            raise ValueError("video source must specify a path or camera index")
        self._capture = cv2.VideoCapture(source_value)
        if not self._capture.isOpened():
            raise ValueError(f"could not open video source: {source_value!r}")

    def read(self):
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    @property
    def fps(self) -> float:
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30.0

    @property
    def frame_size(self) -> tuple[int, int] | None:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return None
        return width, height

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
