from __future__ import annotations

from pathlib import Path

import cv2


class VideoWriter:
    def __init__(self, output_path: str | None, fps: float, frame_size: tuple[int, int] | None) -> None:
        self._writer = None
        if output_path is None:
            return
        if frame_size is None:
            raise ValueError("frame_size is required when output_video is configured")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise ValueError(f"could not open output video writer: {output_path}")

    def write(self, frame) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
