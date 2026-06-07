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
        try:
            self._writer = _FfmpegH264Writer(output_path, fps, frame_size)
        except ImportError:
            self._writer = _OpenCvMp4Writer(output_path, fps, frame_size)

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


class _FfmpegH264Writer:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]) -> None:
        import imageio_ffmpeg

        self._writer = imageio_ffmpeg.write_frames(
            output_path,
            frame_size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-movflags", "+faststart", "-preset", "veryfast", "-crf", "23"],
        )
        self._writer.send(None)

    def write(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._writer.send(rgb_frame)

    def release(self) -> None:
        self._writer.close()


class _OpenCvMp4Writer:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]) -> None:
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
