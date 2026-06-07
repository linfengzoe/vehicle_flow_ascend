from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from vehicle_flow_ascend.video.writer import VideoWriter


class _FakeFfmpegWriter:
    def __init__(self) -> None:
        self.frames = []
        self.started = False
        self.closed = False

    def send(self, frame) -> None:
        if frame is None:
            self.started = True
            return
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


def test_video_writer_prefers_h264_ffmpeg_writer(monkeypatch, tmp_path) -> None:
    fake_writer = _FakeFfmpegWriter()
    calls = []

    def fake_write_frames(path, size, **kwargs):
        calls.append({"path": path, "size": size, "kwargs": kwargs})
        return fake_writer

    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(write_frames=fake_write_frames),
    )
    output = tmp_path / "browser.mp4"
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]

    writer = VideoWriter(str(output), fps=25.0, frame_size=(2, 2))
    writer.write(frame)
    writer.release()

    assert fake_writer.started is True
    assert fake_writer.closed is True
    assert calls == [
        {
            "path": str(output),
            "size": (2, 2),
            "kwargs": {
                "fps": 25.0,
                "codec": "libx264",
                "pix_fmt_in": "rgb24",
                "pix_fmt_out": "yuv420p",
                "macro_block_size": 1,
                "output_params": ["-movflags", "+faststart", "-preset", "veryfast", "-crf", "23"],
            },
        }
    ]
    assert fake_writer.frames[0][0, 0].tolist() == [30, 20, 10]
