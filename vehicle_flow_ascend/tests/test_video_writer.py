from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vehicle_flow_ascend.video import writer as video_writer
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


def test_video_writer_requires_h264_encoder_for_browser_mp4(monkeypatch, tmp_path) -> None:
    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(video_writer.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="H.264"):
        VideoWriter(str(tmp_path / "browser.mp4"), fps=25.0, frame_size=(2, 2))


def test_video_writer_falls_back_to_system_ffmpeg_for_h264(monkeypatch, tmp_path) -> None:
    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(video_writer.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    calls = []

    class FakeStdin:
        def __init__(self) -> None:
            self.writes = []
            self.closed = False

        def write(self, data) -> None:
            self.writes.append(bytes(data))

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, command, stdin, stdout, stderr) -> None:
            self.command = command
            self.stdin = FakeStdin()
            self.returncode = 0
            calls.append(self)

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(video_writer.subprocess, "Popen", FakeProcess)
    output = tmp_path / "browser.mp4"
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]

    writer = VideoWriter(str(output), fps=25.0, frame_size=(2, 2))
    writer.write(frame)
    writer.release()

    assert len(calls) == 1
    command = calls[0].command
    assert command[0] == "/usr/bin/ffmpeg"
    assert "libx264" in command
    assert "+faststart" in command
    assert calls[0].stdin.closed is True
    assert calls[0].stdin.writes == [frame.tobytes()]
