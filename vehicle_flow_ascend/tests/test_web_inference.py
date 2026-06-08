from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.types import Detection
from vehicle_flow_ascend.web import inference
from vehicle_flow_ascend.web.inference import InferenceTaskManager, _source_from_payload


class FakeDetector:
    def detect(self, frame_bgr) -> list[Detection]:
        return []

    def release(self) -> None:
        self.released = True


def wait_until_done(manager: InferenceTaskManager) -> dict:
    for _ in range(100):
        status = manager.status()
        if status["status"] in {"completed", "failed", "stopped"}:
            return status
        time.sleep(0.01)
    raise AssertionError(f"task did not finish: {manager.status()}")


def test_source_from_payload_accepts_video_payload_for_uploaded_file() -> None:
    source = _source_from_payload({"source_type": "video", "source": "data/web_uploads/demo.mp4"})
    assert source.path == "data/web_uploads/demo.mp4"
    assert source.camera is None


def test_source_from_payload_still_accepts_camera_for_backend_compatibility() -> None:
    assert _source_from_payload({"source_type": "camera", "source": "1"}).camera == 1
    assert _source_from_payload({"source_type": "camera", "source": ""}).camera == 0


def test_inference_manager_runs_task_and_reports_counts(monkeypatch, tmp_path) -> None:
    manager = InferenceTaskManager(VehicleFlowConfig(), project_root=tmp_path)

    def fake_create_detector(config):
        assert config.source.path == "data/demo.mp4"
        assert config.display is False
        assert config.output_video is not None
        return FakeDetector()

    def fake_run_app(config, detector, *, show_window, stop_requested, progress_callback):
        assert show_window is False
        progress_callback({"frames": 7, "fps": 18.5, "counts": {"total": 2, "car": 2}})
        assert stop_requested() is False
        Path(config.output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output_video).write_bytes(b"demo")
        return {"total": 2, "car": 2}

    monkeypatch.setattr(inference, "create_detector", fake_create_detector)
    monkeypatch.setattr(inference, "run_app", fake_run_app)

    started = manager.start({"source_type": "video", "source": "data/demo.mp4"})
    assert started["status"] == "running"

    done = wait_until_done(manager)
    assert done["status"] == "completed"
    assert done["frames"] == 7
    assert done["fps"] == 18.5
    assert done["counts"] == {"total": 2, "car": 2}
    assert done["output_video"] is not None


def test_inference_manager_applies_line_override_and_publishes_latest_frame(
    monkeypatch,
    tmp_path,
) -> None:
    manager = InferenceTaskManager(VehicleFlowConfig(), project_root=tmp_path)

    def fake_create_detector(config):
        assert config.line.start == (12, 34)
        assert config.line.end == (320, 210)
        return FakeDetector()

    def fake_run_app(config, detector, *, show_window, stop_requested, progress_callback):
        assert show_window is False
        annotated = np.zeros((16, 20, 3), dtype=np.uint8)
        annotated[:, :] = (0, 120, 255)
        progress_callback(
            {
                "frames": 1,
                "fps": 9.5,
                "counts": {"total": 1, "car": 1},
                "annotated_frame": annotated,
            }
        )
        Path(config.output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output_video).write_bytes(b"demo")
        return {"total": 1, "car": 1}

    monkeypatch.setattr(inference, "create_detector", fake_create_detector)
    monkeypatch.setattr(inference, "run_app", fake_run_app)

    started = manager.start(
        {
            "source_type": "video",
            "source": "data/demo.mp4",
            "line": [[12, 34], [320, 210]],
        }
    )
    frame = manager.wait_for_frame(started["task_id"], last_version=0, timeout=1.0)
    done = wait_until_done(manager)

    assert done["status"] == "completed"
    assert done["frame_version"] == 1
    assert frame is not None
    version, jpeg = frame
    assert version == 1
    assert jpeg.startswith(b"\xff\xd8")


def test_inference_manager_surfaces_worker_errors(monkeypatch, tmp_path) -> None:
    manager = InferenceTaskManager(VehicleFlowConfig(), project_root=tmp_path)

    def fake_create_detector(config):
        raise RuntimeError("model weights missing")

    monkeypatch.setattr(inference, "create_detector", fake_create_detector)

    manager.start({"source_type": "video", "source": "data/demo.mp4"})
    status = wait_until_done(manager)

    assert status["status"] == "failed"
    assert "model weights missing" in status["error"]


def test_inference_upload_saves_file(tmp_path) -> None:
    manager = InferenceTaskManager(VehicleFlowConfig(), project_root=tmp_path)
    source = (tmp_path / "source.mp4")
    source.write_bytes(b"video-bytes")

    with source.open("rb") as stream:
        uploaded = manager.upload("unsafe/../demo.mp4", stream)

    assert uploaded["name"] == "demo.mp4"
    assert uploaded["size"] == len(b"video-bytes")
    assert Path(uploaded["path"]).read_bytes() == b"video-bytes"
