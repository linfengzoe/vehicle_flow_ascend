from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from vehicle_flow_ascend.config import LineConfig, VehicleFlowConfig
from vehicle_flow_ascend.types import Detection
from vehicle_flow_ascend.web import realtime
from vehicle_flow_ascend.web.realtime import RealtimeInferenceManager


class FakeDetector:
    def __init__(self) -> None:
        self.released = False

    def detect(self, frame_bgr) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        return [Detection(1, 1, min(width - 1, 8), min(height - 1, 8), 0.8, "car")]

    def release(self) -> None:
        self.released = True


class BlockingDetector:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release_detection = threading.Event()
        self.released = False

    def detect(self, frame_bgr) -> list[Detection]:
        self.entered.set()
        self.release_detection.wait(timeout=2)
        return []

    def release(self) -> None:
        self.released = True


def jpeg_bytes() -> bytes:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_realtime_manager_processes_jpeg_frame(monkeypatch) -> None:
    monkeypatch.setattr(realtime, "create_detector", lambda config: FakeDetector())
    config = VehicleFlowConfig(line=LineConfig(start=(0, 12), end=(32, 12)))
    manager = RealtimeInferenceManager(config)

    started = manager.start()
    assert started["status"] == "running"
    session_id = started["session_id"]

    status = manager.process_jpeg_frame(session_id, jpeg_bytes())

    assert status["status"] == "running"
    assert status["frames"] == 1
    assert status["fps"] >= 0.0
    assert status["counts"]["total"] == 0
    frame = manager.wait_for_frame(session_id, last_version=0, timeout=0.1)
    assert frame is not None
    version, output = frame
    assert version == 1
    assert output.startswith(b"\xff\xd8")


def test_realtime_manager_rejects_duplicate_session(monkeypatch) -> None:
    monkeypatch.setattr(realtime, "create_detector", lambda config: FakeDetector())
    manager = RealtimeInferenceManager(VehicleFlowConfig())

    manager.start()

    with pytest.raises(RuntimeError, match="realtime inference session is already running"):
        manager.start()


def test_realtime_manager_stop_releases_detector(monkeypatch) -> None:
    detector = FakeDetector()
    monkeypatch.setattr(realtime, "create_detector", lambda config: detector)
    manager = RealtimeInferenceManager(VehicleFlowConfig())
    session_id = manager.start()["session_id"]

    stopped = manager.stop(session_id)

    assert stopped["status"] == "stopped"
    assert detector.released is True


def test_realtime_manager_stop_is_not_blocked_by_inflight_frame(monkeypatch) -> None:
    detector = BlockingDetector()
    monkeypatch.setattr(realtime, "create_detector", lambda config: detector)
    manager = RealtimeInferenceManager(VehicleFlowConfig())
    session_id = manager.start()["session_id"]
    frame_errors = []

    def process_frame() -> None:
        try:
            manager.process_jpeg_frame(session_id, jpeg_bytes())
        except Exception as exc:  # noqa: BLE001 - record worker outcome for assertion context
            frame_errors.append(exc)

    frame_thread = threading.Thread(target=process_frame)
    frame_thread.start()
    assert detector.entered.wait(timeout=1)

    stopped_payload = {}

    def stop_session() -> None:
        stopped_payload.update(manager.stop(session_id))

    stop_thread = threading.Thread(target=stop_session)
    stop_thread.start()
    stop_thread.join(timeout=0.2)

    assert stopped_payload.get("status") == "stopped"

    detector.release_detection.set()
    frame_thread.join(timeout=2)
    stop_thread.join(timeout=2)

    assert detector.released is True
    assert frame_errors == []


def test_realtime_manager_stop_requires_session_id(monkeypatch) -> None:
    monkeypatch.setattr(realtime, "create_detector", lambda config: FakeDetector())
    manager = RealtimeInferenceManager(VehicleFlowConfig())
    manager.start()

    with pytest.raises(ValueError, match="realtime session id is required"):
        manager.stop()


def test_realtime_manager_stop_rejects_wrong_session(monkeypatch) -> None:
    monkeypatch.setattr(realtime, "create_detector", lambda config: FakeDetector())
    manager = RealtimeInferenceManager(VehicleFlowConfig())
    manager.start()

    with pytest.raises(ValueError, match="invalid realtime session"):
        manager.stop("wrong-session")


def test_realtime_manager_reaper_expires_idle_session_without_requests(monkeypatch) -> None:
    detector = FakeDetector()
    monkeypatch.setattr(realtime, "create_detector", lambda config: detector)
    manager = RealtimeInferenceManager(
        VehicleFlowConfig(),
        idle_timeout_seconds=0.05,
    )

    started = manager.start()
    assert started["status"] == "running"
    time.sleep(0.2)

    assert detector.released is True
    status = manager.status()
    assert status["status"] == "stopped"
    assert status["error"] == "realtime session expired after idle timeout"


def test_realtime_manager_rejects_wrong_session(monkeypatch) -> None:
    monkeypatch.setattr(realtime, "create_detector", lambda config: FakeDetector())
    manager = RealtimeInferenceManager(VehicleFlowConfig())
    manager.start()

    with pytest.raises(ValueError, match="invalid realtime session"):
        manager.process_jpeg_frame("wrong-session", jpeg_bytes())
