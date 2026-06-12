from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from vehicle_flow_ascend.app import FrameProcessor
from vehicle_flow_ascend.config import LineConfig, VehicleFlowConfig
from vehicle_flow_ascend.detectors.base import Detector, create_detector


DEFAULT_IDLE_TIMEOUT_SECONDS = 10.0
_IDLE_EXPIRED_ERROR = "realtime session expired after idle timeout"
_REAPER_MAX_SLEEP_SECONDS = 0.25
_REALTIME_IMAGE_SIZE = 320
_DEVBOARD_CAMERA_PROBE_INDICES = tuple(range(6))
_CAMERA_READ_PROBE_ATTEMPTS = 5


@dataclass
class RealtimeState:
    status: str = "idle"
    session_id: str | None = None
    source: int | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    frames: int = 0
    fps: float = 0.0
    counts: dict[str, int] = field(default_factory=lambda: {"total": 0})
    error: str | None = None
    frame_version: int = 0
    updated_at: float | None = None
    last_frame_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "source": self.source,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "frames": self.frames,
            "fps": self.fps,
            "counts": dict(self.counts),
            "error": self.error,
            "frame_version": self.frame_version,
            "updated_at": self.updated_at,
            "last_frame_at": self.last_frame_at,
        }


class RealtimeInferenceManager:
    def __init__(
        self,
        base_config: VehicleFlowConfig,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self._base_config = base_config
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._state = RealtimeState()
        self._detector: Detector | None = None
        self._processor: FrameProcessor | None = None
        self._latest_jpeg: bytes | None = None
        self._last_activity_at: float | None = None
        self._inflight_frames = 0
        self._release_after_inflight = False
        self._reaper_thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._capture_stop_event = threading.Event()

    def start(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._condition:
            self._expire_idle_session_locked()
            if self._state.status == "running":
                raise RuntimeError("realtime inference session is already running")
            if self._inflight_frames:
                raise RuntimeError("realtime inference session is still stopping")

            session_config = replace(
                self._base_config,
                line=_line_from_payload(payload or {}, self._base_config.line),
                image_size=_realtime_image_size(self._base_config),
            )
            detector = create_detector(session_config)
            self._detector = detector
            self._processor = FrameProcessor(session_config, detector)
            self._latest_jpeg = None
            self._last_activity_at = time.monotonic()
            self._release_after_inflight = False
            now = time.time()
            self._state = RealtimeState(
                status="running",
                session_id=uuid.uuid4().hex[:10],
                started_at=now,
                counts={"total": 0},
                updated_at=now,
            )
            self._ensure_reaper_locked()
            self._condition.notify_all()
            return self._state.to_dict()

    def start_devboard_camera(
        self,
        camera_index: int | str = "auto",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._condition:
            self._expire_idle_session_locked()
            if self._state.status == "running":
                raise RuntimeError("realtime inference session is already running")
            if self._inflight_frames:
                raise RuntimeError("realtime inference session is still stopping")

            capture, selected_index = _open_devboard_camera(camera_index)

            session_config = replace(
                self._base_config,
                line=_line_from_payload(payload or {}, self._base_config.line),
                image_size=_realtime_image_size(self._base_config),
            )
            try:
                detector = create_detector(session_config)
            except Exception:
                capture.release()
                raise
            self._detector = detector
            self._processor = FrameProcessor(session_config, detector)
            self._capture = capture
            self._latest_jpeg = None
            self._last_activity_at = time.monotonic()
            self._release_after_inflight = False
            self._capture_stop_event.clear()
            now = time.time()
            self._state = RealtimeState(
                status="running",
                session_id=uuid.uuid4().hex[:10],
                started_at=now,
                counts={"total": 0},
                updated_at=now,
            )
            self._state.source = selected_index
            self._ensure_reaper_locked()
            self._capture_thread = threading.Thread(
                target=self._devboard_capture_loop,
                name="vehicle-flow-devboard-capture",
                daemon=True,
            )
            self._capture_thread.start()
            self._condition.notify_all()
            return self._state.to_dict()

    def _devboard_capture_loop(self) -> None:
        while not self._capture_stop_event.is_set():
            if self._capture is None:
                break
            ok, frame_bgr = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self._condition:
                if self._state.status != "running":
                    break
                if self._processor is None:
                    break
                processor = self._processor
                self._inflight_frames += 1

            processed = None
            output_jpeg = None
            error = None
            try:
                processed = processor.process(frame_bgr)
                output_jpeg = _encode_jpeg(processed.annotated_frame)
            except Exception as exc:  # noqa: BLE001 - publish worker errors to the UI
                error = exc

            with self._condition:
                self._inflight_frames -= 1
                if error is not None:
                    if self._state.status == "running" and self._processor is processor:
                        self._fail_running_session_locked(str(error))
                    if self._inflight_frames == 0:
                        self._release_resources_locked()
                    self._condition.notify_all()
                    break
                if (
                    processed is not None
                    and output_jpeg is not None
                    and self._state.status == "running"
                    and self._processor is processor
                ):
                    self._apply_processed_frame_locked(processed, output_jpeg)
                if self._release_after_inflight and self._inflight_frames == 0:
                    self._release_resources_locked()
                self._condition.notify_all()

    def process_jpeg_frame(self, session_id: str, jpeg_bytes: bytes) -> dict[str, Any]:
        frame_bgr = _decode_jpeg(jpeg_bytes)
        with self._condition:
            self._expire_idle_session_locked()
            self._validate_session(session_id)
            if self._processor is None:
                raise RuntimeError("realtime inference processor is not initialized")
            processor = self._processor
            self._inflight_frames += 1

        processed = None
        output_jpeg = None
        try:
            processed = processor.process(frame_bgr)
            output_jpeg = _encode_jpeg(processed.annotated_frame)
        finally:
            with self._condition:
                self._inflight_frames -= 1
                if (
                    processed is not None
                    and output_jpeg is not None
                    and self._state.status == "running"
                    and self._state.session_id == session_id
                    and self._processor is processor
                ):
                    self._apply_processed_frame_locked(processed, output_jpeg)
                if self._release_after_inflight and self._inflight_frames == 0:
                    self._release_resources_locked()
                self._condition.notify_all()
                state = self._state.to_dict()
        return state

    def _apply_processed_frame_locked(self, processed, output_jpeg: bytes) -> None:
        now = time.time()
        self._last_activity_at = time.monotonic()
        self._latest_jpeg = output_jpeg
        self._state.frames = processed.frames
        self._state.fps = processed.fps
        self._state.counts = dict(processed.counts)
        self._state.frame_version += 1
        self._state.updated_at = now
        self._state.last_frame_at = now

    def _fail_running_session_locked(self, error: str) -> None:
        now = time.time()
        self._state.status = "failed"
        self._state.error = error
        self._state.stopped_at = now
        self._state.updated_at = now

    def status(self) -> dict[str, Any]:
        with self._condition:
            self._expire_idle_session_locked()
            return self._state.to_dict()

    def stop(self, session_id: str | None = None) -> dict[str, Any]:
        with self._condition:
            self._expire_idle_session_locked()
            if session_id is None:
                raise ValueError("realtime session id is required")
            if self._state.session_id != session_id:
                raise ValueError("invalid realtime session")
            if self._state.status == "running":
                now = time.time()
                self._state.status = "stopped"
                self._state.stopped_at = now
                self._state.updated_at = now
            if self._inflight_frames:
                self._release_after_inflight = True
            else:
                self._release_resources_locked()
            self._condition.notify_all()
            return self._state.to_dict()

    def wait_for_frame(
        self, session_id: str, last_version: int, timeout: float = 1.0
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            self._expire_idle_session_locked()
            if self._state.status != "running" and self._state.session_id == session_id:
                return None
            self._validate_session(session_id)
            while self._state.status == "running" and self._state.frame_version <= last_version:
                self._expire_idle_session_locked()
                if self._state.status != "running":
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._expire_idle_session_locked()
                    return None
                idle_remaining = self._idle_remaining_locked()
                wait_time = remaining
                if idle_remaining is not None:
                    wait_time = min(remaining, max(idle_remaining, 0.0))
                if wait_time <= 0:
                    self._expire_idle_session_locked()
                    return None
                self._condition.wait(wait_time)
                if self._state.status != "running":
                    return None
                if self._state.session_id != session_id:
                    raise ValueError("invalid realtime session")

            if self._state.status != "running":
                return None
            if self._latest_jpeg is None or self._state.frame_version <= last_version:
                return None
            return self._state.frame_version, bytes(self._latest_jpeg)

    def _validate_session(self, session_id: str) -> None:
        if self._state.status != "running" or self._state.session_id != session_id:
            raise ValueError("invalid realtime session")

    def _ensure_reaper_locked(self) -> None:
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            name="vehicle-flow-realtime-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def _reaper_loop(self) -> None:
        while True:
            with self._condition:
                while self._state.status != "running":
                    self._condition.wait()

                remaining = self._idle_remaining_locked()
                if remaining is None:
                    self._condition.wait(_REAPER_MAX_SLEEP_SECONDS)
                    continue
                if remaining <= 0:
                    self._expire_idle_session_locked()
                    continue

                wait_time = min(remaining, _REAPER_MAX_SLEEP_SECONDS)
                self._condition.wait(max(wait_time, 0.001))

    def _idle_remaining_locked(self) -> float | None:
        if self._state.status != "running" or self._last_activity_at is None:
            return None
        return self._idle_timeout_seconds - (time.monotonic() - self._last_activity_at)

    def _expire_idle_session_locked(self) -> None:
        remaining = self._idle_remaining_locked()
        if remaining is None or remaining > 0:
            return
        now = time.time()
        self._state.status = "stopped"
        self._state.stopped_at = now
        self._state.updated_at = now
        self._state.error = _IDLE_EXPIRED_ERROR
        if self._inflight_frames:
            self._release_after_inflight = True
        else:
            self._release_resources_locked()
        self._condition.notify_all()

    def _release_resources_locked(self) -> None:
        detector = self._detector
        self._detector = None
        self._processor = None
        self._latest_jpeg = None
        self._last_activity_at = None
        self._release_after_inflight = False
        self._capture_stop_event.set()
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        release = getattr(detector, "release", None)
        if callable(release):
            release()


def _decode_jpeg(jpeg_bytes: bytes):
    if not jpeg_bytes:
        raise ValueError("jpeg frame is empty")
    data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("failed to decode jpeg frame")
    return frame_bgr


def _encode_jpeg(frame_bgr) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        raise ValueError("failed to encode jpeg frame")
    return encoded.tobytes()


def _open_devboard_camera(camera_index: int | str) -> tuple[cv2.VideoCapture, int]:
    indices = _camera_probe_indices(camera_index)
    attempted = []
    for index in indices:
        attempted.append(str(index))
        capture = cv2.VideoCapture(index)
        if capture.isOpened() and _camera_can_read(capture):
            return capture, index
        capture.release()
    raise RuntimeError(f"开发板摄像头不可用 (tried indices: {', '.join(attempted)})")


def _camera_probe_indices(camera_index: int | str) -> tuple[int, ...]:
    if isinstance(camera_index, str):
        value = camera_index.strip().lower()
        if value in {"", "auto"}:
            return _DEVBOARD_CAMERA_PROBE_INDICES
        return (int(value),)
    return (int(camera_index),)


def _camera_can_read(capture: cv2.VideoCapture) -> bool:
    for _ in range(_CAMERA_READ_PROBE_ATTEMPTS):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
        time.sleep(0.05)
    return False


def _realtime_image_size(config: VehicleFlowConfig) -> int:
    if config.backend == "ascend_om":
        return config.image_size
    return min(config.image_size, _REALTIME_IMAGE_SIZE)


def _line_from_payload(payload: dict[str, Any], default_line: LineConfig) -> LineConfig:
    value = payload.get("line")
    if value is None:
        return default_line
    return LineConfig.from_value(value)
