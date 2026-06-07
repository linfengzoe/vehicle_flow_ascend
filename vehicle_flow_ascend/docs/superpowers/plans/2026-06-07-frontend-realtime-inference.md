# Frontend Realtime Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 Web Dashboard：后端复用现有车辆检测/跟踪/计数/叠加逻辑，前端只保留上传视频与浏览器摄像头入口，并展示后台推理后的结果视频或准实时标注画面。

**Architecture:** 抽出 `FrameProcessor` 作为单帧推理单元，让 CLI/上传视频后台任务/浏览器摄像头实时会话共用同一套后端处理逻辑。上传视频继续走现有 `InferenceTaskManager`；浏览器摄像头新增 `RealtimeInferenceManager`，前端用 `getUserMedia()` 抽帧 POST 到后端，结果区域通过 MJPEG 流显示标注帧。

**Tech Stack:** Python 3.10+, OpenCV, stdlib `http.server` / `ThreadingHTTPServer`, static HTML/CSS/JavaScript, pytest.

---

## File Structure

### Backend

- Modify: `src/vehicle_flow_ascend/app.py`
  - Add `ProcessedFrame` dataclass.
  - Add `FrameProcessor` class.
  - Refactor `run_app()` to call `FrameProcessor.process()` instead of duplicating detector/tracker/counter/overlay code.

- Create: `src/vehicle_flow_ascend/web/realtime.py`
  - Own browser-camera realtime sessions.
  - Decode JPEG frames from frontend.
  - Run `FrameProcessor`.
  - Encode annotated frames as JPEG.
  - Expose thread-safe status and latest-frame waiting helpers for MJPEG streaming.

- Modify: `src/vehicle_flow_ascend/web/dashboard.py`
  - Instantiate both `InferenceTaskManager` and `RealtimeInferenceManager`.
  - Add realtime endpoints:
    - `POST /api/realtime/start`
    - `POST /api/realtime/frame?session_id=...`
    - `GET /api/realtime/stream?session_id=...`
    - `GET /api/realtime/status?session_id=...`
    - `POST /api/realtime/stop`
  - Update dashboard payload so the frontend receives presentation-oriented labels/flow only, not a parameter panel.

### Frontend

- Modify: `src/vehicle_flow_ascend/web/static/index.html`
  - Replace current console-like layout.
  - Keep `particleCanvas`.
  - Remove path input, camera number input, and parameter UI.
  - Add upload card, camera card, hidden camera preview video/canvas, result video, realtime stream image, metric elements, and user-facing error/status areas.

- Modify: `src/vehicle_flow_ascend/web/static/app.js`
  - Keep upload-video background inference flow.
  - Add browser camera flow with `getUserMedia()`.
  - Pump canvas JPEG frames to backend at a fixed frontend constant.
  - Display backend MJPEG stream.
  - Render stats/status for both batch and realtime modes.

- Modify: `src/vehicle_flow_ascend/web/static/styles.css`
  - Replace AI-dashboard styling with “城市夜行” visual direction.
  - Add animated particle canvas, warm accent colors, large result stage, responsive layout, and `prefers-reduced-motion` handling.

### Tests and docs

- Create: `tests/test_frame_processor.py`
- Create: `tests/test_web_realtime.py`
- Modify: `tests/test_web_dashboard.py`
- Modify: `tests/test_web_inference.py`
- Modify: `README.md`

---

## Task 1: Extract `FrameProcessor`

**Files:**
- Modify: `src/vehicle_flow_ascend/app.py`
- Create: `tests/test_frame_processor.py`
- Regression: `tests/test_app_pipeline.py`

- [ ] **Step 1: Write failing tests for single-frame processing**

Create `tests/test_frame_processor.py` with:

```python
from __future__ import annotations

import numpy as np

from vehicle_flow_ascend.app import FrameProcessor
from vehicle_flow_ascend.config import LineConfig, VehicleFlowConfig
from vehicle_flow_ascend.types import Detection


class MovingFakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr) -> list[Detection]:
        self.calls += 1
        if self.calls == 1:
            return [Detection(4, 1, 6, 3, 0.9, "car")]
        return [Detection(4, 7, 6, 9, 0.9, "car")]


def test_frame_processor_returns_annotated_frame_and_progress() -> None:
    config = VehicleFlowConfig(line=LineConfig(start=(0, 5), end=(10, 5)))
    processor = FrameProcessor(config, MovingFakeDetector())
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    first = processor.process(frame)
    second = processor.process(frame)

    assert first.annotated_frame.shape == frame.shape
    assert first.frames == 1
    assert first.counts == {"total": 0}
    assert second.annotated_frame.shape == frame.shape
    assert second.frames == 2
    assert second.counts == {"total": 1, "car": 1}
    assert second.fps >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run from `vehicle_flow_ascend/`:

```bash
python -m pytest tests/test_frame_processor.py -q
```

Expected: FAIL because `FrameProcessor` is not defined in `vehicle_flow_ascend.app`.

- [ ] **Step 3: Implement `FrameProcessor` and refactor `run_app()`**

Modify `src/vehicle_flow_ascend/app.py` so it contains this structure. Keep existing imports, and add `dataclass` plus `Any` if missing.

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.counting.line_counter import LineCounter
from vehicle_flow_ascend.detectors.base import Detector
from vehicle_flow_ascend.tracking.centroid import CentroidTracker
from vehicle_flow_ascend.types import Line, Point
from vehicle_flow_ascend.utils.fps import FpsMeter
from vehicle_flow_ascend.video.source import VideoSource
from vehicle_flow_ascend.video.writer import VideoWriter
from vehicle_flow_ascend.visualization.overlay import draw_overlay


@dataclass(frozen=True)
class ProcessedFrame:
    annotated_frame: Any
    frames: int
    fps: float
    counts: dict[str, int]


class FrameProcessor:
    def __init__(self, config: VehicleFlowConfig, detector: Detector) -> None:
        self._detector = detector
        self._tracker = CentroidTracker()
        self._counter = LineCounter(_line_from_config(config))
        self._fps_meter = FpsMeter()
        self._frames = 0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def counts(self) -> dict[str, int]:
        return self._counter.snapshot()

    def process(self, frame_bgr) -> ProcessedFrame:
        detections = self._detector.detect(frame_bgr)
        tracks = self._tracker.update(detections)
        self._counter.update(tracks)
        fps = self._fps_meter.tick()
        counts = self._counter.snapshot()
        annotated = draw_overlay(frame_bgr, detections, tracks, self._counter.line, counts, fps)
        self._frames += 1
        return ProcessedFrame(
            annotated_frame=annotated,
            frames=self._frames,
            fps=fps,
            counts=counts,
        )
```

Then change the core of `run_app()` to use the processor:

```python
    processor = FrameProcessor(config, detector)
    display = config.display if show_window is None else show_window

    try:
        while True:
            if stop_requested is not None and stop_requested():
                break
            if config.max_frames is not None and processor.frames >= config.max_frames:
                break

            frame = source.read()
            if frame is None:
                break

            processed = processor.process(frame)
            writer.write(processed.annotated_frame)

            if progress_callback is not None:
                progress_callback(
                    {
                        "frames": processed.frames,
                        "fps": processed.fps,
                        "counts": processed.counts,
                    }
                )

            if display:
                cv2.imshow("vehicle-flow-ascend", processed.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
```

At the end of `run_app()`, return `processor.counts` instead of a local counter snapshot:

```python
    return processor.counts
```

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/test_frame_processor.py tests/test_app_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Checkpoint**

Run:

```bash
git diff -- src/vehicle_flow_ascend/app.py tests/test_frame_processor.py
```

Expected: diff only contains the processor extraction and tests. Do not commit unless the user explicitly asks for commits.

---

## Task 2: Add realtime backend manager

**Files:**
- Create: `src/vehicle_flow_ascend/web/realtime.py`
- Create: `tests/test_web_realtime.py`

- [ ] **Step 1: Write failing realtime manager tests**

Create `tests/test_web_realtime.py`:

```python
from __future__ import annotations

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


def test_realtime_manager_rejects_wrong_session(monkeypatch) -> None:
    monkeypatch.setattr(realtime, "create_detector", lambda config: FakeDetector())
    manager = RealtimeInferenceManager(VehicleFlowConfig())
    manager.start()

    with pytest.raises(ValueError, match="invalid realtime session"):
        manager.process_jpeg_frame("wrong-session", jpeg_bytes())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_web_realtime.py -q
```

Expected: FAIL because `vehicle_flow_ascend.web.realtime` does not exist.

- [ ] **Step 3: Implement realtime manager**

Create `src/vehicle_flow_ascend/web/realtime.py`:

```python
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from vehicle_flow_ascend.app import FrameProcessor
from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.detectors.base import Detector, create_detector


@dataclass
class RealtimeState:
    status: str = "idle"
    session_id: str | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    frames: int = 0
    fps: float = 0.0
    counts: dict[str, int] = field(default_factory=lambda: {"total": 0})
    error: str | None = None
    frame_version: int = 0
    updated_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "frames": self.frames,
            "fps": self.fps,
            "counts": dict(self.counts),
            "error": self.error,
            "frame_version": self.frame_version,
            "updated_at": self.updated_at,
        }


class RealtimeInferenceManager:
    def __init__(self, base_config: VehicleFlowConfig) -> None:
        self._base_config = base_config
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._state = RealtimeState()
        self._detector: Detector | None = None
        self._processor: FrameProcessor | None = None
        self._latest_jpeg: bytes | None = None

    def start(self) -> dict[str, Any]:
        with self._condition:
            if self._state.status == "running":
                raise RuntimeError("realtime inference session is already running")
            self._release_detector_locked()
            session_id = uuid.uuid4().hex[:10]
            self._detector = create_detector(self._base_config)
            self._processor = FrameProcessor(self._base_config, self._detector)
            self._latest_jpeg = None
            self._state = RealtimeState(
                status="running",
                session_id=session_id,
                started_at=time.time(),
                counts={"total": 0},
            )
            self._condition.notify_all()
            return self._state.to_dict()

    def process_jpeg_frame(self, session_id: str, jpeg_bytes: bytes) -> dict[str, Any]:
        with self._condition:
            self._ensure_session_locked(session_id)
            processor = self._processor
        if processor is None:
            raise RuntimeError("realtime processor is not initialized")

        frame = _decode_jpeg(jpeg_bytes)
        processed = processor.process(frame)
        output_jpeg = _encode_jpeg(processed.annotated_frame)

        with self._condition:
            self._ensure_session_locked(session_id)
            self._latest_jpeg = output_jpeg
            self._state.frames = processed.frames
            self._state.fps = processed.fps
            self._state.counts = processed.counts
            self._state.frame_version += 1
            self._state.updated_at = time.time()
            self._condition.notify_all()
            return self._state.to_dict()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def stop(self, session_id: str | None = None) -> dict[str, Any]:
        with self._condition:
            if session_id is not None and self._state.session_id not in {None, session_id}:
                raise ValueError("invalid realtime session")
            if self._state.status == "running":
                self._state.status = "stopped"
                self._state.stopped_at = time.time()
            self._release_detector_locked()
            self._condition.notify_all()
            return self._state.to_dict()

    def wait_for_frame(
        self,
        session_id: str,
        last_version: int,
        timeout: float = 1.0,
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            self._ensure_session_locked(session_id, allow_stopped=True)
            while True:
                if self._latest_jpeg is not None and self._state.frame_version > last_version:
                    return self._state.frame_version, self._latest_jpeg
                if self._state.status != "running":
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def _ensure_session_locked(self, session_id: str, *, allow_stopped: bool = False) -> None:
        if self._state.session_id != session_id:
            raise ValueError("invalid realtime session")
        allowed = {"running"}
        if allow_stopped:
            allowed.add("stopped")
        if self._state.status not in allowed:
            raise RuntimeError(f"realtime session is not running: {self._state.status}")

    def _release_detector_locked(self) -> None:
        release = getattr(self._detector, "release", None)
        if callable(release):
            release()
        self._detector = None
        self._processor = None


def _decode_jpeg(jpeg_bytes: bytes):
    if not jpeg_bytes:
        raise ValueError("empty realtime frame")
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode realtime JPEG frame")
    return frame


def _encode_jpeg(frame_bgr) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise ValueError("could not encode realtime output frame")
    return encoded.tobytes()
```

- [ ] **Step 4: Run realtime tests**

```bash
python -m pytest tests/test_web_realtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Checkpoint**

```bash
git diff -- src/vehicle_flow_ascend/web/realtime.py tests/test_web_realtime.py
```

Expected: diff contains only realtime manager and its tests. Do not commit unless the user explicitly asks for commits.

---

## Task 3: Wire realtime endpoints into dashboard server

**Files:**
- Modify: `src/vehicle_flow_ascend/web/dashboard.py`
- Modify: `tests/test_web_dashboard.py`

- [ ] **Step 1: Update dashboard tests for presentation payload and static expectations**

Modify `tests/test_web_dashboard.py`:

```python
from pathlib import Path

from vehicle_flow_ascend.config import load_config
from vehicle_flow_ascend.web.dashboard import _dashboard_payload, _media_path_from_query, _media_status_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_payload_exposes_presentation_not_parameter_panel() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pc_demo.yaml")
    inference_status = {
        "status": "running",
        "output_video": "runs/web_inference_demo.mp4",
        "frames": 12,
        "fps": 15.5,
        "counts": {"total": 3, "car": 2, "truck": 1},
    }
    realtime_status = {
        "status": "idle",
        "session_id": None,
        "frames": 0,
        "fps": 0.0,
        "counts": {"total": 0},
    }

    payload = _dashboard_payload(config, inference_status, realtime_status)

    assert payload["title"] == "车辆大类识别与车流统计"
    assert "config" not in payload
    assert "model_path" not in payload.get("runtime", {})
    assert payload["runtime"]["backend"] == "torch_yolov5"
    assert payload["inference"]["status"] == "running"
    assert payload["realtime"]["status"] == "idle"
    assert [item["key"] for item in payload["classes"]] == [
        "car",
        "bus",
        "truck",
        "two_wheeler",
    ]
    assert payload["input_modes"] == [
        {"key": "upload", "label": "上传视频"},
        {"key": "camera", "label": "开启摄像头"},
    ]
    assert payload["pipeline"] == [
        "选择上传视频或浏览器摄像头",
        "后端执行 YOLOv5 推理",
        "车辆跟踪与穿线计数",
        "叠加检测框和统计信息",
        "前端展示结果视频或实时标注画面",
    ]


def test_dashboard_static_assets_match_realtime_frontend() -> None:
    static_dir = PROJECT_ROOT / "src" / "vehicle_flow_ascend" / "web" / "static"

    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    styles_css = (static_dir / "styles.css").read_text(encoding="utf-8")

    assert "particleCanvas" in index_html
    assert "videoFileInput" in index_html
    assert "cameraButton" in index_html
    assert "realtimeStream" in index_html
    assert "cameraPreview" in index_html
    assert "captureCanvas" in index_html
    assert "videoPathInput" not in index_html
    assert "cameraInput" not in index_html
    assert "commandText" not in index_html
    assert "image-size" not in index_html.lower()
    assert "confidence" not in index_html.lower()
    assert "/api/inference/start" in app_js
    assert "/api/inference/upload" in app_js
    assert "/api/realtime/start" in app_js
    assert "/api/realtime/frame" in app_js
    assert "/api/realtime/stream" in app_js
    assert "getUserMedia" in app_js
    assert "initParticles" in app_js
    assert "#particleCanvas" in styles_css
    assert "prefers-reduced-motion" in styles_css
    assert "city-night" in styles_css
```

Keep existing media path/status tests in the same file.

- [ ] **Step 2: Run focused test to verify it fails**

```bash
python -m pytest tests/test_web_dashboard.py -q
```

Expected: FAIL because `_dashboard_payload()` still has old signature/content and static files still contain old controls.

- [ ] **Step 3: Modify `dashboard.py` imports and manager creation**

Add import:

```python
from vehicle_flow_ascend.web.realtime import RealtimeInferenceManager
```

In `run_dashboard()` replace manager setup with:

```python
    task_manager = InferenceTaskManager(config)
    realtime_manager = RealtimeInferenceManager(config)
    handler = _make_handler(config, static_dir, task_manager, realtime_manager)
```

Update `_make_handler()` signature:

```python
def _make_handler(
    config: VehicleFlowConfig,
    static_dir: Path,
    task_manager: InferenceTaskManager,
    realtime_manager: RealtimeInferenceManager,
):
```

- [ ] **Step 4: Add realtime GET routes**

Inside `do_GET()` add before static fallback:

```python
            if parsed.path == "/api/realtime/status":
                self._send_json(realtime_manager.status())
                return
            if parsed.path == "/api/realtime/stream":
                self._handle_realtime_stream(parsed.query)
                return
```

Update `/api/dashboard` route:

```python
            if parsed.path == "/api/dashboard":
                self._send_json(
                    _dashboard_payload(config, task_manager.status(), realtime_manager.status())
                )
                return
```

- [ ] **Step 5: Add realtime POST routes**

Inside `do_POST()` add:

```python
            if parsed.path == "/api/realtime/start":
                self._handle_realtime_start()
                return
            if parsed.path == "/api/realtime/frame":
                self._handle_realtime_frame(parsed.query)
                return
            if parsed.path == "/api/realtime/stop":
                self._handle_realtime_stop()
                return
```

- [ ] **Step 6: Add handler methods**

Inside `DashboardRequestHandler` add:

```python
        def _handle_realtime_start(self) -> None:
            try:
                self._send_json(realtime_manager.start())
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
            except Exception as exc:  # noqa: BLE001 - surface startup issue to UI
                self._send_json({"error": str(exc)}, status=400)

        def _handle_realtime_frame(self, query: str) -> None:
            try:
                session_id = _required_query_param(query, "session_id")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("empty realtime frame")
                jpeg_bytes = self.rfile.read(length)
                self._send_json(realtime_manager.process_jpeg_frame(session_id, jpeg_bytes))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
            except Exception as exc:  # noqa: BLE001 - surface realtime issue to UI
                self._send_json({"error": str(exc)}, status=400)

        def _handle_realtime_stop(self) -> None:
            try:
                payload = self._read_json()
                session_id = payload.get("session_id")
                self._send_json(realtime_manager.stop(str(session_id) if session_id else None))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except Exception as exc:  # noqa: BLE001 - surface stop issue to UI
                self._send_json({"error": str(exc)}, status=400)

        def _handle_realtime_stream(self, query: str) -> None:
            try:
                session_id = _required_query_param(query, "session_id")
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            last_version = 0
            while True:
                try:
                    frame = realtime_manager.wait_for_frame(session_id, last_version, timeout=1.0)
                    if frame is None:
                        status = realtime_manager.status()
                        if status.get("status") != "running":
                            break
                        continue
                    last_version, jpeg_bytes = frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:
                    break
```

- [ ] **Step 7: Add query helper and update payload helper**

Add module helper:

```python
def _required_query_param(query: str, name: str) -> str:
    params = parse_qs(query)
    value = params.get(name, [None])[0]
    if not value:
        raise ValueError(f"missing query parameter: {name}")
    return unquote(value)
```

Change `_dashboard_payload()` signature and body:

```python
def _dashboard_payload(
    config: VehicleFlowConfig,
    inference_status: dict[str, Any] | None = None,
    realtime_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inference_status = inference_status or {}
    realtime_status = realtime_status or {}
    return {
        "project": "Vehicle Flow Ascend",
        "title": "车辆大类识别与车流统计",
        "subtitle": "上传视频或开启浏览器摄像头，后台完成车辆检测、跟踪与车流统计。",
        "runtime": {
            "backend": config.backend,
            "mode": "background-inference",
        },
        "inference": inference_status,
        "realtime": realtime_status,
        "input_modes": [
            {"key": "upload", "label": "上传视频"},
            {"key": "camera", "label": "开启摄像头"},
        ],
        "classes": [
            {"key": "car", "label": "小型车", "accent": "#ffb25d"},
            {"key": "bus", "label": "公交/客车", "accent": "#ffd39b"},
            {"key": "truck", "label": "货车", "accent": "#9aa5ad"},
            {"key": "two_wheeler", "label": "两轮车", "accent": "#f0764f"},
        ],
        "pipeline": [
            "选择上传视频或浏览器摄像头",
            "后端执行 YOLOv5 推理",
            "车辆跟踪与穿线计数",
            "叠加检测框和统计信息",
            "前端展示结果视频或实时标注画面",
        ],
    }
```

- [ ] **Step 8: Run dashboard tests**

```bash
python -m pytest tests/test_web_dashboard.py tests/test_web_realtime.py -q
```

Expected: static test still fails until Task 4 rewrites frontend, but payload and realtime tests should pass. Record the static failures and continue to Task 4.

---

## Task 4: Rewrite frontend HTML and JavaScript

**Files:**
- Modify: `src/vehicle_flow_ascend/web/static/index.html`
- Modify: `src/vehicle_flow_ascend/web/static/app.js`
- Test: `tests/test_web_dashboard.py`

- [ ] **Step 1: Replace HTML with presentation-first layout**

Replace `src/vehicle_flow_ascend/web/static/index.html` with:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>车辆大类识别与车流统计</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='18' fill='%23080a0d'/%3E%3Cpath d='M10 40h44l-6-16H16z' fill='%23ffb25d'/%3E%3Ccircle cx='22' cy='44' r='5' fill='%23080a0d'/%3E%3Ccircle cx='44' cy='44' r='5' fill='%23080a0d'/%3E%3C/svg%3E" />
  <link rel="stylesheet" href="/styles.css" />
</head>
<body class="city-night">
  <canvas id="particleCanvas" aria-hidden="true"></canvas>
  <div class="ambient ambient-one" aria-hidden="true"></div>
  <div class="ambient ambient-two" aria-hidden="true"></div>

  <main class="app-shell">
    <header class="hero-panel surface-card">
      <nav class="top-nav" aria-label="项目导航">
        <div class="brand-lockup">
          <span class="brand-mark" aria-hidden="true"></span>
          <span>Vehicle Flow Observatory</span>
        </div>
        <div class="nav-tags" aria-label="系统能力">
          <span>后台推理</span>
          <span>实时统计</span>
          <span>结果展示</span>
        </div>
      </nav>

      <section class="hero-copy" aria-labelledby="pageTitle">
        <p class="eyebrow">EDGE TRAFFIC OBSERVATORY</p>
        <h1 id="pageTitle">看见道路<br />流动的秩序</h1>
        <p class="hero-text">上传交通视频或开启浏览器摄像头，后端完成车辆检测、跟踪与穿线计数，前端只展示推理后的画面和关键统计。</p>
      </section>
    </header>

    <section class="source-grid" aria-label="选择输入源">
      <article class="source-card surface-card is-highlighted">
        <div>
          <p class="eyebrow">VIDEO FILE</p>
          <h2>上传视频</h2>
          <p>选择本地交通视频，后台生成带检测框和统计信息的结果视频。</p>
        </div>
        <label class="file-picker">
          <input id="videoFileInput" type="file" accept="video/*" />
          <span id="fileLabel">选择视频文件</span>
        </label>
        <button class="primary-action" id="uploadVideoButton" type="button">开始视频推理</button>
      </article>

      <article class="source-card surface-card">
        <div>
          <p class="eyebrow">BROWSER CAMERA</p>
          <h2>开启摄像头</h2>
          <p>浏览器授权后采集画面，逐帧交给后端推理，并显示后端返回的标注画面。</p>
        </div>
        <button class="primary-action secondary" id="cameraButton" type="button">开启实时推理</button>
        <button class="ghost-action" id="stopButton" type="button">停止推理</button>
      </article>
    </section>

    <section class="workbench">
      <article class="result-stage surface-card" aria-label="推理结果展示">
        <div class="stage-head">
          <div>
            <p class="eyebrow">INFERENCE RESULT</p>
            <h2>推理画面</h2>
          </div>
          <div class="status-pill" id="statusPill"><span></span><strong>系统待命</strong></div>
        </div>

        <div class="media-shell" id="mediaShell">
          <div class="empty-state" id="emptyState">
            <span class="empty-orb" aria-hidden="true"></span>
            <strong>等待输入源</strong>
            <p>上传视频或开启摄像头后，这里会显示推理后的画面。</p>
          </div>
          <video id="resultVideo" controls muted playsinline hidden></video>
          <img id="realtimeStream" alt="后端实时推理后的标注画面" hidden />
          <video id="cameraPreview" muted playsinline hidden></video>
          <canvas id="captureCanvas" hidden></canvas>
        </div>
      </article>

      <aside class="insight-stack" aria-label="运行统计">
        <section class="metric-card surface-card">
          <p class="eyebrow">LIVE TELEMETRY</p>
          <div class="metric-total">
            <span>车辆总数</span>
            <strong id="taskTotal">0</strong>
          </div>
          <div class="mini-grid">
            <div><span>状态</span><strong id="taskStatus">idle</strong></div>
            <div><span>帧数</span><strong id="taskFrames">0</strong></div>
            <div><span>FPS</span><strong id="taskFps">0.0</strong></div>
          </div>
          <div class="class-counts" id="classCounts"></div>
        </section>

        <section class="flow-card surface-card">
          <p class="eyebrow">FLOW</p>
          <h2>后台流程</h2>
          <ol id="pipelineList"></ol>
        </section>
      </aside>
    </section>

    <p class="error-line" id="errorLine" role="alert"></p>
  </main>

  <script src="/app.js" defer></script>
</body>
</html>
```

- [ ] **Step 2: Replace JavaScript with upload + realtime flows**

Replace `src/vehicle_flow_ascend/web/static/app.js` with the implementation below:

```javascript
const REALTIME_FPS = 10;

const state = {
  dashboard: null,
  status: { status: 'idle', counts: { total: 0 }, frames: 0, fps: 0 },
  mode: 'idle',
  batchTimer: null,
  realtimeStatusTimer: null,
  frameTimer: null,
  frameInFlight: false,
  cameraStream: null,
  realtimeSessionId: null,
};

const byId = (id) => document.getElementById(id);

const CLASS_LABELS = {
  car: '小型车',
  bus: '公交/客车',
  truck: '货车',
  two_wheeler: '两轮车',
};

const CLASS_COLORS = {
  car: '#ffb25d',
  bus: '#ffd39b',
  truck: '#9aa5ad',
  two_wheeler: '#f0764f',
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    cache: 'no-store',
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function loadDashboard() {
  state.dashboard = await requestJson('/api/dashboard');
  renderPipeline(state.dashboard.pipeline || []);
  renderClassCounts(state.status.counts || {});
  await refreshBatchStatus();
}

async function startUploadedVideo() {
  await stopRealtime({ silent: true });
  setError('');
  const fileInput = byId('videoFileInput');
  if (!fileInput.files || fileInput.files.length === 0) {
    setError('请先选择一个视频文件');
    return;
  }
  setBusy(true, 'upload');
  try {
    showEmpty('正在上传视频并启动后台推理…');
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    const uploaded = await requestJson('/api/inference/upload', { method: 'POST', body: formData });
    state.status = await requestJson('/api/inference/start', {
      method: 'POST',
      body: JSON.stringify({ source_type: 'video', source: uploaded.path }),
    });
    state.mode = 'batch';
    renderStatus(state.status);
    startBatchPolling();
  } catch (error) {
    setError(error.message || '视频推理启动失败');
    showEmpty('视频推理启动失败，请检查后端环境。');
  } finally {
    setBusy(false, 'upload');
  }
}

async function startRealtimeCamera() {
  await stopRealtime({ silent: true });
  setError('');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setError('当前浏览器不支持摄像头调用');
    return;
  }
  setBusy(true, 'camera');
  try {
    showEmpty('正在请求浏览器摄像头权限…');
    state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const preview = byId('cameraPreview');
    preview.srcObject = state.cameraStream;
    await preview.play();

    const started = await requestJson('/api/realtime/start', {
      method: 'POST',
      body: JSON.stringify({}),
    });
    state.realtimeSessionId = started.session_id;
    state.mode = 'realtime';
    state.status = started;
    renderStatus(started);
    showRealtimeStream(started.session_id);
    startFramePump();
    startRealtimeStatusPolling();
  } catch (error) {
    setError(cameraErrorMessage(error));
    await stopRealtime({ silent: true });
    showEmpty('摄像头实时推理启动失败。');
  } finally {
    setBusy(false, 'camera');
  }
}

async function stopRealtime({ silent = false } = {}) {
  clearTimers();
  stopCameraTracks();
  hideRealtimeStream();
  const sessionId = state.realtimeSessionId;
  state.realtimeSessionId = null;
  state.mode = 'idle';
  if (sessionId) {
    try {
      state.status = await requestJson('/api/realtime/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      });
      renderStatus(state.status);
    } catch (error) {
      if (!silent) setError(error.message || '停止实时推理失败');
    }
  } else if (!silent) {
    showEmpty('实时推理已停止。');
  }
}

async function refreshBatchStatus() {
  state.status = await requestJson('/api/inference/status');
  renderStatus(state.status);
  renderClassCounts(state.status.counts || {});
  if (['completed', 'stopped'].includes(state.status.status)) {
    await renderResultVideo(state.status.output_video);
  }
  return state.status;
}

function startBatchPolling() {
  if (state.batchTimer) clearInterval(state.batchTimer);
  state.batchTimer = setInterval(async () => {
    try {
      const status = await refreshBatchStatus();
      if (!['running', 'stopping'].includes(status.status)) {
        clearInterval(state.batchTimer);
        state.batchTimer = null;
      }
    } catch (error) {
      setError(error.message || '状态刷新失败');
    }
  }, 1000);
}

function startRealtimeStatusPolling() {
  if (state.realtimeStatusTimer) clearInterval(state.realtimeStatusTimer);
  state.realtimeStatusTimer = setInterval(async () => {
    try {
      const status = await requestJson('/api/realtime/status');
      state.status = status;
      renderStatus(status);
      renderClassCounts(status.counts || {});
    } catch (error) {
      setError(error.message || '实时状态刷新失败');
    }
  }, 1000);
}

function startFramePump() {
  if (state.frameTimer) clearInterval(state.frameTimer);
  state.frameTimer = setInterval(captureAndSendFrame, Math.round(1000 / REALTIME_FPS));
}

function captureAndSendFrame() {
  if (!state.realtimeSessionId || state.frameInFlight) return;
  const preview = byId('cameraPreview');
  if (!preview.videoWidth || !preview.videoHeight) return;

  const canvas = byId('captureCanvas');
  canvas.width = preview.videoWidth;
  canvas.height = preview.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
  state.frameInFlight = true;

  canvas.toBlob(async (blob) => {
    if (!blob || !state.realtimeSessionId) {
      state.frameInFlight = false;
      return;
    }
    try {
      const status = await requestJson(`/api/realtime/frame?session_id=${encodeURIComponent(state.realtimeSessionId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'image/jpeg' },
        body: blob,
      });
      state.status = status;
      renderStatus(status);
      renderClassCounts(status.counts || {});
    } catch (error) {
      setError(error.message || '实时帧推理失败');
    } finally {
      state.frameInFlight = false;
    }
  }, 'image/jpeg', 0.82);
}

async function renderResultVideo(outputPath) {
  const video = byId('resultVideo');
  if (!outputPath) return;
  const status = await requestJson(`/api/media-status?path=${encodeURIComponent(outputPath)}`);
  if (!status.exists) return;
  hideRealtimeStream();
  byId('emptyState').hidden = true;
  video.hidden = false;
  video.src = `/media/output-video?path=${encodeURIComponent(outputPath)}&t=${Date.now()}`;
  await video.play().catch(() => {});
}

function showRealtimeStream(sessionId) {
  const img = byId('realtimeStream');
  byId('resultVideo').hidden = true;
  byId('emptyState').hidden = true;
  img.hidden = false;
  img.src = `/api/realtime/stream?session_id=${encodeURIComponent(sessionId)}&t=${Date.now()}`;
}

function hideRealtimeStream() {
  const img = byId('realtimeStream');
  img.hidden = true;
  img.removeAttribute('src');
}

function showEmpty(message) {
  byId('resultVideo').hidden = true;
  hideRealtimeStream();
  const empty = byId('emptyState');
  empty.hidden = false;
  empty.querySelector('p').textContent = message;
}

function renderStatus(status) {
  const statusText = status.status || 'idle';
  byId('taskStatus').textContent = statusLabel(statusText).short;
  byId('taskFrames').textContent = String(status.frames || 0);
  byId('taskFps').textContent = Number(status.fps || 0).toFixed(1);
  byId('taskTotal').textContent = String((status.counts || {}).total || 0);
  byId('errorLine').textContent = status.error || '';

  const label = statusLabel(statusText);
  const pill = byId('statusPill');
  pill.dataset.status = statusText;
  pill.innerHTML = `<span></span><strong>${label.title}</strong>`;
}

function statusLabel(status) {
  const labels = {
    idle: { title: '系统待命', short: '待命' },
    uploading: { title: '正在上传', short: '上传中' },
    running: { title: state.mode === 'realtime' ? '实时推理中' : '后台推理中', short: '推理中' },
    stopping: { title: '正在停止', short: '停止中' },
    completed: { title: '推理完成', short: '完成' },
    stopped: { title: '已停止', short: '已停止' },
    failed: { title: '推理失败', short: '失败' },
  };
  return labels[status] || { title: status, short: status };
}

function renderClassCounts(counts) {
  byId('classCounts').innerHTML = Object.entries(CLASS_LABELS)
    .map(([key, label]) => `
      <div class="class-row" style="--class-color:${CLASS_COLORS[key]}">
        <span>${label}</span>
        <strong>${counts[key] || 0}</strong>
      </div>
    `)
    .join('');
}

function renderPipeline(steps) {
  byId('pipelineList').innerHTML = steps.map((step) => `<li>${escapeHtml(step)}</li>`).join('');
}

function setBusy(isBusy, mode) {
  const uploadButton = byId('uploadVideoButton');
  const cameraButton = byId('cameraButton');
  uploadButton.disabled = isBusy;
  cameraButton.disabled = isBusy;
  if (mode === 'upload') uploadButton.textContent = isBusy ? '启动中…' : '开始视频推理';
  if (mode === 'camera') cameraButton.textContent = isBusy ? '连接中…' : '开启实时推理';
}

function setError(message) {
  byId('errorLine').textContent = message || '';
}

function clearTimers() {
  if (state.frameTimer) clearInterval(state.frameTimer);
  if (state.realtimeStatusTimer) clearInterval(state.realtimeStatusTimer);
  state.frameTimer = null;
  state.realtimeStatusTimer = null;
  state.frameInFlight = false;
}

function stopCameraTracks() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  byId('cameraPreview').srcObject = null;
}

function cameraErrorMessage(error) {
  if (error && error.name === 'NotAllowedError') return '浏览器未授予摄像头权限';
  if (error && error.name === 'NotFoundError') return '没有找到可用摄像头';
  return error.message || '摄像头启动失败';
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function initParticles() {
  const canvas = byId('particleCanvas');
  const ctx = canvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const particles = [];
  const particleCount = reduceMotion ? 28 : 88;

  function resize() {
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
  }

  function makeParticle() {
    return {
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.25 * devicePixelRatio,
      vy: (Math.random() - 0.5) * 0.18 * devicePixelRatio,
      r: (Math.random() * 1.8 + 0.7) * devicePixelRatio,
      alpha: Math.random() * 0.45 + 0.18,
    };
  }

  function seed() {
    particles.length = 0;
    for (let i = 0; i < particleCount; i += 1) particles.push(makeParticle());
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 178, 93, ${p.alpha})`;
      ctx.fill();
    }
    if (!reduceMotion) requestAnimationFrame(draw);
  }

  resize();
  seed();
  draw();
  window.addEventListener('resize', () => {
    resize();
    seed();
  });
}

function bindEvents() {
  byId('uploadVideoButton').addEventListener('click', startUploadedVideo);
  byId('cameraButton').addEventListener('click', startRealtimeCamera);
  byId('stopButton').addEventListener('click', () => stopRealtime());
  byId('videoFileInput').addEventListener('change', () => {
    const file = byId('videoFileInput').files?.[0];
    byId('fileLabel').textContent = file ? file.name : '选择视频文件';
  });
}

async function boot() {
  initParticles();
  bindEvents();
  await loadDashboard();
}

boot().catch((error) => setError(error.message));
```

- [ ] **Step 3: Run static frontend test**

```bash
python -m pytest tests/test_web_dashboard.py::test_dashboard_static_assets_match_realtime_frontend -q
```

Expected: FAIL until CSS includes required selectors and `prefers-reduced-motion`.

---

## Task 5: Rewrite city-night CSS

**Files:**
- Modify: `src/vehicle_flow_ascend/web/static/styles.css`
- Test: `tests/test_web_dashboard.py`

- [ ] **Step 1: Replace CSS with city-night theme**

Replace `src/vehicle_flow_ascend/web/static/styles.css` with CSS that defines these exact selectors and concepts:

```css
:root {
  --bg: #07080b;
  --surface: rgba(18, 21, 27, 0.72);
  --surface-strong: rgba(24, 28, 36, 0.88);
  --line: rgba(255, 255, 255, 0.12);
  --ink: #f7efe6;
  --muted: rgba(247, 239, 230, 0.66);
  --accent: #ffb25d;
  --accent-2: #f0764f;
  --steel: #9aa5ad;
  --danger: #ff6f61;
  --shadow: 0 28px 100px rgba(0, 0, 0, 0.42);
  --display: "Bahnschrift", "DIN Condensed", "Microsoft YaHei UI", sans-serif;
  --body: "Aptos", "Microsoft YaHei UI", sans-serif;
  --mono: "Cascadia Mono", "JetBrains Mono", monospace;
}

* { box-sizing: border-box; }

body.city-night {
  min-height: 100vh;
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 12% 12%, rgba(255, 178, 93, 0.18), transparent 32%),
    radial-gradient(circle at 86% 20%, rgba(240, 118, 79, 0.11), transparent 30%),
    linear-gradient(135deg, #111419 0%, #07080b 68%, #030405 100%);
  font-family: var(--body);
  overflow-x: hidden;
}

body.city-night::after {
  position: fixed;
  inset: 0;
  content: "";
  pointer-events: none;
  background-image:
    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(circle at 50% 45%, black 0%, transparent 78%);
}

#particleCanvas {
  position: fixed;
  inset: 0;
  z-index: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
}

.ambient {
  position: fixed;
  z-index: 0;
  pointer-events: none;
  border-radius: 999px;
  filter: blur(44px);
  opacity: 0.52;
  animation: ambient-drift 16s ease-in-out infinite alternate;
}

.ambient-one {
  width: 420px;
  height: 420px;
  left: -120px;
  top: -120px;
  background: rgba(255, 178, 93, 0.18);
}

.ambient-two {
  width: 520px;
  height: 520px;
  right: -160px;
  bottom: 5vh;
  background: rgba(240, 118, 79, 0.12);
  animation-delay: -5s;
}

.app-shell {
  position: relative;
  z-index: 1;
  width: min(1440px, calc(100vw - 40px));
  margin: 0 auto;
  padding: 30px 0 46px;
}

.surface-card {
  border: 1px solid var(--line);
  border-radius: 30px;
  background: linear-gradient(145deg, var(--surface), rgba(8, 10, 14, 0.54));
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(22px);
}

.hero-panel { padding: 28px; overflow: hidden; }
.top-nav { display: flex; justify-content: space-between; align-items: center; gap: 20px; }
.brand-lockup { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: 0.02em; }
.brand-mark { width: 42px; height: 42px; border-radius: 15px; background: linear-gradient(135deg, var(--accent), #5d6268); box-shadow: 0 0 36px rgba(255,178,93,.26); }
.nav-tags { display: flex; flex-wrap: wrap; gap: 10px; }
.nav-tags span { padding: 8px 12px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); background: rgba(255,255,255,.045); font-size: 13px; }

.eyebrow { margin: 0 0 12px; color: var(--accent); font-family: var(--mono); font-size: 12px; letter-spacing: .22em; text-transform: uppercase; }
.hero-copy { margin-top: 52px; max-width: 980px; }
h1, h2 { margin: 0; font-family: var(--display); letter-spacing: -0.045em; line-height: .98; }
h1 { font-size: clamp(54px, 8vw, 112px); }
h2 { font-size: clamp(28px, 2.2vw, 42px); }
.hero-text { max-width: 760px; margin: 22px 0 0; color: var(--muted); font-size: 18px; line-height: 1.8; }

.source-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; margin-top: 22px; }
.source-card { display: grid; gap: 20px; min-height: 250px; padding: 24px; }
.source-card p:not(.eyebrow) { margin: 12px 0 0; color: var(--muted); line-height: 1.65; }
.source-card.is-highlighted { border-color: rgba(255,178,93,.3); }
.file-picker { display: flex; align-items: center; justify-content: center; min-height: 58px; border: 1px dashed rgba(255,178,93,.42); border-radius: 18px; color: #ffd39b; cursor: pointer; background: rgba(255,178,93,.08); }
.file-picker input { position: absolute; inline-size: 1px; block-size: 1px; opacity: 0; pointer-events: none; }
.primary-action, .ghost-action { min-height: 54px; border: 0; border-radius: 18px; font-weight: 800; cursor: pointer; transition: transform .2s ease, opacity .2s ease, background .2s ease; }
.primary-action { color: #15100b; background: linear-gradient(135deg, var(--accent), #ffd39b); box-shadow: 0 18px 42px rgba(255,178,93,.22); }
.primary-action.secondary { color: var(--ink); background: linear-gradient(135deg, rgba(255,178,93,.24), rgba(255,255,255,.08)); border: 1px solid rgba(255,178,93,.3); }
.ghost-action { color: var(--ink); background: rgba(255,255,255,.06); border: 1px solid var(--line); }
.primary-action:hover, .ghost-action:hover { transform: translateY(-2px); }
.primary-action:disabled, .ghost-action:disabled { opacity: .55; cursor: not-allowed; transform: none; }

.workbench { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(320px, .55fr); gap: 22px; margin-top: 22px; }
.result-stage { min-height: 640px; padding: 24px; }
.stage-head { display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 20px; }
.status-pill { display: inline-flex; align-items: center; gap: 10px; padding: 11px 14px; border-radius: 999px; background: rgba(255,178,93,.1); color: #ffd39b; border: 1px solid rgba(255,178,93,.22); font-family: var(--mono); font-size: 12px; }
.status-pill span { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 16px var(--accent); }
.status-pill[data-status="failed"] { color: #ffb7ae; border-color: rgba(255,111,97,.35); background: rgba(255,111,97,.1); }
.status-pill[data-status="failed"] span { background: var(--danger); box-shadow: 0 0 16px var(--danger); }
.media-shell { position: relative; display: grid; place-items: center; min-height: 520px; overflow: hidden; border: 1px solid rgba(255,255,255,.12); border-radius: 28px; background: linear-gradient(145deg, #1d2229, #0c0f14); }
.media-shell::before { content: ""; position: absolute; inset: 16%; border: 1px solid rgba(255,178,93,.16); border-radius: 50%; transform: rotate(-8deg) scaleX(1.35); }
.media-shell video, .media-shell img { position: relative; z-index: 1; width: 100%; height: 520px; object-fit: cover; border-radius: 24px; }
.empty-state { position: relative; z-index: 1; display: grid; place-items: center; gap: 13px; max-width: 520px; padding: 32px; text-align: center; color: var(--muted); }
.empty-state strong { color: var(--ink); font-family: var(--display); font-size: 36px; }
.empty-orb { width: 76px; height: 76px; border-radius: 50%; background: radial-gradient(circle at 35% 30%, #ffd39b, var(--accent) 45%, rgba(255,178,93,.08) 70%); box-shadow: 0 0 56px rgba(255,178,93,.34); }

.insight-stack { display: grid; gap: 22px; }
.metric-card, .flow-card { padding: 24px; }
.metric-total span, .mini-grid span { display: block; color: var(--muted); font-size: 13px; }
.metric-total strong { display: block; margin-top: 6px; font-family: var(--display); font-size: 76px; line-height: .9; }
.mini-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 20px; }
.mini-grid div { padding: 13px; border-radius: 16px; background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.08); }
.mini-grid strong { display: block; margin-top: 7px; }
.class-counts { display: grid; gap: 10px; margin-top: 18px; }
.class-row { display: flex; align-items: center; justify-content: space-between; padding: 12px 13px; border-radius: 16px; color: var(--muted); background: linear-gradient(90deg, color-mix(in srgb, var(--class-color) 16%, transparent), rgba(255,255,255,.045)); border: 1px solid rgba(255,255,255,.08); }
.class-row strong { color: var(--ink); }
.flow-card ol { margin: 18px 0 0; padding-left: 20px; color: var(--muted); line-height: 1.85; }
.error-line { min-height: 24px; margin: 16px 8px 0; color: #ffb7ae; font-weight: 700; }

@keyframes ambient-drift { from { transform: translate3d(0, 0, 0) scale(1); } to { transform: translate3d(24px, -18px, 0) scale(1.08); } }

@media (max-width: 1080px) {
  .source-grid, .workbench { grid-template-columns: 1fr; }
  .result-stage { min-height: auto; }
}

@media (max-width: 720px) {
  .app-shell { width: min(100vw - 22px, 1440px); padding-top: 12px; }
  .top-nav, .stage-head { align-items: flex-start; flex-direction: column; }
  .nav-tags { display: none; }
  .hero-panel, .source-card, .result-stage, .metric-card, .flow-card { border-radius: 22px; padding: 18px; }
  h1 { font-size: clamp(48px, 16vw, 76px); }
  .media-shell, .media-shell video, .media-shell img { min-height: 360px; height: 360px; }
  .mini-grid { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; }
}
```

- [ ] **Step 2: Run static tests**

```bash
python -m pytest tests/test_web_dashboard.py -q
```

Expected: PASS for dashboard tests.

- [ ] **Step 3: Browser smoke check**

Start app manually if model dependencies are available:

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web
```

Open `http://127.0.0.1:8765`; verify the page shows two source cards and no path/camera-number inputs.

---

## Task 6: Update upload-video manager tests and keep existing flow stable

**Files:**
- Modify: `tests/test_web_inference.py`
- Modify: `src/vehicle_flow_ascend/web/inference.py` only if tests reveal a mismatch.

- [ ] **Step 1: Update source parsing test to reflect frontend behavior**

Change `test_source_from_payload_accepts_camera_and_video()` to:

```python
def test_source_from_payload_accepts_video_payload_for_uploaded_file() -> None:
    source = _source_from_payload({"source_type": "video", "source": "data/web_uploads/demo.mp4"})

    assert source.path == "data/web_uploads/demo.mp4"
    assert source.camera is None
```

Add a backend compatibility test for camera payload if desired by CLI/server internals, but the frontend must not depend on camera numbers:

```python
def test_source_from_payload_still_accepts_camera_for_backend_compatibility() -> None:
    assert _source_from_payload({"source_type": "camera", "source": "1"}).camera == 1
    assert _source_from_payload({"source_type": "camera", "source": ""}).camera == 0
```

- [ ] **Step 2: Run upload task tests**

```bash
python -m pytest tests/test_web_inference.py -q
```

Expected: PASS. If it fails because `output_video` can be provided by frontend, remove frontend-side use only; the backend may still accept it internally for compatibility.

- [ ] **Step 3: Run backend web tests together**

```bash
python -m pytest tests/test_web_inference.py tests/test_web_realtime.py tests/test_web_dashboard.py -q
```

Expected: PASS.

---

## Task 7: Update README Web Dashboard docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace Web Dashboard section**

In `README.md`, replace the section starting at `## 交互式可视化前端` with:

```markdown
## 交互式可视化前端

项目内置一个无需 Node/Vite 的 Web Dashboard。前端只负责输入选择、页面效果、状态反馈和推理结果展示；模型路径、输入尺寸、阈值、计数线等工程参数继续由 YAML 配置管理，不在页面暴露。

启动前端：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web
```

浏览器打开：

```text
http://127.0.0.1:8765
```

如需改监听地址和端口：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web --web-host 0.0.0.0 --web-port 8899
```

前端提供两种输入方式：

1. **上传视频文件**：选择本地交通视频，后端保存到 `data/web_uploads/`，随后后台调用现有推理流水线生成标注后结果视频，页面完成后播放结果视频。
2. **开启浏览器摄像头**：浏览器请求摄像头权限，前端按固定帧率抽帧发送给后端，后端执行车辆检测、跟踪、计数和画面叠加，页面显示后端返回的准实时标注画面。

页面展示内容：

- 推理后的视频或实时标注画面；
- 当前状态；
- 已处理帧数；
- FPS；
- 总车流量和分类车流量；
- 简短后台流程说明。
```

- [ ] **Step 2: Search for stale wording**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text(encoding='utf-8')
for needle in ['视频路径', '摄像头编号', '模型路径、输入尺寸、阈值等工程参数继续由 YAML 配置管理，不在前端暴露']:
    print(needle, needle in text)
PY
```

Expected:

- `视频路径 False`
- `摄像头编号 False`
- old wording may be False if replaced.

- [ ] **Step 3: Check README diff**

```bash
git diff -- README.md
```

Expected: Web Dashboard docs match the new upload/camera realtime UX.

---

## Task 8: Full automated verification

**Files:**
- No planned source edits unless tests uncover issues.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 2: Run lint if available**

```bash
python -m ruff check src tests
```

Expected: PASS. If ruff is not installed, install dev extras only if the environment already uses editable dev install; otherwise report that lint could not run.

- [ ] **Step 3: Inspect current diff**

```bash
git status --short
git diff --stat
```

Expected modified/created files are limited to:

- `README.md`
- `src/vehicle_flow_ascend/app.py`
- `src/vehicle_flow_ascend/web/dashboard.py`
- `src/vehicle_flow_ascend/web/realtime.py`
- `src/vehicle_flow_ascend/web/static/index.html`
- `src/vehicle_flow_ascend/web/static/app.js`
- `src/vehicle_flow_ascend/web/static/styles.css`
- `tests/test_frame_processor.py`
- `tests/test_web_dashboard.py`
- `tests/test_web_inference.py`
- `tests/test_web_realtime.py`
- `docs/superpowers/specs/2026-06-07-frontend-realtime-inference-design.md`
- `docs/superpowers/plans/2026-06-07-frontend-realtime-inference.md`

Also expect the pre-existing deleted temporary Word lock file under the Chinese course-doc directory unless separately restored.

---

## Task 9: Manual app verification

**Files:**
- No planned source edits unless manual verification reveals bugs.

- [ ] **Step 1: Start the dashboard**

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web
```

Expected terminal output includes:

```text
Vehicle Flow Dashboard running at http://127.0.0.1:8765
```

- [ ] **Step 2: Verify static UI**

Open `http://127.0.0.1:8765`.

Expected:

- City-night dark visual direction.
- Dynamic warm particles in the background.
- Two source cards: 上传视频 and 开启摄像头.
- No video path input.
- No camera index input.
- No model path, image size, threshold, or command text panel.

- [ ] **Step 3: Verify upload flow**

Choose a short local video file.

Expected:

- File name appears in the picker label.
- Clicking `开始视频推理` uploads the video.
- Status changes to 推理中.
- Frames/FPS/counts update while backend runs.
- On completion, result video plays in the main stage.

- [ ] **Step 4: Verify browser camera flow**

Click `开启实时推理` and grant browser camera permission.

Expected:

- Browser permission dialog appears.
- Status changes to 实时推理中.
- Main stage displays backend MJPEG stream.
- Frames/FPS/counts update.
- Clicking `停止推理` stops frame pumping and releases the browser camera track.

- [ ] **Step 5: Verify error states**

Run these manual checks:

- Click `开始视频推理` without selecting a file.
  - Expected: `请先选择一个视频文件`.
- Deny browser camera permission.
  - Expected: `浏览器未授予摄像头权限`.
- Start realtime twice quickly.
  - Expected: frontend does not leave two active sessions; backend returns a readable conflict if hit.

---

## Self-Review

- Spec coverage:
  - Backend reuse: Task 1 and Task 2.
  - Upload video flow: Task 4 and Task 6.
  - Browser camera realtime MJPEG/continuous annotated frames: Task 2, Task 3, Task 4, Task 9.
  - UI removes engineering parameters: Task 3, Task 4, Task 5.
  - City-night modern UI and particles: Task 4, Task 5, Task 9.
  - Tests and docs: Task 1, Task 2, Task 3, Task 6, Task 7, Task 8.
- Placeholder scan: no TBD/TODO/fill-in placeholders are intentionally left.
- Type consistency:
  - `RealtimeState.session_id`, `RealtimeInferenceManager.process_jpeg_frame()`, `wait_for_frame()` and frontend `realtimeSessionId` all use the same session-id name.
  - `ProcessedFrame.annotated_frame`, `frames`, `fps`, and `counts` are consumed consistently by `run_app()` and realtime manager.
  - Frontend IDs in HTML match `app.js` and static tests.
