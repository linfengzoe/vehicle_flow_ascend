from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from io import BytesIO
from urllib.parse import quote
from urllib.request import Request, urlopen

from vehicle_flow_ascend.config import VehicleFlowConfig, load_config
from vehicle_flow_ascend.web.dashboard import (
    DashboardServerConfig,
    _dashboard_payload,
    _make_handler,
    _media_status_payload,
    _read_multipart_video_upload,
    _resolve_safe_media_path,
    _sanitize_start_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeStatusManager:
    def __init__(self, status: dict | None = None) -> None:
        self._status = status or {}

    def status(self) -> dict:
        return dict(self._status)


class _FakeInferenceStreamManager(_FakeStatusManager):
    def wait_for_frame(self, task_id: str, last_version: int, timeout: float = 1.0):
        assert task_id == "task-1"
        assert last_version == 0
        assert timeout > 0
        return 1, b"\xff\xd8demo-jpeg\xff\xd9"


def test_dashboard_payload_exposes_presentation_not_parameter_panel() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pc_demo.yaml")
    inference_status = {
        "status": "running",
        "output_video": "runs/web_inference_demo.mp4",
        "frames": 12,
        "fps": 15.5,
        "counts": {"total": 3, "car": 2, "truck": 1},
    }
    realtime_status = {"status": "idle"}

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
        {"key": "camera", "label": "开启浏览器摄像头"},
        {"key": "devboard_camera", "label": "使用开发板摄像头"},
    ]
    assert payload["pipeline"] == [
        "上传视频 / 使用浏览器摄像头 / 使用开发板摄像头",
        "后端执行 YOLOv5 推理",
        "车辆跟踪与穿线计数",
        "叠加检测框和统计信息",
        "前端展示结果视频或实时标注画面",
    ]


def test_dashboard_server_config_uses_https_url_when_tls_enabled() -> None:
    config = DashboardServerConfig(
        host="0.0.0.0",
        port=8766,
        certfile="certs/web.crt",
        keyfile="certs/web.key",
    )

    assert config.use_tls is True
    assert config.url == "https://0.0.0.0:8766"


def test_dashboard_static_assets_match_realtime_frontend() -> None:
    static_dir = PROJECT_ROOT / "src" / "vehicle_flow_ascend" / "web" / "static"

    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    styles_css = (static_dir / "styles.css").read_text(encoding="utf-8")
    lower_index = index_html.lower()

    assert (static_dir / "index.html").exists()
    assert (static_dir / "styles.css").exists()
    assert (static_dir / "app.js").exists()
    assert "asciiVehicle" in index_html
    assert "archiveScanline" in index_html
    assert "particleVehicleCanvas" in index_html
    assert "videoFileInput" in index_html
    assert "cameraButton" in index_html
    assert "devboardCameraButton" in index_html
    assert "cameraDeviceSelect" in index_html
    assert "realtimeStream" in index_html
    assert "cameraPreview" in index_html
    assert "lineSetupPanel" in index_html
    assert "linePreviewVideo" in index_html
    assert "lineOverlayCanvas" in index_html
    assert "analysisStartButton" in index_html
    assert "captureCanvas" in index_html
    assert "videoPathInput" not in index_html
    assert "cameraInput" not in index_html
    assert "commandText" not in index_html
    assert "image-size" not in lower_index
    assert "confidence" not in lower_index
    assert "/api/inference/start" in app_js
    assert "/api/inference/upload" in app_js
    assert "/api/realtime/start" in app_js
    assert "/api/realtime/start-devboard-camera" in app_js
    assert "/api/realtime/frame" in app_js
    assert "/api/realtime/stream" in app_js
    assert "AbortController" in app_js
    assert "signal: abortController.signal" in app_js
    assert "refreshCameraDevices" in app_js
    assert "window.isSecureContext" in app_js
    assert "HTTPS" in app_js
    assert "deviceId: { exact: selectedDeviceId }" in app_js
    assert "MAX_REALTIME_FRAME_FAILURES" in app_js
    assert "realtimeFrameFailures" in app_js
    assert "const MAX_CAPTURE_WIDTH = 480;" in app_js
    assert "const REALTIME_CAPTURE_DELAY_MS = 80;" in app_js
    assert "scaleCaptureDimensions" in app_js
    assert "scheduleCaptureTick" in app_js
    assert "requestVideoFrameCallback" in app_js
    assert "window.requestAnimationFrame" in app_js
    assert "initializeLineEditor" in app_js
    assert "lineFromEditor" in app_js
    assert "lineForRealtimeCapture" in app_js
    assert "showInferenceStream" in app_js
    assert "startCameraPreviewFlow" in app_js
    assert "startCameraAnalysisFlow" in app_js
    assert "startDevboardCameraFlow" in app_js
    assert "camera_index: 'auto'" in app_js
    assert "camera-preview" in app_js
    assert "/api/inference/stream" in app_js
    assert "line: lineFromEditor()" in app_js
    assert "line: lineForRealtimeCapture()" in app_js
    assert "getUserMedia" in app_js
    assert "updateAsciiTelemetry" in app_js
    assert "initParticleVehicleField" in app_js
    assert "pauseParticleVehicleField" in app_js
    assert "hideEmptyState" in app_js
    assert "noise2D" in app_js
    assert "sdCar" in app_js
    assert "vehicleParticleState" in app_js
    assert "dragging" in app_js
    assert "pointermove" in app_js
    assert "cancelAnimationFrame(particleState.animationFrame)" in app_js
    assert "canvas.width = 1" in app_js
    assert "hideEmptyState()" in app_js
    assert "STATUS_TAGS" in app_js
    assert "待命" in app_js
    assert "预览" in app_js
    assert "运行" in app_js
    assert "完成" in app_js
    assert "ONLINE" not in app_js
    assert "PREVIEW" not in app_js
    assert "RUNNING" not in app_js
    assert "COMPLETE" not in app_js
    assert "@keyframes ascii-drive" in styles_css
    assert "@keyframes scanline-sweep" in styles_css
    assert "#particleVehicleCanvas" in styles_css
    assert "cursor: grab;" in styles_css
    assert "cursor: grabbing;" in styles_css
    assert ".archive-shell" in styles_css
    assert "prefers-reduced-motion" in styles_css
    result_media_block = styles_css.split("#realtimeStream,", 1)[1].split(".empty-state", 1)[0]
    assert "object-fit: contain;" in result_media_block


def test_media_status_payload_reports_missing_and_existing_file(tmp_path) -> None:
    missing = _media_status_payload(str(tmp_path / "missing.mp4"))
    assert missing["exists"] is False
    assert missing["size"] == 0

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"demo")
    existing = _media_status_payload(str(video))
    assert existing["exists"] is True
    assert existing["size"] == 4


def test_multipart_upload_parser_extracts_video_field_bytes() -> None:
    boundary = "----codex-boundary"
    body = (
        b"------codex-boundary\r\n"
        b'Content-Disposition: form-data; name="video"; filename="demo.mp4"\r\n'
        b"Content-Type: video/mp4\r\n"
        b"\r\n"
        b"abc\x00def\r\n"
        b"------codex-boundary--\r\n"
    )

    filename, source = _read_multipart_video_upload(
        f"multipart/form-data; boundary={boundary}",
        body,
    )

    assert filename == "demo.mp4"
    assert isinstance(source, BytesIO)
    assert source.read() == b"abc\x00def"


def test_sanitize_start_payload_removes_client_output_video() -> None:
    payload = {
        "source": "data/web_uploads/demo.mp4",
        "output_video": "D:/not-allowed/output.mp4",
        "max_frames": 12,
        "line": [[12, 34], [320, 210]],
    }

    sanitized = _sanitize_start_payload(payload, PROJECT_ROOT)

    assert sanitized == {
        "source_type": "video",
        "source": str(PROJECT_ROOT / "data" / "web_uploads" / "demo.mp4"),
        "max_frames": 12,
        "line": [[12, 34], [320, 210]],
    }
    assert payload["output_video"] == "D:/not-allowed/output.mp4"


def test_sanitize_start_payload_rejects_camera_source() -> None:
    try:
        _sanitize_start_payload({"source_type": "camera", "source": "0"})
    except ValueError as exc:
        assert "uploaded video" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("camera source should be rejected by dashboard start payload")


def test_sanitize_start_payload_rejects_unuploaded_video_path() -> None:
    try:
        _sanitize_start_payload({"source_type": "video", "source": "C:/Windows/win.ini"})
    except ValueError as exc:
        assert "uploaded video" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("non-uploaded local path should be rejected")


def test_sanitize_start_payload_rejects_upload_path_traversal() -> None:
    for source in (
        "data/web_uploads/../outside.mp4",
        "C:/tmp/data/web_uploads/../secret.mp4",
    ):
        try:
            _sanitize_start_payload({"source_type": "video", "source": source})
        except ValueError as exc:
            assert "uploaded video" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"path traversal should be rejected: {source}")


def test_safe_media_path_rejects_unlisted_query_path(tmp_path) -> None:
    allowed = tmp_path / "allowed.mp4"
    allowed.write_bytes(b"allowed")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    config = VehicleFlowConfig(output_video=str(allowed))

    resolved = _resolve_safe_media_path(
        config,
        f"path={quote(str(outside))}",
        {"output_video": str(allowed)},
    )

    assert resolved is None


def test_safe_media_path_does_not_resolve_unlisted_query_path(tmp_path, monkeypatch) -> None:
    allowed = tmp_path / "allowed.mp4"
    allowed.write_bytes(b"allowed")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    config = VehicleFlowConfig(output_video=str(allowed))
    original_resolve = Path.resolve
    resolved_inputs: list[str] = []

    def tracking_resolve(self, *args, **kwargs):
        resolved_inputs.append(str(self))
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", tracking_resolve)

    resolved = _resolve_safe_media_path(config, f"path={quote(str(outside))}", {})

    assert resolved is None
    assert str(allowed) in resolved_inputs
    assert str(outside) not in resolved_inputs


def test_safe_media_path_allows_inference_output_query_path(tmp_path) -> None:
    configured = tmp_path / "configured.mp4"
    configured.write_bytes(b"configured")
    inferred = tmp_path / "inferred.mp4"
    inferred.write_bytes(b"inferred")
    config = VehicleFlowConfig(output_video=str(configured))

    resolved = _resolve_safe_media_path(
        config,
        f"path={quote(str(inferred))}",
        {"output_video": str(inferred)},
    )

    assert resolved == inferred
    payload = _media_status_payload(resolved)
    assert payload["exists"] is True
    assert payload["size"] == len(b"inferred")


def test_safe_media_path_defaults_to_inference_status_then_config(tmp_path) -> None:
    configured = tmp_path / "configured.mp4"
    configured.write_bytes(b"configured")
    inferred = tmp_path / "inferred.mp4"
    inferred.write_bytes(b"inferred")
    config = VehicleFlowConfig(output_video=str(configured))

    assert _resolve_safe_media_path(config, "", {}) == configured
    assert _resolve_safe_media_path(
        config,
        "",
        {"output_video": str(inferred)},
    ) == inferred


def test_output_media_endpoint_serves_byte_ranges(tmp_path) -> None:
    output_video = tmp_path / "output.mp4"
    output_video.write_bytes(b"abcdefghij")
    config = VehicleFlowConfig(output_video=str(output_video))
    handler = _make_handler(
        config,
        tmp_path,
        _FakeStatusManager({"output_video": str(output_video)}),
        _FakeStatusManager(),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/media/output-video"
        request = Request(url, headers={"Range": "bytes=2-5"})

        with urlopen(request, timeout=3) as response:
            body = response.read()
            status = response.status
            headers = response.headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert status == 206
    assert headers["Content-Range"] == "bytes 2-5/10"
    assert headers["Content-Length"] == "4"
    assert body == b"cdef"


def test_inference_stream_endpoint_serves_latest_annotated_frame(tmp_path) -> None:
    config = VehicleFlowConfig()
    handler = _make_handler(
        config,
        tmp_path,
        _FakeInferenceStreamManager({"status": "running", "task_id": "task-1"}),
        _FakeStatusManager(),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/inference/stream?task_id=task-1"

        with urlopen(url, timeout=3) as response:
            body = response.read(256)
            status = response.status
            headers = response.headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert status == 200
    assert headers["Content-Type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in body
    assert b"Content-Type: image/jpeg" in body
    assert b"\xff\xd8demo-jpeg\xff\xd9" in body
