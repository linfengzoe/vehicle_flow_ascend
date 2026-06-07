from pathlib import Path
from urllib.parse import quote

from vehicle_flow_ascend.config import VehicleFlowConfig, load_config
from vehicle_flow_ascend.web.dashboard import (
    _dashboard_payload,
    _media_status_payload,
    _resolve_safe_media_path,
    _sanitize_start_payload,
)


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
    lower_index = index_html.lower()

    assert (static_dir / "index.html").exists()
    assert (static_dir / "styles.css").exists()
    assert (static_dir / "app.js").exists()
    assert "particleCanvas" in index_html
    assert "videoFileInput" in index_html
    assert "cameraButton" in index_html
    assert "realtimeStream" in index_html
    assert "cameraPreview" in index_html
    assert "captureCanvas" in index_html
    assert "videoPathInput" not in index_html
    assert "cameraInput" not in index_html
    assert "commandText" not in index_html
    assert "image-size" not in lower_index
    assert "confidence" not in lower_index
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


def test_media_status_payload_reports_missing_and_existing_file(tmp_path) -> None:
    missing = _media_status_payload(str(tmp_path / "missing.mp4"))
    assert missing["exists"] is False
    assert missing["size"] == 0

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"demo")
    existing = _media_status_payload(str(video))
    assert existing["exists"] is True
    assert existing["size"] == 4


def test_sanitize_start_payload_removes_client_output_video() -> None:
    payload = {
        "source": "data/web_uploads/demo.mp4",
        "output_video": "D:/not-allowed/output.mp4",
        "max_frames": 12,
    }

    sanitized = _sanitize_start_payload(payload, PROJECT_ROOT)

    assert sanitized == {
        "source_type": "video",
        "source": str(PROJECT_ROOT / "data" / "web_uploads" / "demo.mp4"),
        "max_frames": 12,
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
