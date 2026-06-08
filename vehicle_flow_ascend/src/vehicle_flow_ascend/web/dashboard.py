from __future__ import annotations

import json
import os
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.web.inference import InferenceTaskManager
from vehicle_flow_ascend.web.realtime import RealtimeInferenceManager


_MAX_JSON_BYTES = 64 * 1024
_MAX_REALTIME_FRAME_BYTES = 4 * 1024 * 1024
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_MEDIA_CHUNK_BYTES = 1024 * 1024
_STREAM_IDLE_TIMEOUTS = 30


@dataclass(frozen=True)
class DashboardServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


def run_dashboard(config: VehicleFlowConfig, server_config: DashboardServerConfig) -> None:
    static_dir = Path(__file__).with_name("static")
    task_manager = InferenceTaskManager(config)
    realtime_manager = RealtimeInferenceManager(config)
    handler = _make_handler(config, static_dir, task_manager, realtime_manager)
    server = ThreadingHTTPServer((server_config.host, server_config.port), handler)
    url = f"http://{server_config.host}:{server_config.port}"
    print(f"Vehicle Flow Dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


def _make_handler(
    config: VehicleFlowConfig,
    static_dir: Path,
    task_manager: InferenceTaskManager,
    realtime_manager: RealtimeInferenceManager,
):
    class DashboardRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def do_GET(self) -> None:  # noqa: N802 - http.server hook name
            parsed = urlparse(self.path)
            if parsed.path == "/api/dashboard":
                self._send_json(
                    _dashboard_payload(config, task_manager.status(), realtime_manager.status())
                )
                return
            if parsed.path == "/api/inference/status":
                self._send_json(task_manager.status())
                return
            if parsed.path == "/api/realtime/status":
                self._send_json(realtime_manager.status())
                return
            if parsed.path == "/api/media-status":
                inference_status = task_manager.status()
                media_path = _resolve_safe_media_path(config, parsed.query, inference_status)
                if media_path is None and _has_media_query_path(parsed.query):
                    self._send_json(
                        _media_status_payload(None, error="media path is not allowed")
                    )
                    return
                self._send_json(_media_status_payload(media_path))
                return
            if parsed.path == "/api/realtime/stream":
                self._handle_realtime_stream(parsed.query)
                return
            if parsed.path == "/api/inference/stream":
                self._handle_inference_stream(parsed.query)
                return
            if parsed.path == "/media/output-video":
                self._send_media(
                    _resolve_safe_media_path(config, parsed.query, task_manager.status())
                )
                return
            if parsed.path == "/" or parsed.path == "":
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - http.server hook name
            parsed = urlparse(self.path)
            if parsed.path == "/api/inference/start":
                self._handle_start()
                return
            if parsed.path == "/api/inference/stop":
                self._send_json(task_manager.stop())
                return
            if parsed.path == "/api/inference/upload":
                self._handle_upload()
                return
            if parsed.path == "/api/realtime/start":
                self._handle_realtime_start()
                return
            if parsed.path == "/api/realtime/frame":
                self._handle_realtime_frame(parsed.query)
                return
            if parsed.path == "/api/realtime/stop":
                self._handle_realtime_stop()
                return
            self.send_error(404, "unknown endpoint")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
            return

        def _handle_start(self) -> None:
            try:
                payload = _sanitize_start_payload(self._read_json(), Path.cwd())
                self._send_json(task_manager.start(payload))
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
            except Exception as exc:  # noqa: BLE001 - surface invalid UI payloads
                self._send_json({"error": str(exc)}, status=400)

        def _handle_upload(self) -> None:
            try:
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("multipart/form-data"):
                    raise ValueError("upload requires multipart/form-data")
                length = int(self.headers.get("Content-Length", "0"))
                if length > _MAX_UPLOAD_BYTES:
                    raise ValueError("uploaded video is too large")
                if length <= 0:
                    raise ValueError("uploaded video body is empty")
                filename, source = _read_multipart_video_upload(
                    content_type,
                    self.rfile.read(length),
                )
                self._send_json(task_manager.upload(filename, source))
            except Exception as exc:  # noqa: BLE001 - return upload issue to UI
                self._send_json({"error": str(exc)}, status=400)

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
                if length > _MAX_REALTIME_FRAME_BYTES:
                    raise ValueError("realtime frame is too large")
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

            status = realtime_manager.status()
            if status.get("status") != "running":
                self._send_json({"error": "realtime inference session is not running"}, status=409)
                return
            if status.get("session_id") != session_id:
                self._send_json({"error": "invalid realtime session"}, status=400)
                return

            self._send_mjpeg_stream(
                wait_for_frame=lambda last_version: realtime_manager.wait_for_frame(
                    session_id,
                    last_version,
                    timeout=1.0,
                ),
                should_continue=lambda: (
                    (status := realtime_manager.status()).get("status") == "running"
                    and status.get("session_id") == session_id
                ),
            )

        def _handle_inference_stream(self, query: str) -> None:
            try:
                task_id = _required_query_param(query, "task_id")
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            status = task_manager.status()
            if status.get("status") not in {"running", "stopping"}:
                self._send_json({"error": "inference task is not running"}, status=409)
                return
            if status.get("task_id") != task_id:
                self._send_json({"error": "invalid inference task"}, status=400)
                return

            self._send_mjpeg_stream(
                wait_for_frame=lambda last_version: task_manager.wait_for_frame(
                    task_id,
                    last_version,
                    timeout=1.0,
                ),
                should_continue=lambda: (
                    (status := task_manager.status()).get("status") in {"running", "stopping"}
                    and status.get("task_id") == task_id
                ),
            )

        def _send_mjpeg_stream(self, wait_for_frame, should_continue) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            last_version = 0
            idle_timeouts = 0
            while True:
                try:
                    frame = wait_for_frame(last_version)
                    if frame is None:
                        if not should_continue():
                            break
                        idle_timeouts += 1
                        if idle_timeouts >= _STREAM_IDLE_TIMEOUTS:
                            break
                        continue
                    idle_timeouts = 0
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

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must not be negative")
            if length > _MAX_JSON_BYTES:
                raise ValueError("JSON body is too large")
            if length == 0:
                return {}
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, output_video: Path | None) -> None:
            if output_video is None:
                self.send_error(404, "output_video is not configured")
                return
            path = output_video
            if not path.exists() or not path.is_file():
                self.send_error(404, f"output video not found: {path}")
                return
            size = path.stat().st_size
            byte_range = self.headers.get("Range")
            if byte_range:
                media_range = _parse_byte_range(byte_range, size)
                if media_range is None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                start, end = media_range
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with path.open("rb") as media_file:
                    media_file.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = media_file.read(min(_MEDIA_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return

            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as media_file:
                while True:
                    chunk = media_file.read(_MEDIA_CHUNK_BYTES)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    return DashboardRequestHandler


def _read_multipart_video_upload(content_type: str, body: bytes) -> tuple[str, BytesIO]:
    boundary = _multipart_boundary(content_type)
    delimiter = b"--" + boundary.encode("utf-8")
    for raw_part in body.split(delimiter):
        part = _normalize_multipart_part(raw_part)
        if not part:
            continue
        separator = b"\r\n\r\n" if b"\r\n\r\n" in part else b"\n\n"
        if separator not in part:
            continue
        header_bytes, payload = part.split(separator, 1)
        headers = BytesParser(policy=email_policy).parsebytes(header_bytes + b"\r\n\r\n")
        if headers.get_content_disposition() != "form-data":
            continue
        if headers.get_param("name", header="content-disposition") != "video":
            continue
        filename = headers.get_filename()
        if not filename:
            raise ValueError("uploaded video filename is empty")
        return filename, BytesIO(payload)
    raise ValueError("missing form field: video")


def _multipart_boundary(content_type: str) -> str:
    message = Message()
    message["Content-Type"] = content_type
    boundary = message.get_boundary()
    if not boundary:
        raise ValueError("multipart/form-data boundary is missing")
    return boundary


def _normalize_multipart_part(part: bytes) -> bytes:
    if not part or part.startswith(b"--"):
        return b""
    if part.startswith(b"\r\n"):
        part = part[2:]
    elif part.startswith(b"\n"):
        part = part[1:]
    if part.endswith(b"\r\n"):
        part = part[:-2]
    elif part.endswith(b"\n"):
        part = part[:-1]
    return part


def _parse_byte_range(range_header: str, size: int) -> tuple[int, int] | None:
    if size <= 0 or not range_header.startswith("bytes="):
        return None
    range_spec = range_header.removeprefix("bytes=").strip()
    if "," in range_spec or "-" not in range_spec:
        return None
    start_text, end_text = [part.strip() for part in range_spec.split("-", 1)]
    try:
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None
            start = max(size - suffix_length, 0)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
            if start < 0 or end < start:
                return None
            end = min(end, size - 1)
    except ValueError:
        return None
    if start >= size:
        return None
    return start, end


def _required_query_param(query: str, name: str) -> str:
    params = parse_qs(query)
    value = params.get(name, [None])[0]
    if not value:
        raise ValueError(f"missing query parameter: {name}")
    return unquote(value)


def _sanitize_start_payload(
    payload: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    source_type = str(payload.get("source_type", "video"))
    source = str(payload.get("source", ""))
    if source_type != "video":
        raise ValueError("start requires an uploaded video source")
    uploaded_source = _uploaded_video_path(source, project_root=project_root)
    if uploaded_source is None:
        raise ValueError("start requires an uploaded video source")
    sanitized = dict(payload)
    sanitized["source_type"] = "video"
    sanitized["source"] = str(uploaded_source)
    sanitized.pop("output_video", None)
    return sanitized


def _uploaded_video_path(source: str, project_root: Path | None = None) -> Path | None:
    if not source:
        return None
    root = os.path.abspath(os.fspath(project_root or Path.cwd()))
    source_text = os.fspath(source)
    if not os.path.isabs(source_text):
        source_text = os.path.join(root, source_text)
    source_path = Path(os.path.abspath(os.path.normpath(source_text)))
    for upload_root in _upload_roots(Path(root)):
        upload_root_path = Path(os.path.abspath(os.path.normpath(os.fspath(upload_root))))
        try:
            relative = source_path.relative_to(upload_root_path)
        except ValueError:
            continue
        if relative.parts:
            return source_path
    return None


def _upload_roots(project_root: Path) -> tuple[Path, Path]:
    return (
        project_root / "data" / "web_uploads",
        project_root / "vehicle_flow_ascend" / "data" / "web_uploads",
    )


def _safe_media_candidates(
    config: VehicleFlowConfig,
    inference_status: dict[str, Any] | None = None,
) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in (
        config.output_video,
        (inference_status or {}).get("output_video"),
    ):
        if value:
            raw_string = str(value)
            if raw_string in seen:
                continue
            seen.add(raw_string)
            candidates.append((raw_string, Path(raw_string).resolve()))
    return candidates


def _resolve_safe_media_path(
    config: VehicleFlowConfig,
    query: str,
    inference_status: dict[str, Any] | None = None,
) -> Path | None:
    candidates = _safe_media_candidates(config, inference_status)
    if not candidates:
        return None

    params = parse_qs(query)
    value = params.get("path", [None])[0]
    if value:
        query_raw = unquote(value)
        query_path_string = str(Path(query_raw))
        for raw_string, resolved_path in candidates:
            if query_raw in {raw_string, str(resolved_path)}:
                return resolved_path
            if query_path_string == str(Path(raw_string)):
                return resolved_path
        return None

    inference_output = (inference_status or {}).get("output_video")
    if inference_output:
        inference_raw = str(inference_output)
        for raw_string, resolved_path in candidates:
            if raw_string == inference_raw:
                return resolved_path
    if config.output_video:
        config_raw = str(config.output_video)
        for raw_string, resolved_path in candidates:
            if raw_string == config_raw:
                return resolved_path
    return None


def _has_media_query_path(query: str) -> bool:
    value = parse_qs(query).get("path", [None])[0]
    return bool(value)


def _media_status_payload(
    media_path: str | Path | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    if media_path is None:
        payload = {"path": None, "exists": False, "size": 0}
        if error is not None:
            payload["error"] = error
        return payload
    path = Path(media_path)
    exists = path.exists() and path.is_file()
    payload = {
        "path": str(media_path),
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
    }
    if error is not None:
        payload["error"] = error
    return payload


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
        "runtime": {"backend": config.backend, "mode": "background-inference"},
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
