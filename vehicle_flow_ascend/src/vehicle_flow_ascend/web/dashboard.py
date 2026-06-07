from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from vehicle_flow_ascend.config import VehicleFlowConfig


@dataclass(frozen=True)
class DashboardServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


def run_dashboard(config: VehicleFlowConfig, server_config: DashboardServerConfig) -> None:
    static_dir = Path(__file__).with_name("static")
    handler = _make_handler(config, static_dir)
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


def _make_handler(config: VehicleFlowConfig, static_dir: Path):
    class DashboardRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def do_GET(self) -> None:  # noqa: N802 - http.server hook name
            parsed = urlparse(self.path)
            if parsed.path == "/api/dashboard":
                self._send_json(_dashboard_payload(config))
                return
            if parsed.path == "/media/output-video":
                self._send_media(config.output_video)
                return
            if parsed.path == "/" or parsed.path == "":
                self.path = "/index.html"
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, output_video: str | None) -> None:
            if output_video is None:
                self.send_error(404, "output_video is not configured")
                return
            path = Path(unquote(output_video))
            if not path.exists() or not path.is_file():
                self.send_error(404, f"output video not found: {path}")
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardRequestHandler


def _dashboard_payload(config: VehicleFlowConfig) -> dict[str, Any]:
    output_path = Path(config.output_video) if config.output_video else None
    source_value = config.source.to_cli_value()
    return {
        "project": "Vehicle Flow Ascend",
        "title": "车辆大类识别与单线车流统计系统",
        "config": config.to_dict(),
        "runtime": {
            "backend": config.backend,
            "model_path": config.model_path,
            "soc_version": config.soc_version,
            "source": source_value,
            "output_video": config.output_video,
            "output_exists": bool(output_path and output_path.exists()),
        },
        "classes": [
            {"key": "car", "label": "小型车", "accent": "#a7ff4f"},
            {"key": "bus", "label": "公交/客车", "accent": "#ffd166"},
            {"key": "truck", "label": "货车", "accent": "#4cc9f0"},
            {"key": "two_wheeler", "label": "两轮车", "accent": "#ff4dba"},
        ],
        "metrics": {
            "fps_target": "10+",
            "input_size": f"{config.image_size}x{config.image_size}",
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
        },
        "pipeline": [
            "视频/摄像头输入",
            "YOLOv5 车辆检测",
            "类别映射",
            "质心跟踪",
            "单线穿越计数",
            "前端监控展示",
        ],
    }
