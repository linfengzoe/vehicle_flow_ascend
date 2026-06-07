from pathlib import Path

from vehicle_flow_ascend.config import load_config
from vehicle_flow_ascend.web.dashboard import _dashboard_payload, _media_path_from_query, _media_status_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_payload_exposes_runtime_and_classes() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pc_demo.yaml")

    payload = _dashboard_payload(config)

    assert payload["title"] == "车辆大类识别与单线车流统计系统"
    assert payload["runtime"]["backend"] == "torch_yolov5"
    assert payload["runtime"]["model_path"] == "models/yolov5n.pt"
    assert payload["metrics"]["input_size"] == "640x640"
    assert [item["key"] for item in payload["classes"]] == [
        "car",
        "bus",
        "truck",
        "two_wheeler",
    ]
    assert payload["pipeline"][-1] == "交互式前端控制台"


def test_dashboard_static_assets_exist() -> None:
    static_dir = PROJECT_ROOT / "src" / "vehicle_flow_ascend" / "web" / "static"

    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    styles_css = (static_dir / "styles.css").read_text(encoding="utf-8")

    assert (static_dir / "index.html").exists()
    assert (static_dir / "styles.css").exists()
    assert (static_dir / "app.js").exists()
    assert "sourceInput" in index_html
    assert "imageSizeInput" in index_html
    assert "socVersionInput" in index_html
    assert "copyButton" in index_html
    assert "refreshPreviewButton" in index_html
    assert "buildCommand" in app_js
    assert "--image-size" in app_js
    assert "--soc-version" in app_js
    assert "/api/media-status" in app_js
    assert ".control-form" in styles_css


def test_media_status_payload_reports_missing_and_existing_file(tmp_path) -> None:
    missing = _media_status_payload(str(tmp_path / "missing.mp4"))
    assert missing["exists"] is False
    assert missing["size"] == 0

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"demo")
    existing = _media_status_payload(str(video))
    assert existing["exists"] is True
    assert existing["size"] == 4


def test_media_path_query_override_falls_back_to_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pc_demo.yaml")

    assert _media_path_from_query(config, "") == "runs/pc_demo_output.mp4"
    assert _media_path_from_query(config, "path=runs/custom.mp4") == "runs/custom.mp4"
