from pathlib import Path

from vehicle_flow_ascend.config import load_config
from vehicle_flow_ascend.web.dashboard import _dashboard_payload


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
    assert payload["pipeline"][-1] == "前端监控展示"


def test_dashboard_static_assets_exist() -> None:
    static_dir = PROJECT_ROOT / "src" / "vehicle_flow_ascend" / "web" / "static"

    assert (static_dir / "index.html").exists()
    assert (static_dir / "styles.css").exists()
    assert (static_dir / "app.js").exists()
