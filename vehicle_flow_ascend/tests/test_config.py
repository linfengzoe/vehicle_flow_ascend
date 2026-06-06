from pathlib import Path

import pytest

from vehicle_flow_ascend.config import (
    SourceConfig,
    VehicleFlowConfig,
    load_config,
    load_config_with_overrides,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_pc_demo_yaml() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pc_demo.yaml")

    assert config.source == SourceConfig(path="data/pc_demo.mp4")
    assert config.backend == "torch_yolov5"
    assert config.model_path == "models/yolov5n.pt"
    assert config.image_size == 640
    assert config.confidence_threshold == pytest.approx(0.35)
    assert config.iou_threshold == pytest.approx(0.45)
    assert config.line.start == (120, 360)
    assert config.line.end == (1160, 360)
    assert config.display is True
    assert config.output_video == "runs/pc_demo_output.mp4"
    assert config.max_frames == 300


def test_defaults_from_empty_mapping() -> None:
    config = VehicleFlowConfig.from_mapping({})

    assert config.source == SourceConfig()
    assert config.backend == "torch_yolov5"
    assert config.model_path is None
    assert config.image_size == 640
    assert config.confidence_threshold == pytest.approx(0.35)
    assert config.iou_threshold == pytest.approx(0.45)
    assert config.line.start == (0, 0)
    assert config.line.end == (0, 0)
    assert config.display is False
    assert config.output_video is None
    assert config.max_frames is None


def test_source_mapping_accepts_path() -> None:
    assert SourceConfig.from_value({"path": "data/demo.mp4"}) == SourceConfig(
        path="data/demo.mp4"
    )


def test_source_mapping_accepts_camera() -> None:
    assert SourceConfig.from_value({"camera": "0"}) == SourceConfig(camera=0)


def test_source_mapping_rejects_path_and_camera() -> None:
    with pytest.raises(ValueError, match="only one of 'path' or 'camera'"):
        SourceConfig.from_value({"path": "data/demo.mp4", "camera": 0})


def test_cli_override_merge() -> None:
    config = load_config_with_overrides(
        PROJECT_ROOT / "configs" / "pc_demo.yaml",
        {
            "source": "0",
            "backend": "ascend_om",
            "display": "false",
            "max_frames": 10,
        },
    )

    assert config.source == SourceConfig(camera=0)
    assert config.backend == "ascend_om"
    assert config.display is False
    assert config.max_frames == 10
    assert config.model_path == "models/yolov5n.pt"
