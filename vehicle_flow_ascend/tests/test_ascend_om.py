from pathlib import Path

import pytest

from vehicle_flow_ascend.config import SourceConfig, load_config
from vehicle_flow_ascend.detectors.ascend_om import AscendOmDetector, _ASCEND_RUNTIME_ERROR
from vehicle_flow_ascend.detectors.base import create_detector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = PROJECT_ROOT / "configs" / "ascend_om.yaml"


def test_load_ascend_om_yaml() -> None:
    config = load_config(PROJECT_CONFIG)

    assert config.source == SourceConfig(path="data/demo.mp4")
    assert config.backend == "ascend_om"
    assert config.model_path == "models/yolov5n.om"
    assert config.soc_version == "Ascend310B4"
    assert config.image_size == 640
    assert config.display is False


def test_ascend_detector_requires_acl_runtime_before_model_file_check() -> None:
    config = load_config(PROJECT_CONFIG)

    with pytest.raises(RuntimeError, match=_ASCEND_RUNTIME_ERROR):
        AscendOmDetector(config)


def test_detector_factory_uses_ascend_backend_and_reports_missing_acl() -> None:
    config = load_config(PROJECT_CONFIG)

    with pytest.raises(RuntimeError, match=_ASCEND_RUNTIME_ERROR):
        create_detector(config)
