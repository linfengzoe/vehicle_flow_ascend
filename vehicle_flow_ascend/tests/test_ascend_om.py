from pathlib import Path

import numpy as np
import pytest

from vehicle_flow_ascend.config import SourceConfig, load_config
from vehicle_flow_ascend.detectors.ascend_om import (
    AscendOmDetector,
    _add_dataset_buffer,
    _ASCEND_RUNTIME_ERROR,
    _DeviceBuffer,
)
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


def test_add_dataset_buffer_accepts_tuple_status_return() -> None:
    class FakeMdl:
        def add_dataset_buffer(self, dataset, data_buffer):
            return data_buffer, 0

    class FakeAcl:
        mdl = FakeMdl()

        def __init__(self) -> None:
            self.destroyed = []

        def create_data_buffer(self, ptr: int, size: int):
            return {"ptr": ptr, "size": size}

        def destroy_data_buffer(self, data_buffer) -> None:
            self.destroyed.append(data_buffer)

    acl = FakeAcl()

    _add_dataset_buffer(acl, object(), _DeviceBuffer(ptr=123, size=456))

    assert acl.destroyed == []


def test_infer_copies_input_array_pointer_to_device() -> None:
    class FakeRt:
        ACL_MEMCPY_HOST_TO_DEVICE = 1

        def __init__(self) -> None:
            self.memcpy_calls = []

        def memcpy(self, dst, dst_size, src, src_size, kind):
            self.memcpy_calls.append((dst, dst_size, src, src_size, kind))
            return 0

    class FakeMdl:
        def execute(self, model_id, input_dataset, output_dataset):
            return 0

    class FakeAcl:
        def __init__(self) -> None:
            self.rt = FakeRt()
            self.mdl = FakeMdl()

    detector = object.__new__(AscendOmDetector)
    detector._acl = FakeAcl()
    detector._input_buffer = _DeviceBuffer(ptr=123, size=16)
    detector._model_id = 1
    detector._input_dataset = object()
    detector._output_dataset = object()
    detector._output_buffers = [_DeviceBuffer(ptr=456, size=4)]
    detector._copy_output_to_host = lambda _buffer: np.array([0], dtype=np.float32)

    detector._infer(np.zeros((1, 1, 1, 4), dtype=np.float32))

    src_arg = detector._acl.rt.memcpy_calls[0][2]
    assert isinstance(src_arg, int)
