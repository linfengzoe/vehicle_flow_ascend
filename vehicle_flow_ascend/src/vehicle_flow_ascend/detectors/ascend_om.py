from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.detectors.base import Detector
from vehicle_flow_ascend.detectors.yolov5_postprocess import postprocess_yolov5_output, preprocess_frame
from vehicle_flow_ascend.types import Detection


_ASCEND_RUNTIME_ERROR = "Ascend OM backend requires Huawei CANN/ACL on the development board."
_COCO_VEHICLE_CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class AscendOmDetector(Detector):
    """YOLOv5 OM detector using Huawei Ascend ACL runtime.

    The ACL import happens only when this backend is instantiated, so PC tests and
    non-Ascend workflows can import the package without CANN installed.
    """

    config: VehicleFlowConfig
    device_id: int = 0
    _acl: Any = field(init=False, repr=False)
    _model_id: int | None = field(default=None, init=False)
    _model_desc: Any = field(default=None, init=False, repr=False)
    _input_dataset: Any = field(default=None, init=False, repr=False)
    _output_dataset: Any = field(default=None, init=False, repr=False)
    _input_buffer: _DeviceBuffer | None = field(default=None, init=False, repr=False)
    _output_buffers: list[_DeviceBuffer] = field(default_factory=list, init=False, repr=False)
    _letterbox_ratio: float = field(default=1.0, init=False)
    _letterbox_pad: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    _last_original_shape: tuple[int, int] = field(default=(0, 0), init=False)

    def __post_init__(self) -> None:
        if self.config.model_path is None:
            raise ValueError("ascend_om backend requires model_path")
        self._acl = _import_acl()
        self._init_runtime(Path(self.config.model_path))

    def detect(self, frame_bgr) -> list[Detection]:
        model_input, letterbox_result = preprocess_frame(
            frame_bgr,
            (self.config.image_size, self.config.image_size),
        )
        self._letterbox_ratio = letterbox_result.ratio
        self._letterbox_pad = letterbox_result.pad
        self._last_original_shape = frame_bgr.shape[:2]

        outputs = self._infer(model_input)
        if not outputs:
            return []
        return postprocess_yolov5_output(
            outputs[0],
            original_shape=self._last_original_shape,
            ratio=self._letterbox_ratio,
            pad=self._letterbox_pad,
            class_names=_COCO_VEHICLE_CLASS_NAMES,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
        )

    def release(self) -> None:
        if self._acl is None:
            return
        acl = self._acl
        if self._input_dataset is not None:
            _destroy_dataset(acl, self._input_dataset)
            self._input_dataset = None
        if self._output_dataset is not None:
            _destroy_dataset(acl, self._output_dataset)
            self._output_dataset = None
        if self._input_buffer is not None:
            self._input_buffer.free(acl)
            self._input_buffer = None
        for buffer in self._output_buffers:
            buffer.free(acl)
        self._output_buffers.clear()
        if self._model_desc is not None:
            acl.mdl.destroy_desc(self._model_desc)
            self._model_desc = None
        if self._model_id is not None:
            acl.mdl.unload(self._model_id)
            self._model_id = None
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    def _init_runtime(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"OM model not found: {model_path}")

        acl = self._acl
        _check_acl(acl.init(), "acl.init")
        _check_acl(acl.rt.set_device(self.device_id), "acl.rt.set_device")
        model_id, ret = acl.mdl.load_from_file(str(model_path))
        _check_acl(ret, "acl.mdl.load_from_file")
        self._model_id = model_id
        self._model_desc = acl.mdl.create_desc()
        _check_acl(acl.mdl.get_desc(self._model_desc, self._model_id), "acl.mdl.get_desc")
        self._create_io_buffers()

    def _create_io_buffers(self) -> None:
        acl = self._acl
        input_count = acl.mdl.get_num_inputs(self._model_desc)
        if input_count != 1:
            raise RuntimeError(f"expected one model input, got {input_count}")
        input_size = acl.mdl.get_input_size_by_index(self._model_desc, 0)
        self._input_buffer = _DeviceBuffer.malloc(acl, input_size)
        self._input_dataset = acl.mdl.create_dataset()
        _add_dataset_buffer(acl, self._input_dataset, self._input_buffer)

        self._output_dataset = acl.mdl.create_dataset()
        output_count = acl.mdl.get_num_outputs(self._model_desc)
        for index in range(output_count):
            output_size = acl.mdl.get_output_size_by_index(self._model_desc, index)
            output_buffer = _DeviceBuffer.malloc(acl, output_size)
            self._output_buffers.append(output_buffer)
            _add_dataset_buffer(acl, self._output_dataset, output_buffer)

    def _infer(self, model_input: np.ndarray) -> list[np.ndarray]:
        if self._input_buffer is None or self._model_id is None:
            raise RuntimeError("Ascend model is not initialized")
        acl = self._acl
        input_bytes = np.ascontiguousarray(model_input.astype(np.float32)).tobytes()
        if len(input_bytes) > self._input_buffer.size:
            raise RuntimeError(
                f"model input bytes ({len(input_bytes)}) exceed allocated input buffer ({self._input_buffer.size})"
            )

        _check_acl(
            acl.rt.memcpy(
                self._input_buffer.ptr,
                self._input_buffer.size,
                input_bytes,
                len(input_bytes),
                getattr(acl.rt, "ACL_MEMCPY_HOST_TO_DEVICE", 1),
            ),
            "acl.rt.memcpy host->device",
        )
        _check_acl(
            acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset),
            "acl.mdl.execute",
        )
        return [self._copy_output_to_host(buffer) for buffer in self._output_buffers]

    def _copy_output_to_host(self, buffer: "_DeviceBuffer") -> np.ndarray:
        acl = self._acl
        host = np.empty(buffer.size // np.dtype(np.float32).itemsize, dtype=np.float32)
        _check_acl(
            acl.rt.memcpy(
                host.ctypes.data,
                buffer.size,
                buffer.ptr,
                buffer.size,
                getattr(acl.rt, "ACL_MEMCPY_DEVICE_TO_HOST", 2),
            ),
            "acl.rt.memcpy device->host",
        )
        return host.reshape(1, -1, _infer_output_width(host.size))

    def __enter__(self) -> "AscendOmDetector":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()


@dataclass
class _DeviceBuffer:
    ptr: int
    size: int

    @classmethod
    def malloc(cls, acl: Any, size: int) -> "_DeviceBuffer":
        ptr, ret = acl.rt.malloc(size, getattr(acl.rt, "ACL_MEM_MALLOC_NORMAL_ONLY", 0))
        _check_acl(ret, "acl.rt.malloc")
        return cls(ptr=ptr, size=size)

    def free(self, acl: Any) -> None:
        if self.ptr:
            _check_acl(acl.rt.free(self.ptr), "acl.rt.free")
            self.ptr = 0


def _import_acl() -> Any:
    try:
        import acl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(_ASCEND_RUNTIME_ERROR) from exc
    return acl


def _check_acl(ret: Any, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{operation} failed with ACL error code {ret}")


def _add_dataset_buffer(acl: Any, dataset: Any, buffer: _DeviceBuffer) -> None:
    data_buffer = acl.create_data_buffer(buffer.ptr, buffer.size)
    ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
    if ret != 0:
        acl.destroy_data_buffer(data_buffer)
        _check_acl(ret, "acl.mdl.add_dataset_buffer")


def _destroy_dataset(acl: Any, dataset: Any) -> None:
    count = acl.mdl.get_dataset_num_buffers(dataset)
    for index in range(count):
        data_buffer = acl.mdl.get_dataset_buffer(dataset, index)
        if data_buffer is not None:
            acl.destroy_data_buffer(data_buffer)
    acl.mdl.destroy_dataset(dataset)


def _infer_output_width(total_float_count: int) -> int:
    # YOLOv5 COCO exports normally use 85 values per candidate (x, y, w, h, obj + 80 classes).
    # If the OM model is exported with a custom class count, keep a flat fallback that still
    # produces a valid ndarray; deployment docs explain updating postprocess metadata if needed.
    return 85 if total_float_count % 85 == 0 else total_float_count
