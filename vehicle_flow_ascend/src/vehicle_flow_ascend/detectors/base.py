from __future__ import annotations

from abc import ABC, abstractmethod

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame_bgr) -> list[Detection]:
        raise NotImplementedError


def create_detector(config: VehicleFlowConfig) -> Detector:
    if config.backend == "torch_yolov5":
        from vehicle_flow_ascend.detectors.torch_yolov5 import TorchYoloV5Detector

        return TorchYoloV5Detector(config)

    if config.backend == "ascend_om":
        raise RuntimeError("ascend_om backend will be available after Task 8 deployment support")

    raise ValueError(f"unsupported detector backend: {config.backend}")
