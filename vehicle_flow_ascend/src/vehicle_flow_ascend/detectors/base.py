from abc import ABC, abstractmethod

from vehicle_flow_ascend.types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame_bgr) -> list[Detection]:
        raise NotImplementedError
