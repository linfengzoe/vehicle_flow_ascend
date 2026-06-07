from __future__ import annotations

import numpy as np

from vehicle_flow_ascend.app import FrameProcessor
from vehicle_flow_ascend.config import LineConfig, VehicleFlowConfig
from vehicle_flow_ascend.types import Detection


class MovingFakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr) -> list[Detection]:
        self.calls += 1
        if self.calls == 1:
            return [Detection(4, 1, 6, 3, 0.9, "car")]
        return [Detection(4, 7, 6, 9, 0.9, "car")]


def test_frame_processor_returns_annotated_frame_and_progress() -> None:
    config = VehicleFlowConfig(line=LineConfig(start=(0, 5), end=(10, 5)))
    processor = FrameProcessor(config, MovingFakeDetector())
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    first = processor.process(frame)
    second = processor.process(frame)

    assert first.annotated_frame.shape == frame.shape
    assert first.frames == 1
    assert first.counts == {"total": 0}
    assert second.annotated_frame.shape == frame.shape
    assert second.frames == 2
    assert second.counts == {"total": 1, "car": 1}
    assert second.fps >= 0.0
