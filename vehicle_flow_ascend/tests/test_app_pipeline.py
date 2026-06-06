import numpy as np

from vehicle_flow_ascend.app import null_video_writer, run_app, sequence_video_source
from vehicle_flow_ascend.config import LineConfig, SourceConfig, VehicleFlowConfig
from vehicle_flow_ascend.detectors.base import Detector
from vehicle_flow_ascend.types import Detection


class FakeDetector(Detector):
    def __init__(self, detections_by_frame: list[list[Detection]]) -> None:
        self._detections_by_frame = detections_by_frame
        self._index = 0

    def detect(self, frame_bgr) -> list[Detection]:
        if self._index >= len(self._detections_by_frame):
            return []
        detections = self._detections_by_frame[self._index]
        self._index += 1
        return detections


def frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def config(max_frames: int | None = None) -> VehicleFlowConfig:
    return VehicleFlowConfig(
        source=SourceConfig(path="unused.mp4"),
        line=LineConfig(start=(0, 50), end=(99, 50)),
        display=False,
        output_video=None,
        max_frames=max_frames,
    )


def box_centered_at(x: float, y: float, class_name: str = "car") -> Detection:
    return Detection(
        x1=x - 5,
        y1=y - 5,
        x2=x + 5,
        y2=y + 5,
        confidence=0.9,
        class_name=class_name,
    )


def test_pipeline_counts_synthetic_vehicle_crossing_line() -> None:
    source = sequence_video_source([frame(), frame(), frame()])
    writer = null_video_writer()
    detector = FakeDetector([
        [box_centered_at(20, 40, "car")],
        [box_centered_at(20, 50, "car")],
        [box_centered_at(20, 60, "car")],
    ])

    counts = run_app(config(), detector, video_source=source, video_writer=writer, show_window=False)

    assert counts == {"total": 1, "car": 1}
    assert len(writer.frames) == 3


def test_pipeline_returns_zero_counts_when_no_detections() -> None:
    source = sequence_video_source([frame(), frame()])
    detector = FakeDetector([[], []])

    counts = run_app(config(), detector, video_source=source, video_writer=null_video_writer(), show_window=False)

    assert counts == {"total": 0}


def test_pipeline_processes_limited_number_of_frames() -> None:
    source = sequence_video_source([frame(), frame(), frame()])
    writer = null_video_writer()
    detector = FakeDetector([
        [box_centered_at(20, 40)],
        [box_centered_at(20, 60)],
        [box_centered_at(20, 80)],
    ])

    counts = run_app(config(max_frames=1), detector, video_source=source, video_writer=writer, show_window=False)

    assert counts == {"total": 0}
    assert len(writer.frames) == 1


def test_pipeline_does_not_require_display_in_test_mode() -> None:
    source = sequence_video_source([frame()])
    detector = FakeDetector([[box_centered_at(20, 40)]])

    counts = run_app(config(), detector, video_source=source, video_writer=null_video_writer(), show_window=False)

    assert counts == {"total": 0}
