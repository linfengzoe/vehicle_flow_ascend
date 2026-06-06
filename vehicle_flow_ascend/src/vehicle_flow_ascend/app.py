from __future__ import annotations

import cv2

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.counting.line_counter import LineCounter
from vehicle_flow_ascend.detectors.base import Detector
from vehicle_flow_ascend.tracking.centroid import CentroidTracker
from vehicle_flow_ascend.types import Line, Point
from vehicle_flow_ascend.utils.fps import FpsMeter
from vehicle_flow_ascend.video.source import VideoSource
from vehicle_flow_ascend.video.writer import VideoWriter
from vehicle_flow_ascend.visualization.overlay import draw_overlay


class _SequenceVideoSource:
    def __init__(self, frames, fps: float = 30.0) -> None:
        self._frames = list(frames)
        self._index = 0
        self.fps = fps
        if self._frames:
            height, width = self._frames[0].shape[:2]
            self.frame_size = (width, height)
        else:
            self.frame_size = None
        self.released = False

    def read(self):
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def release(self) -> None:
        self.released = True


class _NullVideoWriter:
    def __init__(self) -> None:
        self.frames = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def run_app(
    config: VehicleFlowConfig,
    detector: Detector,
    *,
    video_source=None,
    video_writer=None,
    show_window: bool | None = None,
) -> dict[str, int]:
    source = video_source or VideoSource(config.source)
    writer = video_writer
    owns_source = video_source is None
    owns_writer = video_writer is None
    if writer is None:
        writer = VideoWriter(config.output_video, source.fps, source.frame_size)

    tracker = CentroidTracker()
    counter = LineCounter(_line_from_config(config))
    fps_meter = FpsMeter()
    display = config.display if show_window is None else show_window
    processed_frames = 0

    try:
        while True:
            if config.max_frames is not None and processed_frames >= config.max_frames:
                break

            frame = source.read()
            if frame is None:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            counter.update(tracks)
            fps = fps_meter.tick()
            annotated = draw_overlay(frame, detections, tracks, counter.line, counter.snapshot(), fps)
            writer.write(annotated)

            if display:
                cv2.imshow("vehicle-flow-ascend", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed_frames += 1
    finally:
        if display:
            cv2.destroyWindow("vehicle-flow-ascend")
        if owns_writer:
            writer.release()
        if owns_source:
            source.release()

    return counter.snapshot()


def sequence_video_source(frames, fps: float = 30.0):
    return _SequenceVideoSource(frames, fps=fps)


def null_video_writer():
    return _NullVideoWriter()


def _line_from_config(config: VehicleFlowConfig) -> Line:
    return Line(
        start=Point(float(config.line.start[0]), float(config.line.start[1])),
        end=Point(float(config.line.end[0]), float(config.line.end[1])),
    )
