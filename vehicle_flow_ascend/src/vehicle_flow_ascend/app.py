from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


@dataclass
class ProcessedFrame:
    annotated_frame: Any
    frames: int
    fps: float
    counts: dict[str, int]


class FrameProcessor:
    def __init__(self, config: VehicleFlowConfig, detector: Detector) -> None:
        self.detector = detector
        self.tracker = CentroidTracker()
        self.counter = LineCounter(_line_from_config(config))
        self.fps_meter = FpsMeter()
        self.frames = 0

    @property
    def counts(self) -> dict[str, int]:
        return self.counter.snapshot()

    def process(self, frame_bgr) -> ProcessedFrame:
        detections = self.detector.detect(frame_bgr)
        tracks = self.tracker.update(detections)
        self.counter.update(tracks)
        fps = self.fps_meter.tick()
        counts = self.counter.snapshot()
        annotated = draw_overlay(frame_bgr, detections, tracks, self.counter.line, counts, fps)
        self.frames += 1
        return ProcessedFrame(
            annotated_frame=annotated,
            frames=self.frames,
            fps=fps,
            counts=counts,
        )


def run_app(
    config: VehicleFlowConfig,
    detector: Detector,
    *,
    video_source=None,
    video_writer=None,
    show_window: bool | None = None,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    source = video_source or VideoSource(config.source)
    writer = video_writer
    owns_source = video_source is None
    owns_writer = video_writer is None
    if writer is None:
        writer = VideoWriter(config.output_video, source.fps, source.frame_size)

    processor = FrameProcessor(config, detector)
    display = config.display if show_window is None else show_window

    try:
        while True:
            if stop_requested is not None and stop_requested():
                break
            if config.max_frames is not None and processor.frames >= config.max_frames:
                break

            frame = source.read()
            if frame is None:
                break

            processed = processor.process(frame)
            writer.write(processed.annotated_frame)

            if progress_callback is not None:
                progress_callback(
                    {
                        "frames": processed.frames,
                        "fps": processed.fps,
                        "counts": processed.counts,
                    }
                )

            if display:
                cv2.imshow("vehicle-flow-ascend", processed.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if display:
            cv2.destroyWindow("vehicle-flow-ascend")
        if owns_writer:
            writer.release()
        if owns_source:
            source.release()

    return processor.counts


def sequence_video_source(frames, fps: float = 30.0):
    return _SequenceVideoSource(frames, fps=fps)


def null_video_writer():
    return _NullVideoWriter()


def _line_from_config(config: VehicleFlowConfig) -> Line:
    return Line(
        start=Point(float(config.line.start[0]), float(config.line.start[1])),
        end=Point(float(config.line.end[0]), float(config.line.end[1])),
    )
