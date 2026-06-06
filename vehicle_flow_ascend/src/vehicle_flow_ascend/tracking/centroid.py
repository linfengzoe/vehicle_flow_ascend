from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from vehicle_flow_ascend.types import Detection, Point, Track


@dataclass
class _TrackState:
    track: Track
    disappeared: int = 0


@dataclass
class CentroidTracker:
    max_distance: float = 80.0
    max_disappeared: int = 15
    _next_track_id: int = 1
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            self._mark_all_disappeared()
            return self.tracks

        if not self._tracks:
            for detection in detections:
                self._register(detection)
            return self.tracks

        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indexes = set(range(len(detections)))

        for track_id, detection_index in self._match_tracks(detections):
            detection = detections[detection_index]
            self._update_track(track_id, detection)
            unmatched_track_ids.discard(track_id)
            unmatched_detection_indexes.discard(detection_index)

        for track_id in unmatched_track_ids:
            self._mark_disappeared(track_id)

        for detection_index in sorted(unmatched_detection_indexes):
            self._register(detections[detection_index])

        return self.tracks

    @property
    def tracks(self) -> list[Track]:
        return [state.track for _, state in sorted(self._tracks.items())]

    def _register(self, detection: Detection) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1
        centroid = detection.centroid
        self._tracks[track_id] = _TrackState(
            track=Track(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
                previous_centroid=None,
                current_centroid=centroid,
                disappeared=0,
            )
        )

    def _update_track(self, track_id: int, detection: Detection) -> None:
        state = self._tracks[track_id]
        previous = state.track.current_centroid
        state.disappeared = 0
        state.track = Track(
            track_id=track_id,
            class_name=detection.class_name,
            bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
            previous_centroid=previous,
            current_centroid=detection.centroid,
            disappeared=0,
        )

    def _mark_all_disappeared(self) -> None:
        for track_id in list(self._tracks):
            self._mark_disappeared(track_id)

    def _mark_disappeared(self, track_id: int) -> None:
        state = self._tracks[track_id]
        state.disappeared += 1
        if state.disappeared > self.max_disappeared:
            del self._tracks[track_id]
            return

        state.track.disappeared = state.disappeared
        state.track.previous_centroid = state.track.current_centroid

    def _match_tracks(self, detections: list[Detection]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, state in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = _distance(state.track.current_centroid, detection.centroid)
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_index))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            matches.append((track_id, detection_index))
            used_tracks.add(track_id)
            used_detections.add(detection_index)
        return matches


def _distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)
