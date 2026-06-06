from dataclasses import dataclass, field

from vehicle_flow_ascend.counting.geometry import crossed_line, point_on_segment, point_side
from vehicle_flow_ascend.types import Line, Track


@dataclass
class LineCounter:
    line: Line
    counted_track_ids: set[int] = field(default_factory=set)
    total_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    last_nonzero_sides: dict[int, float] = field(default_factory=dict)

    def update(self, tracks: list[Track]) -> None:
        for track in tracks:
            current_side = point_side(self.line, track.current_centroid)

            if track.track_id in self.counted_track_ids:
                if current_side != 0:
                    self.last_nonzero_sides[track.track_id] = current_side
                continue

            if self._track_crossed_line(track, current_side):
                self.counted_track_ids.add(track.track_id)
                self.total_count += 1
                self.class_counts[track.class_name] = self.class_counts.get(track.class_name, 0) + 1

            self._update_last_nonzero_side(track, current_side)

    def _track_crossed_line(self, track: Track, current_side: float) -> bool:
        if crossed_line(self.line, track.previous_centroid, track.current_centroid):
            return True

        if track.previous_centroid is None or track.previous_centroid == track.current_centroid:
            return False

        previous_side = point_side(self.line, track.previous_centroid)
        if previous_side != 0 or current_side == 0:
            return False

        last_side = self.last_nonzero_sides.get(track.track_id)
        return (
            last_side is not None
            and last_side * current_side < 0
            and point_on_segment(self.line, track.previous_centroid)
        )

    def _update_last_nonzero_side(self, track: Track, current_side: float) -> None:
        if current_side != 0:
            self.last_nonzero_sides[track.track_id] = current_side
            return

        if track.previous_centroid is None:
            return

        previous_side = point_side(self.line, track.previous_centroid)
        if previous_side != 0:
            self.last_nonzero_sides[track.track_id] = previous_side

    def snapshot(self) -> dict[str, int]:
        return {"total": self.total_count, **self.class_counts}
