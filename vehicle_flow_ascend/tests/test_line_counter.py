from vehicle_flow_ascend.counting.line_counter import LineCounter
from vehicle_flow_ascend.types import Line, Point, Track


def make_track(
    track_id: int,
    class_name: str,
    previous_centroid: Point | None,
    current_centroid: Point,
) -> Track:
    return Track(
        track_id=track_id,
        class_name=class_name,
        bbox=(0.0, 0.0, 10.0, 10.0),
        previous_centroid=previous_centroid,
        current_centroid=current_centroid,
    )


def test_line_counter_counts_track_once_even_when_updated_repeatedly() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))
    track = make_track(1, "car", Point(5.0, -1.0), Point(5.0, 1.0))

    counter.update([track])
    counter.update([track])

    assert counter.total_count == 1
    assert counter.counted_track_ids == {1}
    assert counter.snapshot() == {"total": 1, "car": 1}


def test_line_counter_tracks_per_class_counts() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))
    tracks = [
        make_track(1, "car", Point(1.0, -1.0), Point(1.0, 1.0)),
        make_track(2, "bus", Point(2.0, -1.0), Point(2.0, 1.0)),
        make_track(3, "truck", Point(3.0, -1.0), Point(3.0, 1.0)),
        make_track(4, "two_wheeler", Point(4.0, -1.0), Point(4.0, 1.0)),
    ]

    counter.update(tracks)

    assert counter.snapshot() == {
        "total": 4,
        "car": 1,
        "bus": 1,
        "truck": 1,
        "two_wheeler": 1,
    }


def test_line_counter_does_not_count_stationary_track() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))
    point = Point(5.0, 1.0)

    counter.update([make_track(1, "car", point, point)])

    assert counter.snapshot() == {"total": 0}
    assert counter.counted_track_ids == set()


def test_line_counter_does_not_count_track_without_previous_centroid() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))

    counter.update([make_track(1, "car", None, Point(5.0, 1.0))])

    assert counter.snapshot() == {"total": 0}
    assert counter.counted_track_ids == set()


def test_line_counter_does_not_count_track_starting_on_line() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))

    counter.update([make_track(1, "car", Point(5.0, 0.0), Point(5.0, 1.0))])

    assert counter.snapshot() == {"total": 0}
    assert counter.counted_track_ids == set()


def test_line_counter_ignores_tracks_that_do_not_cross() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))

    counter.update([make_track(1, "car", Point(5.0, 1.0), Point(6.0, 2.0))])

    assert counter.snapshot() == {"total": 0}
    assert counter.counted_track_ids == set()


def test_line_counter_counts_below_on_line_above_as_one_crossing() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))

    counter.update([make_track(1, "car", Point(5.0, -1.0), Point(5.0, 0.0))])
    counter.update([make_track(1, "car", Point(5.0, 0.0), Point(5.0, 1.0))])

    assert counter.snapshot() == {"total": 1, "car": 1}
    assert counter.counted_track_ids == {1}


def test_line_counter_does_not_double_count_repeated_on_line_jitter() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))

    counter.update([make_track(1, "car", Point(5.0, -1.0), Point(5.0, 0.0))])
    counter.update([make_track(1, "car", Point(5.0, 0.0), Point(5.0, 0.0))])
    counter.update([make_track(1, "car", Point(5.0, 0.0), Point(5.0, 1.0))])
    counter.update([make_track(1, "car", Point(5.0, 1.0), Point(5.0, 0.0))])
    counter.update([make_track(1, "car", Point(5.0, 0.0), Point(5.0, -1.0))])

    assert counter.snapshot() == {"total": 1, "car": 1}
    assert counter.counted_track_ids == {1}


def test_snapshot_returns_copy_of_counts() -> None:
    counter = LineCounter(line=Line(start=Point(0.0, 0.0), end=Point(10.0, 0.0)))
    counter.update([make_track(1, "car", Point(5.0, -1.0), Point(5.0, 1.0))])

    snapshot = counter.snapshot()
    snapshot["car"] = 99
    snapshot["total"] = 99

    assert counter.snapshot() == {"total": 1, "car": 1}
