from vehicle_flow_ascend.tracking.centroid import CentroidTracker
from vehicle_flow_ascend.types import Detection, Point


def detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str = "car",
) -> Detection:
    return Detection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=0.9,
        class_name=class_name,
    )


def test_creates_track_for_first_detection() -> None:
    tracker = CentroidTracker()

    tracks = tracker.update([detection(0, 0, 10, 10)])

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_name == "car"
    assert tracks[0].previous_centroid is None
    assert tracks[0].current_centroid == Point(5, 5)


def test_preserves_track_id_for_nearby_detection() -> None:
    tracker = CentroidTracker(max_distance=20)

    first = tracker.update([detection(0, 0, 10, 10)])[0]
    second = tracker.update([detection(4, 0, 14, 10)])[0]

    assert second.track_id == first.track_id
    assert second.previous_centroid == Point(5, 5)
    assert second.current_centroid == Point(9, 5)


def test_creates_new_track_when_detection_is_too_far() -> None:
    tracker = CentroidTracker(max_distance=10)

    tracker.update([detection(0, 0, 10, 10)])
    tracks = tracker.update([detection(100, 100, 110, 110)])

    assert {track.track_id for track in tracks} == {1, 2}
    assert any(track.disappeared == 1 for track in tracks if track.track_id == 1)


def test_removes_track_after_max_disappeared_frames() -> None:
    tracker = CentroidTracker(max_disappeared=1)

    tracker.update([detection(0, 0, 10, 10)])
    tracks_after_one_miss = tracker.update([])
    tracks_after_two_misses = tracker.update([])

    assert len(tracks_after_one_miss) == 1
    assert tracks_after_one_miss[0].disappeared == 1
    assert tracks_after_two_misses == []


def test_keeps_separate_ids_for_two_vehicles() -> None:
    tracker = CentroidTracker(max_distance=30)

    first_tracks = tracker.update([
        detection(0, 0, 10, 10),
        detection(100, 0, 110, 10, "bus"),
    ])
    second_tracks = tracker.update([
        detection(2, 0, 12, 10),
        detection(102, 0, 112, 10, "bus"),
    ])

    assert {track.track_id for track in first_tracks} == {1, 2}
    assert {track.track_id for track in second_tracks} == {1, 2}
    assert {track.class_name for track in second_tracks} == {"car", "bus"}


def test_updates_class_name_from_detection() -> None:
    tracker = CentroidTracker(max_distance=20)

    tracker.update([detection(0, 0, 10, 10, "car")])
    tracks = tracker.update([detection(1, 0, 11, 10, "truck")])

    assert tracks[0].track_id == 1
    assert tracks[0].class_name == "truck"


def test_unmatched_track_keeps_previous_centroid_for_counter() -> None:
    tracker = CentroidTracker(max_disappeared=2)

    tracker.update([detection(0, 0, 10, 10)])
    tracks = tracker.update([])

    assert tracks[0].previous_centroid == Point(5, 5)
    assert tracks[0].current_centroid == Point(5, 5)
