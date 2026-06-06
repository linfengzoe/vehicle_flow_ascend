import pytest

from vehicle_flow_ascend.detectors.base import Detector
from vehicle_flow_ascend.detectors.class_mapping import (
    filter_project_detections,
    map_coco_class,
)
from vehicle_flow_ascend.types import Detection, Point


@pytest.mark.parametrize(
    ("coco_name", "project_name"),
    [
        ("car", "car"),
        ("bus", "bus"),
        ("truck", "truck"),
    ],
)
def test_map_coco_class_maps_vehicle_classes(coco_name: str, project_name: str) -> None:
    assert map_coco_class(coco_name) == project_name


@pytest.mark.parametrize("coco_name", ["motorcycle", "bicycle"])
def test_map_coco_class_merges_two_wheelers(coco_name: str) -> None:
    assert map_coco_class(coco_name) == "two_wheeler"


@pytest.mark.parametrize("coco_name", ["person", "traffic light", "dog"])
def test_map_coco_class_ignores_non_vehicle_classes(coco_name: str) -> None:
    assert map_coco_class(coco_name) is None


def test_filter_project_detections_preserves_fields_and_maps_class_names() -> None:
    detections = [
        Detection(
            x1=1.0,
            y1=2.0,
            x2=11.0,
            y2=22.0,
            confidence=0.91,
            class_name="motorcycle",
            raw_class_id=3,
        ),
        Detection(
            x1=3.0,
            y1=4.0,
            x2=13.0,
            y2=24.0,
            confidence=0.82,
            class_name="person",
            raw_class_id=0,
        ),
        Detection(
            x1=5.0,
            y1=6.0,
            x2=15.0,
            y2=26.0,
            confidence=0.73,
            class_name="truck",
            raw_class_id=7,
        ),
    ]

    filtered = filter_project_detections(detections)

    assert filtered == [
        Detection(
            x1=1.0,
            y1=2.0,
            x2=11.0,
            y2=22.0,
            confidence=0.91,
            class_name="two_wheeler",
            raw_class_id=3,
        ),
        detections[2],
    ]


def test_filter_project_detections_keeps_existing_project_classes_as_is() -> None:
    detections = [
        Detection(1.0, 2.0, 3.0, 4.0, 0.9, "two_wheeler", raw_class_id=None),
        Detection(5.0, 6.0, 7.0, 8.0, 0.8, "car", raw_class_id=2),
        Detection(9.0, 10.0, 11.0, 12.0, 0.7, "bus", raw_class_id=5),
        Detection(13.0, 14.0, 15.0, 16.0, 0.6, "truck", raw_class_id=7),
    ]

    assert filter_project_detections(detections) == detections


def test_detection_centroid_returns_bbox_center() -> None:
    detection = Detection(
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=60.0,
        confidence=0.95,
        class_name="car",
    )

    assert detection.centroid == Point(x=20.0, y=40.0)


def test_detector_abstract_base_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Detector()


class StubDetector(Detector):
    def detect(self, frame_bgr) -> list[Detection]:
        return [Detection(1.0, 2.0, 3.0, 4.0, 0.5, "car", raw_class_id=2)]


def test_detector_subclass_can_be_instantiated_and_detects() -> None:
    detector = StubDetector()

    assert detector.detect(frame_bgr=object()) == [
        Detection(1.0, 2.0, 3.0, 4.0, 0.5, "car", raw_class_id=2)
    ]
