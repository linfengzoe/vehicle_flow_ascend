from vehicle_flow_ascend.detectors.class_mapping import filter_project_detections
from vehicle_flow_ascend.detectors.torch_yolov5 import (
    _normalize_names,
    detections_from_yolov5_rows,
)


def test_normalize_names_accepts_dict_and_list() -> None:
    assert _normalize_names({0: "car", "1": "bus"}) == {0: "car", 1: "bus"}
    assert _normalize_names(["car", "bus"]) == {0: "car", 1: "bus"}


def test_converts_yolov5_rows_to_detections() -> None:
    detections = detections_from_yolov5_rows(
        [[1, 2, 11, 22, 0.91, 2]],
        {2: "car"},
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.x1 == 1.0
    assert detection.y1 == 2.0
    assert detection.x2 == 11.0
    assert detection.y2 == 22.0
    assert detection.confidence == 0.91
    assert detection.class_name == "car"
    assert detection.raw_class_id == 2


def test_converted_rows_can_be_filtered_to_project_classes() -> None:
    detections = detections_from_yolov5_rows(
        [
            [0, 0, 10, 10, 0.9, 0],
            [20, 20, 30, 30, 0.8, 1],
            [40, 40, 50, 50, 0.7, 2],
        ],
        {0: "person", 1: "bicycle", 2: "truck"},
    )

    filtered = filter_project_detections(detections)

    assert [d.class_name for d in filtered] == ["two_wheeler", "truck"]
    assert filtered[0].raw_class_id == 1
    assert filtered[1].raw_class_id == 2


def test_short_yolov5_rows_are_ignored() -> None:
    assert detections_from_yolov5_rows([[1, 2, 3]], {0: "car"}) == []
