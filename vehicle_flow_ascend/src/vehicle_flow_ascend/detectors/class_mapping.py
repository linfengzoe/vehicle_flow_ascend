from vehicle_flow_ascend.types import Detection


_PROJECT_CLASS_NAMES = {"two_wheeler", "car", "bus", "truck"}
_COCO_TO_PROJECT_CLASS_NAMES = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "two_wheeler",
    "bicycle": "two_wheeler",
}


def map_coco_class(name: str) -> str | None:
    return _COCO_TO_PROJECT_CLASS_NAMES.get(name)


def filter_project_detections(detections: list[Detection]) -> list[Detection]:
    project_detections: list[Detection] = []

    for detection in detections:
        if detection.class_name in _PROJECT_CLASS_NAMES:
            project_detections.append(detection)
            continue

        class_name = map_coco_class(detection.class_name)
        if class_name is None:
            continue

        project_detections.append(
            Detection(
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                confidence=detection.confidence,
                class_name=class_name,
                raw_class_id=detection.raw_class_id,
            )
        )

    return project_detections
