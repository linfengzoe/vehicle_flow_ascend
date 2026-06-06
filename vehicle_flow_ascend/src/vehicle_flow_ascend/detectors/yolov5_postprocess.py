from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from vehicle_flow_ascend.detectors.class_mapping import map_coco_class
from vehicle_flow_ascend.types import Detection


@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray
    ratio: float
    pad: tuple[float, float]


def letterbox(
    frame_bgr: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> LetterboxResult:
    """Resize and pad an image while preserving aspect ratio.

    `new_shape` is `(height, width)`. `pad` is `(pad_x, pad_y)` before splitting across
    both sides; use it with `scale_boxes_to_original()` to recover original coordinates.
    """
    original_height, original_width = frame_bgr.shape[:2]
    target_height, target_width = new_shape
    ratio = min(target_width / original_width, target_height / original_height)

    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))
    pad_x = (target_width - resized_width) / 2.0
    pad_y = (target_height - resized_height) / 2.0

    if (resized_width, resized_height) != (original_width, original_height):
        resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame_bgr

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return LetterboxResult(image=padded, ratio=ratio, pad=(pad_x, pad_y))


def preprocess_frame(
    frame_bgr: np.ndarray,
    input_shape: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, LetterboxResult]:
    """Convert a BGR frame to a normalized NCHW float32 model input."""
    result = letterbox(frame_bgr, input_shape)
    rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    nchw = np.transpose(normalized, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(nchw), result


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    original_shape: tuple[int, int],
    ratio: float,
    pad: tuple[float, float],
) -> np.ndarray:
    """Map `xyxy` boxes from letterboxed input coordinates to original image coordinates."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32)

    boxes = boxes_xyxy.astype(np.float32).copy()
    pad_x, pad_y = pad
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= ratio

    original_height, original_width = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    return boxes


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = boxes_xywh.astype(np.float32).copy()
    half_width = boxes[:, 2] / 2.0
    half_height = boxes[:, 3] / 2.0
    x_center = boxes[:, 0].copy()
    y_center = boxes[:, 1].copy()
    boxes[:, 0] = x_center - half_width
    boxes[:, 1] = y_center - half_height
    boxes[:, 2] = x_center + half_width
    boxes[:, 3] = y_center + half_height
    return boxes


def non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Class-aware NumPy NMS returning kept indexes in descending score order."""
    if boxes_xyxy.size == 0:
        return []

    kept: list[int] = []
    for class_id in sorted(set(int(value) for value in class_ids.tolist())):
        indexes = np.where(class_ids == class_id)[0]
        order = indexes[np.argsort(scores[indexes])[::-1]]
        while order.size > 0:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            ious = _iou(boxes_xyxy[current], boxes_xyxy[rest])
            order = rest[ious <= iou_threshold]

    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept


def postprocess_yolov5_output(
    output: np.ndarray,
    *,
    original_shape: tuple[int, int],
    ratio: float,
    pad: tuple[float, float],
    class_names: dict[int, str] | list[str] | tuple[str, ...],
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Convert exported YOLOv5 output tensors into project `Detection` objects.

    Expected output shape is `(N, 5 + num_classes)` or `(1, N, 5 + num_classes)`,
    where boxes are `xywh`, column 4 is objectness, and remaining columns are class scores.
    """
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.size == 0:
        return []
    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise ValueError("YOLOv5 output must have shape (N, 5 + num_classes)")

    objectness = predictions[:, 4]
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    class_confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    scores = objectness * class_confidences
    keep_confident = scores >= confidence_threshold
    if not np.any(keep_confident):
        return []

    boxes = xywh_to_xyxy(predictions[keep_confident, :4])
    scores = scores[keep_confident]
    class_ids = class_ids[keep_confident]
    boxes = scale_boxes_to_original(boxes, original_shape, ratio, pad)
    keep_nms = non_max_suppression(boxes, scores, class_ids, iou_threshold)

    names = _normalize_class_names(class_names)
    detections: list[Detection] = []
    for index in keep_nms:
        raw_class_id = int(class_ids[index])
        coco_name = names.get(raw_class_id, str(raw_class_id))
        project_name = map_coco_class(coco_name)
        if project_name is None:
            continue
        x1, y1, x2, y2 = boxes[index].tolist()
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(scores[index]),
                class_name=project_name,
                raw_class_id=raw_class_id,
            )
        )
    return detections


def _normalize_class_names(class_names: dict[int, str] | Iterable[str]) -> dict[int, str]:
    if isinstance(class_names, dict):
        return {int(key): str(value) for key, value in class_names.items()}
    return {index: str(value) for index, value in enumerate(class_names)}


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
