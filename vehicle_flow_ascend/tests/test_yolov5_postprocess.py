import numpy as np
import pytest

from vehicle_flow_ascend.detectors.yolov5_postprocess import (
    letterbox,
    non_max_suppression,
    postprocess_yolov5_output,
    preprocess_frame,
    scale_boxes_to_original,
    xywh_to_xyxy,
)


def test_letterbox_preserves_aspect_ratio_and_pads() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    result = letterbox(frame, (640, 640))

    assert result.image.shape == (640, 640, 3)
    assert result.ratio == 3.2
    assert result.pad == (0.0, 160.0)


def test_preprocess_frame_returns_normalized_nchw_rgb() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # B channel in BGR input

    tensor, result = preprocess_frame(frame, (20, 20))

    assert tensor.shape == (1, 3, 20, 20)
    assert tensor.dtype == np.float32
    assert result.ratio == 1.0
    # After BGR->RGB, blue input becomes channel 2.
    assert tensor[0, 2, 5, 5] == 1.0
    assert tensor[0, 0, 5, 5] == 0.0


def test_scale_boxes_back_to_original_frame() -> None:
    boxes = np.array([[320.0, 240.0, 640.0, 560.0]], dtype=np.float32)

    scaled = scale_boxes_to_original(
        boxes,
        original_shape=(100, 200),
        ratio=3.2,
        pad=(0.0, 160.0),
    )

    np.testing.assert_allclose(scaled, np.array([[100.0, 25.0, 200.0, 100.0]], dtype=np.float32))


def test_xywh_to_xyxy() -> None:
    boxes = np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)

    converted = xywh_to_xyxy(boxes)

    np.testing.assert_allclose(converted, np.array([[8.0, 17.0, 12.0, 23.0]], dtype=np.float32))


def test_nms_removes_overlapping_same_class_boxes() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    class_ids = np.array([0, 0, 0], dtype=np.int32)

    kept = non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)

    assert kept == [0, 2]


def test_nms_keeps_overlapping_different_classes() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)

    kept = non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)

    assert kept == [0, 1]


def test_postprocess_filters_low_confidence_predictions() -> None:
    output = np.array(
        [
            [50.0, 50.0, 20.0, 20.0, 0.2, 0.9],
        ],
        dtype=np.float32,
    )

    detections = postprocess_yolov5_output(
        output,
        original_shape=(100, 100),
        ratio=1.0,
        pad=(0.0, 0.0),
        class_names={0: "car"},
        confidence_threshold=0.5,
    )

    assert detections == []


def test_postprocess_maps_coco_vehicle_classes_and_scales_boxes() -> None:
    output = np.array(
        [
            # x, y, w, h, objectness, bicycle score, truck score, person score
            [320.0, 320.0, 64.0, 64.0, 0.9, 0.8, 0.1, 0.1],
            [160.0, 160.0, 32.0, 32.0, 0.8, 0.1, 0.7, 0.1],
            [100.0, 100.0, 30.0, 30.0, 0.99, 0.1, 0.1, 0.95],
        ],
        dtype=np.float32,
    )

    detections = postprocess_yolov5_output(
        output,
        original_shape=(320, 320),
        ratio=2.0,
        pad=(0.0, 0.0),
        class_names={0: "bicycle", 1: "truck", 2: "person"},
        confidence_threshold=0.5,
        iou_threshold=0.45,
    )

    assert [d.class_name for d in detections] == ["two_wheeler", "truck"]
    assert detections[0].raw_class_id == 0
    assert detections[0].confidence == pytest.approx(0.72)
    np.testing.assert_allclose(
        [detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2],
        [144.0, 144.0, 176.0, 176.0],
    )


def test_postprocess_accepts_batched_output() -> None:
    output = np.array([[[50.0, 50.0, 20.0, 20.0, 0.9, 0.8]]], dtype=np.float32)

    detections = postprocess_yolov5_output(
        output,
        original_shape=(100, 100),
        ratio=1.0,
        pad=(0.0, 0.0),
        class_names=["car"],
        confidence_threshold=0.5,
    )

    assert len(detections) == 1
    assert detections[0].class_name == "car"
