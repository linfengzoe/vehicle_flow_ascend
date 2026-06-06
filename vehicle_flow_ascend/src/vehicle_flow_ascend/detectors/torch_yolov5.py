from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vehicle_flow_ascend.config import VehicleFlowConfig
from vehicle_flow_ascend.detectors.base import Detector
from vehicle_flow_ascend.detectors.class_mapping import filter_project_detections
from vehicle_flow_ascend.types import Detection


@dataclass
class TorchYoloV5Detector(Detector):
    config: VehicleFlowConfig

    def __post_init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch_yolov5 backend requires PyTorch. Install torch/torchvision and prepare YOLOv5 weights."
            ) from exc

        if self.config.model_path is None:
            raise ValueError("torch_yolov5 backend requires model_path")

        repo_path = self.config.yolov5_repo_path
        weights_path = self.config.model_path
        if repo_path:
            self._model = torch.hub.load(
                repo_path,
                "custom",
                path=weights_path,
                source="local",
            )
        else:
            self._model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=weights_path,
            )
        self._model.conf = self.config.confidence_threshold
        self._model.iou = self.config.iou_threshold
        self._model_size = self.config.image_size
        self._names = _normalize_names(getattr(self._model, "names", {}))

    def detect(self, frame_bgr) -> list[Detection]:
        results = self._model(frame_bgr, size=self._model_size)
        rows = _extract_xyxy_rows(results)
        detections = detections_from_yolov5_rows(rows, self._names)
        return filter_project_detections(detections)


def _extract_xyxy_rows(results: Any) -> list[list[float]]:
    xyxy = results.xyxy[0]
    if hasattr(xyxy, "detach"):
        xyxy = xyxy.detach()
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu()
    if hasattr(xyxy, "tolist"):
        return xyxy.tolist()
    return list(xyxy)


def _normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list | tuple):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def detections_from_yolov5_rows(rows: list[list[float]], names: dict[int, str]) -> list[Detection]:
    detections: list[Detection] = []
    for row in rows:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, confidence, class_id = row[:6]
        raw_class_id = int(class_id)
        class_name = names.get(raw_class_id, str(raw_class_id))
        detections.append(
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(confidence),
                class_name=class_name,
                raw_class_id=raw_class_id,
            )
        )
    return detections


def default_yolov5_repo_path(project_root: str | Path = ".") -> str:
    return str(Path(project_root) / "third_party" / "yolov5")
