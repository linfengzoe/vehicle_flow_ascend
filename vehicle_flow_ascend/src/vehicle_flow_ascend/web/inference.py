from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2

from vehicle_flow_ascend.app import run_app
from vehicle_flow_ascend.config import LineConfig, SourceConfig, VehicleFlowConfig
from vehicle_flow_ascend.detectors.base import create_detector


@dataclass
class InferenceState:
    status: str = "idle"
    task_id: str | None = None
    source: str | int | None = None
    output_video: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    frames: int = 0
    fps: float = 0.0
    counts: dict[str, int] = field(default_factory=lambda: {"total": 0})
    error: str | None = None
    stop_requested: bool = False
    frame_version: int = 0
    updated_at: float | None = None
    last_frame_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_id": self.task_id,
            "source": self.source,
            "output_video": self.output_video,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "frames": self.frames,
            "fps": self.fps,
            "counts": dict(self.counts),
            "error": self.error,
            "stop_requested": self.stop_requested,
            "frame_version": self.frame_version,
            "updated_at": self.updated_at,
            "last_frame_at": self.last_frame_at,
        }


class InferenceTaskManager:
    def __init__(self, base_config: VehicleFlowConfig, project_root: Path | None = None) -> None:
        self._base_config = base_config
        self._project_root = project_root or Path.cwd()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._state = InferenceState()
        self._latest_jpeg: bytes | None = None
        self._thread: threading.Thread | None = None

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            if self._state.status in {"running", "stopping"}:
                raise RuntimeError("inference task is already running")

            task_id = uuid.uuid4().hex[:10]
            source = _source_from_payload(payload)
            output_video = _output_video_path(payload, task_id)
            line = _line_from_payload(payload, self._base_config.line)
            config = replace(
                self._base_config,
                source=source,
                output_video=output_video,
                line=line,
                display=False,
            )
            now = time.time()
            self._latest_jpeg = None
            self._state = InferenceState(
                status="running",
                task_id=task_id,
                source=source.to_cli_value(),
                output_video=output_video,
                started_at=now,
                counts={"total": 0},
                updated_at=now,
            )
            self._thread = threading.Thread(
                target=self._run_task,
                args=(task_id, config),
                name=f"vehicle-flow-inference-{task_id}",
                daemon=True,
            )
            self._thread.start()
            self._condition.notify_all()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._condition:
            if self._state.status == "running":
                self._state.stop_requested = True
                self._state.status = "stopping"
                self._state.updated_at = time.time()
                self._condition.notify_all()
            return self._state.to_dict()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return self._state.to_dict()

    def wait_for_frame(
        self,
        task_id: str,
        last_version: int,
        timeout: float = 1.0,
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._state.task_id != task_id:
                raise ValueError("invalid inference task")
            while (
                self._state.status in {"running", "stopping"}
                and self._state.frame_version <= last_version
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
                if self._state.task_id != task_id:
                    raise ValueError("invalid inference task")

            if self._latest_jpeg is None or self._state.frame_version <= last_version:
                return None
            return self._state.frame_version, bytes(self._latest_jpeg)

    def upload(self, filename: str, source) -> dict[str, Any]:
        upload_dir = self._project_root / "vehicle_flow_ascend" / "data" / "web_uploads"
        if not (self._project_root / "vehicle_flow_ascend").exists():
            upload_dir = self._project_root / "data" / "web_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or f"upload-{uuid.uuid4().hex[:8]}.mp4"
        target = upload_dir / safe_name
        with target.open("wb") as output:
            shutil.copyfileobj(source, output)
        return {"path": str(target), "name": safe_name, "size": target.stat().st_size}

    def _run_task(self, task_id: str, config: VehicleFlowConfig) -> None:
        detector = None
        try:
            detector = create_detector(config)
            counts = run_app(
                config,
                detector,
                show_window=False,
                stop_requested=self._stop_requested_for(task_id),
                progress_callback=self._progress_callback_for(task_id),
            )
            with self._condition:
                if self._state.task_id != task_id:
                    return
                self._state.counts = counts
                self._state.finished_at = time.time()
                self._state.status = "stopped" if self._state.stop_requested else "completed"
                self._state.updated_at = self._state.finished_at
                self._condition.notify_all()
        except Exception as exc:  # noqa: BLE001 - surface worker errors to UI
            with self._condition:
                if self._state.task_id != task_id:
                    return
                self._state.status = "failed"
                self._state.error = str(exc)
                self._state.finished_at = time.time()
                self._state.updated_at = self._state.finished_at
                self._condition.notify_all()
        finally:
            release = getattr(detector, "release", None)
            if callable(release):
                release()

    def _stop_requested_for(self, task_id: str):
        def stop_requested() -> bool:
            with self._condition:
                return self._state.task_id == task_id and self._state.stop_requested

        return stop_requested

    def _progress_callback_for(self, task_id: str):
        def progress(progress_state: dict[str, Any]) -> None:
            annotated_frame = progress_state.get("annotated_frame")
            latest_jpeg = _encode_jpeg(annotated_frame) if annotated_frame is not None else None
            with self._condition:
                if self._state.task_id != task_id:
                    return
                now = time.time()
                self._state.frames = int(progress_state.get("frames", self._state.frames))
                self._state.fps = float(progress_state.get("fps", self._state.fps))
                self._state.counts = dict(progress_state.get("counts", self._state.counts))
                self._state.updated_at = now
                if latest_jpeg is not None:
                    self._latest_jpeg = latest_jpeg
                    self._state.frame_version += 1
                    self._state.last_frame_at = now
                self._condition.notify_all()

        return progress


def _source_from_payload(payload: dict[str, Any]) -> SourceConfig:
    source_type = str(payload.get("source_type", "video"))
    value = payload.get("source")
    if source_type == "camera":
        camera = 0 if value in (None, "") else int(value)
        return SourceConfig(camera=camera)
    if value in (None, ""):
        raise ValueError("video source path is required")
    return SourceConfig(path=str(value))


def _output_video_path(payload: dict[str, Any], task_id: str) -> str:
    value = payload.get("output_video")
    if value:
        return str(value)
    return str(Path("runs") / f"web_inference_{task_id}.mp4")


def _line_from_payload(payload: dict[str, Any], default_line: LineConfig) -> LineConfig:
    value = payload.get("line")
    if value is None:
        return default_line
    return LineConfig.from_value(value)


def _encode_jpeg(frame_bgr) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        raise ValueError("failed to encode inference preview frame")
    return encoded.tobytes()
