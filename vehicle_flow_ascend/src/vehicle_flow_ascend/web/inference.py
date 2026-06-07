from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from vehicle_flow_ascend.app import run_app
from vehicle_flow_ascend.config import SourceConfig, VehicleFlowConfig
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
        }


class InferenceTaskManager:
    def __init__(self, base_config: VehicleFlowConfig, project_root: Path | None = None) -> None:
        self._base_config = base_config
        self._project_root = project_root or Path.cwd()
        self._lock = threading.RLock()
        self._state = InferenceState()
        self._thread: threading.Thread | None = None

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._state.status in {"running", "stopping"}:
                raise RuntimeError("inference task is already running")

            task_id = uuid.uuid4().hex[:10]
            source = _source_from_payload(payload)
            output_video = _output_video_path(payload, task_id)
            config = replace(
                self._base_config,
                source=source,
                output_video=output_video,
                display=False,
            )
            self._state = InferenceState(
                status="running",
                task_id=task_id,
                source=source.to_cli_value(),
                output_video=output_video,
                started_at=time.time(),
                counts={"total": 0},
            )
            self._thread = threading.Thread(
                target=self._run_task,
                args=(task_id, config),
                name=f"vehicle-flow-inference-{task_id}",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._state.status == "running":
                self._state.stop_requested = True
                self._state.status = "stopping"
            return self._state.to_dict()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

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
            with self._lock:
                if self._state.task_id != task_id:
                    return
                self._state.counts = counts
                self._state.finished_at = time.time()
                self._state.status = "stopped" if self._state.stop_requested else "completed"
        except Exception as exc:  # noqa: BLE001 - surface worker errors to UI
            with self._lock:
                if self._state.task_id != task_id:
                    return
                self._state.status = "failed"
                self._state.error = str(exc)
                self._state.finished_at = time.time()
        finally:
            release = getattr(detector, "release", None)
            if callable(release):
                release()

    def _stop_requested_for(self, task_id: str):
        def stop_requested() -> bool:
            with self._lock:
                return self._state.task_id == task_id and self._state.stop_requested

        return stop_requested

    def _progress_callback_for(self, task_id: str):
        def progress(progress_state: dict[str, Any]) -> None:
            with self._lock:
                if self._state.task_id != task_id:
                    return
                self._state.frames = int(progress_state.get("frames", self._state.frames))
                self._state.fps = float(progress_state.get("fps", self._state.fps))
                self._state.counts = dict(progress_state.get("counts", self._state.counts))

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
