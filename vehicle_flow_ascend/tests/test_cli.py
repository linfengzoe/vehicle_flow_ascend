import json
from pathlib import Path

import pytest

from vehicle_flow_ascend.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_prints_resolved_config(capsys) -> None:
    exit_code = main([
        "--config",
        str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
        "--dry-run",
    ])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == "data/pc_demo.mp4"
    assert output["backend"] == "torch_yolov5"
    assert output["display"] is True
    assert output["max_frames"] == 300


def test_dry_run_applies_cli_overrides(capsys) -> None:
    exit_code = main([
        "--config",
        str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
        "--dry-run",
        "--source",
        "0",
        "--backend",
        "ascend_om",
        "--model-path",
        "models/custom.om",
        "--soc-version",
        "Ascend310B4",
        "--image-size",
        "512",
        "--confidence-threshold",
        "0.42",
        "--iou-threshold",
        "0.51",
        "--line-start",
        "10,20",
        "--line-end",
        "300,400",
        "--display",
        "false",
        "--output-video",
        "runs/custom.mp4",
        "--max-frames",
        "12",
    ])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == 0
    assert output["backend"] == "ascend_om"
    assert output["model_path"] == "models/custom.om"
    assert output["soc_version"] == "Ascend310B4"
    assert output["image_size"] == 512
    assert output["confidence_threshold"] == 0.42
    assert output["iou_threshold"] == 0.51
    assert output["line"] == [[10, 20], [300, 400]]
    assert output["display"] is False
    assert output["output_video"] == "runs/custom.mp4"
    assert output["max_frames"] == 12


def test_line_override_requires_start_and_end() -> None:
    with pytest.raises(SystemExit):
        main([
            "--config",
            str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
            "--dry-run",
            "--line-start",
            "10,20",
        ])


def test_line_override_rejects_invalid_point() -> None:
    with pytest.raises(SystemExit):
        main([
            "--config",
            str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
            "--dry-run",
            "--line-start",
            "bad",
            "--line-end",
            "30,40",
        ])


def test_dry_run_does_not_import_heavy_runtime_modules(capsys) -> None:
    assert main([
        "--config",
        str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
        "--dry-run",
    ]) == 0
    capsys.readouterr()

    import sys

    assert "torch" not in sys.modules
    assert "acl" not in sys.modules


def test_web_run_starts_dashboard(monkeypatch) -> None:
    from vehicle_flow_ascend.web import dashboard

    calls = {}

    def fake_run_dashboard(config, server_config):
        calls["backend"] = config.backend
        calls["host"] = server_config.host
        calls["port"] = server_config.port

    monkeypatch.setattr(dashboard, "run_dashboard", fake_run_dashboard)

    exit_code = main([
        "--config",
        str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
        "--web",
        "--web-host",
        "0.0.0.0",
        "--web-port",
        "8899",
    ])

    assert exit_code == 0
    assert calls == {"backend": "torch_yolov5", "host": "0.0.0.0", "port": 8899}


def test_web_run_passes_tls_certificate_paths(monkeypatch) -> None:
    from vehicle_flow_ascend.web import dashboard

    calls = {}

    def fake_run_dashboard(config, server_config):
        calls["backend"] = config.backend
        calls["certfile"] = server_config.certfile
        calls["keyfile"] = server_config.keyfile
        calls["url"] = server_config.url

    monkeypatch.setattr(dashboard, "run_dashboard", fake_run_dashboard)

    exit_code = main([
        "--config",
        str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
        "--web",
        "--web-host",
        "0.0.0.0",
        "--web-port",
        "8766",
        "--web-certfile",
        "certs/web.crt",
        "--web-keyfile",
        "certs/web.key",
    ])

    assert exit_code == 0
    assert calls == {
        "backend": "torch_yolov5",
        "certfile": "certs/web.crt",
        "keyfile": "certs/web.key",
        "url": "https://0.0.0.0:8766",
    }


def test_normal_run_creates_detector_and_runs_app(monkeypatch, capsys) -> None:
    from vehicle_flow_ascend import app
    from vehicle_flow_ascend.detectors import base

    calls = {}

    class FakeDetector:
        pass

    def fake_create_detector(config):
        calls["backend"] = config.backend
        calls["source"] = config.source.path
        return FakeDetector()

    def fake_run_app(config, detector):
        calls["detector"] = detector
        calls["max_frames"] = config.max_frames
        return {"total": 3, "car": 2, "truck": 1}

    monkeypatch.setattr(base, "create_detector", fake_create_detector)
    monkeypatch.setattr(app, "run_app", fake_run_app)

    exit_code = main([
        "--config",
        str(PROJECT_ROOT / "configs" / "pc_demo.yaml"),
        "--source",
        "data/demo.mp4",
        "--max-frames",
        "5",
    ])

    assert exit_code == 0
    assert calls["backend"] == "torch_yolov5"
    assert calls["source"] == "data/demo.mp4"
    assert calls["max_frames"] == 5
    assert isinstance(calls["detector"], FakeDetector)
    assert json.loads(capsys.readouterr().out) == {"total": 3, "car": 2, "truck": 1}
