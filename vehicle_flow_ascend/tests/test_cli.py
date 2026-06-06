import json
from pathlib import Path

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
        "--display",
        "false",
        "--max-frames",
        "12",
    ])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == 0
    assert output["backend"] == "ascend_om"
    assert output["display"] is False
    assert output["max_frames"] == 12


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
