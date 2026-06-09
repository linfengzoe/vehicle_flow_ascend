from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from vehicle_flow_ascend.config import load_config_with_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vehicle flow counting demo")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--source", help="Override source path or camera index")
    parser.add_argument("--backend", help="Override backend name")
    parser.add_argument("--model-path", help="Override model path")
    parser.add_argument("--yolov5-repo-path", help="Override local YOLOv5 repository path")
    parser.add_argument("--soc-version", help="Override Ascend ATC/OM SOC version")
    parser.add_argument("--image-size", type=int, help="Override square model input size")
    parser.add_argument("--confidence-threshold", type=float, help="Override confidence threshold")
    parser.add_argument("--iou-threshold", type=float, help="Override NMS IoU threshold")
    parser.add_argument("--line-start", help="Override counting line start as x,y")
    parser.add_argument("--line-end", help="Override counting line end as x,y")
    parser.add_argument("--display", choices=("true", "false"), help="Override display flag")
    parser.add_argument("--output-video", help="Override annotated output video path")
    parser.add_argument("--max-frames", type=int, help="Override maximum frames to process")
    parser.add_argument("--web", action="store_true", help="Start the built-in visualization dashboard")
    parser.add_argument("--web-host", default="127.0.0.1", help="Dashboard host, default: 127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765, help="Dashboard port, default: 8765")
    parser.add_argument("--web-certfile", help="TLS certificate file for HTTPS dashboard")
    parser.add_argument("--web-keyfile", help="TLS private key file for HTTPS dashboard")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    line_override = _line_override(args.line_start, args.line_end, parser)
    config = load_config_with_overrides(
        args.config,
        {
            "source": args.source,
            "backend": args.backend,
            "model_path": args.model_path,
            "yolov5_repo_path": args.yolov5_repo_path,
            "soc_version": args.soc_version,
            "image_size": args.image_size,
            "confidence_threshold": args.confidence_threshold,
            "iou_threshold": args.iou_threshold,
            "line": line_override,
            "display": args.display,
            "output_video": args.output_video,
            "max_frames": args.max_frames,
        },
    )

    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.web:
        from vehicle_flow_ascend.web.dashboard import DashboardServerConfig, run_dashboard

        run_dashboard(
            config,
            DashboardServerConfig(
                host=args.web_host,
                port=args.web_port,
                certfile=args.web_certfile,
                keyfile=args.web_keyfile,
            ),
        )
        return 0

    from vehicle_flow_ascend.app import run_app
    from vehicle_flow_ascend.detectors.base import create_detector

    detector = create_detector(config)
    counts = run_app(config, detector)
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    return 0


def _line_override(line_start: str | None, line_end: str | None, parser: argparse.ArgumentParser):
    if line_start is None and line_end is None:
        return None
    if line_start is None or line_end is None:
        parser.error("--line-start and --line-end must be provided together")
    return [_parse_point(line_start, parser), _parse_point(line_end, parser)]


def _parse_point(value: str, parser: argparse.ArgumentParser) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        parser.error(f"line point must be formatted as x,y: {value!r}")
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError:
        parser.error(f"line point must contain integer coordinates: {value!r}")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
