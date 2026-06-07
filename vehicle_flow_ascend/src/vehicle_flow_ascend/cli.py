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
    parser.add_argument("--display", choices=("true", "false"), help="Override display flag")
    parser.add_argument("--max-frames", type=int, help="Override maximum frames to process")
    parser.add_argument("--web", action="store_true", help="Start the built-in visualization dashboard")
    parser.add_argument("--web-host", default="127.0.0.1", help="Dashboard host, default: 127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765, help="Dashboard port, default: 8765")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config_with_overrides(
        args.config,
        {
            "source": args.source,
            "backend": args.backend,
            "display": args.display,
            "max_frames": args.max_frames,
        },
    )

    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.web:
        from vehicle_flow_ascend.web.dashboard import DashboardServerConfig, run_dashboard

        run_dashboard(config, DashboardServerConfig(host=args.web_host, port=args.web_port))
        return 0

    from vehicle_flow_ascend.app import run_app
    from vehicle_flow_ascend.detectors.base import create_detector

    detector = create_detector(config)
    counts = run_app(config, detector)
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
