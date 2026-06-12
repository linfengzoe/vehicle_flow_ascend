from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_devboard_web_start_script_documents_runtime_contract() -> None:
    script = PROJECT_ROOT / "scripts" / "start_devboard_web.sh"

    content = script.read_text(encoding="utf-8")

    assert content.startswith("#!/usr/bin/env bash\n")
    assert 'CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"' in content
    assert 'source "$CANN_ENV"' in content
    assert "export PYTHONPATH=src:${PYTHONPATH:-}" in content
    assert "export PYTHONDONTWRITEBYTECODE=1" in content
    assert "python3 -B -m vehicle_flow_ascend" in content
    assert 'CONFIG_PATH="${CONFIG_PATH:-configs/ascend_om.yaml}"' in content
    assert 'WEB_HOST="${WEB_HOST:-0.0.0.0}"' in content
    assert 'WEB_PORT="${WEB_PORT:-8766}"' in content
    assert 'WEB_CERTFILE="${WEB_CERTFILE:-certs/web.crt}"' in content
    assert 'WEB_KEYFILE="${WEB_KEYFILE:-certs/web.key}"' in content
    assert '--config "$CONFIG_PATH"' in content
    assert '--web-host "$WEB_HOST"' in content
    assert '--web-port "$WEB_PORT"' in content
    assert '--web-certfile "$WEB_CERTFILE"' in content
    assert '--web-keyfile "$WEB_KEYFILE"' in content
    assert "runs/web-https.pid" in content
    assert "runs/web-https.log" in content
    assert "pkill -f" in content
    assert "\r\n" not in content
