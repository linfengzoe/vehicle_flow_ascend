#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/ascend_om.yaml}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8766}"
WEB_CERTFILE="${WEB_CERTFILE:-certs/web.crt}"
WEB_KEYFILE="${WEB_KEYFILE:-certs/web.key}"
PID_FILE="${PID_FILE:-runs/web-https.pid}"
LOG_FILE="${LOG_FILE:-runs/web-https.log}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"

if [[ ! -f "$CANN_ENV" ]]; then
  echo "CANN environment script not found: $CANN_ENV" >&2
  exit 1
fi

if [[ ! -f "$WEB_CERTFILE" || ! -f "$WEB_KEYFILE" ]]; then
  echo "HTTPS certificate files are missing:" >&2
  echo "  cert: $WEB_CERTFILE" >&2
  echo "  key:  $WEB_KEYFILE" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]]; then
    kill "$old_pid" 2>/dev/null || true
  fi
fi
pkill -f "python3.*vehicle_flow_ascend.*--web.*--web-port ${WEB_PORT}" 2>/dev/null || true
sleep 1

source "$CANN_ENV"
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1

nohup python3 -B -m vehicle_flow_ascend \
  --config "$CONFIG_PATH" \
  --web \
  --web-host "$WEB_HOST" \
  --web-port "$WEB_PORT" \
  --web-certfile "$WEB_CERTFILE" \
  --web-keyfile "$WEB_KEYFILE" \
  </dev/null > "$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"
sleep 2

if ! kill -0 "$pid" 2>/dev/null; then
  echo "Vehicle Flow Web failed to start. Log follows:" >&2
  tail -80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "Vehicle Flow Web started."
echo "PID: $pid"
echo "URL: https://$(hostname -I | awk '{print $1}'):${WEB_PORT}/"
echo "Log: $PROJECT_DIR/$LOG_FILE"
