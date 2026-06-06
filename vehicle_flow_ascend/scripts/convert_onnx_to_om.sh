#!/usr/bin/env bash
set -euo pipefail

ONNX_MODEL="${ONNX_MODEL:-models/yolov5n.onnx}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-models/yolov5n}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
INPUT_SHAPE="${INPUT_SHAPE:-images:1,3,640,640}"
INPUT_FORMAT="${INPUT_FORMAT:-NCHW}"

if [[ ! -f "$ONNX_MODEL" ]]; then
  echo "ONNX model not found: $ONNX_MODEL" >&2
  exit 1
fi

if ! command -v atc >/dev/null 2>&1; then
  echo "atc not found. Source the CANN environment first, for example:" >&2
  echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
  exit 1
fi

atc \
  --framework=5 \
  --model="$ONNX_MODEL" \
  --output="$OUTPUT_PREFIX" \
  --input_format="$INPUT_FORMAT" \
  --input_shape="$INPUT_SHAPE" \
  --soc_version="$SOC_VERSION"

echo "OM conversion complete: ${OUTPUT_PREFIX}.om"
