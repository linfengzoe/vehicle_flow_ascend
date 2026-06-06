#!/usr/bin/env bash
set -euo pipefail

YOLOV5_DIR="${YOLOV5_DIR:-third_party/yolov5}"
WEIGHTS="${WEIGHTS:-models/yolov5n.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-models}"
IMG_SIZE="${IMG_SIZE:-640}"
OPSET="${OPSET:-12}"

if [[ ! -d "$YOLOV5_DIR" ]]; then
  echo "YOLOv5 directory not found: $YOLOV5_DIR" >&2
  echo "Run scripts/prepare_pc_model.ps1 on Windows or clone YOLOv5 manually on Linux." >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "YOLOv5 weights not found: $WEIGHTS" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python "$YOLOV5_DIR/export.py" \
  --weights "$WEIGHTS" \
  --include onnx \
  --imgsz "$IMG_SIZE" "$IMG_SIZE" \
  --batch-size 1 \
  --opset "$OPSET" \
  --simplify

echo "ONNX export complete. Check the generated .onnx file next to $WEIGHTS."
