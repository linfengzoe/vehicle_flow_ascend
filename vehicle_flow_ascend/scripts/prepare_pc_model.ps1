param(
    [string]$YoloV5Dir = "third_party/yolov5",
    [string]$WeightsUrl = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
    [string]$WeightsPath = "models/yolov5n.pt"
)

$ErrorActionPreference = "Stop"

Write-Host "Preparing YOLOv5 PC demo assets..."

if (-not (Test-Path "third_party")) {
    New-Item -ItemType Directory -Path "third_party" | Out-Null
}

if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}

if (-not (Test-Path $YoloV5Dir)) {
    git clone https://github.com/ultralytics/yolov5.git $YoloV5Dir
} else {
    Write-Host "YOLOv5 repository already exists at $YoloV5Dir"
}

if (-not (Test-Path $WeightsPath)) {
    Invoke-WebRequest -Uri $WeightsUrl -OutFile $WeightsPath
} else {
    Write-Host "Weights already exist at $WeightsPath"
}

Write-Host "Done. Install runtime dependencies if needed:"
Write-Host "  python -m pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html"
Write-Host "Then run:"
Write-Host "  python -m vehicle_flow_ascend --config configs/pc_demo.yaml --source data/demo.mp4 --display true"
