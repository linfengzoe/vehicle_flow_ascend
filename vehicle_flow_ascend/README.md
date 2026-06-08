# Vehicle Flow Ascend

Vehicle Flow Ascend 是一个面向《人工智能导论》大作业的车辆大类识别与简单车流统计项目。系统支持视频/摄像头输入、YOLOv5 车辆检测、4 类车辆大类映射、质心跟踪、单线穿越计数、结果叠加显示、输出视频保存，并提供 PC PyTorch 后端和昇腾 OM 后端。

当前 Web 版支持：

- 本地视频上传、视频预览、前端手动拖拽穿线、后台推理和 H.264 结果视频播放；
- 视频批处理推理中的 MJPEG 标注帧实时展示；
- 浏览器摄像头设备选择、本地预览手动穿线、准实时帧上传和 MJPEG 标注流展示；
- 总穿线计数、按小型车/公交客车/货车/两轮车拆分的穿线计数；
- 任务启动、停止、异常和摄像头权限状态反馈。

## 文档

- [系统架构说明](docs/architecture.md)：模块划分、处理流程、检测器接口、计数算法和测试策略。
- [昇腾部署说明](docs/ascend_deploy.md)：ONNX 导出、ATC 转 OM、CANN 环境和板端运行步骤。
- [演示检查清单](docs/demo_checklist.md)：答辩前环境、模型、视频、计数线、昇腾端和备用方案检查。
- [大作业报告大纲](docs/report_outline.md)：报告章节、实验表格、部署流程和改进方向参考。

## 安装

从 `vehicle_flow_ascend/` 目录执行：

```bash
python -m pip install -e ".[dev]"
```

## Dry-run 配置检查

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --dry-run
```

覆盖部分配置：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --dry-run --source data/demo.mp4 --backend ascend_om --display false --max-frames 100
```

## 准备 PC YOLOv5 演示模型

Windows PowerShell：

```powershell
.\scripts\prepare_pc_model.ps1
```

该脚本会准备：

- `third_party/yolov5/`：YOLOv5 仓库；
- `models/yolov5n.pt`：默认 PC 演示权重。

如果还没有安装 PyTorch，请根据本机 CUDA/CPU 环境安装 `torch` 和 `torchvision`。当前 PC 演示环境建议使用 `numpy<2`，项目依赖已做约束，避免 PyTorch 2.2.x 与 NumPy 2.x 的 ABI 警告。

## PC 演示命令

准备一段短交通视频到 `data/demo.mp4` 后运行：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --source data/demo.mp4 --display true
```

运行时会显示检测框、车辆类别、计数线、总车流量、分类计数和 FPS，并按配置保存标注后的视频到 `runs/pc_demo_output.mp4`。

`configs/pc_demo.yaml` 的默认计数线已按 `道路监控视频/道路监控视频/1.mp4` 调整到画面下方横线位置，便于演示穿线计数。更换视频视角时，可通过 `--line-start x,y --line-end x,y` 覆盖。

## 交互式可视化前端

项目内置一个无需 Node/Vite 的 Web Dashboard。前端现在只负责输入选择、页面效果、状态反馈和推理结果展示；模型路径、输入尺寸、阈值等工程参数继续由 YAML 配置和 Python 后端统一管理，不在页面暴露。

启动前端：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web
```

浏览器打开：

```text
http://127.0.0.1:8765
```

如需改监听地址和端口：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web --web-host 0.0.0.0 --web-port 8899
```

当前支持两种输入方式：

1. 上传视频文件：浏览器把本地视频上传到后端，后端保存到 `data/web_uploads/`，页面先加载视频预览并显示可拖拽穿线。用户拖动线段或端点后点击“开始分析”，前端会把原始视频坐标系下的穿线坐标提交给后端；
2. 开启浏览器摄像头：页面会列出浏览器可见的摄像头设备，用户可自动选择或指定设备。前端先打开本地摄像头预览并显示可拖拽穿线，点击“开始分析”后再按帧采集画面并发送给后端，后端按手动穿线完成推理和计数后回传 MJPEG 标注流。

视频上传模式的执行路径：

- “上传视频并预览”只负责上传文件和打开主屏穿线编辑层，不会立即启动推理；
- 穿线编辑层按视频原始分辨率保存坐标，即使页面缩放也不会影响后端计数位置；
- “开始分析”会启动批处理任务，主屏立即切换到 `/api/inference/stream` 的 MJPEG 标注流，实时展示已经完成检测、跟踪和穿线计数叠加的帧；
- 后端任务完成后，主屏自动切换到 `/media/output-video` 播放最终 H.264 MP4 结果视频。

摄像头模式需要注意：

- 浏览器摄像头 API `navigator.mediaDevices.getUserMedia` 通常只在安全上下文中可用，例如 `localhost`、`127.0.0.1` 或 HTTPS。若使用 `--web-host 0.0.0.0` 后通过普通 HTTP 从其他设备访问，浏览器可能会禁止摄像头权限；
- 摄像头模式同样需要先在主屏拖动穿线。穿线坐标会按摄像头预览分辨率提交给 `/api/realtime/start`，后端实时 `FrameProcessor` 会使用该线完成穿越计数；
- “准实时”表示当前实现由浏览器按帧采集并上传，后端推理后再回传结果显示。实际 FPS 和延迟取决于浏览器采样、网络或本机传输以及后端推理速度，不等同于原始摄像头帧率的实时预览；
- 前端会对实时帧上传做降采样和连续失败重试，停止摄像头时会中断正在上传的帧，避免停止请求卡住。

页面会持续展示以下信息：

- 推理后视频或准实时标注画面；
- 当前状态与后台处理流程；
- 已处理帧数与 FPS；
- 总穿线计数；
- 按车辆类别拆分的分类穿线计数。

## 验证

推荐在修改后运行：

```bash
python -m pytest -q
python -m ruff check src tests
```

Web 前端可用浏览器打开 `http://127.0.0.1:8765`，上传 `道路监控视频/道路监控视频/1.mp4` 验证结果视频播放和穿线计数；摄像头模式可在页面右侧选择设备后启动。
