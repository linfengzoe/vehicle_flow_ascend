# Vehicle Flow Ascend

Vehicle Flow Ascend 是一个面向《人工智能导论》大作业的车辆大类识别与简单车流统计项目。系统支持视频/摄像头输入、YOLOv5 车辆检测、4 类车辆大类映射、质心跟踪、单线穿越计数、结果叠加显示、输出视频保存，并提供 PC PyTorch 后端和昇腾 OM 后端。

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

如果还没有安装 PyTorch，请根据本机 CUDA/CPU 环境安装 `torch` 和 `torchvision`。

## PC 演示命令

准备一段短交通视频到 `data/demo.mp4` 后运行：

```bash
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --source data/demo.mp4 --display true
```

运行时会显示检测框、车辆类别、计数线、总车流量、分类计数和 FPS，并按配置保存标注后的视频到 `runs/pc_demo_output.mp4`。

## 交互式可视化前端

项目内置一个无需 Node/Vite 的 Web Dashboard，适合答辩时展示系统状态、车辆类别、推理后端、计数线配置、部署链路和输出视频。前端现在支持交互式参数输入：可以修改视频源、后端、模型路径、输出视频、输入尺寸、昇腾 SOC、置信度阈值、IoU 阈值、最大帧数、OpenCV 显示开关和计数线坐标，并实时生成可复制的 PowerShell 运行命令。

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

前端会读取 `/api/dashboard` 获取配置状态；如果 `output_video` 指向的 MP4 已存在，可以点击“刷新预览”在页面中直接展示标注后视频。网页不会直接启动 YOLO 推理进程，需要复制生成的命令到终端执行，这样更适合答辩现场稳定演示。
