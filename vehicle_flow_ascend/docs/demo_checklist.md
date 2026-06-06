# 演示检查清单

本清单用于答辩前确认“车辆大类识别与简单车流统计系统”可以稳定演示。建议至少提前一天完整跑一遍 PC 演示；如果要展示昇腾开发板，也提前完成板端录屏作为备用。

## 1. 基础环境

- [ ] 当前目录为 `vehicle_flow_ascend/`。
- [ ] 已安装项目：

```powershell
python -m pip install -e ".[dev]"
```

- [ ] 全量测试通过：

```powershell
python -m pytest .\tests -q
```

- [ ] `configs/pc_demo.yaml` 能 dry-run：

```powershell
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --dry-run
```

- [ ] 如果演示昇腾端，`configs/ascend_om.yaml` 中的 `soc_version` 与实际开发板一致。

## 2. PC 端模型和视频

- [ ] 已准备 YOLOv5 仓库：`third_party/yolov5/`。
- [ ] 已准备默认模型权重：`models/yolov5n.pt`。
- [ ] 已准备本地演示视频：`data/demo.mp4`。
- [ ] 视频时长建议 30 到 60 秒，车辆要能明显穿过计数线。
- [ ] 如需准备 PC 端模型，运行：

```powershell
.\scripts\prepare_pc_model.ps1
```

## 3. PC 演示命令

推荐先用本地 MP4，避免现场摄像头或真实车流不可控：

```powershell
python -m vehicle_flow_ascend --config configs/pc_demo.yaml --source data/demo.mp4 --display true
```

演示画面应包含：

- [ ] 车辆检测框；
- [ ] 车辆类别；
- [ ] 目标 ID；
- [ ] 黄色计数线；
- [ ] 总计数；
- [ ] `car`、`bus`、`truck`、`two_wheeler` 分类计数；
- [ ] FPS；
- [ ] 输出视频 `runs/pc_demo_output.mp4`。

## 4. 计数线检查

- [ ] 计数线位于车辆必经位置。
- [ ] 车辆中心点能从计数线一侧移动到另一侧。
- [ ] 车辆停在线附近时不会重复计数。
- [ ] 如果计数过少，适当调整 `configs/pc_demo.yaml` 中的 `line`。
- [ ] 如果漏检较多，适当降低 `confidence_threshold` 或改用 YOLOv5s。

## 5. 昇腾端演示准备

如需演示昇腾开发板：

- [ ] 已激活 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- [ ] 已导出 ONNX：`models/yolov5n.onnx`。
- [ ] 已转换 OM：`models/yolov5n.om`。
- [ ] 已复制 `data/demo.mp4` 到开发板。
- [ ] 已确认 `configs/ascend_om.yaml` 的 `model_path`、`source`、`soc_version` 正确。
- [ ] 可运行：

```bash
python -m vehicle_flow_ascend --config configs/ascend_om.yaml
```

## 6. 答辩材料截图/录屏

建议准备以下材料放入报告或 PPT：

- [ ] 系统架构图；
- [ ] 四类车辆类别映射表；
- [ ] PC 端实时检测截图；
- [ ] 车辆穿线计数前后对比截图；
- [ ] 输出视频片段；
- [ ] 昇腾开发板部署照片或终端运行截图；
- [ ] FPS 和计数结果表；
- [ ] 典型错误案例，如遮挡漏检、远处小目标漏检、货车/公交混淆。

## 7. 答辩备用方案

- [ ] 如果摄像头无法使用，改用 `data/demo.mp4`。
- [ ] 如果昇腾板现场环境不稳定，播放提前录制的板端运行视频，并展示部署脚本和配置。
- [ ] 如果 YOLOv5s FPS 不足，切换到 YOLOv5n。
- [ ] 如果现场视频计数不稳定，使用已经标注好计数线的备用视频。

## 8. 现场讲解顺序

1. 说明选题目标：车辆大类识别 + 简单车流统计 + 边缘部署。
2. 展示系统流程：视频输入、YOLO 检测、质心跟踪、穿线计数、结果叠加。
3. 展示类别定义：`car`、`bus`、`truck`、`two_wheeler`。
4. 运行 PC 或昇腾演示。
5. 展示输出计数和 FPS。
6. 说明模型转换：`.pt -> .onnx -> .om`。
7. 说明问题和改进方向。
