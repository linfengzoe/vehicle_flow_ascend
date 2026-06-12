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

- [ ] 如果演示昇腾端，开发板型号为 Atlas 200I DK A2，且 `configs/ascend_om.yaml` 中的 `soc_version` 为 `Ascend310B4` 或已按实际 CANN/ATC 支持值调整。
- [ ] 如果演示昇腾端，PC 能通过网线直连访问 `192.168.137.100`。
- [ ] 如果要从 PC 浏览器使用摄像头访问开发板前端，已准备 HTTPS 地址；普通 `http://192.168.137.100` 只能稳定演示视频上传，不能保证浏览器摄像头权限。
- [ ] 如果要展示开发板 USB 摄像头，后端使用 `root` 启动，或已为 `HwHiAiUser` 配好摄像头设备权限。

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

- [ ] 已按课程资料页确认默认登录信息：`root / Mind@123` 或 `HwHiAiUser / Mind@123`。
- [ ] 已使用 `ssh HwHiAiUser@192.168.137.100` 或 `ssh root@192.168.137.100` 登录过开发板。
- [ ] 项目已放在 `/home/HwHiAiUser/vehicle_flow_ascend_current/`，并包含 `pyproject.toml`、`README.md`、`src/`、`configs/`、`scripts/`、`data/demo.mp4` 和 `models/yolov5n.om`。
- [ ] 已激活 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- [ ] 已导出 ONNX：`models/yolov5n.onnx`。
- [ ] 已转换 OM：`models/yolov5n.om`。
- [ ] 已复制 `data/demo.mp4` 到开发板。
- [ ] 已确认 `configs/ascend_om.yaml` 的 `model_path`、`source`、`soc_version` 正确，其中 Atlas 200I DK A2 默认使用 `Ascend310B4`。
- [ ] 已安装板端依赖：`python3-numpy`、`python3-opencv`，以及 ATC 可能需要的 `python3-decorator`、`python3-sympy`、`python3-scipy`、`python3-attr`、`python3-psutil`。
- [ ] 命令行 5 帧冒烟测试可运行：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH=src:$PYTHONPATH python3 -m vehicle_flow_ascend \
  --config configs/ascend_om.yaml \
  --max-frames 5 \
  --output-video runs/smoke_output.mp4
```

- [ ] 已生成 HTTPS 自签名证书 `certs/web.crt` 和 `certs/web.key`。
- [ ] 一键启动脚本可运行：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current
./scripts/start_devboard_web.sh
```

- [ ] 如果已经进入 `scripts/` 目录，使用 `./start_devboard_web.sh`，不要直接输入 `start_devboard_web.sh`。
- [ ] Web 前端 HTTPS 地址可用：

```text
https://192.168.137.100:8766/
```

- [ ] 首次打开 HTTPS 地址时，已在浏览器自签名证书提示页选择继续访问，并允许摄像头权限。
- [ ] 上传视频模式可以推理并播放最终 H.264 MP4。
- [ ] 浏览器摄像头模式可以授权、上传帧并显示 MJPEG 标注流。
- [ ] 开发板摄像头模式可以连续启动/停止两次，不出现 `acl.init failed with ACL error code 100002`。

如果要展示板端 USB 摄像头，而不是 PC 浏览器摄像头：

- [ ] 已确认 USB 摄像头接在开发板上；
- [ ] 已用 `root` 运行，或已为 `HwHiAiUser` 配好摄像头设备权限；
- [ ] 已先用 `data/demo.mp4` 验证 OM 推理稳定；
- [ ] 摄像头画面中车辆距离、光照和角度适合 COCO 预训练 YOLOv5n 检测；
- [ ] 如果检测不正常，确认 `configs/ascend_om.yaml` 的 `image_size` 与 OM 模型输入一致，默认 `640`。

## 6. 答辩材料截图/录屏

建议准备以下材料放入报告或 PPT：

- [ ] 系统架构图；
- [ ] 四类车辆类别映射表；
- [ ] PC 端实时检测截图；
- [ ] 车辆穿线计数前后对比截图；
- [ ] 输出视频片段；
- [ ] 昇腾开发板部署照片或终端运行截图；
- [ ] 一键启动脚本运行截图和 HTTPS 页面截图；
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
