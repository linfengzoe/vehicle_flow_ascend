# Ascend OM 部署说明

本文档用于将 PC 端 YOLOv5 车辆检测模型导出为 ONNX，再转换为昇腾 OM，并在开发板上运行车流统计演示。

## 0. Atlas 200I DK A2 核对信息

按课程资料页 [熟悉昇腾开发板-260428](https://tnt.gdvzz.com/ailab/aidk260428.html)，本项目对应的开发板为 Atlas 200I DK A2。部署前先核对以下信息：

- 开发板默认直连 IP 为 `192.168.137.100`。PC 与开发板通过网线直连后，先用 `ping 192.168.137.100` 确认连通；
- 默认 SSH 账号为 `root / Mind@123` 和 `HwHiAiUser / Mind@123`。普通项目文件建议放在 `/home/HwHiAiUser/vehicle_flow_ascend/`；
- 资料页登录欢迎信息显示系统基于 Ubuntu 22.04 LTS，架构为 `aarch64`；
- 当前项目的 `configs/ascend_om.yaml` 和 `scripts/convert_onnx_to_om.sh` 默认使用 `Ascend310B4`。如果 ATC 转换报 `soc_version` 不支持，以当前 CANN/ATC 版本实际支持值为准；
- 如需使用板端预置 Jupyter Lab，可登录开发板后进入 `/home/HwHiAiUser/samples/notebooks`，执行 `./start_notebook.sh 192.168.137.100`，再在 PC 浏览器打开终端输出的 `http://192.168.137.100:8888/lab?...` 地址；
- 资料页说明预置摄像头样例需要 `root` 才能访问摄像头。本项目若在开发板上用 USB 摄像头和 OpenCV `source: 0`，需要用 `root` 运行或额外配置设备权限；Web Dashboard 的“开启摄像头”调用的是 PC 浏览器摄像头，不等同于开发板 USB 摄像头输入。

## 1. 准备目录

建议开发板端目录：

```bash
/home/HwHiAiUser/vehicle_flow_ascend/
```

需要复制到开发板的内容：

- `pyproject.toml`
- `README.md`
- `src/`
- `configs/ascend_om.yaml`
- `scripts/export_yolov5_to_onnx.sh`
- `scripts/convert_onnx_to_om.sh`
- `data/demo.mp4`
- `models/yolov5n.om`

## 2. 激活 CANN 环境

在开发板或昇腾工具链所在 Linux 环境执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

检查 ATC 是否可用：

```bash
atc --version
```

## 3. 导出 YOLOv5 ONNX

如果在 Linux 环境中已有 YOLOv5 仓库和 `models/yolov5n.pt`：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend
YOLOV5_DIR=third_party/yolov5 \
WEIGHTS=models/yolov5n.pt \
IMG_SIZE=640 \
bash scripts/export_yolov5_to_onnx.sh
```

导出时固定输入：

```text
batch = 1
channels = 3
height = 640
width = 640
```

## 4. 转换 ONNX 为 OM

根据实际开发板芯片设置 `SOC_VERSION`。常见值示例：

- `Ascend310`
- `Ascend310B4`
- `Ascend310P3`

示例命令：

```bash
ONNX_MODEL=models/yolov5n.onnx \
OUTPUT_PREFIX=models/yolov5n \
SOC_VERSION=Ascend310B4 \
bash scripts/convert_onnx_to_om.sh
```

生成文件：

```text
models/yolov5n.om
```

## 5. 配置文件

编辑 `configs/ascend_om.yaml`：

```yaml
source: data/demo.mp4
backend: ascend_om
model_path: models/yolov5n.om
soc_version: Ascend310B4
image_size: 640
confidence_threshold: 0.35
iou_threshold: 0.45
line:
  - [120, 360]
  - [1160, 360]
display: false
output_video: runs/ascend_om_output.mp4
max_frames: null
```

注意：

- `soc_version` 要与 ATC 转换命令一致；
- `model_path` 必须指向开发板上的 `.om` 文件；
- `source` 建议先用本地 MP4，课堂演示更稳定；
- 摄像头可在稳定后改成 `source: 0`。

## 6. 运行板端演示

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -e .
python -m vehicle_flow_ascend --config configs/ascend_om.yaml
```

输出内容：

- 终端打印最终计数 JSON；
- 按配置写出 `runs/ascend_om_output.mp4`；
- 视频中包含检测框、车辆类别、计数线、分类计数和 FPS。

## 7. 常见问题

### ACL 无法导入

现象：

```text
Ascend OM backend requires Huawei CANN/ACL on the development board.
```

处理：

1. 确认在昇腾开发板或安装 CANN 的 Linux 环境中运行；
2. 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；
3. 确认 Python 能导入 `acl`。

### ATC 转换失败

检查：

- `--soc_version` 是否与硬件一致；
- ONNX 是否固定输入形状 `1,3,640,640`；
- `--input_shape` 中的输入名是否与 ONNX 输入名一致；
- YOLOv5 导出时是否使用常见 opset，例如 12。

### 检测框位置不准

重点检查：

- BGR 是否转换为 RGB；
- 输入是否归一化到 `0..1`；
- 是否使用 letterbox；
- 坐标是否按 ratio 和 padding 还原；
- NMS 是否在 CPU 侧按相同阈值执行。

### 计数不稳定

建议：

- 调整 `line` 位置，让车辆完整穿过计数线；
- 优先使用固定机位本地 MP4；
- 避免车辆在计数线上长时间停留；
- 降低置信度阈值或更换 YOLOv5s 提升召回率。
