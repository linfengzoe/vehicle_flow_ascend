# Ascend OM 部署说明

本文档用于将 PC 端 YOLOv5 车辆检测模型导出为 ONNX，再转换为昇腾 OM，并在开发板上运行车流统计演示。

## 0. Atlas 200I DK A2 核对信息

按课程资料页
[熟悉昇腾开发板-260428](https://tnt.gdvzz.com/ailab/aidk260428.html)，本项目对应的开发板为 Atlas
200I DK A2。部署前先核对以下信息：

- 开发板默认直连 IP 为 `192.168.137.100`。PC 与开发板通过网线直连后，先用
  `ping 192.168.137.100` 确认连通；
- 默认 SSH 账号为 `root / Mind@123` 和
  `HwHiAiUser / Mind@123`。当前演示目录建议使用
  `/home/HwHiAiUser/vehicle_flow_ascend_current/`；
- 资料页登录欢迎信息显示系统基于 Ubuntu 22.04 LTS，架构为 `aarch64`；
- 当前项目的 `configs/ascend_om.yaml` 和 `scripts/convert_onnx_to_om.sh`
  默认使用 `Ascend310B4`。如果 ATC 转换报 `soc_version`
  不支持，以当前 CANN/ATC 版本实际支持值为准；
- 如需使用板端预置 Jupyter Lab，可登录开发板后进入
  `/home/HwHiAiUser/samples/notebooks`，执行
  `./start_notebook.sh 192.168.137.100`，再在 PC 浏览器打开终端输出的
  `http://192.168.137.100:8888/lab?...` 地址；
- 资料页说明预置摄像头样例需要 `root`
  才能访问摄像头。本项目 Web Dashboard 同时支持 PC 浏览器摄像头和开发板 USB 摄像头：
  “开启浏览器摄像头”调用访问页面电脑的摄像头；“使用开发板摄像头”由板端后端直接
  `cv2.VideoCapture` 采集 USB 摄像头，建议用 `root` 启动 Web 后端或额外配置设备权限。

## 1. 准备目录

建议开发板端目录：

```bash
/home/HwHiAiUser/vehicle_flow_ascend_current/
```

也可以使用带版本号的发布目录，并用软链接指向当前版本：

```bash
/home/HwHiAiUser/vehicle_flow_ascend_YYYYMMDD_HHMMSS/
/home/HwHiAiUser/vehicle_flow_ascend_current -> 上面的发布目录
```

需要复制到开发板的内容：

- `pyproject.toml`
- `README.md`
- `src/`
- `configs/ascend_om.yaml`
- `scripts/export_yolov5_to_onnx.sh`
- `scripts/convert_onnx_to_om.sh`
- `scripts/start_devboard_web.sh`
- `data/demo.mp4`
- `models/yolov5n.onnx`
- `models/yolov5n.om`，若已在板端转换完成

如果从 Windows 上传，建议先打一个干净包，避免把
`runs/`、缓存和大压缩包一起传到板子：

```powershell
tar -czf $env:TEMP\vehicle_flow_ascend_deploy.tar.gz -C vehicle_flow_ascend .
scp $env:TEMP\vehicle_flow_ascend_deploy.tar.gz HwHiAiUser@192.168.137.100:/home/HwHiAiUser/
```

板端解包：

```bash
mkdir -p /home/HwHiAiUser/vehicle_flow_ascend_current
tar -xzf /home/HwHiAiUser/vehicle_flow_ascend_deploy.tar.gz \
  -C /home/HwHiAiUser/vehicle_flow_ascend_current
```

## 1.1 网络和 apt 源检查

PC 直连开发板管理口时，PC 以太网手动设置为：

```text
IP 地址: 192.168.137.111
子网掩码: 255.255.255.0
默认网关: 留空
DNS: 留空
```

开发板如果另接一根外网网线，常见拓扑是：

```text
eth1: 192.168.137.100/24，用于 PC SSH/SCP/Web 访问
eth0: 学校或路由器 DHCP 地址，用于 apt 下载
```

若 `apt-get update` 报 `No route to host`，先看默认路由：

```bash
ip route
ip route get repo.huaweicloud.com
```

如果公网流量走到了
`eth1 src 192.168.137.100`，说明管理口被错误当成默认出口。临时修复可以加 split-default 路由，让公网走外网口，同时保留
`192.168.137.0/24` 管理口直连：

```bash
sudo ip route replace 0.0.0.0/1 via 172.18.145.1 dev eth0
sudo ip route replace 128.0.0.0/1 via 172.18.145.1 dev eth0
```

其中 `172.18.145.1` 和 `eth0` 需要按 `ip route` 中实际外网网关和接口替换。

## 1.2 安装板端 Python 依赖

优先使用 Ubuntu 系统包，避免在 aarch64 开发板上从源码编译 OpenCV：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-pip python3-setuptools python3-wheel \
  python3-numpy python3-opencv \
  python3-decorator python3-sympy python3-scipy python3-attr python3-psutil
```

说明：

- `python3-numpy`、`python3-opencv` 是项目运行依赖；
- `python3-decorator`、`python3-sympy`、`python3-scipy`、`python3-attr`、`python3-psutil`
  是 CANN/ATC 在本开发板环境中实际需要的 Python 模块；
- 如果使用 `python -m pip install -e .`，仍需保证 `opencv-python`
  不在板端源码编译失败。课堂演示可直接用 `PYTHONPATH=src:$PYTHONPATH`
  运行项目源码。

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
- 命令行摄像头可在稳定后改成 `source: 0`；
- Web 页面推荐使用“使用开发板摄像头”按钮，不需要修改 YAML 的 `source`。

## 6. 运行板端演示

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=src:$PYTHONPATH
python3 -m vehicle_flow_ascend --config configs/ascend_om.yaml
```

输出内容：

- 终端打印最终计数 JSON；
- 按配置写出 `runs/ascend_om_output.mp4`；
- 视频中包含检测框、车辆类别、计数线、分类计数和 FPS。

建议先跑 5 帧冒烟测试，确认 OM 模型、ACL 运行时、视频读写都可用：

```bash
PYTHONPATH=src:$PYTHONPATH python3 -m vehicle_flow_ascend \
  --config configs/ascend_om.yaml \
  --max-frames 5 \
  --output-video runs/smoke_output.mp4
```

## 6.1 启动 Web Dashboard

普通视频上传和结果播放可以使用 HTTP，但远程浏览器摄像头需要 HTTPS：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=src:$PYTHONPATH
python3 -m vehicle_flow_ascend \
  --config configs/ascend_om.yaml \
  --web \
  --web-host 0.0.0.0 \
  --web-port 8765
```

PC 浏览器打开：

```text
http://192.168.137.100:8765/
```

如果要使用 PC 浏览器摄像头，必须使用 HTTPS。普通远程 HTTP 页面不是安全上下文，Chrome/Edge 会隐藏
`navigator.mediaDevices.getUserMedia`，前端会显示需要 HTTPS。

如果要使用开发板 USB 摄像头，也建议直接使用 HTTPS Web 后端，因为课堂演示通常从 PC 浏览器访问开发板页面，并且页面中可能同时需要浏览器摄像头权限。

生成自签名证书：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current
mkdir -p certs
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout certs/web.key \
  -out certs/web.crt \
  -days 365 \
  -subj "/CN=192.168.137.100" \
  -addext "subjectAltName=IP:192.168.137.100,DNS:localhost"
chmod 600 certs/web.key
```

推荐使用一键脚本启动 HTTPS 服务：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current
./scripts/start_devboard_web.sh
```

PC 浏览器打开：

```text
https://192.168.137.100:8766/
```

首次访问自签名证书会出现风险提示，选择继续访问后再点击“开启摄像头”，浏览器才会弹出摄像头权限。

脚本默认参数：

```text
CONFIG_PATH=configs/ascend_om.yaml
WEB_HOST=0.0.0.0
WEB_PORT=8766
WEB_CERTFILE=certs/web.crt
WEB_KEYFILE=certs/web.key
PID_FILE=runs/web-https.pid
LOG_FILE=runs/web-https.log
CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
```

脚本会自动停止同端口旧服务、加载 CANN 环境、设置 `PYTHONPATH=src:${PYTHONPATH:-}`，并用
`python3 -B -m vehicle_flow_ascend --web` 后台启动。若你已经在 `scripts/`
目录内，必须执行 `./start_devboard_web.sh`；直接输入 `start_devboard_web.sh`
会因为当前目录不在 Linux `PATH` 中而提示 `command not found`。

可用环境变量覆盖默认值：

```bash
WEB_PORT=8899 LOG_FILE=runs/web-8899.log ./scripts/start_devboard_web.sh
```

检查和停止：

```bash
ss -ltnp | grep 8766
tail -n 80 runs/web-https.log
kill $(cat runs/web-https.pid)
```

Web 页面有三种输入模式：

- 上传视频：浏览器上传视频，后端批处理推理，推理中通过 MJPEG 显示标注过程帧，完成后播放 H.264 MP4；
- 开启浏览器摄像头：浏览器 `getUserMedia()` 采集 PC 摄像头，前端抽帧上传，后端推理后返回 MJPEG 标注流；
- 使用开发板摄像头：后端在开发板上自动探测可读摄像头编号，直接采集 USB 摄像头并执行 Ascend OM 推理。

开发板摄像头模式走 `/api/realtime/start-devboard-camera`，后端会自动尝试设备编号 `0..5`。如果需要手动指定，可在前端代码或调试请求中传
`camera_index`。运行用户必须有摄像头设备权限；课程开发板上通常直接用 `root` 启动最省事。

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

如果使用了 `PYTHONPATH=src` 但没有保留 CANN 原来的路径，也会导致 `acl`
导入失败。正确写法：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=src:$PYTHONPATH
```

### ATC 转换失败

检查：

- `--soc_version` 是否与硬件一致；
- ONNX 是否固定输入形状 `1,3,640,640`；
- `--input_shape` 中的输入名是否与 ONNX 输入名一致；
- YOLOv5 导出时是否使用常见 opset，例如 12。

如果 ATC 报缺少 Python 模块，例如 `decorator`、`sympy`、`scipy`、`attr` 或
`psutil`，安装对应系统包：

```bash
sudo apt-get install -y python3-decorator python3-sympy python3-scipy python3-attr python3-psutil
```

YOLOv5n 在 3 到 4
GB 内存的开发板上转换可能会使用 swap，耗时可到十几分钟甚至更久。转换时可用以下命令观察：

```bash
ps -eo pid,etime,pcpu,pmem,args | grep atc
free -h
```

### 浏览器显示不支持摄像头

现象：

```text
当前浏览器不支持摄像头
```

若使用的是
`http://192.168.137.100:8765/`，通常不是浏览器版本问题，而是远程 HTTP 页面不是安全上下文。处理：

1. 按 `6.1 启动 Web Dashboard` 生成证书并启动 HTTPS；
2. 使用 `https://192.168.137.100:8766/` 访问；
3. 首次访问自签名证书时选择继续访问；
4. 浏览器弹出摄像头权限后选择允许。

`localhost` 和 `127.0.0.1`
是浏览器特殊信任地址，所以 PC 本机开发时 HTTP 摄像头可用；开发板 IP 不属于这个范围，必须用 HTTPS。

### 直接运行脚本提示 command not found

现象：

```text
bash: start_devboard_web.sh: command not found
```

处理：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current/scripts
./start_devboard_web.sh
```

或使用绝对路径：

```bash
/home/HwHiAiUser/vehicle_flow_ascend_current/scripts/start_devboard_web.sh
```

Linux 默认不会从当前目录查找可执行文件，必须显式加 `./`。

### Python 3.9 类型注解报错

现象：

```text
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

处理：

- 使用最新代码，项目已为 Python 3.9 运行环境启用延迟注解并移除运行时不兼容的 `isinstance(x, list | tuple)` 写法；
- 启动脚本使用 `python3 -B` 并设置 `PYTHONDONTWRITEBYTECODE=1`，避免旧 `.pyc` 缓存干扰；
- 若手动运行，建议先执行 `find src -name '__pycache__' -type d -prune -exec rm -rf {} +` 清理旧缓存。

### 第二次推理报 acl.init failed 100002

现象：

```text
acl.init failed with ACL error code 100002
```

原因和处理：

- Web 服务是长进程，连续启动/停止视频推理或开发板摄像头推理时，不应在每个 detector 释放时反复 `acl.finalize()`；
- 最新代码在同一 Web 进程内只执行一次 `acl.init()`，每次任务结束只释放模型、数据集、buffer、context 和 device；
- 若仍遇到旧进程加载旧代码，先停止后端 `kill $(cat runs/web-https.pid)`，再运行 `./scripts/start_devboard_web.sh`。

### 开发板摄像头能启动但检测不正常

重点检查：

- `configs/ascend_om.yaml` 中 `image_size` 应与 OM 模型输入一致，默认 `640`；
- 最新代码中 `ascend_om` 实时路径不会再把输入尺寸降到 `320`，避免 640 OM 模型输入不匹配；
- 确认摄像头画面内车辆足够大、光照稳定、视角和训练类别匹配；
- 若画面有明显畸变或方向不对，先用 OpenCV 单独读一帧保存确认摄像头本身输出。

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
