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
- `data/demo.mp4`
- `models/yolov5n.onnx`
- `models/yolov5n.om`，若已在板端转换完成

如果从 Windows 上传，建议先打一个干净包，避免把 `runs/`、缓存和大压缩包一起传到板子：

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

如果公网流量走到了 `eth1 src 192.168.137.100`，说明管理口被错误当成默认出口。临时修复可以加 split-default 路由，让公网走外网口，同时保留 `192.168.137.0/24` 管理口直连：

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
- `python3-decorator`、`python3-sympy`、`python3-scipy`、`python3-attr`、`python3-psutil` 是 CANN/ATC 在本开发板环境中实际需要的 Python 模块；
- 如果使用 `python -m pip install -e .`，仍需保证 `opencv-python` 不在板端源码编译失败。课堂演示可直接用 `PYTHONPATH=src:$PYTHONPATH` 运行项目源码。

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

普通视频上传和结果播放可以使用 HTTP：

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

如果要使用 PC 浏览器摄像头，必须使用 HTTPS。普通远程 HTTP 页面不是安全上下文，Chrome/Edge 会隐藏 `navigator.mediaDevices.getUserMedia`，前端会显示需要 HTTPS。

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

启动 HTTPS 服务：

```bash
cd /home/HwHiAiUser/vehicle_flow_ascend_current
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=src:$PYTHONPATH
python3 -m vehicle_flow_ascend \
  --config configs/ascend_om.yaml \
  --web \
  --web-host 0.0.0.0 \
  --web-port 8766 \
  --web-certfile certs/web.crt \
  --web-keyfile certs/web.key
```

PC 浏览器打开：

```text
https://192.168.137.100:8766/
```

首次访问自签名证书会出现风险提示，选择继续访问后再点击“开启摄像头”，浏览器才会弹出摄像头权限。

后台运行示例：

```bash
mkdir -p runs
nohup bash -lc 'cd /home/HwHiAiUser/vehicle_flow_ascend_current && source /usr/local/Ascend/ascend-toolkit/set_env.sh && export PYTHONPATH=src:$PYTHONPATH && exec python3 -m vehicle_flow_ascend --config configs/ascend_om.yaml --web --web-host 0.0.0.0 --web-port 8766 --web-certfile certs/web.crt --web-keyfile certs/web.key' > runs/web-https.log 2>&1 &
echo $! > runs/web-https.pid
```

检查和停止：

```bash
ss -ltnp | grep 8766
tail -n 80 runs/web-https.log
kill $(cat runs/web-https.pid)
```

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

如果使用了 `PYTHONPATH=src` 但没有保留 CANN 原来的路径，也会导致 `acl` 导入失败。正确写法：

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

如果 ATC 报缺少 Python 模块，例如 `decorator`、`sympy`、`scipy`、`attr` 或 `psutil`，安装对应系统包：

```bash
sudo apt-get install -y python3-decorator python3-sympy python3-scipy python3-attr python3-psutil
```

YOLOv5n 在 3 到 4 GB 内存的开发板上转换可能会使用 swap，耗时可到十几分钟甚至更久。转换时可用以下命令观察：

```bash
ps -eo pid,etime,pcpu,pmem,args | grep atc
free -h
```

### 浏览器显示不支持摄像头

现象：

```text
当前浏览器不支持摄像头
```

若使用的是 `http://192.168.137.100:8765/`，通常不是浏览器版本问题，而是远程 HTTP 页面不是安全上下文。处理：

1. 按 `6.1 启动 Web Dashboard` 生成证书并启动 HTTPS；
2. 使用 `https://192.168.137.100:8766/` 访问；
3. 首次访问自签名证书时选择继续访问；
4. 浏览器弹出摄像头权限后选择允许。

`localhost` 和 `127.0.0.1` 是浏览器特殊信任地址，所以 PC 本机开发时 HTTP 摄像头可用；开发板 IP 不属于这个范围，必须用 HTTPS。

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
