# 系统架构说明

## 1. 项目目标

本项目实现一个基于 YOLOv5 和边缘推理的车辆大类识别与简单车流统计系统。系统输入一路道路视频或摄像头画面，检测车辆目标，识别车辆大类，并通过单条计数线统计总车流量和分类车流量。

目标演示闭环：

```text
视频/摄像头输入 -> YOLO 车辆检测 -> 质心跟踪 -> 单线穿越计数 -> 画面叠加 -> 输出视频/实时显示
```

## 2. 总体流程

```text
VideoSource
  -> Detector
  -> CentroidTracker
  -> LineCounter
  -> OverlayRenderer
  -> VideoWriter / Display
```

对应代码模块：

| 模块 | 文件 | 责任 |
|---|---|---|
| 配置加载 | `src/vehicle_flow_ascend/config.py` | 读取 YAML，解析输入源、模型、阈值、计数线、输出路径等配置 |
| CLI 入口 | `src/vehicle_flow_ascend/cli.py` | 提供 `--dry-run`、配置覆盖和正式运行入口 |
| 应用主循环 | `src/vehicle_flow_ascend/app.py` | 串联视频读取、检测、跟踪、计数、叠加和输出 |
| 视频输入 | `src/vehicle_flow_ascend/video/source.py` | 使用 OpenCV 打开本地视频或摄像头 |
| 视频输出 | `src/vehicle_flow_ascend/video/writer.py` | 保存标注后的演示视频 |
| 检测器接口 | `src/vehicle_flow_ascend/detectors/base.py` | 定义 `Detector.detect(frame_bgr)`，统一 PC 和昇腾后端 |
| PC YOLOv5 后端 | `src/vehicle_flow_ascend/detectors/torch_yolov5.py` | 调用 PyTorch/YOLOv5 权重进行 PC 端检测 |
| 昇腾 OM 后端 | `src/vehicle_flow_ascend/detectors/ascend_om.py` | 懒加载 CANN/ACL，运行 `.om` 模型 |
| 导出模型后处理 | `src/vehicle_flow_ascend/detectors/yolov5_postprocess.py` | letterbox、坐标还原、置信度过滤、NMS、类别映射 |
| 类别映射 | `src/vehicle_flow_ascend/detectors/class_mapping.py` | COCO 类别映射到项目 4 类车辆 |
| 质心跟踪 | `src/vehicle_flow_ascend/tracking/centroid.py` | 根据检测框中心点维护目标 ID |
| 穿线计数 | `src/vehicle_flow_ascend/counting/line_counter.py` | 判断目标是否穿过计数线并防止重复计数 |
| 几何计算 | `src/vehicle_flow_ascend/counting/geometry.py` | 点在线两侧判断、有限线段穿越判断 |
| 可视化叠加 | `src/vehicle_flow_ascend/visualization/overlay.py` | 绘制检测框、类别、ID、计数线、计数和 FPS |
| FPS 统计 | `src/vehicle_flow_ascend/utils/fps.py` | 滑动窗口 FPS 估计 |
| Web Dashboard | `src/vehicle_flow_ascend/web/dashboard.py` | 提供静态前端、上传接口、任务状态、MJPEG 流和可选 HTTPS 服务 |
| Web 批处理任务 | `src/vehicle_flow_ascend/web/inference.py` | 管理上传视频推理任务，向前端推送实时标注帧 |
| Web 实时任务 | `src/vehicle_flow_ascend/web/realtime.py` | 接收浏览器 JPEG 帧或采集开发板 USB 摄像头，调用后端推理并返回 MJPEG 标注流 |
| 开发板一键启动脚本 | `scripts/start_devboard_web.sh` | 加载 CANN 环境，设置 `PYTHONPATH`，停止旧服务并后台启动 HTTPS Web 后端 |

## 3. 车辆类别定义

系统采用 4 类车辆大类：

| 项目类别 | 数据来源类别 | 说明 |
|---|---|---|
| `car` | COCO `car` | 轿车、SUV、面包车等小型车统一作为小型车 |
| `bus` | COCO `bus` | 公交车、大巴车、校车等大型载客车辆 |
| `truck` | COCO `truck` | 卡车、厢式货车、工程车等货运车辆 |
| `two_wheeler` | COCO `motorcycle` + `bicycle` | 摩托车、电动车、自行车等两轮交通工具 |

非车辆类别会被过滤，不进入跟踪和计数流程。

## 4. 检测器设计

所有检测后端都实现统一接口：

```python
class Detector:
    def detect(self, frame_bgr) -> list[Detection]:
        ...
```

这样 PC 端和昇腾端可以复用同一套后续逻辑：

```text
torch_yolov5 -> Detection -> CentroidTracker -> LineCounter
ascend_om    -> Detection -> CentroidTracker -> LineCounter
```

### PC YOLOv5 后端

PC 后端通过 PyTorch 加载 YOLOv5 权重，适合开发调试和答辩前验证。默认配置：

```yaml
backend: torch_yolov5
model_path: models/yolov5n.pt
yolov5_repo_path: third_party/yolov5
```

### 昇腾 OM 后端

昇腾后端通过 CANN/ACL 加载 `.om` 模型。该模块只在选择 `ascend_om` 后端时导入 ACL，因此普通 PC 测试不会依赖开发板环境。

默认配置：

```yaml
backend: ascend_om
model_path: models/yolov5n.om
soc_version: Ascend310B4
```

Web 后端是长进程。为支持连续启动/停止上传视频推理、浏览器摄像头推理和开发板 USB 摄像头推理，`ascend_om`
在进程内只初始化一次 ACL runtime；每个任务结束时释放模型、数据集、device buffer、context 和 device，但不反复
`acl.finalize()`，避免第二次任务出现 `acl.init failed with ACL error code 100002`。

开发板环境当前按 Python 3.9 兼容处理：源码中使用 `|` 类型注解的文件启用延迟注解，运行时 `isinstance()`
类型组使用 `(list, tuple)` 这类 Python 3.9 可执行写法。

## 5. 计数算法

系统使用轻量级质心跟踪和单线穿越判断：

1. 对每个检测框计算中心点；
2. 使用最近距离匹配维护目标 ID；
3. 记录目标上一帧和当前帧中心点；
4. 判断中心点轨迹是否穿过有限计数线段；
5. 每个 `track_id` 只计数一次；
6. 更新总计数和分类计数。

为了减少重复计数，`LineCounter` 会维护：

- `counted_track_ids`：已经计数过的目标 ID；
- `last_nonzero_sides`：目标最近一次位于计数线哪一侧，用于处理中心点刚好落在线上的情况。

## 6. 显示与输出

演示画面包含：

- 检测框；
- 车辆类别；
- 目标 ID；
- 置信度；
- 黄色计数线；
- 总车流量；
- 各类别车流量；
- FPS。

输出视频路径由配置项控制，例如：

```yaml
output_video: runs/pc_demo_output.mp4
```

Web Dashboard 提供三条展示路径：

```text
上传视频 -> 手动拖动穿线 -> 后端批处理 -> MJPEG 过程帧 -> MP4 结果视频
浏览器摄像头 -> 本地预览拖动穿线 -> 浏览器抽帧上传 -> 后端实时推理 -> MJPEG 标注流
开发板摄像头 -> 后端自动探测 USB 摄像头 -> 板端采集 -> Ascend OM 实时推理 -> MJPEG 标注流
```

浏览器摄像头路径调用的是访问页面的那台电脑上的摄像头。若 PC 浏览器访问开发板 IP，摄像头 API 必须使用 HTTPS 页面；普通远程 HTTP 页面会被浏览器判定为非安全上下文，`navigator.mediaDevices` 不可用。后端通过 `--web-certfile` 和 `--web-keyfile` 支持自签名证书 HTTPS 服务。

开发板摄像头路径不依赖浏览器摄像头 API，而是通过 `/api/realtime/start-devboard-camera`
让板端后端直接打开 USB 摄像头。该路径会自动尝试设备编号 `0..5`，需要运行用户具备摄像头设备权限；课程开发板演示通常使用
`root` 启动 Web 后端。为了保持 Ascend OM 模型输入正确，`ascend_om` 实时路径保持配置中的 `image_size`
默认 `640`；非昇腾实时路径仍可降到 `320` 以提升交互帧率。

输出 MP4 使用 H.264 编码，优先使用 `imageio-ffmpeg`，缺失时回退系统 `ffmpeg`。后端 `/media/output-video`
支持 HTTP Range 请求，便于浏览器对结果 MP4 进行 seek 和播放。

## 7. 测试策略

项目测试分为三类：

1. **纯逻辑单元测试**：类别映射、几何计算、计数器、质心跟踪；
2. **流水线测试**：用合成帧和假检测器测试端到端计数；
3. **部署辅助测试**：确保 Ascend 后端在 PC 上懒加载并给出明确错误；
4. **兼容性测试**：检查 Python 3.9 类型注解兼容、开发板启动脚本约定、H.264 输出和 HTTP Range 播放。

运行方式：

```powershell
python -m pytest .\tests -q
```

当前默认测试不需要真实 YOLO 权重，也不需要昇腾开发板。

## 8. 局限性

- 质心跟踪适合简单场景，遮挡严重时可能发生 ID 切换；
- 单线计数对计数线位置敏感，需要根据视频视角调整；
- 默认 COCO 预训练权重没有针对本地道路场景微调；
- 昇腾 OM 推理需要开发板实际验证 FPS、模型输出形状和摄像头视角；
- 开发板摄像头检测效果受光照、车辆距离、摄像头安装角度和 COCO 预训练模型泛化能力影响。

## 9. 可扩展方向

- 使用 YOLOv5s 或本地数据微调提升精度；
- 加入 ROI 区域过滤，减少路边静止车辆误计；
- 加入双向计数；
- 接入 ByteTrack/SORT 提升遮挡场景下的 ID 稳定性；
- 对模型进行 INT8 量化以提升昇腾端 FPS。
