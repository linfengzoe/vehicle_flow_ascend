# 前端重构与实时推理展示设计

日期：2026-06-07

## 当前实现更新说明

截至 2026-06-12，Web Dashboard 已在原“上传视频 + 浏览器摄像头”设计基础上扩展为三种输入模式：

- 上传视频：后端批处理推理，推理中输出 MJPEG 标注过程帧，完成后播放 H.264 MP4；
- 浏览器摄像头：浏览器 `getUserMedia()` 抽帧上传，后端返回 MJPEG 标注流；
- 开发板 USB 摄像头：后端通过 `/api/realtime/start-devboard-camera` 在开发板直接采集摄像头，执行 Ascend OM 推理并输出 MJPEG 标注流。

开发板 HTTPS Web 后端推荐使用 `scripts/start_devboard_web.sh` 一键启动。`ascend_om` 后端在同一 Web 进程内只初始化一次 ACL runtime，以支持连续启动/停止推理任务；开发板 Python 3.9 兼容性由源码和测试覆盖。以下内容保留为 2026-06-07 原始设计记录。

## 背景

当前项目已经具备车辆检测、质心跟踪、穿线计数、画面叠加、视频输出和基础 Web Dashboard。新的目标是重构前后端交互边界：推理继续在 Python 后端完成，前端只负责输入选择、页面动效、状态反馈和推理结果展示。前端不再展示模型路径、输入尺寸、阈值、输出路径等工程参数。

## 目标

1. 复用现有后端推理能力，不重写检测、跟踪、计数算法。
2. 前端只提供两个输入入口：上传本地视频、开启浏览器摄像头。
3. 上传视频走后台任务推理，完成后播放标注后结果视频。
4. 浏览器摄像头走准实时推理：浏览器采集画面，后端逐帧推理并输出标注后的画面流。
5. 前端 UI 重构为“城市夜行”现代视觉风格，加入克制的动态粒子效果。
6. 页面面向课程答辩演示，不再呈现工程控制台式参数表单。

## 非目标

1. 不做 WebRTC 直播系统。
2. 不在前端修改模型、阈值、输入尺寸、计数线等配置。
3. 不新增 Node/Vite 构建链，继续使用静态 HTML、CSS、JavaScript。
4. 不改动 YOLO 后端、Ascend OM 后端和核心检测接口的语义。
5. 不承诺专业低延迟视频会议级实时性，目标是稳定可演示的准实时效果。

## 总体架构

```text
前端页面
  ├─ 上传视频入口
  │    └─ POST /api/inference/upload
  │    └─ POST /api/inference/start
  │    └─ GET  /api/inference/status
  │    └─ GET  /media/output-video
  │
  └─ 浏览器摄像头入口
       └─ getUserMedia() 获取浏览器摄像头
       └─ POST /api/realtime/start 初始化后端实时会话
       └─ POST /api/realtime/frame 按帧提交 JPEG
       └─ GET  /api/realtime/stream 读取后端标注后的 MJPEG 流
       └─ GET  /api/realtime/status 读取 FPS、帧数、车辆统计
       └─ POST /api/realtime/stop 释放实时会话

后端
  ├─ InferenceTaskManager：复用现有上传视频后台任务
  ├─ RealtimeInferenceManager：新增浏览器摄像头实时会话
  ├─ FrameProcessor：抽出单帧推理处理单元
  └─ Detector / Tracker / Counter / Overlay：复用现有核心模块
```

## 后端设计

### FrameProcessor

新增一个单帧处理单元，用来避免在实时摄像头逻辑中复制 `run_app` 的主循环细节。

职责：

- 持有 `Detector`、`CentroidTracker`、`LineCounter`、`FpsMeter`。
- 接收一帧 BGR 图像。
- 调用 `detector.detect(frame)`。
- 更新跟踪器和计数器。
- 调用 `draw_overlay()` 生成标注后图像。
- 返回标注帧、FPS、帧数、车辆统计。

`run_app()` 可以改为使用 `FrameProcessor`，这样上传视频、CLI 推理和实时摄像头共用同一套单帧处理逻辑。

### 上传视频后台任务

保留现有 `InferenceTaskManager`：

- `/api/inference/upload` 保存用户上传的视频到 `data/web_uploads/`。
- `/api/inference/start` 使用上传后的路径启动后台线程。
- 后台线程调用 `run_app()` 生成结果视频。
- `/api/inference/status` 返回状态、帧数、FPS 和车辆计数。
- `/media/output-video` 返回结果视频。

前端不再传 `output_video`，输出路径由后端按任务 ID 生成。

### 浏览器摄像头实时会话

新增 `RealtimeInferenceManager`，用于管理一个浏览器摄像头实时推理会话。

状态字段：

- `status`：`idle`、`running`、`stopping`、`stopped`、`failed`。
- `session_id`：当前实时会话 ID。
- `frames`：已处理帧数。
- `fps`：后端处理 FPS。
- `counts`：总车流量和分类计数。
- `error`：用户可读错误信息。
- `latest_jpeg`：最近一帧标注后的 JPEG bytes。
- `updated_at`：最近更新时间。

接口：

- `POST /api/realtime/start`
  - 创建 detector 和 `FrameProcessor`。
  - 返回会话状态。
- `POST /api/realtime/frame?session_id=...`
  - 请求体为浏览器 canvas 导出的 JPEG bytes。
  - 后端解码为 OpenCV BGR 图像。
  - 调用 `FrameProcessor.process(frame)`。
  - 将标注帧编码为 JPEG，更新 `latest_jpeg` 和统计状态。
  - 返回最新状态 JSON。
- `GET /api/realtime/stream?session_id=...`
  - 返回 `multipart/x-mixed-replace` MJPEG 响应。
  - 持续写出 `latest_jpeg`。
  - 如果短时间没有新帧，保持连接并等待下一帧。
- `GET /api/realtime/status?session_id=...`
  - 返回实时会话状态、FPS、帧数、车辆统计。
- `POST /api/realtime/stop`
  - 停止会话，释放 detector。

并发策略：

- 初版只允许一个实时摄像头会话运行。
- 如果已有实时会话运行，新的 start 请求返回 409。
- 停止会话时释放 detector，避免模型和摄像头资源泄漏。

## 前端设计

### 页面风格

视觉方向为“城市夜行”：

- 深色石墨背景。
- 暖色路灯/车流强调色。
- 大面积结果视频区域。
- 玻璃质感控制卡片，但避免通用 AI Dashboard 的蓝紫霓虹堆叠。
- 背景动态粒子模拟城市车流光点，粒子数量受控，不遮挡视频。
- 支持 `prefers-reduced-motion`，减少动画以避免性能和可访问性问题。

### 页面结构

页面分为四个区域：

1. 顶部品牌与说明
   - 项目名称。
   - 简短说明：上传视频或开启摄像头，后台完成车辆检测与车流统计。
2. 输入源卡片
   - 上传视频。
   - 开启摄像头。
   - 不出现路径输入框和摄像头编号输入框。
3. 结果舞台
   - 上传视频模式：推理中显示进度状态，完成后播放结果视频。
   - 摄像头模式：左侧可短暂显示原始摄像头预览，主区域显示后端标注后的 MJPEG 流。
4. 统计与流程
   - 状态。
   - FPS。
   - 已处理帧数。
   - 车辆总数。
   - 小型车、公交/客车、货车、两轮车计数。
   - 简短流程说明。

### 前端状态机

```text
idle
  ├─ 选择上传视频 -> uploading -> batch-running -> batch-completed
  └─ 开启摄像头 -> requesting-camera -> realtime-running -> realtime-stopped

任意状态遇到错误 -> failed
failed 可回到 idle
```

上传视频模式：

1. 用户选择本地视频文件。
2. 前端调用 `/api/inference/upload`。
3. 上传成功后调用 `/api/inference/start`。
4. 前端轮询 `/api/inference/status`。
5. 完成后加载 `/media/output-video`。

浏览器摄像头模式：

1. 用户点击“开启摄像头”。
2. 前端调用 `navigator.mediaDevices.getUserMedia({ video: true })`。
3. 前端调用 `/api/realtime/start`。
4. 前端用隐藏 canvas 按目标帧率抽取摄像头画面并 POST 到 `/api/realtime/frame`。
5. 主结果区域显示 `/api/realtime/stream`。
6. 前端轮询 `/api/realtime/status` 或直接使用 frame 响应更新统计。
7. 用户点击停止时停止抽帧、停止媒体轨道、调用 `/api/realtime/stop`。

目标抽帧频率初版设为 8–12 FPS，由前端常量控制，不在页面展示。

## 错误处理

前端显示用户可理解的错误，而不是 Python 堆栈或工程参数。

典型错误：

- 未选择视频文件：提示“请先选择一个视频文件”。
- 上传失败：提示“视频上传失败，请重试”。
- 摄像头权限被拒绝：提示“浏览器未授予摄像头权限”。
- 浏览器不支持摄像头 API：提示“当前浏览器不支持摄像头调用”。
- 模型文件缺失：提示“后端模型未准备好，请检查运行环境”。
- 实时会话已运行：提示“已有实时推理任务正在运行，请先停止”。
- 推理失败：显示后端返回的简短错误信息。

后端仍保留详细异常用于测试和调试，但 API 返回前端时转换为简短 `error` 字段。

## 测试设计

### 后端测试

1. `FrameProcessor` 单元测试
   - 使用假 detector 和合成帧。
   - 验证返回帧、FPS、帧数和计数结构。
2. `RealtimeInferenceManager` 单元测试
   - 使用假 detector。
   - 验证 start、process frame、status、stop。
   - 验证重复 start 返回错误。
3. Dashboard API 测试
   - 验证 payload 不再暴露前端不需要展示的工程配置。
   - 验证 class labels、pipeline、presentation 文案存在。
4. 上传视频任务测试
   - 保留现有后台任务测试。
   - 更新断言，确保前端不再依赖路径输入和摄像头编号。

### 前端静态测试

继续使用现有测试读取静态文件，验证：

- `index.html` 包含粒子 canvas。
- 包含上传视频入口和摄像头入口。
- 不包含 `videoPathInput`、`cameraInput`、模型路径、尺寸、阈值等参数控件。
- `app.js` 包含实时摄像头接口调用。
- `styles.css` 包含城市夜行主题和 reduced-motion 处理。

### 手动验证

1. 启动：`python -m vehicle_flow_ascend --config configs/pc_demo.yaml --web`。
2. 打开 `http://127.0.0.1:8765`。
3. 上传短视频，确认后台推理完成并播放结果视频。
4. 开启浏览器摄像头，确认浏览器权限弹窗、后端标注帧、实时统计和停止流程可用。
5. 模拟模型缺失或权限拒绝，确认错误提示清晰。

## 实施边界

这次实现允许修改：

- `src/vehicle_flow_ascend/app.py`
- `src/vehicle_flow_ascend/web/dashboard.py`
- `src/vehicle_flow_ascend/web/inference.py`
- `src/vehicle_flow_ascend/web/static/index.html`
- `src/vehicle_flow_ascend/web/static/app.js`
- `src/vehicle_flow_ascend/web/static/styles.css`
- 相关测试文件
- README 中的 Web Dashboard 说明

这次实现不修改：

- detector 接口语义。
- YOLOv5 推理后端。
- Ascend OM 推理后端。
- YAML 配置格式。

## 成功标准

1. 前端页面不再呈现工程参数面板。
2. 上传视频可以触发后台推理并展示结果视频。
3. 浏览器摄像头可以授权、传帧、后端推理、前端展示标注后的实时画面。
4. 车辆总数和分类计数在上传视频、浏览器摄像头和开发板 USB 摄像头三种输入模式下都能展示。
5. UI 采用城市夜行风格，动态粒子可见但不遮挡内容。
6. 自动测试通过。
7. README 中的前端使用说明与实际页面一致。
