const state = {
  dashboard: null,
  mode: 'idle',
  busyAction: null,
  inferencePollTimer: null,
  realtimePollTimer: null,
  captureTimer: null,
  frameUploadInFlight: false,
  frameAbortController: null,
  realtimeFrameFailures: 0,
  realtimeSessionId: null,
  cameraStream: null,
  cameraDevices: [],
  uploadedVideo: null,
  uploadedVideoUrl: null,
  lineEditor: {
    ready: false,
    source: null,
    dragging: null,
    videoWidth: 0,
    videoHeight: 0,
    points: [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ],
  },
  lastOutputVideo: null,
  lastStatus: { status: 'idle', counts: { total: 0 }, frames: 0, fps: 0 },
  viewToken: 0,
};

const byId = (id) => document.getElementById(id);
const MAX_REALTIME_FRAME_FAILURES = 5;
const REALTIME_RETRY_MESSAGE = '实时帧传输不稳定，正在重试。';
const MAX_CAPTURE_WIDTH = 480;
const REALTIME_CAPTURE_DELAY_MS = 0;

const FALLBACK_CLASSES = [
  { key: 'car', label: '小型车', accent: '#ffb25d' },
  { key: 'bus', label: '公交/客车', accent: '#ffd39b' },
  { key: 'truck', label: '货车', accent: '#9aa5ad' },
  { key: 'two_wheeler', label: '两轮车', accent: '#f0764f' },
];

const FALLBACK_PIPELINE = [
  '选择上传视频或浏览器摄像头',
  '后端执行 YOLOv5 推理',
  '车辆跟踪与穿线计数',
  '叠加检测框和统计信息',
  '前端展示结果视频或实时标注画面',
];

async function requestJson(url, options = {}) {
  const { json, headers = {}, body, ...rest } = options;
  const finalHeaders = new Headers(headers);
  if (!finalHeaders.has('Accept')) {
    finalHeaders.set('Accept', 'application/json');
  }

  let finalBody = body;
  if (json !== undefined) {
    finalBody = JSON.stringify(json);
    if (!finalHeaders.has('Content-Type')) {
      finalHeaders.set('Content-Type', 'application/json');
    }
  }

  const response = await fetch(url, {
    cache: 'no-store',
    ...rest,
    headers: finalHeaders,
    body: finalBody,
  });

  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await response.json() : {};
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function rememberButtonLabels() {
  ['uploadVideoButton', 'analysisStartButton', 'cameraButton', 'stopButton'].forEach((id) => {
    const button = byId(id);
    if (button && !button.dataset.label) {
      button.dataset.label = button.textContent.trim();
    }
  });
}

function syncControls() {
  const uploadButton = byId('uploadVideoButton');
  const analysisStartButton = byId('analysisStartButton');
  const cameraButton = byId('cameraButton');
  const stopButton = byId('stopButton');
  const busy = Boolean(state.busyAction);
  const active = ['batch', 'realtime', 'camera'].includes(state.mode);
  const cameraPreview = state.mode === 'camera-preview';
  const canAnalyze =
    state.lineEditor.ready && (Boolean(state.uploadedVideo) || cameraPreview);

  uploadButton.disabled = busy || active || cameraPreview;
  analysisStartButton.disabled = busy || active || !canAnalyze;
  cameraButton.disabled = busy || active || cameraPreview;
  stopButton.disabled = busy || (!active && !cameraPreview);
}

function setBusy(isBusy, action = '') {
  rememberButtonLabels();
  const buttons = {
    upload: byId('uploadVideoButton'),
    analysis: byId('analysisStartButton'),
    camera: byId('cameraButton'),
    stop: byId('stopButton'),
  };

  Object.values(buttons).forEach((button) => {
    if (button) {
      button.textContent = button.dataset.label || button.textContent;
    }
  });

  state.busyAction = isBusy ? action : null;
  if (isBusy) {
    if (action === 'upload') buttons.upload.textContent = '上传中...';
    if (action === 'analysis') buttons.analysis.textContent = '启动中...';
    if (action === 'camera') buttons.camera.textContent = '连接中...';
    if (action === 'stop') buttons.stop.textContent = '停止中...';
  }

  syncControls();
}

function setError(message) {
  byId('errorLine').textContent = message || '';
}

function setMode(mode) {
  state.mode = mode;
  syncControls();
}

function classesConfig() {
  return state.dashboard?.classes?.length ? state.dashboard.classes : FALLBACK_CLASSES;
}

async function refreshCameraDevices({ silent = true } = {}) {
  const select = byId('cameraDeviceSelect');
  if (!navigator.mediaDevices?.enumerateDevices || !navigator.mediaDevices?.getUserMedia) {
    select.innerHTML = '<option value="">当前浏览器不支持摄像头</option>';
    select.disabled = true;
    if (!silent) {
      setError('当前浏览器不支持摄像头访问，请使用新版 Edge、Chrome 或 HTTPS/localhost 访问。');
    }
    return [];
  }

  try {
    const previousValue = select.value;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === 'videoinput');
    state.cameraDevices = cameras;
    select.disabled = cameras.length === 0;
    select.innerHTML = cameras.length
      ? [
          '<option value="">自动选择摄像头</option>',
          ...cameras.map((device, index) => {
            const label = device.label || `摄像头 ${index + 1}`;
            return `<option value="${escapeHtml(device.deviceId)}">${escapeHtml(label)}</option>`;
          }),
        ].join('')
      : '<option value="">未发现摄像头</option>';
    if (previousValue && cameras.some((device) => device.deviceId === previousValue)) {
      select.value = previousValue;
    }
    if (!silent && cameras.length === 0) {
      setError('未发现可用摄像头，请检查设备连接或浏览器权限。');
    }
    return cameras;
  } catch (error) {
    select.innerHTML = '<option value="">摄像头权限不可用</option>';
    select.disabled = true;
    if (!silent) {
      setError(error.message);
    }
    return [];
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function updateEmptyState(title, description) {
  const emptyState = byId('emptyState');
  const titleNode = emptyState.querySelector('h3');
  const copyNode = emptyState.querySelector('p');
  titleNode.textContent = title;
  copyNode.textContent = description;
}

function nextViewToken() {
  state.viewToken += 1;
  return state.viewToken;
}

function isCurrentViewToken(token) {
  return token === state.viewToken;
}

function canShowResultVideo() {
  return ['batch', 'upload', 'idle'].includes(state.mode);
}

function resetResultVideo() {
  const resultVideo = byId('resultVideo');
  resultVideo.pause();
  resultVideo.removeAttribute('src');
  resultVideo.load();
  resultVideo.hidden = true;
}

function resetRealtimeStream() {
  const realtimeStream = byId('realtimeStream');
  realtimeStream.hidden = true;
  realtimeStream.removeAttribute('src');
}

function resetLineSetup({ clearUpload = false } = {}) {
  const panel = byId('lineSetupPanel');
  const preview = byId('linePreviewVideo');
  panel.hidden = true;
  preview.pause();
  preview.srcObject = null;
  preview.removeAttribute('src');
  preview.load();
  if (clearUpload) {
    if (state.uploadedVideoUrl) {
      URL.revokeObjectURL(state.uploadedVideoUrl);
    }
    state.uploadedVideoUrl = null;
    state.uploadedVideo = null;
  }
  state.lineEditor.ready = false;
  state.lineEditor.source = null;
  syncControls();
}

function showEmptyState(title, description) {
  nextViewToken();
  resetResultVideo();
  resetRealtimeStream();
  resetLineSetup();
  updateEmptyState(title, description);
  byId('emptyState').hidden = false;
}

function showRealtimeStream(sessionId) {
  nextViewToken();
  resetResultVideo();
  resetLineSetup();
  const realtimeStream = byId('realtimeStream');
  byId('emptyState').hidden = true;
  realtimeStream.hidden = false;
  realtimeStream.src = `/api/realtime/stream?session_id=${encodeURIComponent(sessionId)}&t=${Date.now()}`;
}

function showInferenceStream(taskId) {
  nextViewToken();
  resetResultVideo();
  resetLineSetup();
  const realtimeStream = byId('realtimeStream');
  byId('emptyState').hidden = true;
  realtimeStream.hidden = false;
  realtimeStream.src = `/api/inference/stream?task_id=${encodeURIComponent(taskId)}&t=${Date.now()}`;
}

async function showResultVideo(outputPath, token = nextViewToken()) {
  if (!outputPath || !canShowResultVideo()) {
    return;
  }

  const media = await requestJson(`/api/media-status?path=${encodeURIComponent(outputPath)}`);
  if (!isCurrentViewToken(token) || !canShowResultVideo()) {
    return;
  }
  if (!media.exists) {
    showEmptyState('结果视频尚未就绪', '后端任务已返回，但结果文件暂不可用，请稍后重试。');
    return;
  }

  state.lastOutputVideo = outputPath;
  const resultVideo = byId('resultVideo');
  resetResultVideo();
  resetRealtimeStream();
  resetLineSetup();
  byId('emptyState').hidden = true;
  resultVideo.hidden = false;
  resultVideo.src = `/media/output-video?path=${encodeURIComponent(outputPath)}&t=${Date.now()}`;
  resultVideo.load();
  resultVideo.play().catch(() => {});
}

function setCameraPreviewVisible(visible) {
  const dock = byId('cameraPreviewDock');
  const preview = byId('cameraPreview');
  dock.hidden = !visible;
  preview.hidden = !visible;
}

function initializeLineEditor(videoWidth, videoHeight, source = state.lineEditor.source) {
  const editor = state.lineEditor;
  editor.ready = true;
  editor.source = source;
  editor.dragging = null;
  editor.videoWidth = videoWidth;
  editor.videoHeight = videoHeight;
  const y = Math.round(videoHeight * 0.78);
  editor.points = [
    { x: Math.round(videoWidth * 0.12), y },
    { x: Math.round(videoWidth * 0.88), y },
  ];
  drawLineEditor();
  syncControls();
}

function lineFromEditor() {
  const editor = state.lineEditor;
  if (!editor.ready) {
    throw new Error('请先上传视频并设置穿线位置。');
  }
  return editor.points.map((point) => [
    Math.round(clamp(point.x, 0, editor.videoWidth - 1)),
    Math.round(clamp(point.y, 0, editor.videoHeight - 1)),
  ]);
}

function lineForRealtimeCapture() {
  const editor = state.lineEditor;
  const line = lineFromEditor();
  const captureSize = scaleCaptureDimensions(editor.videoWidth, editor.videoHeight);
  const scaleX = captureSize.width / editor.videoWidth;
  const scaleY = captureSize.height / editor.videoHeight;
  return line.map(([x, y]) => [
    Math.round(x * scaleX),
    Math.round(y * scaleY),
  ]);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function lineCanvasMetrics() {
  const canvas = byId('lineOverlayCanvas');
  const video = byId('linePreviewVideo');
  const rect = canvas.getBoundingClientRect();
  const videoWidth = state.lineEditor.videoWidth || video.videoWidth || 1;
  const videoHeight = state.lineEditor.videoHeight || video.videoHeight || 1;
  const scale = Math.min(rect.width / videoWidth, rect.height / videoHeight);
  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  return {
    canvas,
    rect,
    videoWidth,
    videoHeight,
    scale,
    offsetX: (rect.width - drawWidth) / 2,
    offsetY: (rect.height - drawHeight) / 2,
  };
}

function videoPointToCanvas(point, metrics) {
  return {
    x: metrics.offsetX + point.x * metrics.scale,
    y: metrics.offsetY + point.y * metrics.scale,
  };
}

function canvasPointToVideo(clientX, clientY, metrics) {
  return {
    x: clamp((clientX - metrics.rect.left - metrics.offsetX) / metrics.scale, 0, metrics.videoWidth - 1),
    y: clamp((clientY - metrics.rect.top - metrics.offsetY) / metrics.scale, 0, metrics.videoHeight - 1),
  };
}

function resizeLineCanvas() {
  const canvas = byId('lineOverlayCanvas');
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  drawLineEditor();
}

function drawLineEditor() {
  const canvas = byId('lineOverlayCanvas');
  if (!canvas || byId('lineSetupPanel').hidden || !state.lineEditor.ready) {
    return;
  }
  const metrics = lineCanvasMetrics();
  const ratio = window.devicePixelRatio || 1;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, metrics.rect.width, metrics.rect.height);

  const [start, end] = state.lineEditor.points.map((point) => videoPointToCanvas(point, metrics));
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.strokeStyle = 'rgba(5, 7, 13, 0.86)';
  context.lineWidth = 9;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.strokeStyle = '#ffb25d';
  context.lineWidth = 4;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();

  [start, end].forEach((point) => {
    context.beginPath();
    context.arc(point.x, point.y, 10, 0, Math.PI * 2);
    context.fillStyle = '#ffd39b';
    context.fill();
    context.lineWidth = 3;
    context.strokeStyle = 'rgba(5, 7, 13, 0.9)';
    context.stroke();
  });

  byId('lineCoordinateText').textContent = lineFromEditor()
    .map((point) => point.join(','))
    .join(' -> ');
}

function nearestLineTarget(clientX, clientY) {
  const metrics = lineCanvasMetrics();
  const pointer = { x: clientX - metrics.rect.left, y: clientY - metrics.rect.top };
  const canvasPoints = state.lineEditor.points.map((point) => videoPointToCanvas(point, metrics));
  const distances = canvasPoints.map((point) => Math.hypot(point.x - pointer.x, point.y - pointer.y));
  if (distances[0] <= 24) return 'start';
  if (distances[1] <= 24) return 'end';
  return 'line';
}

function setLinePointFromPointer(target, clientX, clientY) {
  const metrics = lineCanvasMetrics();
  const point = canvasPointToVideo(clientX, clientY, metrics);
  if (target === 'start') {
    state.lineEditor.points[0] = point;
  } else if (target === 'end') {
    state.lineEditor.points[1] = point;
  } else {
    const midpoint = {
      x: (state.lineEditor.points[0].x + state.lineEditor.points[1].x) / 2,
      y: (state.lineEditor.points[0].y + state.lineEditor.points[1].y) / 2,
    };
    const dx = point.x - midpoint.x;
    const dy = point.y - midpoint.y;
    state.lineEditor.points = state.lineEditor.points.map((linePoint) => ({
      x: clamp(linePoint.x + dx, 0, state.lineEditor.videoWidth - 1),
      y: clamp(linePoint.y + dy, 0, state.lineEditor.videoHeight - 1),
    }));
  }
  drawLineEditor();
}

function bindLineEditorEvents() {
  const canvas = byId('lineOverlayCanvas');
  canvas.addEventListener('pointerdown', (event) => {
    if (!state.lineEditor.ready) return;
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic pointer events used by browser automation may not own a capture target.
    }
    state.lineEditor.dragging = nearestLineTarget(event.clientX, event.clientY);
    setLinePointFromPointer(state.lineEditor.dragging, event.clientX, event.clientY);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!state.lineEditor.dragging) return;
    setLinePointFromPointer(state.lineEditor.dragging, event.clientX, event.clientY);
  });
  canvas.addEventListener('pointerup', () => {
    state.lineEditor.dragging = null;
  });
  canvas.addEventListener('pointercancel', () => {
    state.lineEditor.dragging = null;
  });
  window.addEventListener('resize', resizeLineCanvas);
}

function renderPipeline(steps) {
  const items = (steps?.length ? steps : FALLBACK_PIPELINE)
    .map((step) => `<li>${escapeHtml(step)}</li>`)
    .join('');
  byId('pipelineList').innerHTML = items;
}

function renderClassCounts(counts = {}) {
  const items = classesConfig().map((item) => {
    const value = counts[item.key] || 0;
    return `
      <div class="class-chip" style="--chip-accent:${escapeHtml(item.accent)}">
        <span>${escapeHtml(item.label)}</span>
        <strong>${value}</strong>
      </div>
    `;
  });
  byId('classCounts').innerHTML = items.join('');
}

function statusPresentation(status, snapshot) {
  const cameraSession = Boolean(snapshot?.session_id);
  const labels = {
    idle: {
      title: '系统待命',
      detail: '选择上传视频或开启摄像头',
    },
    preview: {
      title: '设置穿线',
      detail: '拖动主屏线段后开始分析',
    },
    running: cameraSession
      ? {
          title: '摄像头实时推理中',
          detail: '浏览器以 10 FPS 上传帧，主屏展示后端 MJPEG 标注流',
        }
      : {
          title: '视频任务处理中',
          detail: '后端正在生成标注后结果视频',
        },
    stopping: {
      title: '正在停止',
      detail: '等待后台任务释放资源',
    },
    completed: {
      title: '推理完成',
      detail: '结果视频已可播放',
    },
    stopped: {
      title: '任务已停止',
      detail: '可重新选择上传视频或开启摄像头',
    },
    failed: {
      title: '任务失败',
      detail: '请查看下方错误信息',
    },
  };
  return labels[status] || { title: status || '未知状态', detail: '状态信息不可用' };
}

function renderStatus(snapshot = {}) {
  state.lastStatus = snapshot;
  const status = snapshot.status || 'idle';
  byId('taskStatus').textContent = status;
  byId('taskFrames').textContent = String(snapshot.frames || 0);
  byId('taskFps').textContent = Number(snapshot.fps || 0).toFixed(1);
  byId('taskTotal').textContent = String((snapshot.counts || {}).total || 0);
  if (snapshot.error) {
    setError(snapshot.error);
  }
  renderClassCounts(snapshot.counts || {});

  const pill = byId('statusPill');
  const label = statusPresentation(status, snapshot);
  pill.dataset.status = status;
  pill.innerHTML = `
    <span class="status-pill__dot"></span>
    <div>
      <strong>${escapeHtml(label.title)}</strong>
      <small>${escapeHtml(label.detail)}</small>
    </div>
  `;
}

async function loadDashboard() {
  state.dashboard = await requestJson('/api/dashboard');
  renderPipeline(state.dashboard.pipeline || FALLBACK_PIPELINE);
  refreshCameraDevices().catch(() => {});

  const inference = state.dashboard.inference || { status: 'idle' };
  const realtime = state.dashboard.realtime || { status: 'idle' };

  stopInferencePolling();
  stopRealtimePolling();

  if (realtime.status === 'running' && realtime.session_id) {
    state.realtimeSessionId = realtime.session_id;
    setMode('realtime');
    renderStatus(realtime);
    showRealtimeStream(realtime.session_id);
    setError('摄像头页面已刷新，本地摄像头抽帧不会自动恢复；请停止后重新开启。');
    startRealtimePolling();
    return;
  }

  state.realtimeSessionId = null;

  if (inference.status === 'running' || inference.status === 'stopping') {
    setMode('batch');
    renderStatus(inference);
    if (inference.task_id) {
      showInferenceStream(inference.task_id);
    } else {
      showEmptyState('视频任务正在处理', '后端批处理任务仍在运行，完成后会自动切换为结果视频播放。');
    }
    startInferencePolling();
    return;
  }

  if ((inference.status === 'completed' || inference.status === 'stopped') && inference.output_video) {
    setMode('idle');
    renderStatus(inference);
    await showResultVideo(inference.output_video);
    return;
  }

  const preferred = realtime.status && realtime.status !== 'idle' ? realtime : inference;
  setMode('idle');
  renderStatus(preferred);
  showEmptyState('选择一个入口开始', '上传本地视频以生成结果视频，或开启浏览器摄像头进入实时车流观察。');
}

function stopInferencePolling() {
  if (state.inferencePollTimer) {
    clearInterval(state.inferencePollTimer);
    state.inferencePollTimer = null;
  }
}

function startInferencePolling() {
  stopInferencePolling();
  state.inferencePollTimer = setInterval(async () => {
    try {
      const snapshot = await requestJson('/api/inference/status');
      renderStatus(snapshot);
      if (snapshot.output_video) {
        state.lastOutputVideo = snapshot.output_video;
      }
      if (snapshot.status === 'running' && snapshot.task_id && byId('realtimeStream').hidden) {
        showInferenceStream(snapshot.task_id);
      }
      if (snapshot.status === 'completed' || snapshot.status === 'stopped' || snapshot.status === 'failed') {
        stopInferencePolling();
        setMode('idle');
        setBusy(false);
        if (snapshot.output_video && snapshot.status !== 'failed') {
          await showResultVideo(snapshot.output_video);
        } else if (snapshot.status === 'stopped') {
          showEmptyState('任务已停止', '上传任务已停止，可重新选择视频并再次启动。');
        }
      }
    } catch (error) {
      stopInferencePolling();
      setMode('idle');
      setBusy(false);
      setError(error.message);
    }
  }, 1000);
}

async function startUploadFlow() {
  setError('');
  const file = byId('videoFileInput').files?.[0];
  if (!file) {
    setError('请先选择一个本地视频文件。');
    return;
  }

  nextViewToken();
  resetResultVideo();
  resetRealtimeStream();
  resetLineSetup({ clearUpload: true });
  setBusy(true, 'upload');
  try {
    const formData = new FormData();
    formData.append('video', file);
    const upload = await requestJson('/api/inference/upload', {
      method: 'POST',
      body: formData,
    });

    state.uploadedVideo = upload;
    if (state.uploadedVideoUrl) {
      URL.revokeObjectURL(state.uploadedVideoUrl);
    }
    state.uploadedVideoUrl = URL.createObjectURL(file);
    showLineSetup(file.name);
  } catch (error) {
    setMode('idle');
    setError(error.message);
  } finally {
    setBusy(false);
  }
}

function showLineSetup(fileName) {
  nextViewToken();
  resetResultVideo();
  resetRealtimeStream();
  byId('emptyState').hidden = true;
  const panel = byId('lineSetupPanel');
  const preview = byId('linePreviewVideo');
  panel.hidden = false;
  state.lineEditor.source = 'upload';
  preview.srcObject = null;
  preview.src = state.uploadedVideoUrl;
  armLinePreviewMetadata('upload');
  preview.load();
  updateEmptyState('设置穿线位置', fileName || '拖动主屏线段两端后开始分析。');
}

function showCameraLineSetup(stream) {
  nextViewToken();
  resetResultVideo();
  resetRealtimeStream();
  byId('emptyState').hidden = true;
  const panel = byId('lineSetupPanel');
  const preview = byId('linePreviewVideo');
  panel.hidden = false;
  state.lineEditor.source = 'camera';
  preview.removeAttribute('src');
  preview.srcObject = stream;
  armLinePreviewMetadata('camera');
  preview.play().catch(() => {});
  updateEmptyState('设置摄像头穿线', '拖动主屏线段两端后开始实时分析。');
}

function armLinePreviewMetadata(source) {
  const preview = byId('linePreviewVideo');
  const initialize = () => {
    initializeLineEditor(preview.videoWidth || 1920, preview.videoHeight || 1080, source);
    resizeLineCanvas();
    setError('');
  };
  if (preview.videoWidth && preview.videoHeight) {
    initialize();
    return;
  }
  preview.addEventListener('loadedmetadata', initialize, { once: true });
}

async function startAnalysisFlow() {
  setError('');
  if (state.mode === 'camera-preview') {
    await startCameraAnalysisFlow();
    return;
  }
  if (!state.uploadedVideo?.path) {
    setError('请先上传视频并设置穿线位置。');
    return;
  }

  nextViewToken();
  setBusy(true, 'analysis');
  try {
    const started = await requestJson('/api/inference/start', {
      method: 'POST',
      json: {
        source_type: 'video',
        source: state.uploadedVideo.path,
        line: lineFromEditor(),
      },
    });

    setMode('batch');
    renderStatus(started);
    if (started.task_id) {
      showInferenceStream(started.task_id);
    } else {
      showEmptyState('视频任务已启动', '后端正在处理上传视频，完成后会自动切换为结果视频播放。');
    }
    startInferencePolling();
  } catch (error) {
    setMode('idle');
    setError(error.message);
  } finally {
    setBusy(false);
  }
}

function stopRealtimePolling() {
  if (state.realtimePollTimer) {
    clearInterval(state.realtimePollTimer);
    state.realtimePollTimer = null;
  }
}

function cleanupLocalRealtimeSession() {
  stopCaptureLoop();
  stopRealtimePolling();
  stopLocalTracks();
  resetRealtimeStream();
  state.realtimeSessionId = null;
  setMode('idle');
  setBusy(false);
}

function stopCaptureLoop(abortCurrent = false) {
  if (state.captureTimer) {
    clearTimeout(state.captureTimer);
    state.captureTimer = null;
  }
  if (abortCurrent && state.frameAbortController) {
    state.frameAbortController.abort();
    state.frameAbortController = null;
  }
  state.frameUploadInFlight = false;
  state.realtimeFrameFailures = 0;
}

function stopLocalTracks() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  const preview = byId('cameraPreview');
  preview.pause();
  preview.srcObject = null;
  setCameraPreviewVisible(false);
}

async function stopRealtimeSession(callApi = true, silent = false) {
  const sessionId = state.realtimeSessionId;
  stopCaptureLoop(true);
  stopRealtimePolling();
  stopLocalTracks();
  resetRealtimeStream();

  if (!callApi || !sessionId) {
    state.realtimeSessionId = null;
    setMode('idle');
    setBusy(false);
    return true;
  }

  try {
    const stopped = await requestJson('/api/realtime/stop', {
      method: 'POST',
      json: { session_id: sessionId },
    });
    if (state.realtimeSessionId === sessionId) {
      state.realtimeSessionId = null;
    }
    renderStatus(stopped);
    setMode('idle');
    setBusy(false);
    return true;
  } catch (error) {
    if (!silent) {
      setError(error.message);
    }
    state.realtimeSessionId = sessionId;
    setMode('realtime');
    setBusy(false);
    return false;
  }
}

function blobFromCanvas(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error('无法生成摄像头 JPEG 帧。'));
    }, type, quality);
  });
}

function scaleCaptureDimensions(width, height) {
  const scale = Math.min(1, MAX_CAPTURE_WIDTH / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

async function captureAndSendFrame() {
  if (state.frameUploadInFlight || !state.realtimeSessionId) {
    return;
  }

  const sessionId = state.realtimeSessionId;
  const preview = byId('cameraPreview');
  if (!preview.videoWidth || !preview.videoHeight) {
    return;
  }

  state.frameUploadInFlight = true;
  const abortController = new AbortController();
  state.frameAbortController = abortController;
  const canvas = byId('captureCanvas');
  const context = canvas.getContext('2d', { alpha: false });
  const captureSize = scaleCaptureDimensions(preview.videoWidth, preview.videoHeight);
  if (canvas.width !== captureSize.width || canvas.height !== captureSize.height) {
    canvas.width = captureSize.width;
    canvas.height = captureSize.height;
  }

  try {
    context.drawImage(preview, 0, 0, canvas.width, canvas.height);
    const blob = await blobFromCanvas(canvas, 'image/jpeg', 0.72);
    const snapshot = await requestJson(
      `/api/realtime/frame?session_id=${encodeURIComponent(sessionId)}`,
      {
        method: 'POST',
        body: blob,
        signal: abortController.signal,
        headers: {
          'Content-Type': 'image/jpeg',
        },
      },
    );
    if (state.realtimeSessionId !== sessionId) {
      return;
    }
    state.realtimeFrameFailures = 0;
    if (byId('errorLine').textContent === REALTIME_RETRY_MESSAGE) {
      setError('');
    }
    renderStatus(snapshot);
  } catch (error) {
    if (error.name === 'AbortError' || state.realtimeSessionId !== sessionId) {
      return;
    }
    state.realtimeFrameFailures += 1;
    if (state.realtimeFrameFailures < MAX_REALTIME_FRAME_FAILURES) {
      setError(REALTIME_RETRY_MESSAGE);
      return;
    }
    setError(error.message);
    const stopped = await stopRealtimeSession(true);
    showEmptyState(
      stopped ? '摄像头任务已中断' : '摄像头停止请求失败',
      stopped
        ? '浏览器与后端之间的实时传输失败，已尝试停止后端会话。'
        : '浏览器与后端之间的实时传输失败，且后端会话可能仍在运行，请再次点击停止。',
    );
  } finally {
    if (state.frameAbortController === abortController) {
      state.frameAbortController = null;
    }
    if (state.realtimeSessionId === sessionId) {
      state.frameUploadInFlight = false;
    }
  }
}

function startCaptureLoop() {
  stopCaptureLoop(true);
  const tick = async () => {
    if (!state.realtimeSessionId) {
      state.captureTimer = null;
      return;
    }
    try {
      await captureAndSendFrame();
    } catch (error) {
      setError(error.message);
    }
    if (state.realtimeSessionId) {
      state.captureTimer = setTimeout(tick, REALTIME_CAPTURE_DELAY_MS);
    }
  };
  state.captureTimer = setTimeout(tick, 0);
}

function startRealtimePolling() {
  stopRealtimePolling();
  state.realtimePollTimer = setInterval(async () => {
    const localSessionId = state.realtimeSessionId;
    if (!localSessionId) {
      return;
    }
    try {
      const snapshot = await requestJson('/api/realtime/status');
      const backendSessionId = snapshot.session_id;
      const sessionChanged = snapshot.status === 'running' && backendSessionId && backendSessionId !== localSessionId;
      const sessionEnded = snapshot.status !== 'running';

      if (sessionChanged || sessionEnded) {
        renderStatus(snapshot);
        cleanupLocalRealtimeSession();
        showEmptyState(
          sessionChanged ? '实时会话已在其他页面变更' : '摄像头任务已结束',
          sessionChanged
            ? '当前页面持有的实时会话已失效，未停止其他页面的新会话。请重新开启摄像头。'
            : '实时推理会话已停止或过期，可重新开启浏览器摄像头。',
        );
        return;
      }

      renderStatus(snapshot);
    } catch (error) {
      setError(error.message);
    }
  }, 1000);
}

async function startCameraPreviewFlow() {
  setError('');
  nextViewToken();
  resetResultVideo();
  resetRealtimeStream();
  resetLineSetup();
  setBusy(true, 'camera');
  let stream;

  try {
    const cameras = await refreshCameraDevices({ silent: false });
    if (cameras.length === 0) {
      throw new Error('未发现可用摄像头。');
    }
    const selectedDeviceId = byId('cameraDeviceSelect').value;
    const video = selectedDeviceId
      ? {
          deviceId: { exact: selectedDeviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        }
      : {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        };
    stream = await navigator.mediaDevices.getUserMedia({
      video,
      audio: false,
    });
    refreshCameraDevices().catch(() => {});

    state.cameraStream = stream;
    setMode('camera-preview');
    renderStatus({ status: 'preview', counts: { total: 0 }, frames: 0, fps: 0 });
    showCameraLineSetup(stream);
  } catch (error) {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      state.cameraStream = null;
    }

    setMode('idle');
    setError(error.message);
    showEmptyState(
      '无法开启摄像头',
      '请确认浏览器已获得摄像头权限，然后重试。',
    );
  } finally {
    setBusy(false);
  }
}

async function startCameraAnalysisFlow() {
  setError('');
  if (!state.cameraStream) {
    setError('请先开启摄像头预览并设置穿线位置。');
    return;
  }

  setBusy(true, 'analysis');
  try {
    const started = await requestJson('/api/realtime/start', {
      method: 'POST',
      json: {
        line: lineForRealtimeCapture(),
      },
    });
    state.realtimeSessionId = started.session_id;

    const preview = byId('cameraPreview');
    preview.srcObject = state.cameraStream;
    setCameraPreviewVisible(true);
    await preview.play().catch(() => {});

    setMode('camera');
    renderStatus(started);
    showRealtimeStream(started.session_id);
    startCaptureLoop();
    startRealtimePolling();
  } catch (error) {
    const sessionId = state.realtimeSessionId;
    let backendStopped = true;
    if (sessionId) {
      backendStopped = await stopRealtimeSession(true);
    }
    setError(
      backendStopped
        ? error.message
        : `${error.message} 后端实时会话可能仍在运行，请点击停止后重试。`,
    );
    showEmptyState(
      backendStopped ? '无法开始实时分析' : '摄像头启动未完成',
      backendStopped
        ? '请确认穿线位置有效，然后重试。'
        : '本地启动失败，且后端实时会话停止请求失败；请点击停止清理会话。',
    );
  } finally {
    setBusy(false);
  }
}

async function stopActiveFlow() {
  setError('');
  setBusy(true, 'stop');

  try {
    if (state.mode === 'camera-preview') {
      stopLocalTracks();
      resetLineSetup();
      setMode('idle');
      showEmptyState('摄像头预览已关闭', '本地摄像头已释放，可重新选择视频或开启摄像头。');
      return;
    }

    if (state.mode === 'camera' || state.mode === 'realtime' || state.realtimeSessionId) {
      const stopped = await stopRealtimeSession(true);
      if (stopped) {
        showEmptyState('摄像头已停止', '浏览器摄像头与实时推理会话都已释放。');
      } else {
        showEmptyState('摄像头停止请求失败', '后端实时会话可能仍在运行，请再次点击停止或刷新后重试。');
      }
      return;
    }

    if (state.mode === 'batch' || state.mode === 'upload') {
      const stopped = await requestJson('/api/inference/stop', {
        method: 'POST',
        json: {},
      });
      renderStatus(stopped);
      startInferencePolling();
      return;
    }
  } catch (error) {
    setError(error.message);
  } finally {
    if (state.mode !== 'camera' && state.mode !== 'realtime') {
      setBusy(false);
    }
  }
}

function notifyRealtimeStopOnUnload() {
  const sessionId = state.realtimeSessionId;
  if (!sessionId) {
    return;
  }

  const body = JSON.stringify({ session_id: sessionId });
  fetch('/api/realtime/stop', {
    method: 'POST',
    body,
    headers: { 'Content-Type': 'application/json' },
    keepalive: true,
  }).catch(() => {});
}

function bindEvents() {
  rememberButtonLabels();
  byId('uploadVideoButton').addEventListener('click', () => {
    startUploadFlow().catch((error) => setError(error.message));
  });
  byId('analysisStartButton').addEventListener('click', () => {
    startAnalysisFlow().catch((error) => setError(error.message));
  });
  byId('cameraButton').addEventListener('click', () => {
    startCameraPreviewFlow().catch((error) => setError(error.message));
  });
  byId('stopButton').addEventListener('click', () => {
    stopActiveFlow().catch((error) => setError(error.message));
  });
  byId('videoFileInput').addEventListener('change', () => {
    if (byId('videoFileInput').files?.[0]) {
      setError('');
      resetLineSetup({ clearUpload: true });
      showEmptyState('视频已选择', '点击上传视频并预览，然后拖动主屏穿线位置。');
    }
  });
  byId('cameraDeviceSelect').addEventListener('change', () => {
    setError('');
  });
  window.addEventListener('beforeunload', () => {
    notifyRealtimeStopOnUnload();
    stopCaptureLoop();
    stopRealtimePolling();
    stopLocalTracks();
  });
}

function initParticles() {
  const canvas = byId('particleCanvas');
  const context = canvas.getContext('2d');
  const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  let animationFrame = 0;
  let particles = [];

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * ratio);
    canvas.height = Math.floor(window.innerHeight * ratio);
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
  }

  function seed() {
    const ratio = window.devicePixelRatio || 1;
    const count = mediaQuery.matches ? 18 : 64;
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.22 * ratio,
      vy: (Math.random() - 0.5) * 0.14 * ratio,
      radius: (Math.random() * 2.4 + 0.8) * ratio,
      warm: Math.random() > 0.55,
    }));
  }

  function drawParticle(particle) {
    context.beginPath();
    context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
    context.fillStyle = particle.warm
      ? 'rgba(255, 178, 93, 0.52)'
      : 'rgba(255, 211, 155, 0.36)';
    context.fill();
  }

  function drawLinks() {
    if (mediaQuery.matches) {
      return;
    }
    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        const maxDistance = 150 * (window.devicePixelRatio || 1);
        if (distance > maxDistance) {
          continue;
        }
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.strokeStyle = `rgba(255, 178, 93, ${0.18 * (1 - distance / maxDistance)})`;
        context.lineWidth = 1;
        context.stroke();
      }
    }
  }

  function tick() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach((particle) => {
      if (!mediaQuery.matches) {
        particle.x += particle.vx;
        particle.y += particle.vy;
        if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
        if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
      }
      drawParticle(particle);
    });
    drawLinks();
    if (!mediaQuery.matches) {
      animationFrame = window.requestAnimationFrame(tick);
    }
  }

  function redraw() {
    window.cancelAnimationFrame(animationFrame);
    resize();
    seed();
    tick();
  }

  redraw();
  window.addEventListener('resize', redraw);
  mediaQuery.addEventListener('change', redraw);
}

async function boot() {
  initParticles();
  bindEvents();
  bindLineEditorEvents();
  syncControls();
  await loadDashboard();
}

boot().catch((error) => {
  setError(error.message);
  showEmptyState('页面初始化失败', '静态资源已加载，但无法读取后端状态，请检查服务是否已启动。');
});
