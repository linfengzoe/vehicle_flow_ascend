const state = {
  loaded: false,
  payload: null,
};

const byId = (id) => document.getElementById(id);

const escapeHtml = (value) => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#39;');

const formatValue = (value) => {
  if (value === null || value === undefined || value === '') return '未配置';
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const quotePowerShell = (value) => {
  const text = String(value ?? '');
  return `"${text.replaceAll('`', '``').replaceAll('"', '`"')}"`;
};

async function loadDashboard() {
  const response = await fetch('/api/dashboard', { cache: 'no-store' });
  if (!response.ok) throw new Error(`API failed: ${response.status}`);
  return response.json();
}

function renderStatus(payload) {
  const runtime = payload.runtime;
  const outputExists = runtime.output_exists;
  const status = byId('runtimeStatus');
  status.innerHTML = `
    <span class="status-dot"></span>
    <div>
      <strong>${outputExists ? 'OUTPUT VIDEO READY' : 'CONFIG ONLINE'}</strong>
      <small>${escapeHtml(runtime.backend)} · ${escapeHtml(runtime.model_path || 'no model path')}</small>
    </div>
  `;
  byId('backendChip').textContent = runtime.backend;
}

function outputVideoPath() {
  return byId('outputVideoInput')?.value?.trim() || state.payload?.runtime?.output_video || '';
}

async function outputExists(path) {
  if (!path) return false;
  const response = await fetch(`/api/media-status?path=${encodeURIComponent(path)}`, { cache: 'no-store' });
  if (!response.ok) return false;
  const data = await response.json();
  return Boolean(data.exists);
}

async function renderVideo(payload, { preferFormPath = false } = {}) {
  const stage = byId('videoStage');
  const runtime = payload.runtime;
  const selectedOutput = preferFormPath ? outputVideoPath() : runtime.output_video;
  const exists = preferFormPath ? await outputExists(selectedOutput) : runtime.output_exists;

  if (!exists) {
    if (stage.querySelector('video')) {
      stage.innerHTML = placeholderMarkup();
    }
    byId('videoTitle').textContent = '等待输出视频';
    byId('videoHint').textContent = `运行演示后生成 ${selectedOutput || 'runs/*.mp4'}，点击“刷新预览”即可展示。`;
    return;
  }

  stage.innerHTML = `
    <video src="/media/output-video?path=${encodeURIComponent(selectedOutput)}&t=${Date.now()}" controls autoplay muted loop playsinline></video>
    <div class="placeholder-copy video-caption">
      <strong>输出视频已就绪</strong>
      <span>${escapeHtml(selectedOutput)}</span>
    </div>
  `;
}

function placeholderMarkup() {
  return `
    <div class="road-grid" aria-hidden="true">
      <span class="count-line"></span>
      <span class="vehicle vehicle-a"></span>
      <span class="vehicle vehicle-b"></span>
      <span class="vehicle vehicle-c"></span>
    </div>
    <div class="placeholder-copy">
      <strong id="videoTitle">等待输出视频</strong>
      <span id="videoHint">运行 PC 或昇腾演示后，将输出视频写入 runs/。</span>
    </div>
  `;
}

function renderClasses(payload) {
  byId('classList').innerHTML = payload.classes
    .map((item) => `
      <div class="class-item" style="--class-color: ${item.accent}">
        <strong>${escapeHtml(item.label)}</strong>
        <small>${escapeHtml(item.key.toUpperCase())} / COCO mapped vehicle class</small>
      </div>
    `)
    .join('');
}

function renderPipeline(payload) {
  byId('pipelineList').innerHTML = payload.pipeline
    .map((step) => `<li>${escapeHtml(step)}</li>`)
    .join('');
}

function renderConfig(payload) {
  const runtime = payload.runtime;
  const metrics = payload.metrics;
  const config = payload.config;
  const rows = [
    ['source', runtime.source],
    ['backend', runtime.backend],
    ['model', runtime.model_path],
    ['soc', runtime.soc_version],
    ['input', metrics.input_size],
    ['conf', metrics.confidence_threshold],
    ['iou', metrics.iou_threshold],
    ['line', config.line],
    ['output', runtime.output_video],
    ['fps target', metrics.fps_target],
  ];
  byId('configList').innerHTML = rows
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(formatValue(value))}</dd>`)
    .join('');
}

function hydrateForm(payload) {
  const config = payload.config;
  const runtime = payload.runtime;
  byId('sourceInput').value = runtime.source === null || runtime.source === undefined ? '' : formatValue(runtime.source);
  byId('backendSelect').value = runtime.backend || 'torch_yolov5';
  byId('modelPathInput').value = runtime.model_path || '';
  byId('outputVideoInput').value = runtime.output_video || '';
  byId('imageSizeInput').value = config.image_size ?? '';
  byId('socVersionInput').value = runtime.soc_version || '';
  byId('confidenceInput').value = config.confidence_threshold ?? '';
  byId('iouInput').value = config.iou_threshold ?? '';
  byId('maxFramesInput').value = config.max_frames ?? '';
  byId('displaySelect').value = String(Boolean(config.display));
  byId('lineStartInput').value = Array.isArray(config.line) ? config.line[0].join(',') : '';
  byId('lineEndInput').value = Array.isArray(config.line) ? config.line[1].join(',') : '';
}

function formValues() {
  return {
    source: byId('sourceInput').value.trim(),
    backend: byId('backendSelect').value,
    modelPath: byId('modelPathInput').value.trim(),
    outputVideo: byId('outputVideoInput').value.trim(),
    imageSize: byId('imageSizeInput').value.trim(),
    socVersion: byId('socVersionInput').value.trim(),
    confidence: byId('confidenceInput').value.trim(),
    iou: byId('iouInput').value.trim(),
    maxFrames: byId('maxFramesInput').value.trim(),
    display: byId('displaySelect').value,
    lineStart: byId('lineStartInput').value.trim(),
    lineEnd: byId('lineEndInput').value.trim(),
  };
}

function buildCommand() {
  const values = formValues();
  const configPath = values.backend === 'ascend_om' ? 'configs/ascend_om.yaml' : 'configs/pc_demo.yaml';
  const parts = ['python', '-m', 'vehicle_flow_ascend', '--config', quotePowerShell(configPath)];
  if (values.source) parts.push('--source', quotePowerShell(values.source));
  if (values.backend) parts.push('--backend', quotePowerShell(values.backend));
  if (values.modelPath) parts.push('--model-path', quotePowerShell(values.modelPath));
  if (values.outputVideo) parts.push('--output-video', quotePowerShell(values.outputVideo));
  if (values.imageSize) parts.push('--image-size', values.imageSize);
  if (values.socVersion) parts.push('--soc-version', quotePowerShell(values.socVersion));
  if (values.confidence) parts.push('--confidence-threshold', values.confidence);
  if (values.iou) parts.push('--iou-threshold', values.iou);
  if (values.maxFrames) parts.push('--max-frames', values.maxFrames);
  if (values.display) parts.push('--display', values.display);
  if (values.lineStart && values.lineEnd) {
    parts.push('--line-start', quotePowerShell(values.lineStart));
    parts.push('--line-end', quotePowerShell(values.lineEnd));
  }
  return parts.join(' ');
}

function renderCommand() {
  byId('commandText').textContent = buildCommand();
  byId('copyStatus').textContent = '已更新';
}

async function copyCommand() {
  const command = byId('commandText').textContent;
  try {
    await navigator.clipboard.writeText(command);
    byId('copyStatus').textContent = '已复制';
  } catch {
    byId('copyStatus').textContent = '复制失败，请手动选择';
  }
}

function bindInteractions() {
  const form = byId('controlForm');
  form.addEventListener('input', renderCommand);
  form.addEventListener('change', renderCommand);
  byId('generateButton').addEventListener('click', renderCommand);
  byId('copyButton').addEventListener('click', copyCommand);
  byId('refreshPreviewButton').addEventListener('click', async () => {
    if (state.payload) await renderVideo(state.payload, { preferFormPath: true });
  });
}

async function boot() {
  try {
    const payload = await loadDashboard();
    state.payload = payload;
    renderStatus(payload);
    renderClasses(payload);
    renderPipeline(payload);
    renderConfig(payload);
    if (!state.loaded) {
      hydrateForm(payload);
      bindInteractions();
      renderCommand();
    }
    await renderVideo(payload, { preferFormPath: state.loaded });
    state.loaded = true;
  } catch (error) {
    byId('runtimeStatus').innerHTML = `
      <span class="status-dot" style="background: var(--danger); box-shadow: 0 0 22px var(--danger)"></span>
      <div>
        <strong>DASHBOARD API ERROR</strong>
        <small>${escapeHtml(error.message)}</small>
      </div>
    `;
  }
}

boot();
setInterval(boot, 5000);
