const state = {
  loaded: false,
};

const formatValue = (value) => {
  if (value === null || value === undefined || value === '') return '未配置';
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const byId = (id) => document.getElementById(id);

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
      <small>${runtime.backend} · ${runtime.model_path || 'no model path'}</small>
    </div>
  `;
  byId('backendChip').textContent = runtime.backend;
}

function renderVideo(payload) {
  const stage = byId('videoStage');
  const runtime = payload.runtime;
  if (!runtime.output_exists) {
    byId('videoTitle').textContent = '等待输出视频';
    byId('videoHint').textContent = `运行演示后生成 ${runtime.output_video || 'runs/*.mp4'}，页面会展示标注后视频。`;
    return;
  }

  stage.innerHTML = `
    <video src="/media/output-video" controls autoplay muted loop playsinline></video>
    <div class="placeholder-copy video-caption">
      <strong>输出视频已就绪</strong>
      <span>${runtime.output_video}</span>
    </div>
  `;
}

function renderClasses(payload) {
  byId('classList').innerHTML = payload.classes
    .map((item) => `
      <div class="class-item" style="--class-color: ${item.accent}">
        <strong>${item.label}</strong>
        <small>${item.key.toUpperCase()} / COCO mapped vehicle class</small>
      </div>
    `)
    .join('');
}

function renderPipeline(payload) {
  byId('pipelineList').innerHTML = payload.pipeline
    .map((step) => `<li>${step}</li>`)
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
    .map(([key, value]) => `<dt>${key}</dt><dd>${formatValue(value)}</dd>`)
    .join('');
}

async function boot() {
  try {
    const payload = await loadDashboard();
    renderStatus(payload);
    renderVideo(payload);
    renderClasses(payload);
    renderPipeline(payload);
    renderConfig(payload);
    state.loaded = true;
  } catch (error) {
    byId('runtimeStatus').innerHTML = `
      <span class="status-dot" style="background: var(--danger); box-shadow: 0 0 22px var(--danger)"></span>
      <div>
        <strong>DASHBOARD API ERROR</strong>
        <small>${error.message}</small>
      </div>
    `;
  }
}

boot();
setInterval(boot, 5000);
