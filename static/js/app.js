/* =====================================================
   The Meter Whisperer — frontend logic
   Talks to the Flask backend on the same origin.
   ===================================================== */

const API_BASE = "";
const SR = 100;          // sampling rate used for the simulated demo
const DURATION = 8;      // seconds
const N = SR * DURATION; // 800 samples

// UI scenarios -> backend classes supported by the current trained model.
const SCENARIOS = {
  normal:   { label: "Normal",          backend: "normal",               color: "#9fb0c0", rgba: "rgba(159,176,192,0.85)" },
  tap:      { label: "Tap",             backend: "tap_water_leak",       color: "#7fd1c0", rgba: "rgba(127,209,192,0.85)" },
  washing:  { label: "Washing machine", backend: "washing_machine_leak", color: "#6ea8d8", rgba: "rgba(110,168,216,0.85)" },
  toilet:   { label: "Toilet",          backend: "toilet_leak",          color: "#e07878", rgba: "rgba(224,120,120,0.85)" },
  pipe:     { label: "Pipe / fitting",  backend: "pipe_fitting",         color: "#f5a65b", rgba: "rgba(245,166,91,0.85)" },
};

const SEEDS = { normal: 11, tap: 22, washing: 33, toilet: 44, pipe: 55 };

// =====================================================
// STATE
// =====================================================
let currentScenario = "normal";
let currentMode = "raw";
let currentWaveform = null;   // { time_s, acc_x_m_s2, acc_y_m_s2, acc_z_m_s2 }
let currentMagnitude = null;  // number[]
let recordingClasses = {};    // class -> [filenames]
let suppressRecordingEvents = false;

const $ = (id) => document.getElementById(id);

function scenarioForBackend(className) {
  return Object.keys(SCENARIOS).find((key) => SCENARIOS[key].backend === className);
}

function currentColor() {
  return (SCENARIOS[currentScenario] || SCENARIOS.normal).color;
}

function currentRgba() {
  return (SCENARIOS[currentScenario] || SCENARIOS.normal).rgba;
}

// =====================================================
// DETERMINISTIC PRNG (mulberry32)
// =====================================================
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const round6 = (v) => Math.round(v * 1e6) / 1e6;

// =====================================================
// WAVEFORM GENERATION (simulated sensor)
// =====================================================
function envelope(key, i, time, rnd) {
  switch (key) {
    case "normal":
      return 0.008 + rnd() * 0.004;
    case "tap":
      // quiet baseline with occasional short high-energy bursts
      return (i % 200 < 40) ? 0.05 + rnd() * 0.02 : 0.008 + rnd() * 0.004;
    case "washing":
      // cyclical activity (drum rotation)
      return 0.03 + 0.025 * Math.sin(2 * Math.PI * time * 0.25) + rnd() * 0.006;
    case "toilet":
      // persistent repeated water movement (leak)
      return 0.045 + 0.02 * Math.sin(2 * Math.PI * time * 0.5) + rnd() * 0.008;
    case "pipe":
      // irregular mechanical pattern
      return (rnd() > 0.75) ? 0.06 + rnd() * 0.02 : 0.012 + rnd() * 0.006;
    default:
      return 0.01;
  }
}

const FREQ = { normal: 9, tap: 18, washing: 30, toilet: 20, pipe: 24 };

function generateWaveform(key) {
  const rnd = mulberry32(SEEDS[key]);
  const dt = 1 / SR;
  const time_s = [], x = [], y = [], z = [];

  for (let i = 0; i < N; i++) {
    const time = i * dt;
    time_s.push(round6(time));
    const env = envelope(key, i, time, rnd);
    const f = FREQ[key];
    const noise = rnd() - 0.5;
    x.push(round6(env * Math.sin(2 * Math.PI * f * time) + noise * env));
    y.push(round6(env * Math.sin(2 * Math.PI * f * time + 2.1) + noise * env * 0.8));
    z.push(round6(env * Math.sin(2 * Math.PI * f * time + 4.2) + noise * env * 0.9));
  }

  return { time_s, acc_x_m_s2: x, acc_y_m_s2: y, acc_z_m_s2: z };
}

function magnitudeOf(w) {
  const m = [];
  for (let i = 0; i < w.acc_x_m_s2.length; i++) {
    m.push(Math.sqrt(
      w.acc_x_m_s2[i] ** 2 +
      w.acc_y_m_s2[i] ** 2 +
      w.acc_z_m_s2[i] ** 2
    ));
  }
  return m;
}

// =====================================================
// SIMPLE SPECTRUM (DFT for a small set of bins)
// =====================================================
function dftBars(signal, bins = 48) {
  const n = signal.length;
  const step = Math.max(1, Math.floor(n / 400)); // downsample for speed
  const ds = [];
  for (let i = 0; i < n; i += step) ds.push(signal[i]);

  const out = [];
  const m = ds.length;
  for (let b = 0; b < bins; b++) {
    const f = (b / bins) * (SR / 2);
    let re = 0, im = 0;
    for (let i = 0; i < m; i++) {
      const angle = 2 * Math.PI * f * (i / SR);
      re += ds[i] * Math.cos(angle);
      im -= ds[i] * Math.sin(angle);
    }
    out.push(Math.sqrt(re * re + im * im) / m);
  }
  return out;
}

// =====================================================
// CANVAS RENDERING
// =====================================================
const CAPTIONS = {
  raw: "raw acceleration",
  processed: "processed signal",
  frequency: "frequency signature",
  interpretation: "AI interpretation",
};

function draw() {
  const canvas = $("wave-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // subtle grid
  ctx.strokeStyle = "rgba(34,48,63,0.5)";
  ctx.lineWidth = 1;
  for (let gy = 0; gy <= 4; gy++) {
    const y = (H / 4) * gy;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  if (!currentWaveform) return;

  if (currentMode === "raw" || currentMode === "processed") {
    const data = currentMode === "raw" ? currentWaveform.acc_x_m_s2 : currentMagnitude;
    const stroke = currentColor();
    drawLine(ctx, data, W, H, stroke);
  } else if (currentMode === "frequency") {
    const bars = dftBars(currentMagnitude);
    const max = Math.max(...bars, 1e-9);
    const bw = W / bars.length;
    ctx.fillStyle = currentRgba();
    bars.forEach((v, i) => {
      const h = (v / max) * (H - 16);
      ctx.fillRect(i * bw + 1, H - h, bw - 2, h);
    });
  } else {
    // interpretation: envelope bars
    const bars = downSample(currentMagnitude, 16);
    const max = Math.max(...bars, 1e-9);
    const bw = W / bars.length;
    bars.forEach((v, i) => {
      const h = (v / max) * (H - 16);
      const y = H - h;
      ctx.fillStyle = currentRgba();
      ctx.fillRect(i * bw + 3, y, bw - 6, h);
    });
  }
}

function drawLine(ctx, data, W, H, color) {
  const max = Math.max(...data.map(Math.abs), 1e-9);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H / 2 - (v / max) * (H / 2 - 12);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function downSample(data, bins) {
  const out = [];
  const step = Math.max(1, Math.floor(data.length / bins));
  for (let i = 0; i < data.length; i += step) {
    out.push(data[i]);
    if (out.length === bins) break;
  }
  while (out.length < bins) out.push(out[out.length - 1] || 0);
  return out;
}

// =====================================================
// API CLIENT
// =====================================================
async function api(path, options) {
  const res = await fetch(API_BASE + path, options);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

async function checkHealth() {
  const el = $("api-status");
  try {
    const { ok, data } = await api("/api/health");
    if (ok && data.status === "online") {
      el.textContent = "● online";
      el.className = "api-status online";
      $("demo-disclaimer").textContent =
        "Live inference — results come from the trained model.";
      return true;
    }
    throw new Error("unhealthy");
  } catch (err) {
    el.textContent = "● offline";
    el.className = "api-status offline";
    $("demo-disclaimer").textContent =
      "Backend unreachable. Start it with `python3 app.py`.";
    return false;
  }
}

async function predict(payload) {
  const { ok, status, data } = await api("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return { ok, status, data };
}

async function loadRecordings() {
  const { ok, data } = await api("/api/recordings");
  if (!ok) return;
  recordingClasses = data.recordings || {};
  const classSel = $("recording-class");
  classSel.innerHTML = "";
  Object.entries(SCENARIOS).forEach(([scenarioKey, sc]) => {
    if (!recordingClasses[sc.backend]) return;
    const opt = document.createElement("option");
    opt.value = sc.backend;
    opt.textContent = sc.label;
    opt.dataset.scenario = scenarioKey;
    classSel.appendChild(opt);
  });

  selectScenarioRecording(currentScenario, { autoLoad: true });
}

function populateFiles(selectedFile = null) {
  const cls = $("recording-class").value;
  const fileSel = $("recording-file");
  fileSel.innerHTML = "";
  (recordingClasses[cls] || []).forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f;
    fileSel.appendChild(opt);
  });

  if (selectedFile && recordingClasses[cls]?.includes(selectedFile)) {
    fileSel.value = selectedFile;
  } else if (fileSel.options.length > 0) {
    fileSel.selectedIndex = 0;
  }

  return fileSel.value;
}

function selectScenarioRecording(key, { autoLoad = true } = {}) {
  const sc = SCENARIOS[key];
  if (!sc || !recordingClasses[sc.backend]) return false;

  const classSel = $("recording-class");
  suppressRecordingEvents = true;
  classSel.value = sc.backend;
  const file = populateFiles();
  suppressRecordingEvents = false;

  if (autoLoad && file) {
    loadRecording(sc.backend, file);
  }

  return Boolean(file);
}

async function loadRecording(cls, file) {
  const { ok, data } = await api(`/api/recording/${cls}/${file}`);
  if (!ok) {
    renderError(data.error || "Could not load recording.");
    return;
  }
  currentWaveform = {
    time_s: data.time_s,
    acc_x_m_s2: data.acc_x_m_s2,
    acc_y_m_s2: data.acc_y_m_s2,
    acc_z_m_s2: data.acc_z_m_s2,
  };
  currentMagnitude = magnitudeOf(currentWaveform);
  const scenarioKey = scenarioForBackend(cls);
  if (scenarioKey) {
    currentScenario = scenarioKey;
    renderScenarioButtons();
  }
  $("lab-status-text").textContent = `Real recording · ${file}`;
  runPrediction(currentWaveform);
  draw();
}


// =====================================================
// RENDERING
// =====================================================
function renderScenarioButtons() {
  const container = $("activity-options");
  container.innerHTML = "";
  Object.entries(SCENARIOS).forEach(([key, sc]) => {
    const btn = document.createElement("button");
    btn.className = key === currentScenario ? "selected" : "";
    btn.style.setProperty("--activity-color", sc.color);
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(key === currentScenario));
    btn.innerHTML = `<span class="activity-radio"></span>${sc.label}`;
    btn.addEventListener("click", () => setScenario(key));
    container.appendChild(btn);
  });
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".view-tab").forEach((t) => {
    t.classList.toggle("selected", t.dataset.mode === mode);
  });
  $("display-caption").textContent = CAPTIONS[mode];
  draw();
}

function setScenario(key) {
  currentScenario = key;
  renderScenarioButtons();

  if (selectScenarioRecording(key, { autoLoad: true })) {
    return;
  }

  // Fallback while recordings are still loading or if the backend is offline.
  currentWaveform = generateWaveform(key);
  currentMagnitude = magnitudeOf(currentWaveform);
  $("lab-status-text").textContent = `Simulated sensor window · ${DURATION.toFixed(1)}s`;
  draw();
  runPrediction(currentWaveform);
}

function renderPatternBars(values) {
  const el = $("pattern-bars");
  el.innerHTML = "";
  el.style.setProperty("--pattern-color", currentColor());
  const max = Math.max(...values, 1e-9);
  values.forEach((v) => {
    const span = document.createElement("span");
    span.style.height = `${Math.max(4, (v / max) * 90)}px`;
    el.appendChild(span);
  });
}

function renderInterpretation(data) {
  const output = $("demo-output");
  const tone = data.tone || "quiet";
  output.className = `demo-output tone-${tone}`;

  $("output-label").textContent = data.source || data.predicted_class || "—";
  $("output-severity").textContent = `Severity: ${data.severity || "—"}`;
  $("output-confidence").textContent =
    `Confidence: ${data.confidence_percent != null ? data.confidence_percent + "%" : "—"}`;
  $("output-rms").textContent =
    `RMS: ${data.average_vibration_rms != null ? data.average_vibration_rms : "—"}`;
  $("output-explanation").textContent = data.explanation || "";
  $("output-action").textContent = "";

  // fallback pattern bars from the current magnitude
  if (currentMagnitude) renderPatternBars(downSample(currentMagnitude, 16));
}

function renderError(message) {
  const output = $("demo-output");
  output.className = "demo-output tone-red";
  $("output-label").textContent = "Error";
  $("output-severity").textContent = "Severity: —";
  $("output-confidence").textContent = "Confidence: —";
  $("output-rms").textContent = "RMS: —";
  $("output-explanation").textContent = message;
  $("output-action").textContent = "";
}

async function runPrediction(payload) {
  // mark "thinking" state
  $("output-label").textContent = "Listening…";
  $("output-severity").textContent = "Severity: —";
  $("output-confidence").textContent = "Confidence: —";
  $("output-rms").textContent = "RMS: —";
  $("output-explanation").textContent = "";
  $("output-action").textContent = "";

  const { ok, data } = await predict(payload);

  if (!ok) {
    renderError(data.error || "Prediction failed.");
    return;
  }

  if (data.success) {
    renderInterpretation(data);
  } else {
    renderError(data.error || "Prediction failed.");
  }
}

// =====================================================
// INIT
// =====================================================
function init() {
  renderScenarioButtons();

  document.querySelectorAll(".view-tab").forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  });

  $("recording-class").addEventListener("change", () => {
    if (suppressRecordingEvents) return;
    const cls = $("recording-class").value;
    const scenarioKey = scenarioForBackend(cls);
    if (scenarioKey) {
      currentScenario = scenarioKey;
      renderScenarioButtons();
    }
    const file = populateFiles();
    if (cls && file) loadRecording(cls, file);
  });

  $("recording-file").addEventListener("change", () => {
    if (suppressRecordingEvents) return;
    const cls = $("recording-class").value;
    const file = $("recording-file").value;
    if (cls && file) loadRecording(cls, file);
  });

  // start with the default scenario and check backend health
  checkHealth();
  setScenario(currentScenario);
  loadRecordings();
}

document.addEventListener("DOMContentLoaded", init);

