"use strict";

let selectedFile = null;
let datasetValid = false;
let pollTimer = null;
let activeJobId = null;
let liveMode = false; // true while watching a running job (enables fail popups)
let charts = {}; // canvasId -> Chart instance
let loadedArtifacts = new Set(); // run_ids whose artifacts were already fetched

const $ = (id) => document.getElementById(id);

const THEME = { text: "#e6e9ef", muted: "#8b97a7", grid: "rgba(255,255,255,0.07)" };
const PALETTE = ["#4f8cff", "#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22", "#e84393"];

function showModal(title, body) {
  $("modalTitle").textContent = title;
  $("modalBody").textContent = body;
  $("modal").classList.remove("hidden");
}

function destroyCharts() {
  Object.values(charts).forEach((c) => c.destroy());
  charts = {};
}

// --- 1. Upload + validation -------------------------------------------------
const dropzone = $("dropzone");
const fileInput = $("fileInput");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  selectedFile = file;
  dropzone.querySelector("p").innerHTML = `<strong>${file.name}</strong> — validating…`;
  const fd = new FormData();
  fd.append("file", file);
  let data;
  try {
    data = await fetch("/api/validate", { method: "POST", body: fd }).then((r) => r.json());
  } catch (err) {
    return showModal("Network error", String(err));
  }
  const box = $("validation");
  box.classList.remove("hidden", "ok", "err");
  if (data.ok) {
    box.classList.add("ok");
    box.textContent = `✓ Valid dataset — ${data.rows} rows` + (data.stays ? `, ${data.stays} stays` : "");
    datasetValid = true;
    $("trainBtn").disabled = false;
  } else {
    box.classList.add("err");
    box.textContent = "✗ " + data.error;
    datasetValid = false;
    $("trainBtn").disabled = true;
    showModal("Dataset validation failed", data.error);
  }
  dropzone.querySelector("p").innerHTML = `<strong>${file.name}</strong> — click to choose another`;
}

// --- 2. Launch training -----------------------------------------------------
$("trainBtn").addEventListener("click", async () => {
  if (!selectedFile) return;
  const types = [...document.querySelectorAll('input[name="retrain"]:checked')].map((c) => c.value);
  if (!types.length) return showModal("Pick a retrain type", "Select at least one retrain type.");

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("retrain_types", types.join(","));
  fd.append("epochs", $("epochs").value);
  fd.append("batch_size", $("batch_size").value);
  fd.append("learning_rate", $("learning_rate").value);
  fd.append("seed", $("seed").value);

  $("trainBtn").disabled = true;
  let job;
  try {
    const res = await fetch("/api/train", { method: "POST", body: fd });
    job = await res.json();
    if (!res.ok) throw new Error(job.detail || "Failed to start job");
  } catch (err) {
    $("trainBtn").disabled = false;
    return showModal("Could not start training", String(err.message || err));
  }
  openJob(job.id, true);
});

$("cancelBtn").addEventListener("click", async () => {
  const id = $("cancelBtn").dataset.job;
  if (id) await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
});

// --- History: list, reopen, delete -----------------------------------------
$("refreshHistory").addEventListener("click", loadHistory);

async function loadHistory() {
  let jobs;
  try {
    jobs = await fetch("/api/jobs").then((r) => r.json());
  } catch {
    return;
  }
  const ul = $("jobList");
  if (!jobs.length) {
    ul.innerHTML = '<li class="muted">No runs yet.</li>';
    return;
  }
  ul.innerHTML = "";
  jobs.forEach((job) => {
    const li = document.createElement("li");
    li.className = "job-item" + (job.id === activeJobId ? " active" : "");
    li.addEventListener("click", () => openJob(job.id, job.status === "running"));
    const when = job.created_at.replace("T", " ").replace("+00:00", " UTC");

    const left = document.createElement("div");
    left.className = "ji-left";
    left.innerHTML =
      `<span class="badge ${job.status}">${job.status}</span>` +
      `<span>${job.params.data_filename} · [${job.params.retrain_types.join(", ")}]</span>`;

    const right = document.createElement("div");
    right.className = "ji-right";
    right.textContent = `${when} · ${job.params.epochs} epochs · lr ${job.params.learning_rate}`;

    const del = document.createElement("button");
    del.className = "ghost del";
    del.title = "Delete run";
    del.textContent = "🗑";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteJob(job.id);
    });

    li.append(left, right, del);
    ul.appendChild(li);
  });
}

async function deleteJob(jobId) {
  if (!confirm("Delete this run from the history and from MLflow? This cannot be undone.")) return;
  await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
  if (jobId === activeJobId) {
    activeJobId = null;
    if (pollTimer) clearInterval(pollTimer);
    ["jobCard", "metricsCard", "resultsCard"].forEach((id) => $(id).classList.add("hidden"));
  }
  loadHistory();
}

// Open a job (fresh or historical). `live` starts polling on an interval.
function openJob(jobId, live) {
  activeJobId = jobId;
  liveMode = !!live;
  $("jobCard").classList.remove("hidden");
  $("metricsCard").classList.remove("hidden");
  $("resultsCard").classList.add("hidden");
  $("cancelBtn").dataset.job = jobId;
  $("cancelBtn").disabled = !live;
  destroyCharts();
  loadedArtifacts = new Set();
  $("charts").innerHTML = "";
  $("results").innerHTML = "";
  $("steps").innerHTML = "";
  $("log").textContent = "";
  if (pollTimer) clearInterval(pollTimer);
  poll(jobId);
  if (live) pollTimer = setInterval(() => poll(jobId), 2500);
  loadHistory();
}

async function poll(jobId) {
  const [job, logData, metricData] = await Promise.all([
    fetch(`/api/jobs/${jobId}`).then((r) => r.json()),
    fetch(`/api/jobs/${jobId}/log`).then((r) => r.json()),
    fetch(`/api/jobs/${jobId}/metrics`).then((r) => r.json()),
  ]);
  if (jobId !== activeJobId) return; // a newer openJob() superseded this poll

  renderStatus(job);
  renderLog(logData.lines);
  renderCharts(metricData);
  renderResults(metricData); // live: results/artifacts fill in as runs finish

  const finished = ["succeeded", "failed", "cancelled", "interrupted"].includes(job.status);
  if (finished) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    $("cancelBtn").disabled = true;
    $("trainBtn").disabled = !datasetValid; // re-enable so you can launch again
    if (liveMode && job.status === "failed") showModal("Training failed", job.error || "See logs.");
    liveMode = false;
    loadHistory();
  }
}

function renderStatus(job) {
  const badge = $("jobStatus");
  badge.textContent = job.status;
  badge.className = "badge " + job.status;
  const ol = $("steps");
  ol.innerHTML = "";
  job.steps.forEach((label, i) => {
    const li = document.createElement("li");
    li.textContent = label;
    if (job.status === "succeeded" || i < job.current_step) li.className = "done";
    else if (i === job.current_step && job.status === "running") li.className = "active";
    else li.className = "pending";
    ol.appendChild(li);
  });
}

function renderLog(lines) {
  const pre = $("log");
  pre.textContent = lines.join("\n");
  pre.scrollTop = pre.scrollHeight;
}

// --- 4. Live training curves (per run: loss | classification metrics) -------
function lineChartOpts(yTitle, legendPos = "top") {
  return {
    animation: false,
    parsing: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    scales: {
      x: { type: "linear", title: { display: true, text: "epoch", color: THEME.muted },
           ticks: { color: THEME.muted }, grid: { color: THEME.grid } },
      y: { title: { display: true, text: yTitle, color: THEME.muted },
           ticks: { color: THEME.muted }, grid: { color: THEME.grid } },
    },
    plugins: {
      // pointStyle:"line" draws a line sample (not a square) in the legend and
      // reflects each dataset's dash, so dashed (val_*) vs solid (train) shows.
      legend: { position: legendPos, align: "center",
                labels: { color: THEME.text, usePointStyle: true, pointStyle: "line",
                          boxWidth: 26, padding: 12 } },
      tooltip: { mode: "index", intersect: false },
    },
  };
}

// Pair each metric with its val_ counterpart: same colour, train solid /
// validation dashed. Avoids the "circus" of one colour per line.
function lineDatasets(history, keys) {
  const name = (k) => k.split("/").pop();           // strip "mort/" / "disch/"
  const base = (k) => name(k).replace(/^val_/, ""); // metric without val_ prefix
  const bases = [...new Set(keys.map(base))];
  // Group train+val of the same metric together, train before val.
  const ordered = [...keys].sort((a, b) =>
    base(a) === base(b)
      ? (name(a).startsWith("val_") ? 1 : 0) - (name(b).startsWith("val_") ? 1 : 0)
      : bases.indexOf(base(a)) - bases.indexOf(base(b))
  );
  return ordered.map((k) => {
    const isVal = name(k).startsWith("val_");
    const color = PALETTE[bases.indexOf(base(k)) % PALETTE.length];
    return {
      label: name(k),
      data: history[k].map((p) => ({ x: p.step, y: p.value })),
      borderColor: color,
      backgroundColor: "transparent",
      borderWidth: 2,
      borderDash: isVal ? [6, 4] : [],
      pointRadius: 0,
      tension: 0.25,
    };
  });
}

function upsertChart(id, type, data, options) {
  if (charts[id]) {
    charts[id].data = data;
    charts[id].update("none");
  } else {
    charts[id] = new Chart($(id), { type, data, options });
  }
}

// Idempotent tab: returns the panel for `key`, creating button+panel once.
// Existing tabs/selection are never destroyed, so live updates don't reset
// which tab the user is looking at.
function ensureTab(bar, panels, key, label) {
  const panId = panels.id + "--" + key;
  let panel = document.getElementById(panId);
  if (panel) return panel;
  const first = bar.children.length === 0;
  const btn = document.createElement("button");
  btn.className = "tab" + (first ? " active" : "");
  btn.textContent = label;
  panel = document.createElement("div");
  panel.id = panId;
  panel.className = "tab-panel" + (first ? "" : " hidden");
  btn.addEventListener("click", () => {
    [...bar.children].forEach((b) => b.classList.remove("active"));
    [...panels.children].forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    panel.classList.remove("hidden");
    Object.values(charts).forEach((c) => c.resize()); // fix 0-size in hidden panels
  });
  bar.appendChild(btn);
  panels.appendChild(panel);
  return panel;
}

function ensureTabSkeleton(host, id) {
  let bar = document.getElementById(id + "-bar");
  if (!bar) {
    host.innerHTML = `<div id="${id}-bar" class="tab-bar"></div><div id="${id}-panels" class="tab-panels"></div>`;
    bar = document.getElementById(id + "-bar");
  }
  return [bar, document.getElementById(id + "-panels")];
}

function modelLabel(run) {
  if (run.step === "mortality") return "Mortality";
  if (run.step === "discharge") return "Discharge";
  return run.run_name;
}

// Group runs by retrain_type, preserving first-seen order.
function groupByType(runs, keep) {
  const order = [];
  const groups = {};
  for (const run of runs) {
    if (!keep(run)) continue;
    const rt = run.retrain_type || "—";
    if (!groups[rt]) { groups[rt] = []; order.push(rt); }
    groups[rt].push(run);
  }
  return { order, groups };
}

function renderCharts(metricData) {
  const host = $("charts");
  if (!metricData.mlflow_enabled) {
    host.innerHTML = '<p class="badge warn">MLflow is OFF — no live metrics. Models still train and are saved to disk.</p>';
    return;
  }
  const [bar, panels] = ensureTabSkeleton(host, "mt");
  const { order, groups } = groupByType(metricData.runs, (r) => Object.keys(r.history || {}).length);
  for (const rt of order) {
    const panel = ensureTab(bar, panels, rt, rt.toUpperCase());
    for (const run of groups[rt]) {
      const keys = Object.keys(run.history);
      const lossKeys = keys.filter((k) => ["loss", "val_loss"].includes(k.split("/").pop()));
      const metricKeys = keys.filter((k) => !["loss", "val_loss"].includes(k.split("/").pop()));
      const lossId = "loss-" + run.run_id;
      const metId = "met-" + run.run_id;
      if (!$(lossId)) {
        const block = document.createElement("div");
        block.className = "run-block sub";
        block.innerHTML =
          `<h4>${modelLabel(run)}</h4>` +
          `<div class="chart-row">` +
          `<div class="chart-col"><canvas id="${lossId}"></canvas></div>` +
          `<div class="chart-col"><canvas id="${metId}"></canvas></div>` +
          `</div>`;
        panel.appendChild(block);
      }
      if (lossKeys.length) upsertChart(lossId, "line", { datasets: lineDatasets(run.history, lossKeys) }, lineChartOpts("loss", "top"));
      if (metricKeys.length) upsertChart(metId, "line", { datasets: lineDatasets(run.history, metricKeys) }, lineChartOpts("score", "right"));
    }
  }
}

// --- 5. Final metrics (bar charts) + artifacts (with image preview) ---------
const CLS_METRICS = ["auc", "f1", "precision", "recall"];

function classificationBars(metrics) {
  const strip = (k) => k.replace(/^test\//, "").replace(/^inf\/[^/]+\//, "");
  const get = (grp, met) => {
    for (const k in metrics) if (strip(k) === `${grp}_${met}`) return metrics[k];
    return null;
  };
  const mort = CLS_METRICS.map((m) => get("mort", m));
  const disch = CLS_METRICS.map((m) => get("disch", m));
  if (mort.every((v) => v == null) && disch.every((v) => v == null)) return null;
  return { mort, disch };
}

function barChartOpts() {
  return {
    animation: false,
    maintainAspectRatio: false,
    scales: {
      x: { ticks: { color: THEME.text }, grid: { display: false } },
      y: { min: 0, max: 1, ticks: { color: THEME.muted }, grid: { color: THEME.grid } },
    },
    plugins: { legend: { position: "right", labels: { color: THEME.text, usePointStyle: true, boxWidth: 8, padding: 12 } } },
  };
}

function renderResults(metricData) {
  $("resultsCard").classList.remove("hidden");
  const container = $("results");
  container.innerHTML = "";
  if (!metricData.mlflow_enabled) {
    container.innerHTML = '<p class="badge warn">MLflow OFF — results are on disk under results/&lt;retrain_type&gt;/ and models/.</p>';
    return;
  }
  const hasFinal = (r) => Object.keys(r.metrics).some((k) => !k.startsWith("mort/") && !k.startsWith("disch/"));
  const { order, groups } = groupByType(metricData.runs, hasFinal);
  if (!order.length) {
    container.innerHTML = '<p class="muted">No results yet.</p>';
    return;
  }

  // One tab per retrain type; each panel holds that model's metrics + inference.
  const bar = document.createElement("div");
  bar.className = "tab-bar";
  const panels = document.createElement("div");
  panels.className = "tab-panels";
  container.append(bar, panels);

  order.forEach((rt, idx) => {
    const btn = document.createElement("button");
    btn.className = "tab" + (idx === 0 ? " active" : "");
    btn.textContent = rt.toUpperCase();
    const panel = document.createElement("div");
    panel.className = "tab-panel" + (idx === 0 ? "" : " hidden");
    btn.addEventListener("click", () => {
      bar.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      panels.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      panel.classList.remove("hidden");
      // Charts built inside a hidden panel render at size 0; fix on reveal.
      Object.values(charts).forEach((c) => c.resize());
    });
    bar.appendChild(btn);
    panels.appendChild(panel);
    groups[rt].forEach((run) => buildResultBlock(run, panel));
  });
}

async function buildResultBlock(run, parent) {
  const finalKeys = Object.keys(run.metrics).filter((k) => !k.startsWith("mort/") && !k.startsWith("disch/"));
  const block = document.createElement("div");
  block.className = "run-block";
  block.innerHTML = `<h3>${run.run_name} <span class="badge ${run.status === "FINISHED" ? "succeeded" : ""}">${run.status}</span></h3>`;
  parent.appendChild(block);

  // Grouped bar chart for classification metrics (mortality vs discharge).
  const bars = classificationBars(run.metrics);
  if (bars) {
    const id = "bars-" + run.run_id;
    const wrap = document.createElement("div");
    wrap.className = "chart-col bars";
    wrap.innerHTML = `<canvas id="${id}"></canvas>`;
    block.appendChild(wrap);
    upsertChart(id, "bar",
      {
        labels: CLS_METRICS.map((m) => m.toUpperCase()),
        datasets: [
          { label: "Mortality", data: bars.mort, backgroundColor: "#e74c3c" },
          { label: "Discharge", data: bars.disch, backgroundColor: "#4f8cff" },
        ],
      },
      barChartOpts());
  }

  // Full numeric table (collapsible).
  const det = document.createElement("details");
  let t = "<summary>All metrics</summary><table><tr><th>metric</th><th>value</th></tr>";
  finalKeys.sort().forEach((k) => {
    const v = run.metrics[k];
    t += `<tr><td>${k}</td><td>${v == null ? "—" : v.toFixed(4)}</td></tr>`;
  });
  det.innerHTML = t + "</table>";
  block.appendChild(det);

  await appendArtifacts(run.run_id, block);
}

const IMG_RE = /\.(png|jpe?g|gif|webp|svg)$/i;

async function appendArtifacts(runId, block, path = "") {
  let data;
  try {
    data = await fetch(`/api/runs/${runId}/artifacts?path=${encodeURIComponent(path)}`).then((r) => r.json());
  } catch {
    return;
  }
  let gallery = null;
  for (const art of data.artifacts) {
    if (art.is_dir) {
      await appendArtifacts(runId, block, art.path);
      continue;
    }
    const url = `/api/runs/${runId}/download?path=${encodeURIComponent(art.path)}`;
    if (IMG_RE.test(art.path)) {
      if (!gallery) {
        gallery = document.createElement("div");
        gallery.className = "gallery";
        block.appendChild(gallery);
      }
      const fig = document.createElement("figure");
      fig.innerHTML =
        `<a href="${url}" target="_blank" rel="noopener"><img src="${url}" loading="lazy" alt="${art.path}" /></a>` +
        `<figcaption><a class="dl" href="${url}" download>⬇ ${art.path.split("/").pop()}</a></figcaption>`;
      gallery.appendChild(fig);
    } else {
      const a = document.createElement("a");
      a.className = "dl";
      a.href = url;
      a.download = "";
      a.textContent = "⬇ " + art.path;
      a.style.display = "block";
      block.appendChild(a);
    }
  }
}

// --- on load: show any past runs -------------------------------------------
loadHistory();
