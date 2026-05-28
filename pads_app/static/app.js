"use strict";

let selectedFile = null;
let datasetValid = false;
let pollTimer = null;
let activeJobId = null;
let liveMode = false; // true while watching a running job (enables fail popups)
let charts = {}; // canvasId -> Chart instance
let loadedArtifacts = new Set(); // run_ids whose artifacts were already fetched
let comparisonRenderedCount = 0; // finished inference runs already in the comparison

const $ = (id) => document.getElementById(id);

const THEME = { text: "#e6e9ef", muted: "#8b97a7", grid: "rgba(255,255,255,0.07)" };
const PALETTE = ["#4f8cff", "#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22", "#e84393"];

// FastAPI errors come back as {detail: "..."} or, for request-validation errors,
// {detail: [{loc, msg, ...}]}. Flatten either into a readable string so the popup
// never shows "[object Object]".
function detailToText(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
  return JSON.stringify(detail);
}

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
    if (data.dropped_stays > 0) {
      const warn = document.createElement("div");
      warn.className = "badge warn";
      warn.style.marginTop = "0.5rem";
      warn.textContent = `⚠ ${data.dropped_stays} stay(s) will be dropped: missing icu_expire_flag.`;
      box.appendChild(warn);
    }
    if (data.short_stays > 0) {
      const info = document.createElement("div");
      info.className = "badge info";
      info.style.marginTop = "0.5rem";
      info.textContent = `ℹ ${data.short_stays} stay(s) will be dropped: stay shorter than 48 h.`;
      box.appendChild(info);
    }
    if (data.final_stays != null) {
      const total = document.createElement("div");
      total.className = "badge ok";
      total.style.marginTop = "0.5rem";
      total.style.fontWeight = "600";
      total.textContent = `✓ ${data.final_stays} patient(s) will be used.`;
      box.appendChild(total);
    }
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

// Advanced options: a toggle reveals per-model LR / early stopping / normalizer
// source. When it's off we fall back to the basic learning rate for both models.
$("advancedToggle").addEventListener("change", (e) => {
  $("advancedFields").classList.toggle("hidden", !e.target.checked);
});

// --- 2. Launch training -----------------------------------------------------
$("trainBtn").addEventListener("click", async () => {
  if (!selectedFile) return;
  const types = [...document.querySelectorAll('input[name="retrain"]:checked')].map((c) => c.value);
  const evaluateOriginal = $("evaluateOriginal").checked;
  // Need something to do: at least one retrain type, or the baseline on its own.
  if (!types.length && !evaluateOriginal) {
    return showModal("Nothing to run", "Select at least one retrain type, or enable the baseline evaluation.");
  }

  const basicLR = $("learning_rate").value;
  const advanced = $("advancedToggle").checked;
  // Empty per-model LR fields reuse the basic learning rate.
  const lrMort = advanced ? ($("lr_mort").value || basicLR) : basicLR;
  const lrDisch = advanced ? ($("lr_disch").value || basicLR) : basicLR;
  const esp = advanced ? $("early_stopping_patience").value : 50;
  const normSrc = advanced ? $("normalizer_source").value : "fitted";

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("retrain_types", types.join(","));
  fd.append("epochs", $("epochs").value);
  fd.append("batch_size", $("batch_size").value);
  fd.append("learning_rate_mort", lrMort);
  fd.append("learning_rate_disch", lrDisch);
  fd.append("early_stopping_patience", esp);
  fd.append("normalizer_source", normSrc);
  fd.append("evaluate_original", evaluateOriginal);
  fd.append("seed", $("seed").value);

  $("trainBtn").disabled = true;
  let job;
  try {
    const res = await fetch("/api/train", { method: "POST", body: fd });
    job = await res.json();
    if (!res.ok) throw new Error(detailToText(job.detail) || "Failed to start job");
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
    // Show the baseline alongside any retrain types (it may be the only thing run).
    const labels = [...job.params.retrain_types];
    if (job.params.evaluate_original) labels.unshift("original");
    left.innerHTML =
      `<span class="badge ${job.status}">${job.status}</span>` +
      `<span>${job.params.data_filename} · [${labels.join(", ")}]</span>`;

    const right = document.createElement("div");
    right.className = "ji-right";
    // One LR if both models share it, else "mort/disch".
    const lrM = job.params.learning_rate_mort, lrD = job.params.learning_rate_disch;
    const lr = lrM === lrD ? lrM : `${lrM}/${lrD}`;
    right.textContent = `${when} · ${job.params.epochs} epochs · lr ${lr}`;

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
  $("resultsCard").classList.remove("hidden");
  $("cancelBtn").dataset.job = jobId;
  $("cancelBtn").disabled = !live;
  destroyCharts();
  loadedArtifacts = new Set();
  // Loaders shown until the first poll renders each section over them.
  $("charts").innerHTML = '<div class="loader">Cargando…</div>';
  $("results").innerHTML = '<div class="loader">Cargando…</div>';
  $("comparisonCard").classList.add("hidden");
  $("comparison").innerHTML = "";
  comparisonRenderedCount = 0;
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
  // Idempotent: results fill in live as each run finishes, without rebuilding the
  // section — the open tab is preserved and each run's images are fetched once.
  renderResults(metricData);
  // Grows live: re-renders whenever another inference run has finished.
  maybeRenderComparison(jobId, metricData);

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
function lineChartOpts(yTitle, legendPos = "top", title = "") {
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
      title: { display: !!title, text: title, color: THEME.text },
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

function upsertChart(id, type, data, options, plugins) {
  if (charts[id]) {
    charts[id].data = data;
    charts[id].update("none");
  } else {
    charts[id] = new Chart($(id), { type, data, options, plugins: plugins || [] });
  }
}

// Paints a solid background behind a chart so exported PNGs aren't transparent
// (otherwise the light-on-dark text is invisible on a white viewer).
const solidBgPlugin = {
  id: "solidBg",
  beforeDraw(chart) {
    const { ctx, width, height } = chart;
    ctx.save();
    ctx.globalCompositeOperation = "destination-over";
    ctx.fillStyle = "#1a2029";
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  },
};

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
      if (lossKeys.length) upsertChart(lossId, "line", { datasets: lineDatasets(run.history, lossKeys) }, lineChartOpts("loss", "top", "Loss"));
      if (metricKeys.length) upsertChart(metId, "line", { datasets: lineDatasets(run.history, metricKeys) }, lineChartOpts("score", "right", "Metrics"));
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
    plugins: {
      title: { display: true, text: "Classification metrics", color: THEME.text },
      legend: { position: "right", labels: { color: THEME.text, usePointStyle: true, boxWidth: 8, padding: 12 } },
    },
  };
}

// Section headers inside a retrain-type tab, keyed by the run's step.
const RESULT_SECTIONS = {
  calculate_metrics: {
    title: "Metrics",
    subtitle: "Test set, scored on the same windows the models were trained on: for mortality, the single 48 h window just before discharge; for discharge, the 3 time points (start, middle and end of the stay).",
  },
  inference: {
    title: "Inference",
    subtitle: "Test set too, but over the whole stay — one prediction every hour from h48 onward, not just the training windows. Reuses the thresholds chosen in Metrics and adds the error-severity bars and the predicted-vs-real heatmap.",
  },
};

// Adds the "Metrics" / "Inference" header (once) before that step's run block.
function ensureSectionHeader(panel, rt, step) {
  const sec = RESULT_SECTIONS[step];
  if (!sec) return;
  const id = `sec-${rt}-${step}`;
  if (document.getElementById(id)) return;
  const h = document.createElement("div");
  h.id = id;
  h.className = "result-section";
  h.innerHTML = `<h3>${sec.title}</h3><p class="hint">${sec.subtitle}</p>`;
  panel.appendChild(h);
}

function renderResults(metricData) {
  $("resultsCard").classList.remove("hidden");
  const host = $("results");
  const dlAll = $("downloadAllBtn");
  if (!metricData.mlflow_enabled) {
    dlAll.classList.add("hidden");
    if (!host.querySelector(".badge.warn")) {
      host.innerHTML = '<p class="badge warn">MLflow OFF — results are on disk under results/&lt;retrain_type&gt;/ and models/.</p>';
    }
    return;
  }
  const hasFinal = (r) => Object.keys(r.metrics).some((k) => !k.startsWith("mort/") && !k.startsWith("disch/"));
  const { order, groups } = groupByType(metricData.runs, hasFinal);
  // One button bundles every artifact of the whole job (models live in different
  // runs than metrics), shown once at least one run has produced results.
  if (metricData.runs.length && activeJobId) {
    dlAll.href = `/api/jobs/${activeJobId}/download-all`;
    dlAll.classList.remove("hidden");
  } else {
    dlAll.classList.add("hidden");
  }
  if (!order.length) {
    // Nothing to show yet; only paint the placeholder once (don't wipe later).
    if (!document.getElementById("rs-bar") && !host.querySelector(".muted")) {
      host.innerHTML = '<p class="muted">No results yet.</p>';
    }
    return;
  }

  // One tab per retrain type, created once; ensureTab keeps the open tab on
  // updates and ensureTabSkeleton replaces any "No results yet" placeholder.
  const [bar, panels] = ensureTabSkeleton(host, "rs");
  for (const rt of order) {
    const panel = ensureTab(bar, panels, rt, rt.toUpperCase());
    for (const run of groups[rt]) {
      ensureSectionHeader(panel, rt, run.step);
      buildResultBlock(run, panel);
    }
  }
}

async function buildResultBlock(run, parent) {
  const finalKeys = Object.keys(run.metrics).filter((k) => !k.startsWith("mort/") && !k.startsWith("disch/"));

  // Create the block (and its sub-nodes) once, then update in place on each poll
  // so the live refresh never tears down what the user is looking at.
  const blockId = "rblock-" + run.run_id;
  let block = document.getElementById(blockId);
  if (!block) {
    block = document.createElement("div");
    block.id = blockId;
    block.className = "run-block";
    block.innerHTML = `<h3>${run.run_name} <span class="badge run-status"></span></h3>`;
    parent.appendChild(block);
  }

  // Status badge: refreshed each poll (running → finished).
  const badge = block.querySelector(".run-status");
  badge.textContent = run.status;
  badge.className = "badge run-status" + (run.status === "FINISHED" ? " succeeded" : "");

  // Grouped bar chart for classification metrics (mortality vs discharge).
  // Lives in row 1, sharing it with the ROC chart (added once the run finishes).
  const bars = classificationBars(run.metrics);
  if (bars) {
    const id = "bars-" + run.run_id;
    if (!document.getElementById(id)) {
      let row1 = block.querySelector(".result-row1");
      if (!row1) {
        row1 = document.createElement("div");
        row1.className = "chart-row result-row1";
        block.appendChild(row1);
      }
      const wrap = document.createElement("div");
      wrap.className = "chart-col bars";
      wrap.innerHTML = `<canvas id="${id}"></canvas>`;
      row1.appendChild(wrap);
    }
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

  // Full numeric table (collapsible). Rewriting only its innerHTML keeps the
  // <details> open/closed state (an attribute on the element itself).
  let det = block.querySelector("details.metrics");
  if (!det) {
    det = document.createElement("details");
    det.className = "metrics";
    block.appendChild(det);
  }
  let t = "<summary>All metrics</summary><table><tr><th>metric</th><th>value</th></tr>";
  finalKeys.sort().forEach((k) => {
    const v = run.metrics[k];
    t += `<tr><td>${k}</td><td>${v == null ? "—" : v.toFixed(4)}</td></tr>`;
  });
  det.innerHTML = t + "</table>";

  // Full parameter table (collapsible): every config value this run used —
  // models, normalizers, seed, learning rates, etc. Lets you confirm from the
  // UI exactly which normalizer/model produced a given AUC.
  const params = run.params || {};
  const paramKeys = Object.keys(params).sort();
  if (paramKeys.length) {
    let pd = block.querySelector("details.params");
    if (!pd) {
      pd = document.createElement("details");
      pd.className = "params";
      block.appendChild(pd);
    }
    let p = "<summary>All params</summary><table><tr><th>param</th><th>value</th></tr>";
    paramKeys.forEach((k) => {
      p += `<tr><td>${k}</td><td>${params[k]}</td></tr>`;
    });
    pd.innerHTML = p + "</table>";
  }

  // ROC + error plots, drawn in JS (replaces the static PNGs). Fetched once per
  // run when it finishes — the underlying CSVs are immutable by then.
  if (run.status === "FINISHED" && !loadedArtifacts.has(run.run_id)) {
    loadedArtifacts.add(run.run_id);
    await appendRunCharts(run.run_id, block);
  }
}

const ERROR_COLORS = { 0: "#3ab06a", 1: "#f4c430", 2: "#ef8a3a", 3: "#e4572e" };

// Per-run charts: ROC (metrics + inference), error-severity bars and the
// predicted-vs-real bubble heatmap — all from the run's prediction CSV.
async function appendRunCharts(runId, block) {
  const det = block.querySelector("details.metrics");
  const loading = document.createElement("div");
  loading.className = "loader";
  loading.textContent = "Cargando gráficas…";
  block.insertBefore(loading, det);

  let data;
  try {
    data = await fetch(`/api/runs/${runId}/charts`).then((r) => r.json());
  } catch {
    loading.remove();
    return;
  }
  loading.remove();

  const hasRoc = data.roc && (data.roc.mort || data.roc.disch);
  if (!hasRoc && !data.error_bars && !data.error_heatmap) return;

  // Row 1: ROC next to the classification bars (created in buildResultBlock).
  if (hasRoc) {
    let row1 = block.querySelector(".result-row1");
    if (!row1) {
      row1 = document.createElement("div");
      row1.className = "chart-row result-row1";
      block.insertBefore(row1, det);
    }
    const col = document.createElement("div");
    col.className = "chart-col roc";
    col.innerHTML = `<canvas id="roc-${runId}"></canvas>`;
    row1.appendChild(col);
    upsertChart(`roc-${runId}`, "line", { datasets: rocPairDatasets(data.roc) },
      rocChartOpts("ROC — Mortality & Discharge"), [solidBgPlugin]);
  }

  // Row 2 (inference only): error-severity bars + predicted-vs-real heatmap.
  if (data.error_bars || data.error_heatmap) {
    const row2 = document.createElement("div");
    row2.className = "chart-row result-row2";
    if (data.error_bars) row2.insertAdjacentHTML("beforeend", `<div class="chart-col roc"><canvas id="eb-${runId}"></canvas></div>`);
    if (data.error_heatmap) row2.insertAdjacentHTML("beforeend", `<div class="chart-col roc"><canvas id="hm-${runId}"></canvas></div>`);
    block.insertBefore(row2, det);
    if (data.error_bars) {
      const eb = data.error_bars;
      upsertChart(`eb-${runId}`, "bar",
        { labels: eb.groups.map(String),
          datasets: [{ data: eb.proportions, backgroundColor: eb.groups.map((g) => ERROR_COLORS[g] || "#888") }] },
        errorBarOpts(eb.mean), [solidBgPlugin, barValuePlugin]);
    }
    if (data.error_heatmap) {
      const h = heatmapData(data.error_heatmap);
      upsertChart(`hm-${runId}`, "bubble", h.data, h.options, [solidBgPlugin, bubbleCountPlugin]);
    }
  }
}

// ROC datasets for one run: mortality + discharge curves, their operating-point
// dots, and the chance diagonal.
function rocPairDatasets(roc) {
  const sets = [];
  for (const [key, name, color] of [["mort", "Mortality", "#e74c3c"], ["disch", "Discharge", "#4f8cff"]]) {
    const r = roc[key];
    if (!r) continue;
    sets.push({
      label: `${name} (AUC ${r.auc.toFixed(3)})`,
      data: r.fpr.map((x, j) => ({ x, y: r.tpr[j] })),
      borderColor: color, backgroundColor: "transparent",
      borderWidth: 2, pointRadius: 0, tension: 0,
    });
    if (r.op) {
      sets.push({
        type: "scatter", label: `thr ${r.op.threshold.toFixed(2)}`,
        data: [{ x: r.op.fpr, y: r.op.tpr }],
        backgroundColor: color, borderColor: "#fff", borderWidth: 2, pointRadius: 6,
      });
    }
  }
  sets.push({
    label: "chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: THEME.muted, borderWidth: 1, borderDash: [5, 5], pointRadius: 0,
  });
  return sets;
}

function errorBarOpts(mean) {
  return {
    animation: false, maintainAspectRatio: true, aspectRatio: 1.2,
    scales: {
      x: { title: { display: true, text: "Error severity", color: THEME.muted },
           ticks: { color: THEME.muted }, grid: { display: false } },
      y: { min: 0, title: { display: true, text: "Proportion", color: THEME.muted },
           ticks: { color: THEME.muted }, grid: { color: THEME.grid } },
    },
    plugins: {
      title: { display: true, text: `Error per severity · mean ${mean.toFixed(4)}`, color: THEME.text },
      legend: { display: false },
    },
  };
}

function heatmapData(hm) {
  const cats = hm.categories;
  const maxCount = Math.max(...hm.cells.map((c) => c.count), 1);
  const points = hm.cells
    .map((c) => ({ x: cats.indexOf(c.pred), y: cats.indexOf(c.real),
                   r: 8 + (c.count / maxCount) * 26, count: c.count, error: c.error }))
    .filter((p) => p.x >= 0 && p.y >= 0);
  return {
    data: { datasets: [{
      data: points,
      backgroundColor: points.map((p) => (ERROR_COLORS[p.error] || "#888") + "99"),
      borderColor: "#000", borderWidth: 1,
    }] },
    options: {
      animation: false, maintainAspectRatio: true, aspectRatio: 1,
      scales: {
        x: { type: "linear", min: -0.5, max: cats.length - 0.5,
             afterBuildTicks: (axis) => { axis.ticks = cats.map((_, i) => ({ value: i })); },
             title: { display: true, text: "Predicted", color: THEME.muted },
             ticks: { color: THEME.muted, callback: (v) => cats[v] || "" }, grid: { color: THEME.grid } },
        y: { type: "linear", min: -0.5, max: cats.length - 0.5, reverse: true,
             afterBuildTicks: (axis) => { axis.ticks = cats.map((_, i) => ({ value: i })); },
             title: { display: true, text: "Real", color: THEME.muted },
             ticks: { color: THEME.muted, callback: (v) => cats[v] || "" }, grid: { color: THEME.grid } },
      },
      plugins: {
        title: { display: true, text: "Predicted vs real (size = count)", color: THEME.text },
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `count ${ctx.raw.count} · error ${ctx.raw.error}` } },
      },
    },
  };
}

// Draws the proportion on top of each error bar.
const barValuePlugin = {
  id: "barValue",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.fillStyle = THEME.text;
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    meta.data.forEach((bar, i) => ctx.fillText(chart.data.datasets[0].data[i].toFixed(3), bar.x, bar.y - 4));
    ctx.restore();
  },
};

// Draws the count inside each heatmap bubble.
const bubbleCountPlugin = {
  id: "bubbleCount",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.fillStyle = "#000";
    ctx.font = "10px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    chart.getDatasetMeta(0).data.forEach((pt, i) => ctx.fillText(chart.data.datasets[0].data[i].count, pt.x, pt.y));
    ctx.restore();
  },
};

// --- 6. Model comparison: ROC overlay (mortality / discharge) + mean error --
// Re-fetched as each retrained inference finishes, so the section fills in live.
function maybeRenderComparison(jobId, metricData) {
  const done = metricData.runs.filter(
    (r) => r.step === "inference" && r.status === "FINISHED"
  ).length;
  if (done === 0 || done === comparisonRenderedCount) return;
  comparisonRenderedCount = done;
  const hasRetrained = metricData.runs.some(
    (r) => r.step === "inference" && r.status === "FINISHED" && r.retrain_type && r.retrain_type !== "original"
  );
  if (!hasRetrained) return; // nothing to compare against the baseline yet
  // First time: show a loader while the per-model CSVs download.
  if (!$("cmp-mort")) {
    $("comparisonCard").classList.remove("hidden");
    $("comparison").innerHTML = '<div class="loader">Cargando…</div>';
  }
  renderComparison(jobId);
}

async function renderComparison(jobId) {
  let data;
  try {
    data = await fetch(`/api/jobs/${jobId}/comparison`).then((r) => r.json());
  } catch {
    return;
  }
  if (jobId !== activeJobId) return;

  const models = (data.models || []).filter((m) => m.mort && m.disch);
  // Need at least one retrained model to compare against the baseline.
  if (!models.some((m) => m.retrain_type !== "original")) return;

  $("comparisonCard").classList.remove("hidden");
  const host = $("comparison");
  // Build the skeleton once; later updates refresh the charts + table in place
  // so the section grows live as each retrained model finishes.
  if (!$("cmp-mort")) {
    host.innerHTML =
      `<div class="chart-row">` +
      `<div class="chart-col roc"><canvas id="cmp-mort"></canvas></div>` +
      `<div class="chart-col roc"><canvas id="cmp-disch"></canvas></div>` +
      `</div><div id="cmp-table"></div>`;
  }
  upsertChart("cmp-mort", "line", { datasets: rocDatasets(models, "mort") }, rocChartOpts("Mortality ROC"), [solidBgPlugin]);
  upsertChart("cmp-disch", "line", { datasets: rocDatasets(models, "disch") }, rocChartOpts("Discharge ROC"), [solidBgPlugin]);

  let t = "<table><tr><th>Model</th><th>Mortality AUC</th><th>Discharge AUC</th><th>Mean error</th></tr>";
  models.forEach((m) => {
    t += `<tr><td>${m.retrain_type}</td>` +
      `<td>${m.mort.auc.toFixed(3)}</td>` +
      `<td>${m.disch.auc.toFixed(3)}</td>` +
      `<td>${m.mean_error == null ? "—" : m.mean_error.toFixed(3)}</td></tr>`;
  });
  $("cmp-table").innerHTML = t + "</table>";
}

function rocDatasets(models, key) {
  const sets = models.map((m, i) => ({
    label: `${m.retrain_type} (AUC ${m[key].auc.toFixed(3)})`,
    data: m[key].fpr.map((x, j) => ({ x, y: m[key].tpr[j] })),
    borderColor: PALETTE[i % PALETTE.length],
    backgroundColor: "transparent",
    borderWidth: 2,
    pointRadius: 0,
    tension: 0,
  }));
  // Diagonal "chance" reference.
  sets.push({
    label: "chance",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: THEME.muted,
    borderWidth: 1,
    borderDash: [5, 5],
    pointRadius: 0,
  });
  return sets;
}

function rocChartOpts(title) {
  return {
    animation: false,
    parsing: false,
    maintainAspectRatio: true,
    aspectRatio: 1,
    scales: {
      x: { type: "linear", min: 0, max: 1,
           title: { display: true, text: "False positive rate", color: THEME.muted },
           ticks: { color: THEME.muted }, grid: { color: THEME.grid } },
      y: { min: 0, max: 1,
           title: { display: true, text: "True positive rate", color: THEME.muted },
           ticks: { color: THEME.muted }, grid: { color: THEME.grid } },
    },
    plugins: {
      title: { display: true, text: title, color: THEME.text },
      legend: { position: "bottom",
                labels: { color: THEME.text, usePointStyle: true, pointStyle: "line", boxWidth: 26, padding: 10,
                          filter: (item) => item.text && item.text !== "chance" && !item.text.startsWith("thr ") } },
      tooltip: { enabled: false },
    },
  };
}

// --- Download all: fetch the zip via JS so we can show a building state -----
$("downloadAllBtn").addEventListener("click", async (e) => {
  e.preventDefault();
  const btn = e.currentTarget;
  if (btn.classList.contains("loading")) return;
  const label = btn.textContent;
  btn.classList.add("loading");
  btn.textContent = "⏳ Generando zip…";
  try {
    const resp = await fetch(btn.href);
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(detailToText(data.detail) || resp.statusText);
    }
    const blob = await resp.blob();
    const cd = resp.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="?([^"]+)"?/);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = m ? m[1] : "pads_job.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showModal("Download failed", String(err));
  } finally {
    btn.classList.remove("loading");
    btn.textContent = label;
  }
});

// --- on load: show any past runs -------------------------------------------
loadHistory();
