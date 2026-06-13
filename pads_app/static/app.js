"use strict";

let selectedFile = null;
let datasetValid = false;
let pollTimer = null;
let activeJobId = null;
let liveMode = false; // true while watching a running job (enables fail popups)
let charts = {}; // canvasId -> Chart instance
let loadedArtifacts = new Set(); // run_ids whose artifacts were already fetched
let comparisonRenderedCount = 0; // finished inference runs already in the comparison
let comparisonModels = []; // cached comparison data so the test-type filter repaints without refetch
let comparisonTestType = null; // inference window currently shown in the comparison
let selectedModel = null; // retrain type shown in both Retrain (4) and Results (5)
let historyExpanded = false; // History shows the latest HISTORY_PAGE runs until expanded
const HISTORY_PAGE = 10;

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
    if (data.has_episode_id) {
      const ep = document.createElement("div");
      ep.className = "badge warn";
      ep.style.marginTop = "0.5rem";
      ep.textContent =
        `⚠ hospital_episode_id found — ${data.multi_stay_patients} patient(s) with more than one ICU stay. ` +
        `Stays of the same hospital episode are kept in the same split.`;
      box.appendChild(ep);
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
  const esp = advanced ? $("early_stopping_patience").value : 20;
  const monitorMetric = advanced ? $("monitor_metric").value : "loss";
  const normSrc = advanced ? $("normalizer_source").value : "fitted";
  const thrMethod = advanced ? $("threshold_method").value : "precision_recall";

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("retrain_types", types.join(","));
  fd.append("epochs", $("epochs").value);
  fd.append("batch_size", $("batch_size").value);
  fd.append("learning_rate_mort", lrMort);
  fd.append("learning_rate_disch", lrDisch);
  fd.append("early_stopping_patience", esp);
  fd.append("monitor_metric", monitorMetric);
  fd.append("normalizer_source", normSrc);
  fd.append("threshold_method", thrMethod);
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
  // Re-enable straight away so more runs can be queued while this one is busy.
  // A submission while another job is active comes back as "queued".
  $("trainBtn").disabled = !datasetValid;
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
  // Show only the latest HISTORY_PAGE runs until the user expands the list.
  const visible = historyExpanded ? jobs : jobs.slice(0, HISTORY_PAGE);
  visible.forEach((job) => {
    const li = document.createElement("li");
    li.className = "job-item" + (job.id === activeJobId ? " active" : "");
    // Poll live for both running and queued jobs (a queued one will flip to
    // running on its own once the slot frees, then stream as usual).
    const liveStatus = job.status === "running" || job.status === "queued";
    const when = job.created_at.replace("T", " ").replace("+00:00", " UTC");

    const row = document.createElement("div");
    row.className = "ji-row";
    row.addEventListener("click", () => openJob(job.id, liveStatus));

    const left = document.createElement("div");
    left.className = "ji-left";
    // Show the baseline alongside any retrain types (it may be the only thing run).
    const labels = [...job.params.retrain_types];
    if (job.params.evaluate_original) labels.unshift("original");
    left.innerHTML =
      `<span class="badge ${job.status}">${job.status}</span>` +
      `<span>${job.params.data_filename} · [${labels.join(", ")}]</span>`;

    // Row shows just the timestamp; the parameters live in the dropdown below
    // and only for the selected run.
    const right = document.createElement("div");
    right.className = "ji-right";
    right.textContent = when;

    const del = document.createElement("button");
    del.className = "ghost del";
    del.title = "Delete run";
    del.textContent = "🗑";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteJob(job.id);
    });

    row.append(left, right, del);
    li.appendChild(row);

    // Selected run only: drop its parameters down just below the row.
    if (job.id === activeJobId) {
      const params = document.createElement("div");
      params.className = "ji-params";
      params.innerHTML = historyParamsHtml(job);
      li.appendChild(params);
    }

    ul.appendChild(li);
  });

  // "Show more / less" toggle when there are more than one page of runs.
  if (jobs.length > HISTORY_PAGE) {
    const li = document.createElement("li");
    li.className = "show-more";
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = historyExpanded
      ? "Show less"
      : `Show more (${jobs.length - HISTORY_PAGE} more)`;
    btn.addEventListener("click", () => {
      historyExpanded = !historyExpanded;
      loadHistory();
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

// Parameters of a History run, as a compact key/value grid (the dropdown shown
// under the selected run). Training params only apply when something was retrained;
// an original-only run just shows the evaluation-relevant fields.
function historyParamsHtml(job) {
  const p = job.params;
  const card = (k, v) => `<div class="lb-param"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  const cards = [];
  if (p.retrain_types.length) {
    const lr = p.learning_rate_mort === p.learning_rate_disch
      ? p.learning_rate_mort : `${p.learning_rate_mort} / ${p.learning_rate_disch}`;
    cards.push(card("Retrain types", p.retrain_types.join(", ")));
    cards.push(card("Epochs", p.epochs));
    cards.push(card("Batch size", p.batch_size));
    cards.push(card("Learning rate", lr));
    cards.push(card("Early stop", p.early_stopping_patience));
    if (p.monitor_metric != null) cards.push(card("Optimize", p.monitor_metric));
    cards.push(card("Normalizer", p.normalizer_source));
    if (p.threshold_method != null) cards.push(card("Threshold", p.threshold_method));
    cards.push(card("Seed", p.seed));
  } else {
    cards.push(card("Model", "Original baseline (no retraining)"));
    cards.push(card("Normalizer", p.normalizer_source));
    cards.push(card("Seed", p.seed));
  }
  if (p.evaluate_original && p.retrain_types.length) cards.push(card("Baseline", "original evaluated"));
  return `<div class="lb-params">${cards.join("")}</div>`;
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
  // Reset the shared model selector for the new job (revealed once models load).
  selectedModel = null;
  $("modelTabs").innerHTML = "";
  $("modelTabs").dataset.sig = "";
  $("modelSelectorCard").classList.add("hidden");
  $("comparisonCard").classList.add("hidden");
  $("comparison").innerHTML = "";
  comparisonRenderedCount = 0;
  comparisonModels = [];
  comparisonTestType = null;
  $("steps").innerHTML = '<li class="loader">Cargando…</li>';
  // Hide status badge, Cancel and Logs until the first poll renders the run.
  ["jobStatus", "cancelBtn", "logDetails"].forEach((id) => $(id).classList.add("hidden"));
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
  // First poll arrived: reveal the controls hidden behind the loader.
  ["jobStatus", "cancelBtn", "logDetails"].forEach((id) => $(id).classList.remove("hidden"));
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

// --- Shared model selector: one set of tabs (above Retrain) drives the visible
// retrain type in BOTH section 4 (Retrain) and section 5 (Results). -----------

// Ensure a bare panels container inside `host` (no per-section tab bar — the
// shared #modelTabs bar controls visibility). Replaces any loader/placeholder.
function ensurePanelsContainer(host, id) {
  let panels = document.getElementById(id + "-panels");
  if (!panels) {
    host.innerHTML = `<div id="${id}-panels" class="tab-panels"></div>`;
    panels = document.getElementById(id + "-panels");
  }
  return panels;
}

// Idempotent per-retrain-type panel inside a section's panels container. Tagged
// with data-rt and hidden by default; applyModelSelection() reveals the active one.
function ensureModelPanel(panels, prefix, rt) {
  const panId = prefix + "-panels--" + rt;
  let panel = document.getElementById(panId);
  if (!panel) {
    panel = document.createElement("div");
    panel.id = panId;
    panel.className = "tab-panel hidden";
    panel.dataset.rt = rt;
    panels.appendChild(panel);
  }
  return panel;
}

// Rebuild the shared model tabs from whatever retrain types currently have a
// panel in either section (Results order first, then any Retrain-only types).
// Cheap and guarded by a signature so the live poll doesn't flicker the bar.
function syncModelTabs() {
  const order = [];
  const add = (rt) => { if (rt && !order.includes(rt)) order.push(rt); };
  // Tabs follow the run order (Results order): the "original" baseline runs first,
  // so it lands first; then the retrained types in the order they were launched.
  const rsP = document.getElementById("rs-panels");
  const mtP = document.getElementById("mt-panels");
  if (rsP) [...rsP.children].forEach((p) => add(p.dataset.rt));
  if (mtP) [...mtP.children].forEach((p) => add(p.dataset.rt));

  const bar = $("modelTabs");
  if (!order.length) {
    bar.innerHTML = ""; bar.dataset.sig = "";
    $("modelSelectorCard").classList.add("hidden");
    return;
  }
  $("modelSelectorCard").classList.remove("hidden");
  if (!selectedModel || !order.includes(selectedModel)) {
    // Default to the first retrained model (one that has training curves) so
    // section 4 isn't empty; fall back to the first tab (e.g. original-only runs).
    selectedModel = order.find((rt) => document.getElementById("mt-panels--" + rt)) || order[0];
  }

  const sig = order.join(",") + "|" + selectedModel;
  if (bar.dataset.sig !== sig) {
    bar.innerHTML = "";
    order.forEach((rt) => {
      const btn = document.createElement("button");
      btn.className = "tab" + (rt === selectedModel ? " active" : "");
      btn.textContent = rt.toUpperCase();
      btn.dataset.rt = rt;
      btn.addEventListener("click", () => selectModel(rt));
      bar.appendChild(btn);
    });
    bar.dataset.sig = sig;
  }
  applyModelSelection();
}

function selectModel(rt) {
  selectedModel = rt;
  [...$("modelTabs").children].forEach((b) => b.classList.toggle("active", b.dataset.rt === rt));
  applyModelSelection();
}

// Show only the selected retrain type's panel in each section, hide the rest.
function applyModelSelection() {
  ["mt-panels", "rs-panels"].forEach((pid) => {
    const panels = document.getElementById(pid);
    if (!panels) return;
    [...panels.children].forEach((p) => p.classList.toggle("hidden", p.dataset.rt !== selectedModel));
  });
  Object.values(charts).forEach((c) => c.resize()); // fix 0-size in just-shown panels
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
  const panels = ensurePanelsContainer(host, "mt");
  const { order, groups } = groupByType(metricData.runs, (r) => Object.keys(r.history || {}).length);
  for (const rt of order) {
    const panel = ensureModelPanel(panels, "mt", rt);
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
    // One params table per retrain tab — the mortality and discharge runs share
    // the same config, so render it once at the panel level (not per model).
    if (groups[rt].length) renderParams(groups[rt][0], panel);
  }
  syncModelTabs();
}

// Collapsible "All params" table for a retrain tab: every config value the run
// used — epochs, learning rates, batch size, seed, normalizers, models, etc.
// Rendered once per tab (mortality/discharge share the same config). Only retrain
// runs receive the real training flags, so these match what was actually used
// (the metrics/inference runs log config defaults instead).
function renderParams(run, block) {
  if (!block) return;
  const params = run.params || {};
  // Drop unset params (MLflow logs a None config value as the string "None"),
  // e.g. fixed_th_mort/fixed_th_disch on retrained models — showing them is noise.
  const paramKeys = Object.keys(params)
    .filter((k) => params[k] != null && params[k] !== "None" && params[k] !== "")
    .sort();
  if (!paramKeys.length) return;
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
    if (!document.getElementById("rs-panels") && !host.querySelector(".muted")) {
      host.innerHTML = '<p class="muted">No results yet.</p>';
    }
    return;
  }

  // Per-retrain-type panels (no own tab bar — the shared #modelTabs selector
  // drives which one is visible, in sync with section 4).
  const panels = ensurePanelsContainer(host, "rs");
  for (const rt of order) {
    const panel = ensureModelPanel(panels, "rs", rt);
    for (const run of groups[rt]) {
      ensureSectionHeader(panel, rt, run.step);
      if (run.step === "inference") {
        // One sub-tab per inference window (full / last_48h / …) instead of
        // stacking the blocks, mirroring the per-model tabs in Retrain.
        const [subBar, subPanels] = ensureInferenceTabs(panel, rt);
        const subPanel = ensureInferenceTab(subBar, subPanels, inferenceTestType(run));
        buildResultBlock(run, subPanel);
      } else {
        buildResultBlock(run, panel);
      }
    }
  }
  syncModelTabs();
}

// Nested tab skeleton for the Inference section of a retrain-type panel, created
// once per retrain type. It appends to the panel (never wipes it — the panel
// already holds the Metrics block + section headers).
function ensureInferenceTabs(panel, rt) {
  const barId = `inf-bar-${rt}`;
  let bar = document.getElementById(barId);
  if (!bar) {
    const wrap = document.createElement("div");
    wrap.className = "inf-tabs";
    wrap.innerHTML = `<div id="${barId}" class="tab-bar"></div><div id="inf-panels-${rt}" class="tab-panels"></div>`;
    panel.appendChild(wrap);
    bar = document.getElementById(barId);
  }
  return [bar, document.getElementById(`inf-panels-${rt}`)];
}

// The inference window an inference run scored. Prefer the logged param; fall
// back to parsing the run name "inference_<retrain_type>_<test_type>".
function inferenceTestType(run) {
  const p = run.params || {};
  return p.test_type_active || p.test_type
    || (run.run_name || "").replace(/^inference_[^_]+_/, "") || "full";
}

// Fixed display order for the inference sub-tabs; unknown windows go last.
const INFERENCE_ORDER = ["full", "first_48h", "last_96h", "last_48h"];
const inferenceRank = (tt) => {
  const i = INFERENCE_ORDER.indexOf(tt);
  return i === -1 ? INFERENCE_ORDER.length : i;
};

// Like ensureTab, but inserts the new tab at its canonical position (by
// INFERENCE_ORDER) instead of appending — so the windows always read full →
// first_48h → last_48h → last_96h regardless of which finished first.
function ensureInferenceTab(bar, panels, tt) {
  const panId = panels.id + "--" + tt;
  let panel = document.getElementById(panId);
  if (panel) return panel;
  const first = bar.children.length === 0;
  const btn = document.createElement("button");
  btn.className = "tab" + (first ? " active" : "");
  btn.textContent = tt.toUpperCase();
  btn.dataset.tt = tt;
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
  // Insert before the first existing tab that ranks after this one (panels stay
  // parallel to the bar, so the same index applies to both).
  const refIdx = [...bar.children].findIndex((b) => inferenceRank(b.dataset.tt) > inferenceRank(tt));
  if (refIdx === -1) {
    bar.appendChild(btn);
    panels.appendChild(panel);
  } else {
    bar.insertBefore(btn, bar.children[refIdx]);
    panels.insertBefore(panel, panels.children[refIdx]);
  }
  return panel;
}

// `idPrefix` namespaces the DOM ids so the same run can be rendered in two places
// at once (the live results section uses "", the Best-runs window uses "lb-")
// without colliding on rblock-/bars-/roc- ids or the loadedArtifacts guard.
async function buildResultBlock(run, parent, idPrefix = "") {
  const finalKeys = Object.keys(run.metrics).filter((k) => !k.startsWith("mort/") && !k.startsWith("disch/"));

  // Create the block (and its sub-nodes) once, then update in place on each poll
  // so the live refresh never tears down what the user is looking at.
  const blockId = idPrefix + "rblock-" + run.run_id;
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
    const id = idPrefix + "bars-" + run.run_id;
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

  // ROC + error plots, drawn in JS (replaces the static PNGs). Fetched once per
  // run when it finishes — the underlying CSVs are immutable by then.
  if (run.status === "FINISHED" && !loadedArtifacts.has(idPrefix + run.run_id)) {
    loadedArtifacts.add(idPrefix + run.run_id);
    await appendRunCharts(run.run_id, block, idPrefix);
  }
}

const ERROR_COLORS = { 0: "#3ab06a", 1: "#f4c430", 2: "#ef8a3a", 3: "#e4572e" };

// Per-run charts: ROC (metrics + inference), error-severity bars and the
// predicted-vs-real bubble heatmap — all from the run's prediction CSV.
async function appendRunCharts(runId, block, idPrefix = "") {
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
    col.innerHTML = `<canvas id="${idPrefix}roc-${runId}"></canvas>`;
    row1.appendChild(col);
    upsertChart(`${idPrefix}roc-${runId}`, "line", { datasets: rocPairDatasets(data.roc) },
      rocChartOpts("ROC — Mortality & Discharge"), [solidBgPlugin]);
    // Operating point shown next to the curve: fixed for the original model,
    // the data-derived optimum for retrained ones (whatever produced this ROC).
    col.insertAdjacentHTML("beforeend", rocThresholdCaption(data.roc));
  }

  // Row 2 (inference only): error-severity bars + predicted-vs-real heatmap.
  if (data.error_bars || data.error_heatmap) {
    const row2 = document.createElement("div");
    row2.className = "chart-row result-row2";
    if (data.error_bars) row2.insertAdjacentHTML("beforeend", `<div class="chart-col roc"><canvas id="${idPrefix}eb-${runId}"></canvas></div>`);
    if (data.error_heatmap) row2.insertAdjacentHTML("beforeend", `<div class="chart-col roc"><canvas id="${idPrefix}hm-${runId}"></canvas></div>`);
    block.insertBefore(row2, det);
    if (data.error_bars) {
      const eb = data.error_bars;
      upsertChart(`${idPrefix}eb-${runId}`, "bar",
        { labels: eb.groups.map(String),
          datasets: [{ data: eb.proportions, backgroundColor: eb.groups.map((g) => ERROR_COLORS[g] || "#888") }] },
        errorBarOpts(eb.mean), [solidBgPlugin, barValuePlugin]);
    }
    if (data.error_heatmap) {
      const h = heatmapData(data.error_heatmap);
      upsertChart(`${idPrefix}hm-${runId}`, "bubble", h.data, h.options, [solidBgPlugin, bubbleCountPlugin]);
    }
  }
}

// ROC datasets for one run: mortality + discharge curves and the chance
// diagonal. (The per-threshold operating-point dots were removed by request.)
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
  }
  sets.push({
    label: "chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: THEME.muted, borderWidth: 1, borderDash: [5, 5], pointRadius: 0,
  });
  return sets;
}

// Caption under a ROC chart with each model's decision threshold (the curve's
// operating point). Empty string when no threshold is available for either model.
function rocThresholdCaption(roc) {
  const fmt = (r) => (r && r.op && r.op.threshold != null) ? r.op.threshold.toFixed(4) : null;
  const m = fmt(roc.mort), d = fmt(roc.disch);
  if (m == null && d == null) return "";
  const parts = [];
  if (m != null) parts.push(`<span><b style="color:#e74c3c">Mortality</b> ${m}</span>`);
  if (d != null) parts.push(`<span><b style="color:#4f8cff">Discharge</b> ${d}</span>`);
  return `<div class="roc-threshold">Optimal threshold · ${parts.join(" · ")}</div>`;
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

  comparisonModels = (data.models || []).filter((m) => m.mort && m.disch);
  // Need at least one retrained model to compare against the baseline.
  if (!comparisonModels.some((m) => m.retrain_type !== "original")) return;
  paintComparison();
}

// Draw the comparison for the selected inference window. Models of other windows
// are filtered out (comparing e.g. full vs last_96h ROCs would be apples-to-oranges).
function paintComparison() {
  const models = comparisonModels;
  if (!models.length) return;
  $("comparisonCard").classList.remove("hidden");
  const host = $("comparison");
  // Build the skeleton once; later updates refresh the charts + table in place
  // so the section grows live as each retrained model finishes.
  if (!$("cmp-mort")) {
    host.innerHTML =
      `<div class="cmp-filter"><span class="cmp-filter-label">Inference window</span>` +
      `<div id="cmp-filter-bar" class="tab-bar"></div></div>` +
      `<div class="chart-row">` +
      `<div class="chart-col roc"><canvas id="cmp-mort"></canvas></div>` +
      `<div class="chart-col roc"><canvas id="cmp-disch"></canvas></div>` +
      `</div><div id="cmp-table"></div>`;
  }

  // Available inference windows, ordered full → first_48h → last_48h → last_96h.
  const testTypes = [...new Set(models.map((m) => m.test_type || "full"))]
    .sort((a, b) => inferenceRank(a) - inferenceRank(b));
  if (!comparisonTestType || !testTypes.includes(comparisonTestType)) comparisonTestType = testTypes[0];
  const filterRow = host.querySelector(".cmp-filter");
  filterRow.style.display = testTypes.length > 1 ? "" : "none"; // only worth showing with >1
  const bar = $("cmp-filter-bar");
  const sig = testTypes.join(",") + "|" + comparisonTestType;
  if (bar.dataset.sig !== sig) {
    bar.innerHTML = "";
    testTypes.forEach((tt) => {
      const btn = document.createElement("button");
      btn.className = "tab" + (tt === comparisonTestType ? " active" : "");
      btn.textContent = tt.toUpperCase();
      btn.addEventListener("click", () => { comparisonTestType = tt; paintComparison(); });
      bar.appendChild(btn);
    });
    bar.dataset.sig = sig;
  }

  const shown = models.filter((m) => (m.test_type || "full") === comparisonTestType);
  upsertChart("cmp-mort", "line", { datasets: rocDatasets(shown, "mort") }, rocChartOpts("Mortality ROC"), [solidBgPlugin]);
  upsertChart("cmp-disch", "line", { datasets: rocDatasets(shown, "disch") }, rocChartOpts("Discharge ROC"), [solidBgPlugin]);

  let t = "<table><tr><th>Model</th><th>Mortality AUC</th><th>Discharge AUC</th><th>Mean error</th></tr>";
  shown.forEach((m) => {
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
// Shared by the live results section and the Best-runs window. `href` is the
// job's download-all URL; `btn` gets a transient "building" state.
async function downloadZip(href, btn) {
  if (btn.classList.contains("loading")) return;
  const label = btn.textContent;
  btn.classList.add("loading");
  btn.textContent = "⏳ Generando zip…";
  try {
    const resp = await fetch(href);
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
}

$("downloadAllBtn").addEventListener("click", (e) => {
  e.preventDefault();
  downloadZip(e.currentTarget.href, e.currentTarget);
});

// --- Best runs window: every successful inference run ranked by mean error ---
let lbSelectedRunId = null;

$("openLeaderboard").addEventListener("click", () => {
  $("leaderboard").classList.remove("hidden");
  loadLeaderboard();
});
$("lbClose").addEventListener("click", () => $("leaderboard").classList.add("hidden"));
$("lbRefresh").addEventListener("click", loadLeaderboard);
// Click on the dim backdrop (but not the box) closes the window.
$("leaderboard").addEventListener("click", (e) => {
  if (e.target === $("leaderboard")) $("leaderboard").classList.add("hidden");
});

async function loadLeaderboard() {
  const list = $("lbList");
  list.innerHTML = '<li class="muted">Loading…</li>';
  let runs;
  try {
    runs = await fetch("/api/successful-runs").then((r) => r.json()).then((d) => d.runs);
  } catch {
    list.innerHTML = '<li class="muted">Could not load runs.</li>';
    return;
  }
  if (!runs.length) {
    list.innerHTML = '<li class="muted">No successful runs yet.</li>';
    $("lbDetails").innerHTML = '<p class="muted">Select a run on the left to see its results and download its artifacts.</p>';
    return;
  }
  list.innerHTML = "";
  runs.forEach((entry, i) => {
    const li = document.createElement("li");
    li.className = "lb-item" + (entry.run_id === lbSelectedRunId ? " active" : "");
    const when = (entry.created_at || "").replace("T", " ").replace("+00:00", " UTC");
    const crit = entry.critical_error_rate != null ? ` · crit ${entry.critical_error_rate.toFixed(3)}` : "";
    li.innerHTML =
      `<span class="lb-rank">${i + 1}</span>` +
      `<div class="lb-main">` +
      `<div><span class="lb-err">mean error ${entry.mean_error.toFixed(4)}</span>${crit}</div>` +
      `<div class="lb-sub">${entry.retrain_type} · ${entry.test_type || "full"} · ${entry.data_filename || "?"} · ${when}</div>` +
      `</div>`;
    li.addEventListener("click", () => {
      [...list.children].forEach((c) => c.classList.remove("active"));
      li.classList.add("active");
      renderLeaderboardDetails(entry);
    });
    list.appendChild(li);
  });
}

async function renderLeaderboardDetails(entry) {
  lbSelectedRunId = entry.run_id;
  const host = $("lbDetails");
  // Tear down this window's previous charts + artifact guards (lb- namespace) so
  // each selection renders fresh without touching the live results section.
  Object.keys(charts).filter((id) => id.startsWith("lb-")).forEach((id) => {
    charts[id].destroy();
    delete charts[id];
  });
  [...loadedArtifacts].filter((k) => k.startsWith("lb-")).forEach((k) => loadedArtifacts.delete(k));
  host.innerHTML = "";

  const when = (entry.created_at || "").replace("T", " ").replace("+00:00", " UTC");
  const critFig = entry.critical_error_rate != null
    ? `<div class="lb-figure"><span class="k">Critical error rate</span><span class="v">${entry.critical_error_rate.toFixed(4)}</span></div>`
    : "";
  const head = document.createElement("div");
  head.className = "lb-detail-head";
  head.innerHTML =
    `<h3>${entry.run_name} <span class="badge succeeded">${entry.retrain_type}</span>` +
    `<span class="badge info">${entry.test_type || "full"}</span></h3>` +
    `<p class="lb-sub">${entry.data_filename || "?"} · ${when}</p>` +
    `<div class="lb-figure"><span class="k">Mean error severity</span><span class="v good">${entry.mean_error.toFixed(4)}</span></div>` +
    critFig;
  const dl = document.createElement("button");
  dl.className = "dl-all";
  dl.textContent = "⬇ Download all (.zip)";
  dl.addEventListener("click", () => downloadZip(`/api/jobs/${entry.job_id}/download-all`, dl));
  head.appendChild(dl);
  host.appendChild(head);

  // Parameters used, shown next to the charts.
  const paramsHtml = leaderboardParamsHtml(entry);
  if (paramsHtml) {
    const pSection = document.createElement("div");
    pSection.innerHTML = `<p class="lb-section-label">Parameters</p>${paramsHtml}`;
    host.appendChild(pSection);
  }

  const resultsEl = document.createElement("div");
  resultsEl.innerHTML = '<p class="lb-section-label">Results</p>';
  host.appendChild(resultsEl);
  // Reuse the live results renderer with the "lb-" id namespace.
  const run = {
    run_id: entry.run_id, run_name: entry.run_name, status: "FINISHED",
    step: "inference", retrain_type: entry.retrain_type, metrics: entry.metrics,
  };
  await buildResultBlock(run, resultsEl, "lb-");
}

// Compact key/value grid of the params a run used. The "original" baseline trains
// nothing, so it only shows the evaluation-relevant fields.
function leaderboardParamsHtml(entry) {
  const p = entry.params || {};
  const card = (k, v) => `<div class="lb-param"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  const cards = [];
  if (entry.retrain_type === "original") {
    cards.push(card("Model", "Shipped base (no retraining)"));
    if (p.normalizer_source != null) cards.push(card("Normalizer", p.normalizer_source));
    cards.push(card("Thresholds", "Fixed (published)"));
  } else {
    const lr = p.learning_rate_mort === p.learning_rate_disch
      ? p.learning_rate_mort : `${p.learning_rate_mort} / ${p.learning_rate_disch}`;
    if (p.epochs != null) cards.push(card("Epochs", p.epochs));
    if (p.batch_size != null) cards.push(card("Batch size", p.batch_size));
    if (p.learning_rate_mort != null) cards.push(card("Learning rate", lr));
    if (p.early_stopping_patience != null) cards.push(card("Early stop", p.early_stopping_patience));
    if (p.monitor_metric != null) cards.push(card("Optimize", p.monitor_metric));
    if (p.normalizer_source != null) cards.push(card("Normalizer", p.normalizer_source));
    if (p.threshold_method != null) cards.push(card("Threshold", p.threshold_method));
    if (p.seed != null) cards.push(card("Seed", p.seed));
  }
  return cards.length ? `<div class="lb-params">${cards.join("")}</div>` : "";
}

// --- on load: show any past runs -------------------------------------------
loadHistory();
