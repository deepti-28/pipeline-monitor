"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const label = (c) => (c || "").replace(/_/g, " ");
const COLORS = { red: "#d92d20", orange: "#f79009", grey: "#98a2b3", blue: "#2f5bea", green: "#12b76a" };

let trendChart = null, evalCharts = [], unread = 0, current = null, evalLoaded = false;

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 3500);
}

// ---------------------------------------------------------------- status & controls
function renderStatus(s) {
  current = s;
  const day = Math.max(0, s.cursor - s.warmup_days);
  const total = s.total_days - s.warmup_days;
  $("#status").textContent = s.done ? `all ${total} days processed` : `day ${day} of ${total} · next ${s.next_date}`;
  const b = $("#banner");
  if (s.pending_fault) { b.hidden = false; b.textContent = `Queued: "${label(s.pending_fault)}" will be injected into the next day you run.`; }
  else b.hidden = true;
  $("#run").disabled = s.done;
}

async function refreshAll() {
  renderStatus(await api("/api/status"));
  await Promise.all([loadTrend(), loadAlerts()]);
}

async function runNext() {
  const btn = $("#run");
  btn.disabled = true;
  try {
    const r = await post("/run-day");
    if (r.done) toast("No more days to process.");
    else if (!r.anomaly) toast(`${r.date}: all clear`);
    await refreshAll();
  } catch (e) { toast("Error: " + e.message); }
  finally { btn.disabled = current && current.done; }
}

async function queueFault() {
  const f = $("#fault").value;
  try { renderStatus(await post("/inject-fault", { fault: f || null })); toast(f ? `Queued ${label(f)}` : "Cleared queued fault"); }
  catch (e) { toast("Error: " + e.message); }
}

async function resetAll() {
  if (!confirm("Reset the demo? Alerts and processed days will be cleared.")) return;
  try { await post("/reset"); unread = 0; updateBadge(); $("#detail").innerHTML = '<p class="hint">Select an alert from the live feed.</p>'; await refreshAll(); toast("Reset"); }
  catch (e) { toast("Error: " + e.message); }
}

// ---------------------------------------------------------------- overview chart
async function loadFeatures() {
  const feats = await api("/features");
  const sel = $("#feature");
  sel.innerHTML = feats.map((f) => `<option value="${esc(f)}">${esc(f)}</option>`).join("");
  sel.value = feats.includes("row_count") ? "row_count" : feats[0];
}

async function loadTrend() {
  const feature = $("#feature").value;
  if (!feature) return;
  const { points } = await api("/metrics?feature=" + encodeURIComponent(feature));
  const colorOf = (p) => (p.flagged ? COLORS.red : p.injected ? COLORS.orange : p.warmup ? COLORS.grey : COLORS.blue);
  const data = {
    labels: points.map((p) => p.date.slice(5)),
    datasets: [{
      label: feature, data: points.map((p) => p.value), borderColor: COLORS.blue + "88", borderWidth: 1.5, tension: 0.2,
      pointBackgroundColor: points.map(colorOf), pointBorderColor: points.map(colorOf),
      pointRadius: points.map((p) => (p.flagged || p.injected ? 7 : 3.5)),
    }],
  };
  const tip = { callbacks: { afterLabel: (c) => { const p = points[c.dataIndex]; return [p.flagged ? "ALERT" : "", p.injected ? "injected: " + label(p.injected) : ""].filter(Boolean); } } };
  if (trendChart) { trendChart.data = data; trendChart.update(); return; }
  trendChart = new Chart($("#trend"), { type: "line", data, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: tip }, scales: { y: { grace: "10%" } } } });
}

// ---------------------------------------------------------------- alerts feed
function alertCard(a, isNew) {
  const tag = a.correct === true ? '<span class="tag ok">agent correct</span>' : a.correct === false ? '<span class="tag bad">agent wrong</span>' : "";
  return `<div class="card clickable" data-id="${a.id}">
    <div class="top"><span><span class="cause">${esc(label(a.cause))}</span> <span class="muted">· ${esc(a.date)}${isNew ? " · new" : ""}</span></span>
    <span>${tag} <span class="tag">${esc(a.status)}</span></span></div>
    <div class="muted mono" style="margin-top:6px">${esc(a.text)}</div></div>`;
}

async function loadAlerts() {
  const alerts = await api("/alerts");
  $("#feed").innerHTML = alerts.length ? alerts.map((a) => alertCard(a)).join("") : '<p class="hint">No alerts yet. Queue a fault and run the next day.</p>';
}

function updateBadge() { const b = $("#badge"); b.hidden = unread === 0; b.textContent = unread; }

// ---------------------------------------------------------------- alert detail
async function showAlert(id) {
  const a = await api("/alerts/" + id);
  const inj = a.injected ? `${label(a.injected.fault)}${a.injected.column ? " on " + a.injected.column : ""}` : "none (organic)";
  const trace = (a.trace || []).map((t) => `<tr><td>${t.step}</td><td class="mono">${esc(t.tool)}</td><td class="mono">${esc(JSON.stringify(t.args))}</td>
    <td><details><summary>result</summary><pre class="mono">${esc(JSON.stringify(t.result, null, 1))}</pre></details></td></tr>`).join("");
  $("#detail").innerHTML = `
    <div class="card">
      <div class="top"><h2 style="margin:0">${esc(label(a.cause))}</h2>
        <span>confidence <span class="bar"><i style="width:${Math.round(a.confidence * 100)}%"></i></span> ${Math.round(a.confidence * 100)}%</span></div>
      <p class="muted">${esc(a.date)} · detector: ${esc(a.detector)} · ${a.tool_calls} tool calls</p>
      <p><b>Ground truth (demo):</b> ${esc(inj)} ${a.correct === true ? '<span class="tag ok">match</span>' : a.correct === false ? '<span class="tag bad">mismatch</span>' : ""}</p>
      <p><b>Alert:</b> <span class="mono">${esc(a.text)}</span></p>
      <h2>Evidence</h2><ul class="ev">${a.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
      <h2>Suggested fix</h2><p>${esc(a.suggested_fix)}</p>
      <p>Status: ${["open", "acknowledged", "resolved"].map((s) => `<button data-status="${s}" data-id="${a.id}" ${a.status === s ? "disabled" : ""}>${s}</button>`).join(" ")}</p>
    </div>
    <h2>Agent tool-call trace</h2>
    <table><thead><tr><th>#</th><th>Tool</th><th>Args</th><th></th></tr></thead><tbody>${trace}</tbody></table>`;
  switchTab("detail");
}

// ---------------------------------------------------------------- evaluation
const pct = (r) => (r.rate == null ? "n/a" : `${Math.round(r.rate * 100)}% <span class="muted">(${r.k}/${r.n})</span>`);

async function loadEval() {
  const box = $("#eval");
  let res;
  try { res = await api("/eval-results"); } catch (e) { box.innerHTML = `<p class="hint">Could not load results: ${esc(e.message)}</p>`; return; }
  evalLoaded = true;
  const faults = res.meta.faults;
  const det = res.detection, ag = res.agent;
  box.innerHTML = `
    <p class="hint">${res.meta.n_faulty} faulty days and ${res.meta.n_clean} clean days, each with ${res.meta.history_days} clean history days. Data: ${esc(res.meta.data)}. ${esc(res.meta.note)}</p>
    <h2>Detection</h2>
    <table><thead><tr><th>Detector</th><th>Detection rate</th><th>False alarm rate</th></tr></thead><tbody>
    ${det.map((d) => `<tr><td>${esc(d.name)}</td><td>${pct(d.detection)}</td><td>${pct(d.false_alarms)}</td></tr>`).join("")}</tbody></table>
    <h2>Root-cause accuracy</h2>
    <table><thead><tr><th>Setup</th><th>Cause accuracy (given alert)</th><th>End-to-end</th><th>Avg tool calls</th></tr></thead><tbody>
    ${ag.map((a) => `<tr><td>${esc(a.name)}</td><td>${pct(a.cause_accuracy_given_alert)}</td><td>${pct(a.end_to_end_accuracy)}</td><td>${a.avg_tool_calls}</td></tr>`).join("")}</tbody></table>
    <div class="grid2" style="margin-top:16px">
      <div class="chartbox"><canvas id="c-det"></canvas></div>
      <div class="chartbox"><canvas id="c-ag"></canvas></div>
    </div>`;
  evalCharts.forEach((c) => c.destroy()); evalCharts = [];
  const palette = [COLORS.grey, COLORS.orange, COLORS.blue];
  const mk = (id, title, groups) => new Chart($(id), {
    type: "bar",
    data: { labels: faults.map(label), datasets: groups.map((g, i) => ({ label: g.name, data: faults.map((f) => Math.round((g.per_fault[f].rate || 0) * 100)), backgroundColor: palette[i % 3] })) },
    options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: title }, legend: { position: "bottom" } }, scales: { y: { min: 0, max: 100, title: { display: true, text: "%" } } } },
  });
  evalCharts.push(mk("#c-det", "Detection rate by fault type", det), mk("#c-ag", "Cause accuracy by fault type", ag));
}

// ---------------------------------------------------------------- tabs & wiring
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
  if (name === "alerts") { unread = 0; updateBadge(); }
  if (name === "eval" && !evalLoaded) loadEval();
}

function connectWS() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/alerts");
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "alert") {
      const feed = $("#feed");
      if (feed.querySelector(".hint")) feed.innerHTML = "";
      feed.insertAdjacentHTML("afterbegin", alertCard(m.alert, true));
      if (!$("#tab-alerts").hidden) return;
      unread++; updateBadge(); toast(`Alert: ${label(m.alert.cause)} (${m.alert.date})`);
    }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
  setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 25000);
}

document.addEventListener("click", (e) => {
  const tab = e.target.closest(".tab"); if (tab) return switchTab(tab.dataset.tab);
  const card = e.target.closest(".card.clickable"); if (card) return showAlert(card.dataset.id);
  const st = e.target.closest("button[data-status]");
  if (st) post(`/alerts/${st.dataset.id}/status`, { status: st.dataset.status }).then(() => { showAlert(st.dataset.id); loadAlerts(); }).catch((x) => toast(x.message));
});

(async function init() {
  $("#run").onclick = runNext; $("#inject").onclick = queueFault; $("#reset").onclick = resetAll;
  $("#feature").onchange = loadTrend;
  try {
    const s = await api("/api/status");
    $("#fault").innerHTML = '<option value="">no fault</option>' + s.faults.map((f) => `<option value="${f}">${label(f)}</option>`).join("");
    await loadFeatures(); renderStatus(s); await Promise.all([loadTrend(), loadAlerts()]);
  } catch (e) { $("#status").textContent = "backend waking up — refresh in a few seconds"; }
  connectWS();
})();
