"use strict";

// Local vectors are sufficient for inspection; no tile, routing or CDN request is made.
const $ = (id) => document.getElementById(id);
const canvas = $("mapCanvas");
const ctx = canvas.getContext("2d");
const state = { connected: false, modelReady: false, busy: false, data: null, gps: [], roads: [], routes: [],
  origin: [0, 0], center: [0, 0], scale: 1, selected: -1, width: 1, height: 1, drag: null };
const numberFormat = new Intl.NumberFormat("en-IN");
const pct = (v) => typeof v === "number" && Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : "—";
const numeric = (v, digits = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
const setText = (id, value) => { $(id).textContent = value == null ? "—" : String(value); };

let savedServer;
try { savedServer = localStorage.getItem("pathsnap-server"); } catch (_) { /* Private mode may deny storage. */ }
$("serverUrl").value = savedServer || (location.port === "8000" ? location.origin : "http://127.0.0.1:8000");

function serverBase() {
  const value = $("serverUrl").value.trim().replace(/\/+$/, "");
  let parsed;
  try { parsed = new URL(value); } catch (_) { throw new Error("Enter an HTTP server URL, for example http://192.168.1.10:8000."); }
  if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("The server URL must use http:// or https://.");
  if (parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error("Use the server's base URL without credentials, queries or fragments.");
  return value;
}

async function api(path, options = {}, timeout = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`${serverBase()}${path}`, { ...options, signal: controller.signal });
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(`The server returned a non-JSON response (${response.status}). Check the server address.`); }
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `API request failed (${response.status}). See the receiving laptop's log.`);
    return result;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The API request timed out. Check the receiving laptop's terminal before retrying.");
    if (error instanceof TypeError) throw new Error("Cannot reach the API. Check its address, LAN connection and CORS configuration.");
    throw error;
  } finally { clearTimeout(timer); }
}

function message(text, kind = "") { setText("message", text); $("message").className = `message ${kind}`; }
function connection(text, kind = "") {
  const badge = $("connectionStatus"); badge.className = `connection-status ${kind}`;
  badge.replaceChildren(Object.assign(document.createElement("span"), { className: "dot" }), document.createTextNode(text));
}

async function connect() {
  if (state.busy) return;
  state.connected = false; $("runButton").disabled = true; $("connectButton").disabled = true;
  connection("Connecting…"); message("Checking server health and discovering available trips…");
  try {
    const health = await api("/health", {}, 10000);
    const result = await api("/trips");
    const trips = Array.isArray(result) ? result : result.trips;
    if (!Array.isArray(trips)) throw new Error("The API did not return a trips list.");
    $("tripSelect").replaceChildren();
    for (const trip of trips) {
      const id = typeof trip === "string" ? trip : String(trip.id ?? trip.trip_id ?? "");
      if (!id) continue;
      const observations = trip.observations ?? trip.gps_observations;
      const label = `Trip ${id}${Number.isFinite(observations) ? ` · ${numberFormat.format(observations)} points` : ""}`;
      $("tripSelect").append(new Option(label, id));
    }
    const count = $("tripSelect").options.length;
    const partition = result.partition || health.partition || "available trips";
    state.modelReady = health.model_ready !== false && health.model_loaded !== false;
    state.connected = true; $("tripSelect").disabled = count === 0; $("runButton").disabled = count === 0 || !state.modelReady;
    connection("Server connected", "connected"); setText("partitionBadge", partition.replaceAll("_", " "));
    setText("modelStatus", health.model_name || health.model || (!state.modelReady ? "Trained model not ready — reconnect after training" : "Inference server is reachable"));
    try { localStorage.setItem("pathsnap-server", serverBase()); } catch (_) { /* Optional preference only. */ }
    if (!count) throw new Error("Server is reachable but no trips were discovered. Check PATHSNAP_DATA on Laptop 1.");
    message(state.modelReady ? `Connected · ${count} trips discovered in ${partition.replaceAll("_", " ")}. Choose a trip to run genuine inference.` : `Server connected · ${count} trips discovered. Complete model training on Laptop 1, then select Connect again.`, state.modelReady ? "success" : "");
  } catch (error) {
    state.connected = false; $("runButton").disabled = true; $("tripSelect").disabled = true;
    connection("Connection needs attention", "error"); setText("modelStatus", "Check the inference server"); message(error.message, "error");
  } finally { $("connectButton").disabled = false; }
}

async function run() {
  if (!state.connected || !state.modelReady || state.busy || !$("tripSelect").value) return;
  const tripId = $("tripSelect").value;
  state.busy = true; document.body.classList.add("is-busy");
  $("runButton").disabled = true; $("connectButton").disabled = true; $("tripSelect").disabled = true;
  setText("runButtonText", "Reconstructing…"); setText("runState", "Running");
  message(`Running trip ${tripId} on the receiving laptop. Candidate generation, scoring and decoding may take a moment.`);
  try {
    const result = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: tripId }) }, 300000);
    renderResult(result);
    setText("runState", "Complete"); message(`Trip ${result.trip_id || tripId} complete. Results below come from this API request.`, "success");
  } catch (error) {
    setText("runState", "Failed"); message(`${error.message}${state.data ? " The previous successful result remains displayed." : ""}`, "error");
  } finally {
    state.busy = false; document.body.classList.remove("is-busy");
    $("runButton").disabled = !state.connected || !state.modelReady; $("connectButton").disabled = false; $("tripSelect").disabled = false;
    setText("runButtonText", "Reconstruct route");
  }
}

function geoLines(value) {
  if (!value) return [];
  if (value.type === "FeatureCollection") return (value.features || []).flatMap(geoLines);
  if (value.type === "Feature") return geoLines(value.geometry);
  if (value.type === "GeometryCollection") return (value.geometries || []).flatMap(geoLines);
  if (value.type === "LineString") return [value.coordinates];
  if (value.type === "MultiLineString") return value.coordinates;
  return [];
}
const validCoord = (p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]);
function project(p) {
  return [(p[0] - state.origin[0]) * 111320 * Math.cos(state.origin[1] * Math.PI / 180), (p[1] - state.origin[1]) * 111320];
}
function screen(p) { return [(p[0] - state.center[0]) * state.scale + state.width / 2, state.height / 2 - (p[1] - state.center[1]) * state.scale]; }

function renderResult(result) {
  if (!Array.isArray(result.raw_gps)) throw new Error("The prediction response has no raw GPS observations.");
  state.data = result;
  const raw = result.raw_gps.filter(validCoord);
  if (!raw.length) throw new Error("The API returned no finite GPS coordinates to display.");
  // Use a local metric approximation only for rendering, never for model features or evaluation.
  state.origin = raw.reduce((s, p) => [s[0] + p[0] / raw.length, s[1] + p[1] / raw.length], [0, 0]);
  state.gps = result.raw_gps.map(p => validCoord(p) ? project(p) : null);
  state.roads = geoLines(result.road_network).map(line => line.filter(validCoord).map(project)).filter(line => line.length > 1);
  state.routes = geoLines(result.predicted_route).map(line => line.filter(validCoord).map(project)).filter(line => line.length > 1);
  state.selected = -1;
  $("mapEmpty").hidden = true; $("fitButton").disabled = false;
  setText("mapTitle", `Trip ${result.trip_id || $("tripSelect").value}`);
  setText("mapCounts", `${numberFormat.format(result.raw_gps.length)} observations · ${numberFormat.format(state.roads.length)} road geometries`);
  const m = result.metrics || {};
  const continuity = typeof m.route_continuity === "object" && m.route_continuity ? m.route_continuity.value ?? m.route_continuity.continuity : m.route_continuity;
  setText("metricAccuracy", pct(m.top1_accuracy)); setText("metricContinuity", pct(continuity)); setText("metricSequence", numeric(m.sequence_lcs_f1, 3));
  setText("accuracyBasis", Number.isFinite(m.legacy_top1_accuracy)
    ? `Full-route labels · earlier labels ${pct(m.legacy_top1_accuracy)}`
    : "Against derived reference labels");
  const seconds = result.timing?.inference_seconds;
  setText("metricTime", Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : "—");
  setText("detailTrip", result.trip_id); setText("detailModel", result.model_name); setText("detailObservations", numberFormat.format(result.raw_gps.length));
  setText("detailCoverage", pct(m.prediction_coverage)); setText("detailMiss", pct(m.candidate_miss_rate)); setText("detailRun", result.run_id);
  $("pointDetails").hidden = true; $("pointHint").hidden = false; renderWarnings(result.warnings || []); fit();
}

function renderWarnings(warnings) {
  const list = $("warningsList"); list.replaceChildren();
  setText("warningCount", warnings.length);
  if (!warnings.length) {
    const li = document.createElement("li"); li.className = "muted"; li.textContent = "No warnings returned for this run."; list.append(li); return;
  }
  warnings.slice(0, 80).forEach(warning => {
    const li = document.createElement("li");
    if (typeof warning === "string") li.textContent = warning;
    else {
      const index = warning.point_index ?? warning.index;
      li.textContent = `${Number.isInteger(index) ? `Point ${index}: ` : ""}${warning.reason || warning.message || "Unspecified warning"}`;
    }
    list.append(li);
  });
  if (warnings.length > 80) { const li = document.createElement("li"); li.textContent = `${warnings.length - 80} additional warnings are retained in the API response and server evidence.`; list.append(li); }
}

function fit() {
  const points = state.gps.filter(Boolean).concat(state.routes.flat());
  if (!points.length) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]); }
  state.center = [(minX + maxX) / 2, (minY + maxY) / 2];
  state.scale = Math.min((state.width - 75) / Math.max(maxX - minX, 60), (state.height - 75) / Math.max(maxY - minY, 60));
  draw();
}

function drawLine(line, color, width) {
  if (line.length < 2) return;
  ctx.beginPath();
  line.forEach((p, i) => { const s = screen(p); if (i === 0) ctx.moveTo(...s); else ctx.lineTo(...s); });
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
}

function draw() {
  const { width: w, height: h } = state;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#0c1521"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#172335"; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 0; x <= w; x += 44) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (let y = 0; y <= h; y += 44) { ctx.moveTo(0, y); ctx.lineTo(w, y); } ctx.stroke();
  if (!state.data) return;
  if ($("layerRoads").checked) state.roads.forEach(line => drawLine(line, "#3a4f65", 1.1));
  if ($("layerRoute").checked) {
    state.routes.forEach(line => drawLine(line, "#113c38", 8));
    state.routes.forEach(line => drawLine(line, "#64e4c4", 3));
  }
  if ($("layerGps").checked) {
    const radius = state.gps.length > 5000 ? 1.6 : 2.4;
    ctx.fillStyle = "#73acffbb"; ctx.strokeStyle = "#a4c9ff"; ctx.lineWidth = .5;
    state.gps.forEach(p => { if (!p) return; const [x, y] = screen(p); if (x < -5 || x > w + 5 || y < -5 || y > h + 5) return; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill(); });
  }
  if (state.selected >= 0 && state.gps[state.selected]) {
    const [x, y] = screen(state.gps[state.selected]); ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
  }
  const finiteGps = state.gps.filter(Boolean);
  if (finiteGps.length) {
    for (const [point, color] of [[finiteGps[0], "#64e4c4"], [finiteGps[finiteGps.length - 1], "#f1bf70"]]) {
      const [x, y] = screen(point); ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = "#eff9ff"; ctx.lineWidth = 1.4; ctx.stroke();
    }
  }
  const rawDistance = 95 / state.scale;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawDistance)));
  const niceDistance = [1, 2, 5, 10].map(x => x * magnitude).filter(x => x <= rawDistance).pop() || magnitude;
  $("mapScale").hidden = false; $("mapScale").style.width = `${niceDistance * state.scale}px`;
  setText("mapScale", niceDistance >= 1000 ? `${numeric(niceDistance / 1000, niceDistance % 1000 ? 1 : 0)} km` : `${Math.round(niceDistance)} m`);
}

function resize() {
  const bounds = $("mapWrap").getBoundingClientRect(); const ratio = window.devicePixelRatio || 1;
  state.width = bounds.width; state.height = bounds.height;
  canvas.width = Math.round(bounds.width * ratio); canvas.height = Math.round(bounds.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
}
function zoom(factor, anchorX = state.width / 2, anchorY = state.height / 2) {
  if (!state.data) return;
  const next = Math.min(50, Math.max(.00005, state.scale * factor));
  state.center[0] += (anchorX - state.width / 2) * (1 / state.scale - 1 / next);
  state.center[1] -= (anchorY - state.height / 2) * (1 / state.scale - 1 / next);
  state.scale = next; draw();
}
function inspect(index) {
  if (!state.data || index < 0) return;
  state.selected = index; $("pointHint").hidden = true; $("pointDetails").hidden = false;
  const p = state.data.raw_gps[index], d = state.data.diagnostics || {};
  const rows = [["Point index", index], ["Longitude", numeric(p[0], 6)], ["Latitude", numeric(p[1], 6)], ["Predicted arc", state.data.predictions?.[index] ?? "Missing"]];
  if (Array.isArray(d.candidate_count)) rows.push(["Candidates", d.candidate_count[index]]);
  if (Array.isArray(d.heading_reliability)) rows.push(["Heading reliability", numeric(d.heading_reliability[index], 3)]);
  if (Array.isArray(d.score_margin)) rows.push(["Relative score margin", numeric(d.score_margin[index], 3)]);
  $("pointDetails").replaceChildren(...rows.map(([label, value]) => {
    const div = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = label; dd.textContent = String(value ?? "—"); div.append(dt, dd); return div;
  })); draw();
}

canvas.addEventListener("pointerdown", event => { if (!state.data) return; canvas.setPointerCapture(event.pointerId); state.drag = { x: event.clientX, y: event.clientY, initialX: event.clientX, initialY: event.clientY, moved: false }; canvas.classList.add("dragging"); });
canvas.addEventListener("pointermove", event => {
  if (!state.drag) return; const dx = event.clientX - state.drag.x, dy = event.clientY - state.drag.y;
  state.drag.moved ||= Math.hypot(event.clientX - state.drag.initialX, event.clientY - state.drag.initialY) > 4;
  state.center[0] -= dx / state.scale; state.center[1] += dy / state.scale; state.drag.x = event.clientX; state.drag.y = event.clientY; draw();
});
canvas.addEventListener("pointerup", event => {
  const drag = state.drag; state.drag = null; canvas.classList.remove("dragging"); if (!drag || drag.moved) return;
  const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
  let nearest = -1, distance = 16;
  state.gps.forEach((p, i) => { if (!p) return; const s = screen(p), dist = Math.hypot(s[0] - x, s[1] - y); if (dist < distance) { nearest = i; distance = dist; } });
  if (nearest >= 0) inspect(nearest);
});
canvas.addEventListener("pointercancel", () => { state.drag = null; canvas.classList.remove("dragging"); });
canvas.addEventListener("wheel", event => { if (!state.data) return; event.preventDefault(); const rect = canvas.getBoundingClientRect(); zoom(Math.exp(-event.deltaY * .0015), event.clientX - rect.left, event.clientY - rect.top); }, { passive: false });
canvas.addEventListener("keydown", event => {
  if (!state.data) return; const move = 50 / state.scale;
  if (event.key === "+" || event.key === "=") zoom(1.4);
  else if (event.key === "-") zoom(1 / 1.4);
  else if (event.key === "ArrowLeft") state.center[0] -= move;
  else if (event.key === "ArrowRight") state.center[0] += move;
  else if (event.key === "ArrowUp") state.center[1] += move;
  else if (event.key === "ArrowDown") state.center[1] -= move;
  else return; event.preventDefault(); draw();
});
$("connectButton").addEventListener("click", connect);
$("serverUrl").addEventListener("change", () => { state.connected = false; $("runButton").disabled = true; connection("Reconnect to changed server"); });
$("serverUrl").addEventListener("keydown", event => { if (event.key === "Enter") connect(); });
$("runButton").addEventListener("click", run);
$("fitButton").addEventListener("click", fit);
$("zoomIn").addEventListener("click", () => zoom(1.5)); $("zoomOut").addEventListener("click", () => zoom(1 / 1.5));
["layerRoads", "layerGps", "layerRoute"].forEach(id => $(id).addEventListener("change", draw));
new ResizeObserver(resize).observe($("mapWrap"));
resize(); connect();
