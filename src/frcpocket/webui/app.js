// Plate Pocketer front end. Talks to frcpocket.web; geometry arrives in inches.

const $ = (id) => document.getElementById(id);
const MM = 25.4;
const fields = ["wall", "rib", "boss", "corner", "depth"];

const state = {
  units: "in",
  depth: "through",
  density: "medium",
  plate: null,
  job: null,
  pick: null,
  view: null,
  fit: null,
};

// ---------- units ----------

function fmtLen(valueIn) {
  if (valueIn == null) return "–";
  return state.units === "mm" ? (valueIn * MM).toFixed(1) + " mm" : valueIn.toFixed(3) + '"';
}

function fraction(valueIn) {
  for (const denom of [2, 4, 8, 16, 32, 64]) {
    const top = Math.round(valueIn * denom);
    if (top > 0 && Math.abs(top / denom - valueIn) < 0.0006) {
      const whole = Math.floor(top / denom), rest = top % denom;
      return rest ? (whole ? whole + " " : "") + `${rest}/${denom}"` : `${whole}"`;
    }
  }
  return "";
}

function refreshFractions() {
  document.querySelectorAll(".frac").forEach((span) => {
    const value = parseFloat($(span.dataset.for).value);
    span.textContent = state.units === "in" && value > 0 ? fraction(value) : "";
  });
}

function setUnits(units) {
  if (units === state.units) return;
  for (const id of fields) {
    const value = parseFloat($(id).value);
    if (isFinite(value)) $(id).value = units === "mm" ? +(value * MM).toFixed(2) : +(value / MM).toFixed(4);
  }
  state.units = units;
  document.querySelectorAll("[data-units]").forEach((b) => b.classList.toggle("on", b.dataset.units === units));
  refreshFractions();
  showFacts();
  showSummary();
}

// ---------- helpers ----------

function status(text, isError) {
  $("status").textContent = text;
  $("status").classList.toggle("err", !!isError);
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* not json */ }
  if (!response.ok) throw new Error((body && body.detail) || `Something went wrong (${response.status}).`);
  return body;
}

function post(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}

function pathOf(points) {
  return points.map(([x, y], i) => (i ? "L" : "M") + x.toFixed(4) + " " + (-y).toFixed(4)).join("") + "Z";
}

function bounds(points) {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}

function el(tag, attrs, parent) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (parent) parent.appendChild(node);
  return node;
}

function drawPlate(svg, pockets) {
  const p = state.plate;
  el("path", { d: pathOf(p.outline), class: "plate" }, svg);
  for (const hole of p.holes) el("path", { d: pathOf(hole), class: "hole" }, svg);
  for (const pocket of pockets) el("path", { d: pathOf(pocket), class: "pocket" }, svg);
}

const pct = (value) => (value == null ? "–" : Math.round(value * 100) + "%");
const letter = (index) => String.fromCharCode(65 + index);

// ---------- upload ----------

async function upload(file) {
  if (!file) return;
  $("dropTitle").textContent = file.name;
  $("dropHint").textContent = "reading…";
  status("");
  const form = new FormData();
  form.append("file", file);
  try {
    state.plate = await api("/api/plate", { method: "POST", body: form });
  } catch (error) {
    $("dropTitle").textContent = "Choose a STEP file";
    $("dropHint").textContent = "or drag it here";
    status(error.message, true);
    return;
  }
  $("dropHint").textContent = "click to pick a different file";
  state.job = null;
  state.pick = null;
  for (const id of ["cands", "actions", "checks", "originalToggle"]) $(id).hidden = true;
  $("empty").hidden = true;
  $("goButton").disabled = false;
  const b = bounds(state.plate.outline);
  const pad = Math.max(b.w, b.h) * 0.06;
  state.fit = [b.x0 - pad, -b.y1 - pad, b.w + 2 * pad, b.h + 2 * pad];
  state.view = state.fit.slice();
  showFacts();
  draw();
  showSummary();
}

function showFacts() {
  const p = state.plate;
  if (!p) return;
  const size = state.units === "mm"
    ? `${(p.widthIn * MM).toFixed(0)} × ${(p.heightIn * MM).toFixed(0)} mm`
    : `${p.widthIn.toFixed(2)} × ${p.heightIn.toFixed(2)}"`;
  $("plateFacts").textContent = `${size}, ${fmtLen(p.thicknessIn)} thick, ${p.holes.length} holes, ${p.weightLb.toFixed(2)} lb`;
  $("plateFacts").hidden = false;
}

// ---------- generate ----------

async function generate() {
  if (!state.plate) return;
  const v = {};
  for (const id of fields) v[id] = parseFloat($(id).value);
  if (!["wall", "rib", "boss", "corner"].every((id) => v[id] > 0)) {
    status("Wall, rib, ring and tool radius all need a number above zero.", true);
    return;
  }
  const button = $("goButton");
  button.disabled = true;
  button.textContent = "Working…";
  const started = Date.now();
  const tick = () => {
    const s = Math.round((Date.now() - started) / 1000);
    status(`This usually takes 1–3 minutes (big plates take longer). ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`);
  };
  tick();
  const clock = setInterval(tick, 1000);
  try {
    state.job = await post("/api/generate", {
      plateId: state.plate.plateId, units: state.units,
      wall: v.wall, rib: v.rib, boss: v.boss, corner: v.corner,
      depth: state.depth === "blind" ? v.depth : null,
      density: state.density, count: Math.max(2, Math.min(8, +$("count").value || 6)),
    });
  } catch (error) {
    status(error.message, true);
    return;
  } finally {
    clearInterval(clock);
    button.disabled = false;
    button.textContent = "Generate";
  }
  status(`Done in ${Math.round(state.job.seconds)} s.` +
    (state.job.relaxed ? ` Nothing passed the strength check with your settings, so these use ${state.job.relaxed}.` : ""));
  buildCards();
  choose(0);
}

function buildCards() {
  const list = $("cands");
  list.innerHTML = "";
  const b = bounds(state.plate.outline);
  const pad = Math.max(b.w, b.h) * 0.04;
  state.job.candidates.forEach((cand, index) => {
    const item = document.createElement("li");
    item.className = "cand";
    item.tabIndex = 0;
    const svg = el("svg", { viewBox: `${b.x0 - pad} ${-b.y1 - pad} ${b.w + 2 * pad} ${b.h + 2 * pad}`, preserveAspectRatio: "xMidYMid meet" });
    drawPlate(svg, cand.pockets);
    item.appendChild(svg);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<b>${letter(index)}</b><span>−${cand.savedLb.toFixed(2)} lb</span>` +
      (cand.sound ? `<span>${pct(cand.stiffnessKept)} stiff</span>` : `<span class="weak">weak</span>`);
    item.appendChild(meta);
    item.addEventListener("click", () => choose(index));
    item.addEventListener("keydown", (e) => { if (e.key === "Enter") choose(index); });
    list.appendChild(item);
  });
  list.hidden = false;
}

function choose(index) {
  state.pick = index;
  document.querySelectorAll(".cand").forEach((item, k) => item.classList.toggle("on", k === index));
  for (const id of ["actions", "originalToggle"]) $(id).hidden = false;
  $("checks").hidden = true;
  $("rated").textContent = "";
  $("rateUp").classList.remove("picked");
  $("rateDown").classList.remove("picked");
  $("showOriginal").checked = false;
  draw();
  showSummary();
}

function current() {
  return state.job && state.pick != null ? state.job.candidates[state.pick] : null;
}

function showSummary() {
  const cand = current();
  if (!cand) { $("summary").textContent = ""; return; }
  const p = state.plate;
  const walls = [cand.minWallIn, cand.minCutoutWallIn].filter((v) => v != null);
  $("summary").innerHTML =
    `<b>Design ${letter(state.pick)}</b> — ${cand.savedLb.toFixed(2)} lb lighter (${pct(cand.removed)} removed, ` +
    `${p.weightLb.toFixed(2)} → ${(p.weightLb - cand.savedLb).toFixed(2)} lb), ` +
    `keeps ${pct(cand.stiffnessKept)} of the stiffness, thinnest wall ${walls.length ? fmtLen(Math.min(...walls)) : "–"}.` +
    (cand.sound ? "" : ` <span class="weak">Below the stiffness floor.</span>`);
}

// ---------- preview ----------

function draw() {
  const svg = $("view");
  svg.innerHTML = "";
  if (!state.plate) return;
  svg.setAttribute("viewBox", state.view.join(" "));
  const cand = current();
  drawPlate(svg, cand && !$("showOriginal").checked ? cand.pockets : []);
}

function toView(event) {
  const rect = $("view").getBoundingClientRect();
  const [vx, vy, vw, vh] = state.view;
  const scale = Math.max(vw / rect.width, vh / rect.height);
  return {
    x: vx - (rect.width * scale - vw) / 2 + (event.clientX - rect.left) * scale,
    y: vy - (rect.height * scale - vh) / 2 + (event.clientY - rect.top) * scale,
    scale,
  };
}

function setupZoom() {
  const svg = $("view");
  const apply = () => svg.setAttribute("viewBox", state.view.join(" "));
  svg.addEventListener("wheel", (event) => {
    if (!state.view) return;
    event.preventDefault();
    const point = toView(event);
    const [vx, vy, vw, vh] = state.view;
    const nw = Math.min(Math.max(vw * Math.exp(event.deltaY * 0.0015), state.fit[2] * 0.05), state.fit[2] * 4);
    const f = nw / vw;
    state.view = [point.x - (point.x - vx) * f, point.y - (point.y - vy) * f, vw * f, vh * f];
    apply();
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (event) => {
    if (!state.view) return;
    drag = { x: event.clientX, y: event.clientY, view: state.view.slice(), scale: toView(event).scale };
    svg.setPointerCapture(event.pointerId);
    svg.classList.add("panning");
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const dx = (event.clientX - drag.x) * drag.scale, dy = (event.clientY - drag.y) * drag.scale;
    state.view = [drag.view[0] - dx, drag.view[1] - dy, drag.view[2], drag.view[3]];
    apply();
  });
  const end = () => { drag = null; svg.classList.remove("panning"); };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("dblclick", () => { if (state.fit) { state.view = state.fit.slice(); apply(); } });
}

// ---------- rate + export ----------

async function rate(value) {
  if (!current()) return;
  try {
    await post("/api/rate", { jobId: state.job.jobId, index: state.pick, rating: value });
  } catch (error) {
    $("rated").textContent = error.message;
    return;
  }
  $("rateUp").classList.toggle("picked", value > 0);
  $("rateDown").classList.toggle("picked", value < 0);
  $("rated").textContent = "Thanks, saved.";
}

async function exportStep() {
  if (!current()) return;
  const button = $("exportButton");
  button.disabled = true;
  button.textContent = "Making the STEP…";
  $("checks").hidden = true;
  let result;
  try {
    result = await post("/api/export", { jobId: state.job.jobId, index: state.pick });
  } catch (error) {
    status(error.message, true);
    return;
  } finally {
    button.disabled = false;
    button.textContent = "Download STEP";
  }
  const failed = result.checks.filter((c) => !c.passed);
  const list = $("checkList");
  list.innerHTML = "";
  for (const check of failed) {
    const li = document.createElement("li");
    li.className = "fail";
    li.textContent = check.rule.replace(/^\d\s/, "");
    list.appendChild(li);
  }
  list.hidden = !failed.length;
  const link = $("download");
  if (result.ok) {
    $("checkSummary").textContent = `All ${result.checks.length} checks passed.`;
    link.href = result.url;
    link.download = result.name;
    link.textContent = `Didn't start? Download ${result.name}`;
    link.hidden = false;
    link.click();
  } else {
    $("checkSummary").textContent = "Couldn't export this one. It failed:";
    link.hidden = true;
  }
  $("checks").hidden = false;
}

// ---------- wiring ----------

const drop = $("drop");
$("fileInput").addEventListener("change", (e) => upload(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

document.querySelectorAll("[data-units]").forEach((b) => b.addEventListener("click", () => setUnits(b.dataset.units)));
document.querySelectorAll("#depthSeg button").forEach((b) => b.addEventListener("click", () => {
  state.depth = b.dataset.depth;
  document.querySelectorAll("#depthSeg button").forEach((x) => x.classList.toggle("on", x === b));
  $("depth").disabled = state.depth !== "blind";
}));
document.querySelectorAll("#densitySeg button").forEach((b) => b.addEventListener("click", () => {
  state.density = b.dataset.density;
  document.querySelectorAll("#densitySeg button").forEach((x) => x.classList.toggle("on", x === b));
}));
for (const id of fields) $(id).addEventListener("input", refreshFractions);
$("goButton").addEventListener("click", generate);
$("showOriginal").addEventListener("change", draw);
$("rateUp").addEventListener("click", () => rate(1));
$("rateDown").addEventListener("click", () => rate(-1));
$("exportButton").addEventListener("click", exportStep);
document.addEventListener("keydown", (e) => {
  if (!state.job || e.target.tagName === "INPUT") return;
  const n = state.job.candidates.length;
  if (e.key === "ArrowRight") choose((state.pick + 1) % n);
  if (e.key === "ArrowLeft") choose((state.pick - 1 + n) % n);
});
setupZoom();
refreshFractions();
