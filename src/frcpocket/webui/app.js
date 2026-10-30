// Plate Pocketer front end. Talks to frcpocket.web; all geometry arrives in inches.

const $ = (id) => document.getElementById(id);
const MM = 25.4;

const state = {
  units: "in",
  depth: "through",
  density: "dense",
  plate: null,      // upload response
  job: null,        // generate response
  pick: null,       // selected candidate index
  view: null,       // current viewBox [x, y, w, h]
  fit: null,        // viewBox that fits the plate
};

// ---------- units ----------

const fields = ["wall", "rib", "boss", "corner", "depth"];

function inches(value) {
  return state.units === "mm" ? value / MM : value;
}

function fmtLen(valueIn, digits) {
  if (valueIn == null) return "–";
  if (state.units === "mm") return (valueIn * MM).toFixed(digits ?? 1) + " mm";
  return valueIn.toFixed(digits ?? 3) + '"';
}

function fraction(valueIn) {
  for (const denom of [2, 4, 8, 16, 32, 64]) {
    const top = Math.round(valueIn * denom);
    if (top > 0 && Math.abs(top / denom - valueIn) < 0.0006) {
      const whole = Math.floor(top / denom);
      const rest = top % denom;
      if (!rest) return `${whole}"`;
      return (whole ? whole + " " : "") + `${rest}/${denom}"`;
    }
  }
  return "";
}

function refreshFractions() {
  document.querySelectorAll(".frac").forEach((span) => {
    const value = parseFloat($(span.dataset.for).value);
    if (!(value > 0)) { span.textContent = ""; return; }
    span.textContent = state.units === "in" ? fraction(value) : "≈ " + (value / MM).toFixed(3) + '"';
  });
}

function setUnits(units) {
  if (units === state.units) return;
  for (const id of fields) {
    const input = $(id);
    const value = parseFloat(input.value);
    if (!isFinite(value)) continue;
    input.value = units === "mm" ? +(value * MM).toFixed(2) : +(value / MM).toFixed(4);
  }
  state.units = units;
  document.querySelectorAll("[data-units]").forEach((b) => b.classList.toggle("on", b.dataset.units === units));
  refreshFractions();
  showFacts();
  draw();
  fillTitleBlock();
}

// ---------- small helpers ----------

function setStep(n) {
  document.querySelectorAll(".steps span").forEach((span) => {
    const k = +span.dataset.step;
    span.classList.toggle("on", k === n);
    span.classList.toggle("done", k < n);
  });
}

function status(text, isError) {
  const el = $("status");
  el.innerHTML = text;
  el.classList.toggle("err", !!isError);
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* not json */ }
  if (!response.ok) throw new Error((body && body.detail) || `Server said ${response.status}`);
  return body;
}

function pathOf(points) {
  let d = "";
  for (let i = 0; i < points.length; i++) {
    d += (i ? "L" : "M") + points[i][0].toFixed(4) + " " + (-points[i][1]).toFixed(4);
  }
  return d + "Z";
}

function bounds(points) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of points) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}

function el(tag, attrs, parent) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (parent) parent.appendChild(node);
  return node;
}

function pct(value) {
  return value == null ? "–" : Math.round(value * 100) + "%";
}

// ---------- plate upload ----------

async function upload(file) {
  if (!file) return;
  const drop = $("drop");
  drop.classList.add("busy");
  $("dropTitle").textContent = file.name;
  $("dropHint").textContent = "Reading the plate";
  status("Reading the STEP file…");
  const form = new FormData();
  form.append("file", file);
  try {
    state.plate = await api("/api/plate", { method: "POST", body: form });
  } catch (error) {
    drop.classList.remove("busy", "loaded");
    $("dropTitle").textContent = "Drop a plate STEP here";
    $("dropHint").textContent = "or click to browse. One flat plate per file; holes, slots and bosses are kept.";
    status(error.message, true);
    return;
  }
  drop.classList.remove("busy");
  drop.classList.add("loaded");
  $("dropHint").textContent = "Click to swap for a different plate.";
  state.job = null;
  state.pick = null;
  $("tray").hidden = true;
  $("actions").hidden = true;
  $("checks").hidden = true;
  $("goButton").disabled = false;
  $("empty").hidden = true;
  const b = bounds(state.plate.outline);
  const pad = Math.max(b.w, b.h) * 0.16;
  // room under the plate for the width dimension and the title block
  state.fit = [b.x0 - pad, -b.y1 - pad, b.w + pad * 2.2, b.h + pad * 3.4];
  state.view = state.fit.slice();
  showFacts();
  draw();
  fillTitleBlock();
  setStep(2);
  status("Plate loaded. Check the rules, then generate.");
}

function showFacts() {
  const p = state.plate;
  if (!p) return;
  $("plateFacts").hidden = false;
  $("factSize").textContent = state.units === "mm"
    ? `${(p.widthIn * MM).toFixed(0)} × ${(p.heightIn * MM).toFixed(0)} mm`
    : `${p.widthIn.toFixed(2)} × ${p.heightIn.toFixed(2)}"`;
  $("factThick").textContent = fmtLen(p.thicknessIn, state.units === "mm" ? 2 : 3);
  $("factHoles").textContent = p.holes.length;
  $("factWeight").textContent = p.weightLb.toFixed(2) + " lb";
}

// ---------- generate ----------

let clock = null;

async function generate() {
  if (!state.plate) return;
  const values = {};
  for (const id of ["wall", "rib", "boss", "corner", "depth"]) values[id] = parseFloat($(id).value);
  for (const id of ["wall", "rib", "boss", "corner"]) {
    if (!(values[id] > 0)) { status(`${$(id).previousElementSibling.textContent} needs a number above zero.`, true); return; }
  }
  const body = {
    plateId: state.plate.plateId,
    units: state.units,
    wall: values.wall, rib: values.rib, boss: values.boss, corner: values.corner,
    depth: state.depth === "blind" ? values.depth : null,
    density: state.density,
    count: +$("count").value,
  };
  const button = $("goButton");
  button.disabled = true;
  button.textContent = "Working…";
  const started = Date.now();
  const tick = () => {
    const s = Math.round((Date.now() - started) / 1000);
    status(`Generating and running FEA on every design. Usually 1–3 minutes. <span class="clock">${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}</span>`);
  };
  tick();
  clock = setInterval(tick, 1000);
  try {
    state.job = await api("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (error) {
    status(error.message, true);
    return;
  } finally {
    clearInterval(clock);
    button.disabled = false;
    button.textContent = "Generate designs";
  }
  const job = state.job;
  const weak = job.candidates.filter((c) => !c.sound).length;
  status(`${job.candidates.length} designs in ${Math.round(job.seconds)} s.` +
    (weak ? ` ${weak} fall below the stiffness floor and are marked.` : ""));
  buildTray();
  choose(0);
  setStep(3);
}

// ---------- tray ----------

function buildTray() {
  const list = $("cands");
  list.innerHTML = "";
  const p = state.plate;
  const b = bounds(p.outline);
  const pad = Math.max(b.w, b.h) * 0.04;
  state.job.candidates.forEach((cand, index) => {
    const item = document.createElement("li");
    item.className = "cand";
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    const svg = el("svg", { viewBox: `${b.x0 - pad} ${-b.y1 - pad} ${b.w + 2 * pad} ${b.h + 2 * pad}`, preserveAspectRatio: "xMidYMid meet" });
    el("path", { d: pathOf(p.outline), class: "plate" }, svg);
    for (const hole of p.holes) el("path", { d: pathOf(hole), class: "hole" }, svg);
    for (const pocket of cand.pockets) el("path", { d: pathOf(pocket), class: "pocket" }, svg);
    item.innerHTML = `<span class="letter">${String.fromCharCode(65 + index)}</span><span class="src">${cand.source}</span>`;
    item.appendChild(svg);
    const stats = document.createElement("div");
    stats.className = "stats";
    stats.innerHTML = `<span>−${cand.savedLb.toFixed(2)} lb</span><span>${pct(cand.stiffnessKept)} stiff</span>`;
    item.appendChild(stats);
    if (!cand.sound) {
      const flag = document.createElement("span");
      flag.className = "flag";
      flag.textContent = "weak";
      item.appendChild(flag);
    }
    item.addEventListener("click", () => choose(index));
    item.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(index); } });
    list.appendChild(item);
  });
  $("tray").hidden = false;
  $("trayNote").textContent = "Best first. Stiffness is relative to the solid plate.";
}

function choose(index) {
  state.pick = index;
  document.querySelectorAll(".cand").forEach((item, k) => item.classList.toggle("on", k === index));
  $("actions").hidden = false;
  $("checks").hidden = true;
  $("rated").textContent = "";
  $("rateUp").classList.remove("picked");
  $("rateDown").classList.remove("picked");
  $("showOriginal").checked = false;
  draw();
  fillTitleBlock();
}

// ---------- drawing ----------

function draw() {
  const svg = $("view");
  svg.innerHTML = "";
  const p = state.plate;
  if (!p) return;
  svg.setAttribute("viewBox", state.view.join(" "));
  const cand = state.job && state.pick != null ? state.job.candidates[state.pick] : null;
  const showPockets = cand && !$("showOriginal").checked;

  el("path", { d: pathOf(p.outline), class: "plate" }, svg);
  for (const hole of p.holes) el("path", { d: pathOf(hole), class: "hole" }, svg);
  if (showPockets) for (const pocket of cand.pockets) el("path", { d: pathOf(pocket), class: "pocket" }, svg);
  if (state.job && $("showWall").checked) {
    for (const ring of state.job.wallLine) el("path", { d: pathOf(ring), class: "wall-line" }, svg);
  }
  if ($("showDims").checked) drawDims(svg, bounds(p.outline));
}

function drawDims(svg, b) {
  const size = Math.max(b.w, b.h);
  const gap = size * 0.05;
  const tick = size * 0.012;
  const font = size * 0.024;
  const g = el("g", { class: "dim" }, svg);
  const top = -b.y1, bottom = -b.y0;

  // width, under the plate
  const y = bottom + gap;
  el("line", { x1: b.x0, y1: bottom + tick, x2: b.x0, y2: y + tick }, g);
  el("line", { x1: b.x1, y1: bottom + tick, x2: b.x1, y2: y + tick }, g);
  el("line", { x1: b.x0, y1: y, x2: b.x1, y2: y }, g);
  arrow(g, b.x0, y, 1, 0, tick);
  arrow(g, b.x1, y, -1, 0, tick);
  const wText = el("text", { x: (b.x0 + b.x1) / 2, y: y + font * 1.25, "font-size": font, "text-anchor": "middle" }, g);
  wText.textContent = fmtLen(b.w, state.units === "mm" ? 1 : 3);

  // height, right of the plate
  const x = b.x1 + gap;
  el("line", { x1: b.x1 + tick, y1: top, x2: x + tick, y2: top }, g);
  el("line", { x1: b.x1 + tick, y1: bottom, x2: x + tick, y2: bottom }, g);
  el("line", { x1: x, y1: top, x2: x, y2: bottom }, g);
  arrow(g, x, top, 0, 1, tick);
  arrow(g, x, bottom, 0, -1, tick);
  const cy = (top + bottom) / 2;
  const tx = x + font * 1.1;
  const hText = el("text", { x: tx, y: cy, "font-size": font, "text-anchor": "middle", transform: `rotate(-90 ${tx} ${cy})` }, g);
  hText.textContent = fmtLen(b.h, state.units === "mm" ? 1 : 3);
}

function arrow(g, x, y, dx, dy, size) {
  const px = -dy, py = dx;
  const d = `M${x} ${y} L${x + dx * size * 2 + px * size * 0.55} ${y + dy * size * 2 + py * size * 0.55} ` +
            `L${x + dx * size * 2 - px * size * 0.55} ${y + dy * size * 2 - py * size * 0.55} Z`;
  el("path", { d, fill: "var(--pencil)", stroke: "none" }, g);
}

// zoom + pan

function toView(event) {
  const svg = $("view");
  const rect = svg.getBoundingClientRect();
  const [vx, vy, vw, vh] = state.view;
  const scale = Math.max(vw / rect.width, vh / rect.height);
  const offX = (rect.width * scale - vw) / 2;
  const offY = (rect.height * scale - vh) / 2;
  return { x: vx - offX + (event.clientX - rect.left) * scale, y: vy - offY + (event.clientY - rect.top) * scale, scale };
}

function setupView() {
  const svg = $("view");
  svg.addEventListener("wheel", (event) => {
    if (!state.view) return;
    event.preventDefault();
    const point = toView(event);
    const factor = Math.exp(event.deltaY * 0.0015);
    const [vx, vy, vw, vh] = state.view;
    const nw = Math.min(Math.max(vw * factor, state.fit[2] * 0.05), state.fit[2] * 4);
    const f = nw / vw;
    state.view = [point.x - (point.x - vx) * f, point.y - (point.y - vy) * f, vw * f, vh * f];
    svg.setAttribute("viewBox", state.view.join(" "));
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
    const dx = (event.clientX - drag.x) * drag.scale;
    const dy = (event.clientY - drag.y) * drag.scale;
    state.view = [drag.view[0] - dx, drag.view[1] - dy, drag.view[2], drag.view[3]];
    svg.setAttribute("viewBox", state.view.join(" "));
  });
  const end = () => { drag = null; svg.classList.remove("panning"); };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("dblclick", () => {
    if (!state.fit) return;
    state.view = state.fit.slice();
    svg.setAttribute("viewBox", state.view.join(" "));
  });
}

// ---------- title block ----------

function fillTitleBlock() {
  const p = state.plate;
  $("tbDate").textContent = new Date().toISOString().slice(0, 10);
  if (!p) return;
  $("tbPart").textContent = p.name.replace(/\.(step|stp)$/i, "");
  $("tbThick").textContent = fmtLen(p.thicknessIn, state.units === "mm" ? 2 : 3);
  const cand = state.job && state.pick != null ? state.job.candidates[state.pick] : null;
  const n = state.job ? state.job.candidates.length : 1;
  $("tbSheet").textContent = cand ? `${String.fromCharCode(65 + state.pick)} of ${n}` : "1 of 1";
  if (!cand) {
    $("tbDesign").textContent = "unpocketed";
    $("tbWeight").textContent = p.weightLb.toFixed(2) + " lb";
    for (const id of ["tbRemoved", "tbStiff", "tbWall", "tbStress"]) $(id).textContent = "–";
    return;
  }
  $("tbDesign").textContent = `${String.fromCharCode(65 + state.pick)} · ${cand.source}`;
  $("tbWeight").textContent = `${(p.weightLb - cand.savedLb).toFixed(2)} lb`;
  $("tbRemoved").textContent = `${pct(cand.removed)}  −${cand.savedLb.toFixed(2)} lb`;
  $("tbStiff").textContent = pct(cand.stiffnessKept);
  $("tbStiff").classList.toggle("bad", !cand.sound);
  $("tbStress").textContent = cand.stressIncrease == null ? "–" : "×" + cand.stressIncrease.toFixed(1);
  const walls = [cand.minWallIn, cand.minCutoutWallIn].filter((v) => v != null);
  $("tbWall").textContent = walls.length ? fmtLen(Math.min(...walls), state.units === "mm" ? 2 : 3) : "–";
}

// ---------- rate + export ----------

async function rate(value) {
  if (!state.job || state.pick == null) return;
  try {
    await api("/api/rate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId: state.job.jobId, index: state.pick, rating: value }) });
  } catch (error) {
    $("rated").textContent = error.message;
    return;
  }
  $("rateUp").classList.toggle("picked", value > 0);
  $("rateDown").classList.toggle("picked", value < 0);
  $("rated").textContent = "Saved. It counts next time the scorer is retrained.";
}

async function exportStep() {
  if (!state.job || state.pick == null) return;
  const button = $("exportButton");
  button.disabled = true;
  button.textContent = "Cutting the solid…";
  $("checks").hidden = true;
  let result;
  try {
    result = await api("/api/export", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId: state.job.jobId, index: state.pick }) });
  } catch (error) {
    status(error.message, true);
    return;
  } finally {
    button.disabled = false;
    button.textContent = "Export STEP";
  }
  const list = $("checkList");
  list.innerHTML = "";
  for (const check of result.checks) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="${check.passed ? "mark-ok" : "mark-no"}">${check.passed ? "OK" : "FAIL"}</span>`;
    li.append(check.rule.replace(/^\d\s/, ""));
    list.appendChild(li);
  }
  const link = $("download");
  if (result.ok) {
    link.href = result.url;
    link.download = result.name;
    link.textContent = `Download ${result.name}`;
    link.hidden = false;
    setStep(4);
  } else {
    link.hidden = true;
    status("Export blocked: the exact geometry failed a check (listed below). Try another design.", true);
  }
  $("checks").hidden = false;
}

// ---------- wiring ----------

function wire() {
  const drop = $("drop");
  $("fileInput").addEventListener("change", (event) => upload(event.target.files[0]));
  ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (event) => { event.preventDefault(); upload(event.dataTransfer.files[0]); });
  // dropping anywhere on the page still works
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => { event.preventDefault(); if (event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]); });

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
  $("count").addEventListener("input", () => { $("countOut").textContent = $("count").value; });
  $("goButton").addEventListener("click", generate);
  for (const id of ["showOriginal", "showWall", "showDims"]) $(id).addEventListener("change", draw);
  $("rateUp").addEventListener("click", () => rate(1));
  $("rateDown").addEventListener("click", () => rate(-1));
  $("exportButton").addEventListener("click", exportStep);
  document.addEventListener("keydown", (event) => {
    if (!state.job || event.target.tagName === "INPUT") return;
    const n = state.job.candidates.length;
    if (event.key === "ArrowRight") choose((state.pick + 1) % n);
    if (event.key === "ArrowLeft") choose((state.pick - 1 + n) % n);
  });
  setupView();
  refreshFractions();
  fillTitleBlock();
}

wire();
