"use strict";

const state = { reports: [], galleries: [], gallery: null, frame: -1, show: "frame", role: "", context: null, branch: null };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path) {
  const response = await fetch(path);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

const fmt = (x, digits = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—"
  : (Math.abs(x) >= 1000 || (Math.abs(x) > 0 && Math.abs(x) < 1e-3) ? Number(x).toExponential(2) : Number(x).toFixed(digits));
const short = (value, n = 90) => { const t = JSON.stringify(value); return t && t.length > n ? t.slice(0, n - 3) + "..." : t; };
const stimulusText = (s) => s ? `${s.frames} frames × ${fmt(s.interval, 3)} s, ${s.height}×${s.width}, observe +${s.observe_after}` : "—";

function showView(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
}

/* ------------------------------------------------------------------ reports */
function renderReports() {
  const table = document.getElementById("reports-table");
  table.replaceChildren(el("tr", {}, ["report", "study", "subject", "masker", "stimulus", "checks", "outcomes", "keyframes"]
    .map((h) => el("th", {}, h))));
  if (!state.reports.length) { table.append(el("tr", {}, el("td", { colspan: 8, class: "empty" }, "No reports in reports/."))); return; }
  for (const r of state.reports) {
    const checks = r.checks.failed.length
      ? el("span", { class: "bad", title: r.checks.failed.join(", ") }, `${r.checks.passed}/${r.checks.total}`)
      : el("span", { class: "ok" }, `${r.checks.passed}/${r.checks.total}`);
    const kf = r.keyframes ? el("span", {}, el("span", { class: `pill ${kfClass(r.keyframes.verdict)}` }, r.keyframes.verdict),
      ` max rms ${fmt(r.keyframes.max_rms)}`, r.keyframes.warnings ? el("span", { class: "warn" }, ` · ${r.keyframes.warnings} flagged`) : "") : "—";
    const row = el("tr", { class: "clickable", onclick: () => showReport(r.name, row) },
      el("td", {}, el("div", {}, r.experiment_id), el("div", { class: "muted" }, r.name), r.sealed ? "" : el("span", { class: "bad" }, "seal broken")),
      el("td", {}, r.study || el("span", { class: "muted" }, "—")),
      el("td", {}, r.subjects.join(", ") || "—"),
      el("td", {}, r.masker),
      el("td", {}, stimulusText(r.stimulus)),
      el("td", {}, checks),
      el("td", {}, Object.entries(r.outcomes).map(([task, value]) => el("div", {}, `${task}: ${value}`))),
      el("td", {}, kf));
    table.append(row);
  }
}

const kfClass = (verdict) => ({ identical: "ok", within_tolerance: "ok", reported: "", free: "", differs: "bad", exceeds_tolerance: "bad", not_captured: "warn" })[verdict] ?? "";

async function showReport(name, row) {
  document.querySelectorAll("#reports-table tr").forEach((r) => r.classList.remove("selected"));
  row.classList.add("selected");
  const detail = await api(`/api/report?name=${encodeURIComponent(name)}`);
  const checks = el("table", { class: "grid" }, el("tr", {}, el("th", {}, "check"), el("th", {}, "title"), el("th", {}, "result")),
    detail.checks.map((c) => el("tr", {}, el("td", {}, c.id), el("td", {}, c.title),
      el("td", {}, c.passed ? el("span", { class: "ok" }, "pass") : el("span", { class: "bad" }, `FAIL ${c.detail}`)))));
  const r2 = el("table", { class: "grid" }, el("tr", {}, el("th", {}, "task"), el("th", {}, "stage"), el("th", {}, "descriptive R²")),
    Object.entries(detail.r2).flatMap(([task, roles]) => Object.entries(roles).map(([role, value]) =>
      el("tr", {}, el("td", {}, task), el("td", {}, role), el("td", {}, fmt(value))))));
  const kf = detail.keyframes;
  const keyframes = kf ? el("div", {},
    el("p", {}, el("span", { class: `pill ${kfClass(kf.verdict)}` }, kf.verdict), ` policy ${short(kf.policy)} · ${kf.status}`),
    el("table", { class: "grid" }, el("tr", {}, ["stage", "branches", "identical", "max rms", "mean rms"].map((h) => el("th", {}, h))),
      Object.entries(kf.summary || {}).map(([role, s]) => el("tr", {}, el("td", {}, role), el("td", {}, s.branches),
        el("td", {}, s.identical), el("td", {}, fmt(s.max_rms)), el("td", {}, fmt(s.mean_rms))))),
    kf.warnings && kf.warnings.length ? el("p", { class: "warn" }, `flagged: ${kf.warnings.slice(0, 12).join(", ")}${kf.warnings.length > 12 ? " …" : ""}`) : "")
    : el("p", { class: "muted" }, "No keyframe section (no stimulus declared).");
  document.getElementById("report-detail").replaceChildren(el("div", { class: "columns" },
    el("div", { class: "card" }, el("h3", {}, `${detail.summary.experiment_id}`), el("p", { class: "muted" }, detail.summary.question || ""), r2,
      detail.masking ? el("p", {}, `masker ${detail.masking.rule?.masker_id ?? "?"} · coverage ${short(detail.masking.coverage)}`) : ""),
    el("div", { class: "card" }, el("h3", {}, "Keyframe alignment"), keyframes),
    el("div", { class: "card" }, el("h3", {}, "Conformance checks"), checks),
    el("div", { class: "card" }, el("h3", {}, "Locked protocol fields (default)"), el("pre", {}, JSON.stringify(detail.locked, null, 1)))));
}

/* ------------------------------------------------------------------ compare */
function fillCompare(studies) {
  const [a, b] = ["compare-a", "compare-b"].map((id) => document.getElementById(id));
  for (const select of [a, b]) select.replaceChildren(...state.reports.map((r) => el("option", { value: r.name }, `${r.experiment_id} (${r.name})`)));
  if (state.reports.length > 1) b.selectedIndex = 1;
  const study = document.getElementById("compare-study");
  study.replaceChildren(el("option", { value: "" }, "(default fields)"), ...studies.filter((s) => !s.error).map((s) => el("option", { value: s.name }, s.study_id)));
  [a, b, study].forEach((s) => s.addEventListener("change", runCompare));
  if (state.reports.length > 1) runCompare();
}

async function runCompare() {
  const a = document.getElementById("compare-a").value, b = document.getElementById("compare-b").value;
  const study = document.getElementById("compare-study").value;
  const out = document.getElementById("compare-result");
  if (!a || !b) { out.replaceChildren(el("p", { class: "empty" }, "Need two reports.")); return; }
  const result = await api(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}${study ? `&study=${encodeURIComponent(study)}` : ""}`);
  const banner = el("div", { class: `banner ${result.comparable ? "ok" : "bad"}` },
    result.comparable ? "COMPARABLE: every compared protocol field agrees." : `NOT COMPARABLE: differs in ${result.different.join(", ")}`);
  const fields = el("table", { class: "grid" }, el("tr", {}, ["field", "", "A", "B"].map((h) => el("th", {}, h))),
    result.fields.map((row) => el("tr", {}, el("td", {}, row.field), el("td", {}, row.same ? el("span", { class: "ok" }, "same") : el("span", { class: "bad" }, "DIFF")),
      el("td", { class: row.same ? "" : "diff" }, el("code", {}, short(row.a, 160))), el("td", { class: row.same ? "" : "diff" }, el("code", {}, short(row.b, 160))))));
  const varying = el("table", { class: "grid" }, el("tr", {}, ["may vary", "A", "B"].map((h) => el("th", {}, h))),
    ["subjects", "worlds", "interfaces", "masker", "observed_channels", "observation"].map((key) => el("tr", {}, el("td", {}, key),
      el("td", {}, el("code", {}, short(result.varying.a[key], 160))), el("td", {}, el("code", {}, short(result.varying.b[key], 160))))));
  const studyLine = el("p", { class: "muted" }, `study A: ${result.study.a ?? "—"} · study B: ${result.study.b ?? "—"} · same lock: ${result.study.same_lock}`);
  out.replaceChildren(banner, studyLine, el("div", { class: "card" }, el("h3", {}, "Locked protocol fields"), fields),
    el("div", { class: "card" }, el("h3", {}, "What varies"), varying));
}

/* ------------------------------------------------------------------ studies */
function renderStudies(studies) {
  const root = document.getElementById("studies");
  if (!studies.length) { root.replaceChildren(el("p", { class: "empty" }, "No study locks in studies/.")); return; }
  root.replaceChildren(...studies.map((s) => s.error ? el("div", { class: "card" }, el("h3", {}, s.name), el("p", { class: "bad" }, s.error)) :
    el("div", { class: "card" }, el("h3", {}, s.study_id), el("p", { class: "muted" }, `${s.description || ""} · ${s.name} · digest ${s.digest.slice(0, 16)}…`),
      el("p", {}, "locked: ", s.fields.map((f) => el("span", { class: "pill" }, f))),
      el("table", { class: "grid" }, el("tr", {}, ["report", "belongs", "declares study", "differs in"].map((h) => el("th", {}, h))),
        s.members.map((m) => el("tr", {}, el("td", {}, m.report), el("td", {}, m.belongs ? el("span", { class: "ok" }, "yes") : el("span", { class: "bad" }, "no")),
          el("td", {}, m.declares ?? "—"), el("td", {}, m.different.join(", ") || "—")))),
      el("details", {}, el("summary", {}, "locked values"), el("pre", {}, JSON.stringify(s.locked, null, 1))))));
}

/* ---------------------------------------------------------------- keyframes */
async function loadGallery(name) {
  state.gallery = await api(`/api/gallery?name=${encodeURIComponent(name)}`);
  const g = state.gallery;
  document.getElementById("frame").replaceChildren(...g.frame_labels.map((label, i) => el("option", { value: i }, label)));
  const keyframe = g.frame_labels.findIndex((label) => label.includes("(keyframe)"));
  state.frame = keyframe >= 0 ? keyframe : g.frame_labels.length - 1;
  document.getElementById("frame").value = state.frame;
  const roles = [...new Set(g.contexts.map((c) => c.role))];
  document.getElementById("role").replaceChildren(el("option", { value: "" }, "all"), ...roles.map((r) => el("option", { value: r }, r)));
  state.role = ""; state.context = g.contexts.length ? g.contexts[0].context_id : null; state.branch = null;
  const p = g.meta.policy || {};
  document.getElementById("policy").textContent = `policy: ${p.mode ?? "?"}${p.warn != null ? ` · warn ${p.warn}` : ""}${p.tolerance != null ? ` · tolerance ${p.tolerance}` : ""}`;
  renderKeyframes();
}

const axisLabel = () => state.gallery.meta.axis_label || state.gallery.meta.handle_symbol || "value";
const valueText = (values) => values.map((v) => fmt(v, 4)).join(", ");
const visibleContexts = () => state.gallery.contexts.filter((c) => !state.role || c.role === state.role);
const thresholds = () => state.gallery.meta.policy || {};

function renderKeyframes() {
  if (!state.gallery) return;
  renderChart(); renderContexts(); renderStrip(); renderInspect();
}

function renderContexts() {
  const f = state.frame, warn = thresholds().warn, tol = thresholds().tolerance;
  document.getElementById("contexts").replaceChildren(...visibleContexts().map((c) => {
    const rms = c.max_rms[f];
    const cls = tol != null && rms > tol ? "bad" : warn != null && rms > warn ? "warn" : rms === 0 ? "ok" : "";
    return el("div", { class: `ctx ${c.context_id === state.context ? "active" : ""}`, onclick: () => { state.context = c.context_id; state.branch = null; renderKeyframes(); } },
      el("span", {}, c.context_id, el("span", { class: "muted" }, ` ${c.role}`)), el("span", { class: cls }, rms === 0 ? "identical" : fmt(rms)));
  }));
}

function imageUrl(branch, kind) {
  const vmax = state.gallery.frame_max_abs[state.frame] || 1;
  return `/api/image?gallery=${encodeURIComponent(state.gallery.name)}&branch=${branch}&frame=${state.frame}&kind=${kind}&vmax=${vmax}`;
}

function currentContext() { return state.gallery.contexts.find((c) => c.context_id === state.context); }

function renderStrip() {
  const strip = document.getElementById("strip"), context = currentContext();
  if (!context) { strip.replaceChildren(el("p", { class: "empty" }, "No context selected.")); return; }
  const warn = thresholds().warn, tol = thresholds().tolerance;
  strip.replaceChildren(...context.branches.map((b) => {
    const rms = b.rms[state.frame];
    const cls = b.reference ? "reference" : tol != null && rms > tol ? "exceeds" : warn != null && rms > warn ? "flagged" : "";
    const kind = state.show === "diff" && !b.reference ? "diff" : "frame";
    return el("div", { class: `thumb ${cls} ${state.branch === b.index ? "active" : ""}`, onclick: () => { state.branch = b.index; renderStrip(); renderInspect(); renderChart(); } },
      el("img", { src: imageUrl(b.index, kind), alt: `branch ${b.index}`, loading: "lazy" }),
      el("div", { class: "value" }, `${axisLabel()} = ${valueText(b.values)}`),
      el("div", { class: "metric" }, b.reference ? el("span", { class: "pill ref" }, "reference") : `rms ${rms === 0 ? "0 (identical)" : fmt(rms)}`));
  }));
}

function renderInspect() {
  const out = document.getElementById("inspect"), context = currentContext();
  const branch = context && context.branches.find((b) => b.index === state.branch);
  if (!branch) { out.replaceChildren(el("p", { class: "muted" }, "Click a branch to compare it with the reference branch.")); return; }
  const reference = context.branches.find((b) => b.reference && b.seed === branch.seed);
  const figure = (src, caption) => el("figure", {}, el("img", { src }), el("figcaption", {}, caption));
  out.replaceChildren(el("div", { class: "card" },
    el("h3", {}, `${context.context_id} · ${state.gallery.frame_labels[state.frame]} · ${axisLabel()} = ${valueText(branch.values)}`),
    el("div", { class: "inspect-row" },
      reference ? figure(imageUrl(reference.index, "frame"), `reference (${axisLabel()} = ${valueText(reference.values)})`) : "",
      figure(imageUrl(branch.index, "frame"), "this branch"),
      branch.reference ? "" : figure(imageUrl(branch.index, "diff"), `|difference|, colour scale 0 … ${fmt(state.gallery.frame_max_abs[state.frame])}`)),
    el("p", {}, `rms ${fmt(branch.rms[state.frame])} · max |Δ| ${fmt(branch.max_abs[state.frame])} · seed ${branch.seed}`)));
}

function renderChart() {
  const host = document.getElementById("axis-chart"), f = state.frame;
  const contexts = visibleContexts();
  const points = contexts.flatMap((c) => c.branches.map((b) => ({ c: c.context_id, x: b.values[0], y: b.rms[f] ?? 0, i: b.index, ref: b.reference })));
  if (!points.length) { host.replaceChildren(); return; }
  const W = 900, H = 210, L = 58, R = 14, T = 14, B = 34;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const pol = thresholds();
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(...ys, pol.warn ?? 0, pol.tolerance ?? 0) || 1;
  const sx = (x) => L + (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin)) * (W - L - R);
  const sy = (y) => H - B - (y / ymax) * (H - T - B);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%");
  const add = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text !== undefined) n.textContent = text; svg.append(n); return n; };
  add("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "#9aa5b4" });
  add("line", { x1: L, y1: T, x2: L, y2: H - B, stroke: "#9aa5b4" });
  add("text", { x: L - 6, y: sy(ymax) + 4, "text-anchor": "end", "font-size": 11, fill: "#687384" }, fmt(ymax));
  add("text", { x: L - 6, y: H - B + 4, "text-anchor": "end", "font-size": 11, fill: "#687384" }, "0");
  add("text", { x: L, y: H - 8, "font-size": 11, fill: "#687384" }, fmt(xmin, 4));
  add("text", { x: W - R, y: H - 8, "text-anchor": "end", "font-size": 11, fill: "#687384" }, fmt(xmax, 4));
  add("text", { x: (L + W) / 2, y: H - 8, "text-anchor": "middle", "font-size": 12, fill: "#1d232b" }, `${axisLabel()} (counterfactual axis) → rms of ${state.gallery.frame_labels[f]} vs reference`);
  for (const [value, colour, label] of [[pol.warn, "#b7791f", "warn"], [pol.tolerance, "#c43d3d", "tolerance"]]) {
    if (value == null) continue;
    add("line", { x1: L, x2: W - R, y1: sy(value), y2: sy(value), stroke: colour, "stroke-dasharray": "5 4" });
    add("text", { x: W - R - 4, y: sy(value) - 4, "text-anchor": "end", "font-size": 11, fill: colour }, `${label} ${value}`);
  }
  const selected = points.filter((p) => p.c === state.context).sort((a, b) => a.x - b.x);
  if (selected.length > 1) add("polyline", { points: selected.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" "), fill: "none", stroke: "#2f6fd6", "stroke-width": 1.5 });
  for (const p of points) {
    const mine = p.c === state.context;
    const dot = add("circle", { cx: sx(p.x), cy: sy(p.y), r: mine ? (p.i === state.branch ? 6 : 4) : 2.5,
      fill: p.ref ? "#6b46c1" : mine ? "#2f6fd6" : "#b9c3d0", opacity: mine ? 1 : 0.7, style: "cursor:pointer" });
    const title = document.createElementNS(ns, "title"); title.textContent = `${p.c} · ${axisLabel()}=${fmt(p.x, 4)} · rms ${fmt(p.y)}`; dot.append(title);
    dot.addEventListener("click", () => { state.context = p.c; state.branch = p.i; renderKeyframes(); });
  }
  host.replaceChildren(svg);
}

/* --------------------------------------------------------------------- boot */
async function boot() {
  document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  const info = await api("/api/workspace");
  document.getElementById("workspace").textContent = `${info.title} — ${info.reports} reports · ${info.studies} studies · ${info.keyframes} galleries`;
  const [reports, studies] = await Promise.all([api("/api/reports"), api("/api/studies")]);
  state.reports = reports;
  renderReports(); fillCompare(studies); renderStudies(studies);
  const hash = location.hash.slice(1);
  if (["reports", "compare", "studies", "keyframes"].includes(hash)) showView(hash);
  const galleries = await api("/api/galleries");
  state.galleries = galleries;
  const select = document.getElementById("gallery");
  const usable = galleries.filter((g) => !g.error);
  select.replaceChildren(...usable.map((g) => el("option", { value: g.name }, `${g.meta.experiment_id || g.name} (${g.contexts} contexts, ${g.branches} branches)`)));
  select.addEventListener("change", () => loadGallery(select.value));
  document.getElementById("frame").addEventListener("change", (e) => { state.frame = Number(e.target.value); renderKeyframes(); });
  document.getElementById("show").addEventListener("change", (e) => { state.show = e.target.value; renderStrip(); });
  document.getElementById("role").addEventListener("change", (e) => { state.role = e.target.value; renderKeyframes(); });
  if (usable.length) {
    document.getElementById("strip").replaceChildren(el("p", { class: "empty" }, "Loading gallery…"));
    await loadGallery(usable[0].name);
  } else document.getElementById("strip").replaceChildren(el("p", { class: "empty" }, "No galleries in keyframes/."));
}

boot().catch((error) => { document.querySelector("main").prepend(el("div", { class: "banner bad" }, `Error: ${error.message}`)); });
