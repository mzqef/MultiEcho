"use strict";
// Runs: every run of the workspace, and one run's stages in the order the protocol enforces.

const view = { id: null, record: null, timer: null, methods: {} };
const STEP_TITLES = {
  measure: ["Measure discovery", "Every discovery context is measured at the reference and at each declared value; determinism is checked by repeating the reference branch; a data-driven masker is calibrated and frozen here."],
  freeze: ["Freeze laws", "Fit one method per task on discovery responses only, then freeze it. A frozen law cannot be refitted, and held-out contexts cannot be measured before every task is frozen."],
  test: ["Test held-out contexts", "Validation and confirmation contexts are measured now. Each frozen law sees a context's reference and calibration responses and must predict the queries; it is judged against fixed controls."],
  seal: ["Seal the report", "The report is sealed with the declaration, stages, evaluations and outcomes, written to reports/, and checked by the 22 conformance checks."],
};

/* ---------------------------------------------------------------- list */
PAGES.runs = {
  async show() {
    const runs = await api("/api/runs");
    const root = document.getElementById("view-runs");
    mount(root, el("h1", {}, "Runs"),
      el("p", { class: "lead" }, "Each run measures one design. A run stays live while the workbench runs; its record, report and keyframes stay in the workspace."),
      el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Run", "Design", "Experiment", "Created", "Stage", "Outcomes", "Report"].map((h) => el("th", {}, h)))),
        el("tbody", {}, runs.length ? runs.map((run) => el("tr", { class: "clickable", onclick: () => go("run", run.id) },
          el("td", {}, run.id, run.live ? "" : el("div", { class: "note" }, "record only")),
          el("td", {}, run.design), el("td", {}, run.experiment_id), el("td", {}, when(run.created_at)),
          el("td", {}, run.busy ? badge(`${run.busy}…`) : badge(run.stage), run.error ? el("div", { class: "bad small" }, run.error) : ""),
          el("td", {}, Object.entries(run.outcomes || {}).map(([task, value]) => el("div", {}, `${task}: `, badge(value)))),
          el("td", {}, run.report || "—"))) : [emptyRow("No runs yet. Open an experiment and press Run.", 7)])));
  },
};

/* --------------------------------------------------------------- charts */
function ticks(lo, hi, count = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count, power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function extent(values, pad = 0.06) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return [0, 1];
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (lo === hi) { lo -= Math.abs(lo) * 0.1 || 1; hi += Math.abs(hi) * 0.1 || 1; }
  const margin = (hi - lo) * pad;
  return [lo - margin, hi + margin];
}

function plot({ title, x, y, xlabel, ylabel, draw, width = 440, height = 270, legend }) {
  const L = 54, R = 12, T = 12, B = 40;
  const sx = (v) => L + ((v - x[0]) / (x[1] - x[0])) * (width - L - R);
  const sy = (v) => height - B - ((v - y[0]) / (y[1] - y[0])) * (height - T - B);
  const text = (px, py, value, extra = {}) => svg("text", { x: px, y: py, "font-size": 10.5, fill: "#66707e", ...extra }, value);
  const label = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(1) : String(+v.toPrecision(4)));
  return el("figure", { class: "chart" }, el("figcaption", {}, title),
    svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height },
      ticks(x[0], x[1]).map((v) => [svg("line", { x1: sx(v), x2: sx(v), y1: T, y2: height - B, stroke: "#eef1f5" }), text(sx(v), height - B + 14, label(v), { "text-anchor": "middle" })]),
      ticks(y[0], y[1]).map((v) => [svg("line", { x1: L, x2: width - R, y1: sy(v), y2: sy(v), stroke: "#eef1f5" }), text(L - 6, sy(v) + 3, label(v), { "text-anchor": "end" })]),
      svg("line", { x1: L, y1: height - B, x2: width - R, y2: height - B, stroke: "#9aa5b4" }),
      svg("line", { x1: L, y1: T, x2: L, y2: height - B, stroke: "#9aa5b4" }),
      text((L + width - R) / 2, height - 6, xlabel, { "text-anchor": "middle", fill: "#1c2229" }),
      text(12, (T + height - B) / 2, ylabel, { "text-anchor": "middle", fill: "#1c2229", transform: `rotate(-90 12 ${(T + height - B) / 2})` }),
      draw(sx, sy)),
    legend ? el("div", { class: "legend" }, legend) : "");
}

const valueColour = (v, lo, hi) => `hsl(${Math.round(220 - 220 * (hi > lo ? (v - lo) / (hi - lo) : 0.5))}, 70%, 45%)`;
const ROLE_COLOURS = { validation: "#2f6fd6", confirmation: "#c43d3d" };
const tip = (text) => svg("title", {}, text);

function discoveryCharts(d) {
  const axis = d.axes[0] || "value", branches = d.contexts.flatMap((c) => c.branches.map((b) => ({ ...b, context: c.context_id })));
  const xs = branches.map((b) => b.values[0]), ref = d.reference[0];
  const x = extent([...xs, ref]), y = extent([0, ...branches.map((b) => b.effect_rms)]);
  const [lo, hi] = [Math.min(...xs), Math.max(...xs)];
  const effect = plot({ title: "Effect size along the axis (one line per discovery context)", x, y, xlabel: axis,
    ylabel: "effect RMS", draw: (sx, sy) => [
      d.contexts.map((c) => {
        const points = [...c.branches.map((b) => [b.values[0], b.effect_rms]), [ref, 0]].sort((a, b) => a[0] - b[0]);
        return svg("polyline", { points: points.map(([v, e]) => `${sx(v)},${sy(e)}`).join(" "), fill: "none", stroke: "#9fb4d6", "stroke-width": 1.2 }, tip(c.context_id));
      }),
      branches.map((b) => svg("circle", { cx: sx(b.values[0]), cy: sy(b.effect_rms), r: 3, fill: b.kind === "calibration" ? "#6b46c1" : "#2f6fd6" },
        tip(`${b.context} · ${axis} = ${b.values.join(", ")} · ${b.kind} · effect RMS ${fmt(b.effect_rms, 4)}`))),
      svg("line", { x1: sx(ref), x2: sx(ref), y1: sy(y[0]), y2: sy(y[1]), stroke: "#1d8a4e", "stroke-dasharray": "4 4" }, tip(`reference ${ref}`))],
    legend: [el("span", {}, el("i", { style: "background:#6b46c1" }), "calibration"), el("span", {}, el("i", { style: "background:#2f6fd6" }), "query"),
      el("span", {}, el("i", { style: "background:#1d8a4e" }), "reference (effect 0 by definition)")] });
  const pcs = branches.filter((b) => b.pc.length === 2);
  const scatter = pcs.length ? plot({ title: `Effects in their two main directions (${(100 * (d.explained[0] || 0)).toFixed(0)}% and ${(100 * (d.explained[1] || 0)).toFixed(0)}% of the variance)`,
    x: extent(pcs.map((b) => b.pc[0])), y: extent(pcs.map((b) => b.pc[1])), xlabel: "direction 1", ylabel: "direction 2",
    draw: (sx, sy) => [
      d.contexts.map((c) => svg("polyline", { points: c.branches.filter((b) => b.pc.length === 2).map((b) => `${sx(b.pc[0])},${sy(b.pc[1])}`).join(" "),
        fill: "none", stroke: "#dde2e8" })),
      pcs.map((b) => svg("circle", { cx: sx(b.pc[0]), cy: sy(b.pc[1]), r: 3.5, fill: valueColour(b.values[0], lo, hi) },
        tip(`${b.context} · ${axis} = ${b.values.join(", ")}`)))],
    legend: [el("span", {}, `colour: ${axis} from `, el("b", { style: `color:${valueColour(lo, lo, hi)}` }, fmt(lo, 3)), " to ",
      el("b", { style: `color:${valueColour(hi, lo, hi)}` }, fmt(hi, 3)))] }) : "";
  return el("div", { class: "charts" }, effect, scatter);
}

function testCharts(tests, axis) {
  const rows = Object.entries(tests.diagnostics).flatMap(([role, items]) => items.map((item) => ({ ...item, role })));
  const out = [];
  const backward = rows.filter((r) => r.backward && r.backward.recovered.length);
  if (backward.length) {
    const all = backward.flatMap((r) => [r.values[0], r.backward.recovered[0]]);
    const range = extent(all);
    out.push(plot({ title: "Backward: the value recovered from each held-out query response", x: range, y: range,
      xlabel: `true ${axis}`, ylabel: `recovered ${axis}`, draw: (sx, sy) => [
        svg("line", { x1: sx(range[0]), y1: sy(range[0]), x2: sx(range[1]), y2: sy(range[1]), stroke: "#9aa5b4", "stroke-dasharray": "4 4" }),
        backward.map((r) => svg("circle", { cx: sx(r.values[0]), cy: sy(r.backward.recovered[0]), r: 3.5, fill: ROLE_COLOURS[r.role], opacity: 0.85 },
          tip(`${r.context_id} (${r.role}) · true ${r.values[0]} · recovered ${fmt(r.backward.recovered[0], 4)}`)))],
      legend: [...Object.entries(ROLE_COLOURS).map(([role, colour]) => el("span", {}, el("i", { style: `background:${colour}` }), role)),
        el("span", {}, "dashed: perfect recovery")] }));
  }
  const forward = rows.filter((r) => r.forward && r.forward.observed_pc.length);
  if (forward.length) {
    const all = forward.flatMap((r) => [r.forward.observed_pc[0], r.forward.predicted_pc[0]]);
    const range = extent(all);
    out.push(plot({ title: "Forward: predicted and observed responses along the main discovery direction", x: range, y: range,
      xlabel: "observed (direction 1)", ylabel: "predicted (direction 1)", draw: (sx, sy) => [
        svg("line", { x1: sx(range[0]), y1: sy(range[0]), x2: sx(range[1]), y2: sy(range[1]), stroke: "#9aa5b4", "stroke-dasharray": "4 4" }),
        forward.map((r) => svg("circle", { cx: sx(r.forward.observed_pc[0]), cy: sy(r.forward.predicted_pc[0]), r: 3.5, fill: ROLE_COLOURS[r.role], opacity: 0.85 },
          tip(`${r.context_id} (${r.role}) · ${axis} = ${r.values.join(", ")} · error RMS ${fmt(r.forward.error_rms, 4)} of effect RMS ${fmt(r.effect_rms, 4)}`)))],
      legend: [...Object.entries(ROLE_COLOURS).map(([role, colour]) => el("span", {}, el("i", { style: `background:${colour}` }), role)),
        el("span", {}, "dashed: perfect prediction")] }));
  }
  return el("div", { class: "charts" }, out);
}

/* -------------------------------------------------------------- sections */
function evaluationTable(task, roles) {
  const stages = Object.entries(roles).filter(([role]) => role !== "regimes");
  return el("div", {}, el("table", { class: "grid" },
    el("thead", {}, el("tr", {}, ["Stage", "Queries", "Descriptive R²", "Mean squared error", "Control", "Advantage over control", "Interval", "Evidence"].map((h) => el("th", {}, h)))),
    el("tbody", {}, stages.flatMap(([role, e]) => e.comparisons.map((c, i) => el("tr", {},
      i === 0 ? el("td", { rowspan: e.comparisons.length }, role) : "", i === 0 ? el("td", { rowspan: e.comparisons.length }, e.queries) : "",
      i === 0 ? el("td", { rowspan: e.comparisons.length, class: "num" }, fmt(e.descriptive_r2)) : "",
      i === 0 ? el("td", { rowspan: e.comparisons.length, class: "num" }, fmt(e.group_weighted_mse, 5)) : "",
      el("td", {}, c.comparator), el("td", { class: "num" }, fmt(c.advantage, 5)),
      el("td", { class: "num" }, c.interval ? `[${fmt(c.interval[0], 5)}, ${fmt(c.interval[1], 5)}] at ${(100 * c.confidence).toFixed(1)}%` : "—"),
      el("td", {}, badge(c.status))))))),
  Object.keys(roles.regimes || {}).length ? el("div", { class: "note" }, "By condition: ", Object.entries(roles.regimes).map(([name, values]) =>
    Object.entries(values).map(([value, stages]) => el("span", { class: "pill" }, `${name} = ${value}: `,
      Object.entries(stages).map(([stage, comparisons]) => `${stage} ${Object.values(comparisons).every((s) => s === "supported") ? "supported" : Object.values(comparisons).join("/")}`).join("; "))))) : "");
}

function stepCard(key, done, allowed, busy, content, action) {
  const [title, text] = STEP_TITLES[key];
  const status = busy ? "running" : done ? "done" : allowed ? "ready" : "waiting";
  return el("section", { class: `stage ${status}` },
    el("div", { class: "stage-head" }, el("span", { class: "stepnum" }, ["measure", "freeze", "test", "seal"].indexOf(key) + 1),
      el("h2", {}, title), el("span", { class: `pill ${status === "done" ? "ok" : status === "running" ? "warn" : ""}` }, status),
      el("span", { class: "spacer" }), action || ""),
    el("p", { class: "note" }, text), content || "");
}

function progress(record) {
  const p = record.progress;
  if (!record.busy) return "";
  const share = p && p.total ? Math.min(1, p.done / p.total) : null;
  return el("div", { class: "progress" }, el("div", { class: "bar", style: `width:${share === null ? 100 : Math.round(100 * share)}%` }),
    el("span", {}, p ? `${p.phase}: ${p.done}${p.total ? ` of ${p.total}` : ""}` : `${record.busy}…`));
}

async function act(action, extra = {}) {
  try {
    await post("/api/run/act", { id: view.id, action, ...extra });
    await refresh();
  } catch (error) { notice(error.message); }
}

function renderRun() {
  const r = view.record, root = document.getElementById("view-run");
  if (!r) return;
  const allowed = r.allowed || {}, busy = r.busy;
  const tasks = r.tasks || [];
  const d = r.discovery;
  const measured = d ? el("div", {},
    el("div", { class: "stats" },
      el("div", { class: "stat" }, el("span", {}, "pairs measured"), el("strong", {}, d.pairs)),
      el("div", { class: "stat" }, el("span", {}, "observed values"), el("strong", {}, d.dims)),
      el("div", { class: "stat" }, el("span", {}, "effect RMS"), el("strong", {}, fmt(d.effect_rms, 4))),
      el("div", { class: "stat" }, el("span", {}, "determinism"), el("strong", { class: d.determinism.identical ? "ok" : "bad" },
        d.determinism.identical ? "identical" : `differs by ${fmt(d.determinism.max_abs_difference)}`))),
    discoveryCharts(d),
    d.masking && d.masking.frozen ? el("p", {}, "Masker frozen on discovery: ", el("b", {}, d.masking.frozen.masker_id),
      ` (${d.masking.frozen.kind}; needs ${d.masking.frozen.needs})`,
      Object.entries(d.masking.coverage || {}).map(([role, share]) => ` · keeps ${fmt(100 * share, 1)}% of the observation in ${role} contexts`).join(""),
      el("details", {}, el("summary", {}, "frozen rule"), el("pre", {}, JSON.stringify(d.masking.frozen, null, 1)))) : "",
    d.keyframes ? el("p", {}, "Keyframes: ", badge(d.keyframes.verdict), ` ${d.keyframes.status}`,
      Object.entries(d.keyframes.summary || {}).map(([role, s]) => ` · ${role}: ${s.branches} branches, ${s.identical} identical, max RMS ${fmt(s.max_rms, 4)}`).join(""),
      (d.keyframes.warnings || []).length ? el("span", { class: "warn" }, ` · ${(d.keyframes.warnings || []).length} flagged above the warning level`) : "") : "") : "";

  const laws = tasks.map((task) => {
    const law = r.laws[task];
    if (law) {
      return el("div", { class: "law" }, el("h4", {}, task === "forward" ? "Forward law" : "Backward law", " ", badge("frozen")),
        el("p", {}, el("b", {}, law.law_id), ` — ${law.hypothesis}`),
        el("p", { class: "note" }, `method ${law.method.plugin.split(":").pop()} ${Object.keys(law.method.params).length ? short(law.method.params) : ""} · `,
          `validity: ${law.validity_domain} · fitted on ${law.fitted_on} discovery contexts · digest ${digest(law.digest, 16)}`));
    }
    view.methods[task] = view.methods[task] || JSON.parse(JSON.stringify((r.methods || {})[task] || { plugin: null, params: {} }));
    return el("div", { class: "law" }, el("h4", {}, task === "forward" ? "Forward: predict the response" : "Backward: recover the intervention"),
      allowed.freeze ? pluginPicker(view.methods[task], task, "Method") : el("p", { class: "note" }, "Available after discovery is measured."),
      el("button", { class: "primary", disabled: !allowed.freeze || busy || !view.methods[task].plugin,
        onclick: () => act("freeze", { task, method: view.methods[task] }) }, `Fit and freeze the ${task} law`));
  });

  const tested = r.tests ? el("div", {}, Object.entries(r.tests.tasks).map(([task, roles]) => el("div", {}, el("h4", {}, `${task} law on held-out contexts`), evaluationTable(task, roles))),
    testCharts(r.tests, (d && d.axes[0]) || "value")) : "";

  const rep = r.report;
  const sealed = rep ? el("div", {},
    Object.entries(rep.outcomes).map(([task, o]) => el("div", { class: "outcome" }, el("h4", {}, `${task}: `, badge(o.value)),
      el("p", {}, o.meaning), o.claim ? el("p", { class: "note" }, `Claim: ${o.claim}`) : "",
      Object.keys(o.holds_in || {}).length ? el("p", { class: "note" }, `Holds in: ${Object.entries(o.holds_in).map(([k, v]) => `${k} = ${[].concat(v).join(", ")}`).join("; ")}`) : "")),
    el("p", {}, "Conformance: ", el("b", { class: rep.checks.failed.length ? "bad" : "ok" }, `${rep.checks.passed}/${rep.checks.total} checks passed`),
      rep.checks.failed.length ? el("span", { class: "bad" }, ` · failed ${rep.checks.failed.map((c) => c.id).join(", ")}`) : ""),
    el("p", {}, el("button", { onclick: () => go("reports", rep.name) }, "Open the report"), " ",
      rep.gallery ? el("button", { onclick: () => go("keyframes", rep.gallery) }, "Open the keyframes") : "",
      el("span", { class: "note" }, ` seal ${digest(rep.seal, 16)}`))) : "";

  mount(root, 
    el("div", { class: "title-row" }, el("h1", {}, r.id), badge(busy ? `${busy}…` : r.stage),
      r.live || r.stage === "sealed" ? "" : el("span", { class: "pill warn" }, "record only: the workbench restarted; start a new run of the design"),
      el("span", { class: "spacer" }),
      el("button", { class: "primary", disabled: busy || !r.live || r.stage === "sealed", onclick: () => act("all", { methods: view.methods }) }, "Run every remaining stage")),
    el("p", { class: "lead" }, "Design ", el("a", { href: `#experiment/${encodeURIComponent(r.design)}` }, r.design), ` · experiment ${r.experiment_id} · created ${when(r.created_at)}`,
      ` · chain ${digest(r.chain_digest)} · declaration ${digest(r.declaration_digest)}`),
    r.error ? el("div", { class: "banner bad" }, r.error, r.traceback ? el("details", {}, el("summary", {}, "traceback"), el("pre", {}, r.traceback)) : "") : "",
    progress(r),
    stepCard("measure", Boolean(d), allowed.measure, busy === "measure" || (busy === "all" && !d), measured,
      el("button", { class: "primary", disabled: !allowed.measure || busy, onclick: () => act("measure") }, "Measure discovery")),
    stepCard("freeze", tasks.every((t) => r.laws[t]), allowed.freeze, busy === "freeze", el("div", { class: "laws" }, laws)),
    stepCard("test", Boolean(r.tests), allowed.test, busy === "test" || (busy === "all" && d && !r.tests), tested,
      el("button", { class: "primary", disabled: !allowed.test || busy, onclick: () => act("test") }, "Measure held-out contexts and test")),
    stepCard("seal", Boolean(rep), allowed.seal, busy === "seal", sealed,
      el("button", { class: "primary", disabled: !allowed.seal || busy, onclick: () => act("seal") }, "Seal the report")),
    el("details", { class: "log" }, el("summary", {}, `Log (${(r.log || []).length})`), el("table", { class: "grid" },
      (r.log || []).map((entry) => el("tr", {}, el("td", {}, when(entry.at)), el("td", {}, entry.stage), el("td", {}, entry.note),
        el("td", { class: "num" }, entry.seconds !== undefined ? `${entry.seconds} s` : ""))))));
}

async function refresh() {
  view.record = await api(`/api/run?id=${encodeURIComponent(view.id)}`);
  renderRun();
  clearTimeout(view.timer);
  if (view.record.busy) view.timer = setTimeout(() => refresh().catch((error) => notice(error.message)), 700);
}

PAGES.run = {
  async show([id]) {
    if (!id) { go("runs"); return; }
    if (view.id !== id) Object.assign(view, { id, record: null, methods: {} });
    hooks.changed = () => {};
    hooks.render = () => renderRun();
    await refresh();
  },
  leave() { clearTimeout(view.timer); },
};
