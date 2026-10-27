"use strict";
// Records: sealed reports and their checks, report comparison, study locks, and keyframe galleries.

const records = { reports: [], studies: [], galleries: [], gallery: null, frame: -1, show: "frame", role: "", context: null, branch: null };
const stimulusText = (s) => (s ? `${s.frames} frames × ${fmt(s.interval, 3)} s, ${s.height}×${s.width}` : "—");
const table = (heads, rows) => el("table", { class: "grid" }, el("thead", {}, el("tr", {}, heads.map((h) => el("th", {}, h)))), el("tbody", {}, rows));

/* ----------------------------------------------------------------- reports */
PAGES.reports = {
  async show([name]) {
    [records.reports, records.studies] = await Promise.all([api("/api/reports"), api("/api/studies")]);
    const sub = name === "compare" ? "compare" : "list";
    document.querySelectorAll("#view-reports .subnav button").forEach((b) => {
      b.classList.toggle("active", b.dataset.sub === sub);
      b.onclick = () => go("reports", b.dataset.sub === "compare" ? "compare" : "");
    });
    document.getElementById("reports-list").classList.toggle("active", sub === "list");
    document.getElementById("reports-compare").classList.toggle("active", sub === "compare");
    renderReports(sub === "list" ? name : null);
    fillCompare();
    if (sub === "list" && name) await showReport(name);
    else document.getElementById("report-detail").replaceChildren();
  },
};

function renderReports(selected) {
  const host = document.getElementById("reports-table");
  const rows = records.reports.map((r) => el("tr", { class: `clickable ${r.name === selected ? "selected" : ""}`, onclick: () => go("reports", r.name) },
    el("td", {}, el("b", {}, r.experiment_id), el("div", { class: "note" }, r.name), r.sealed ? "" : el("div", { class: "bad" }, "seal broken")),
    el("td", {}, r.worlds.join(" → ")),
    el("td", {}, r.masker),
    el("td", {}, stimulusText(r.stimulus)),
    el("td", {}, r.study || "—"),
    el("td", {}, Object.entries(r.outcomes).map(([task, value]) => el("div", {}, `${task}: `, badge(value)))),
    el("td", {}, el("span", { class: r.checks.failed.length ? "bad" : "ok", title: r.checks.failed.join(", ") }, `${r.checks.passed}/${r.checks.total}`)),
    el("td", {}, r.keyframes ? el("span", {}, badge(r.keyframes.verdict), r.keyframes.warnings ? el("span", { class: "note" }, ` ${r.keyframes.warnings} flagged`) : "") : "—")));
  mount(host, el("thead", {}, el("tr", {}, ["Experiment", "Chain", "Masker", "Stimulus", "Study", "Outcomes", "Checks", "Keyframes"].map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.length ? rows : emptyRow("No reports yet. Runs write their sealed reports here.", 8)));
}

async function showReport(name) {
  const detail = await api(`/api/report?name=${encodeURIComponent(name)}`);
  const d = detail.declaration, s = detail.summary;
  const analyses = Object.entries(detail.analyses).map(([task, a]) => el("div", { class: "card" },
    el("h3", {}, `${task}: `, badge(a.outcome && a.outcome.value)),
    el("p", {}, a.outcome && a.outcome.meaning), a.claim ? el("p", { class: "note" }, `Claim: ${a.claim}`) : "",
    a.law ? el("p", { class: "note" }, `Law ${a.law}: ${a.hypothesis || ""}`) : "",
    table(["Stage", "Queries", "Descriptive R²", "Control", "Advantage", "Interval", "Evidence"],
      Object.entries(a.evaluations || {}).flatMap(([role, e]) => Object.entries(e.controls || {}).map(([control, c], i, list) => el("tr", {},
        i === 0 ? el("td", { rowspan: list.length }, role) : "", i === 0 ? el("td", { rowspan: list.length }, e.queries) : "",
        i === 0 ? el("td", { rowspan: list.length, class: "num" }, fmt(e.descriptive_r2)) : "",
        el("td", {}, control), el("td", { class: "num" }, fmt(c.advantage, 5)),
        el("td", { class: "num" }, c.interval ? `[${fmt(c.interval[0], 5)}, ${fmt(c.interval[1], 5)}]` : "—"), el("td", {}, badge(c.status))))))));
  const kf = detail.keyframes;
  const keyframes = kf ? el("div", {}, el("p", {}, badge(kf.verdict), ` ${kf.status} · policy ${short(kf.policy)}`),
    table(["Stage", "Branches", "Identical", "Max RMS", "Mean RMS"], Object.entries(kf.summary || {}).map(([role, x]) =>
      el("tr", {}, el("td", {}, role), el("td", {}, x.branches), el("td", {}, x.identical), el("td", {}, fmt(x.max_rms)), el("td", {}, fmt(x.mean_rms))))),
    (kf.warnings || []).length ? el("p", { class: "warn" }, `${kf.warnings.length} flagged: ${kf.warnings.slice(0, 10).join(", ")}${kf.warnings.length > 10 ? " …" : ""}`) : "")
    : el("p", { class: "note" }, "No keyframe section (no stimulus declared).");
  document.getElementById("report-detail").replaceChildren(
    el("div", { class: "title-row" }, el("h2", {}, s.experiment_id), s.sealed ? badge("sealed") : el("span", { class: "pill bad" }, "seal broken"),
      el("span", { class: "note" }, name)),
    el("p", { class: "lead" }, d.question || ""),
    el("p", { class: "note" }, `Intervention: ${d.handle} · observed: ${d.observation_time} · independence unit: ${d.independence_unit} · `,
      `contexts ${Object.entries(d.contexts).map(([role, n]) => `${role} ${n}`).join(", ")} · draws ${d.draws} (${d.coupling})`),
    el("div", { class: "columns" }, analyses),
    el("div", { class: "columns" },
      el("div", { class: "card" }, el("h3", {}, "Conformance checks"), table(["Check", "Title", "Result"], detail.checks.map((c) => el("tr", {},
        el("td", {}, c.id), el("td", {}, c.title), el("td", {}, c.passed ? el("span", { class: "ok" }, "pass") : el("span", { class: "bad" }, `fail: ${c.detail}`)))))),
      el("div", { class: "card" }, el("h3", {}, "Keyframe alignment"), keyframes,
        detail.masking ? el("div", {}, el("h3", {}, "Masking"), el("pre", {}, JSON.stringify(detail.masking, null, 1))) : ""),
      el("div", { class: "card" }, el("h3", {}, "Locked protocol fields (default lock)"), el("pre", {}, JSON.stringify(detail.locked, null, 1)))));
}

/* ----------------------------------------------------------------- compare */
function fillCompare() {
  const [a, b, study] = ["compare-a", "compare-b", "compare-study"].map((id) => document.getElementById(id));
  const keep = [a.value, b.value, study.value];
  for (const select of [a, b]) select.replaceChildren(...records.reports.map((r) => el("option", { value: r.name }, `${r.experiment_id} (${r.name})`)));
  study.replaceChildren(el("option", { value: "" }, "the default locked fields"),
    ...records.studies.filter((s) => !s.error).map((s) => el("option", { value: s.name }, `the fields of ${s.study_id}`)));
  if (keep[0] && records.reports.some((r) => r.name === keep[0])) [a.value, b.value, study.value] = keep;
  else if (records.reports.length > 1) b.selectedIndex = 1;
  [a, b, study].forEach((select) => { select.onchange = runCompare; });
  runCompare();
}

async function runCompare() {
  const a = document.getElementById("compare-a").value, b = document.getElementById("compare-b").value;
  const study = document.getElementById("compare-study").value, out = document.getElementById("compare-result");
  if (!a || !b || records.reports.length < 2) { mount(out, el("p", { class: "empty" }, "Comparing needs two reports.")); return; }
  try {
    const result = await api(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}${study ? `&study=${encodeURIComponent(study)}` : ""}`);
    mount(out, 
      el("div", { class: `banner ${result.comparable ? "ok" : "bad"}` }, result.comparable ? "Comparable: every compared protocol field agrees."
        : `Not comparable: they differ in ${result.different.join(", ")}.`),
      el("p", { class: "note" }, `Study of A: ${result.study.a ?? "—"} · study of B: ${result.study.b ?? "—"} · same lock: ${result.study.same_lock}`),
      el("div", { class: "card" }, el("h3", {}, "Protocol fields"), table(["Field", "", "A", "B"], result.fields.map((row) => el("tr", {},
        el("td", {}, row.field), el("td", {}, el("span", { class: row.same ? "ok" : "bad" }, row.same ? "same" : "differs")),
        el("td", { class: row.same ? "" : "diff" }, el("code", {}, short(row.a, 160))), el("td", { class: row.same ? "" : "diff" }, el("code", {}, short(row.b, 160))))))),
      el("div", { class: "card" }, el("h3", {}, "What may vary"), table(["Field", "A", "B"], ["subjects", "worlds", "interfaces", "masker", "observed_channels", "observation"].map((key) =>
        el("tr", {}, el("td", {}, key), el("td", {}, el("code", {}, short(result.varying.a[key], 160))), el("td", {}, el("code", {}, short(result.varying.b[key], 160))))))));
  } catch (error) { mount(out, el("div", { class: "banner bad" }, error.message)); }
}

/* ----------------------------------------------------------------- studies */
PAGES.studies = {
  async show() {
    [records.studies, records.reports] = await Promise.all([api("/api/studies"), api("/api/reports")]);
    const root = document.getElementById("view-studies");
    const lockable = state.info.lockable || {}, chosen = new Set(state.info.default_locked || []);
    const report = el("select", {}, records.reports.map((r) => el("option", { value: r.name }, `${r.experiment_id} (${r.name})`)));
    const studyId = el("input", { placeholder: "my-study" }), description = el("input", { placeholder: "what the study compares", size: 40 });
    const boxes = Object.entries(lockable).map(([key, text]) => el("label", { class: "tick", title: text },
      el("input", { type: "checkbox", checked: chosen.has(key), onchange: (e) => (e.target.checked ? chosen.add(key) : chosen.delete(key)) }), ` ${key}`));
    const create = async () => {
      try {
        const made = await post("/api/study/create", { report: report.value, study_id: studyId.value.trim(), description: description.value, fields: [...chosen] });
        notice(`Created the study lock ${made.name}. Declare it in a design (Declaration → Study lock) to admit only comparable experiments.`, "ok");
        PAGES.studies.show();
      } catch (error) { notice(error.message); }
    };
    mount(root, 
      el("h1", {}, "Study locks"),
      el("p", { class: "lead" }, "A study lock freezes the protocol content that experiments must share to be compared. ",
        "Subjects, maskers and methods may vary; a declaration that differs in a locked field is refused before anything runs."),
      el("div", { class: "card" }, el("h3", {}, "Create a lock from a report"),
        records.reports.length ? el("div", {}, el("div", { class: "controls" }, el("label", {}, "Report ", report), el("label", {}, "Study id ", studyId),
          el("label", {}, "Description ", description)), el("div", { class: "rowlist" }, boxes),
        el("button", { class: "primary", onclick: create }, "Create the study lock")) : el("p", { class: "note" }, "Run an experiment first: a lock is taken from a sealed report.")),
      records.studies.length ? records.studies.map((s) => (s.error ? el("div", { class: "card" }, el("h3", {}, s.name), el("p", { class: "bad" }, s.error))
        : el("div", { class: "card" }, el("h3", {}, s.study_id), el("p", { class: "note" }, `${s.description || ""} · ${s.name} · ${digest(s.digest, 16)}`),
          el("p", {}, "Locked: ", s.fields.map((f) => el("span", { class: "pill" }, f))),
          table(["Report", "Belongs", "Declares study", "Differs in"], s.members.map((m) => el("tr", {}, el("td", {}, m.report),
            el("td", {}, el("span", { class: m.belongs ? "ok" : "bad" }, m.belongs ? "yes" : "no")),
            el("td", {}, m.declares ?? "—"), el("td", {}, m.different.join(", ") || "—")))),
          el("details", {}, el("summary", {}, "Locked values"), el("pre", {}, JSON.stringify(s.locked, null, 1))))))
        : el("p", { class: "empty" }, "No study locks yet."));
  },
};

/* --------------------------------------------------------------- keyframes */
PAGES.keyframes = {
  async show([name]) {
    records.galleries = await api("/api/galleries");
    const usable = records.galleries.filter((g) => !g.error);
    const select = document.getElementById("gallery");
    select.replaceChildren(...usable.map((g) => el("option", { value: g.name }, `${g.meta.experiment_id || g.name} (${g.contexts} contexts × ${g.branches / Math.max(1, g.contexts)} branches) · ${g.name}`)));
    select.onchange = () => go("keyframes", select.value);
    document.getElementById("frame").onchange = (e) => { records.frame = Number(e.target.value); renderKeyframes(); };
    document.getElementById("show").onchange = (e) => { records.show = e.target.value; renderKeyframes(); };
    document.getElementById("role").onchange = (e) => { records.role = e.target.value; renderKeyframes(); };
    const chosen = usable.find((g) => g.name === name) || usable[0];
    if (!chosen) { records.gallery = null; renderKeyframes(); return; }
    select.value = chosen.name;
    if (!records.gallery || records.gallery.name !== chosen.name) await loadGallery(chosen.name);
    renderKeyframes();
  },
};

async function loadGallery(name) {
  records.gallery = await api(`/api/gallery?name=${encodeURIComponent(name)}`);
  const g = records.gallery, keyframe = g.frame_labels.findIndex((label) => label.includes("(keyframe)"));
  records.frame = keyframe >= 0 ? keyframe : g.frame_labels.length - 1;
  records.role = ""; records.context = g.contexts.length ? g.contexts[0].context_id : null; records.branch = null;
}

const axisLabel = () => records.gallery.meta.axis_label || records.gallery.meta.handle_symbol || records.gallery.meta.handle || "value";
const valueText = (values) => values.map((v) => fmt(v, 4)).join(", ");
const visibleContexts = () => records.gallery.contexts.filter((c) => !records.role || c.role === records.role);
const policy = () => records.gallery.meta.policy || {};
const currentContext = () => records.gallery.contexts.find((c) => c.context_id === records.context);

function renderKeyframes() {
  const g = records.gallery;
  if (!g) {
    ["axis-chart", "contexts", "inspect"].forEach((id) => document.getElementById(id).replaceChildren());
    document.getElementById("strip").replaceChildren(el("p", { class: "empty" }, "No keyframe galleries yet. Runs that declare kept frames write one when sealed."));
    return;
  }
  document.getElementById("frame").replaceChildren(...g.frame_labels.map((label, i) => el("option", { value: i, selected: i === records.frame }, label)));
  document.getElementById("show").replaceChildren(...[["frame", "frames"], ["diff", "difference from the reference branch"]].map(([value, text]) =>
    el("option", { value, selected: value === records.show }, text)));
  document.getElementById("role").replaceChildren(el("option", { value: "" }, "all"),
    ...[...new Set(g.contexts.map((c) => c.role))].map((r) => el("option", { value: r, selected: r === records.role }, r)));
  const p = policy();
  document.getElementById("policy").textContent = `policy: ${p.mode ?? "?"}${p.warn != null ? ` · warn above ${p.warn}` : ""}${p.tolerance != null ? ` · tolerance ${p.tolerance}` : ""}`;
  renderChart(); renderContexts(); renderStrip(); renderInspect();
}

function renderContexts() {
  const f = records.frame, { warn, tolerance } = policy();
  document.getElementById("contexts").replaceChildren(...visibleContexts().map((c) => {
    const rms = c.max_rms[f];
    const cls = tolerance != null && rms > tolerance ? "bad" : warn != null && rms > warn ? "warn" : rms === 0 ? "ok" : "";
    return el("div", { class: `ctx ${c.context_id === records.context ? "active" : ""}`, onclick: () => { records.context = c.context_id; records.branch = null; renderKeyframes(); } },
      el("span", {}, c.context_id, el("span", { class: "muted" }, ` ${c.role}`)), el("span", { class: cls }, rms === 0 ? "identical" : fmt(rms)));
  }));
}

function imageUrl(branch, kind) {
  const vmax = records.gallery.frame_max_abs[records.frame] || 1;
  return `/api/image?gallery=${encodeURIComponent(records.gallery.name)}&branch=${branch}&frame=${records.frame}&kind=${kind}&vmax=${vmax}`;
}

function renderStrip() {
  const strip = document.getElementById("strip"), context = currentContext();
  if (!context) { strip.replaceChildren(el("p", { class: "empty" }, "No context selected.")); return; }
  const { warn, tolerance } = policy();
  strip.replaceChildren(...context.branches.map((b) => {
    const rms = b.rms[records.frame];
    const cls = b.reference ? "reference" : tolerance != null && rms > tolerance ? "exceeds" : warn != null && rms > warn ? "flagged" : "";
    return el("div", { class: `thumb ${cls} ${records.branch === b.index ? "active" : ""}`, onclick: () => { records.branch = b.index; renderKeyframes(); } },
      el("img", { src: imageUrl(b.index, records.show === "diff" && !b.reference ? "diff" : "frame"), alt: "", loading: "lazy" }),
      el("div", { class: "value" }, `${axisLabel()} = ${valueText(b.values)}`),
      el("div", { class: "metric" }, b.reference ? el("span", { class: "pill ref" }, "reference") : `rms ${rms === 0 ? "0 (identical)" : fmt(rms)}`));
  }));
}

function renderInspect() {
  const out = document.getElementById("inspect"), context = currentContext();
  const branch = context && context.branches.find((b) => b.index === records.branch);
  if (!branch) { mount(out, el("p", { class: "note" }, "Click a branch to compare it with its reference branch.")); return; }
  const reference = context.branches.find((b) => b.reference && b.seed === branch.seed);
  const figure = (src, caption) => el("figure", {}, el("img", { src }), el("figcaption", {}, caption));
  mount(out, el("div", { class: "card" },
    el("h3", {}, `${context.context_id} · ${records.gallery.frame_labels[records.frame]} · ${axisLabel()} = ${valueText(branch.values)}`),
    el("div", { class: "inspect-row" },
      reference ? figure(imageUrl(reference.index, "frame"), `reference (${axisLabel()} = ${valueText(reference.values)})`) : "",
      figure(imageUrl(branch.index, "frame"), "this branch"),
      branch.reference ? "" : figure(imageUrl(branch.index, "diff"), `|difference|, colour scale 0 … ${fmt(records.gallery.frame_max_abs[records.frame])}`)),
    el("p", {}, `rms ${fmt(branch.rms[records.frame])} · max |Δ| ${fmt(branch.max_abs[records.frame])} · draw seed ${branch.seed}`)));
}

function renderChart() {
  const host = document.getElementById("axis-chart"), f = records.frame;
  const points = visibleContexts().flatMap((c) => c.branches.map((b) => ({ c: c.context_id, x: b.values[0], y: b.rms[f] ?? 0, i: b.index, ref: b.reference })));
  if (!points.length) { mount(host, ); return; }
  const W = 900, H = 210, L0 = 58, R = 14, T = 14, B = 34, pol = policy();
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymax = Math.max(...ys, pol.warn ?? 0, pol.tolerance ?? 0) || 1;
  const sx = (x) => L0 + (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin)) * (W - L0 - R);
  const sy = (y) => H - B - (y / ymax) * (H - T - B);
  const text = (x, y, value, extra = {}) => svg("text", { x, y, "font-size": 11, fill: "#687384", ...extra }, value);
  const selected = points.filter((p) => p.c === records.context).sort((a, b) => a.x - b.x);
  mount(host, svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" },
    svg("line", { x1: L0, y1: H - B, x2: W - R, y2: H - B, stroke: "#9aa5b4" }), svg("line", { x1: L0, y1: T, x2: L0, y2: H - B, stroke: "#9aa5b4" }),
    text(L0 - 6, sy(ymax) + 4, fmt(ymax), { "text-anchor": "end" }), text(L0 - 6, H - B + 4, "0", { "text-anchor": "end" }),
    text(L0, H - 8, fmt(xmin, 4)), text(W - R, H - 8, fmt(xmax, 4), { "text-anchor": "end" }),
    text((L0 + W) / 2, H - 8, `${axisLabel()} → RMS of ${records.gallery.frame_labels[f]} against the reference branch`, { "text-anchor": "middle", fill: "#1d232b" }),
    [[pol.warn, "#b7791f", "warn"], [pol.tolerance, "#c43d3d", "tolerance"]].filter(([value]) => value != null).map(([value, colour, label]) => [
      svg("line", { x1: L0, x2: W - R, y1: sy(value), y2: sy(value), stroke: colour, "stroke-dasharray": "5 4" }),
      text(W - R - 4, sy(value) - 4, `${label} ${value}`, { "text-anchor": "end", fill: colour })]),
    selected.length > 1 ? svg("polyline", { points: selected.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" "), fill: "none", stroke: "#2f6fd6", "stroke-width": 1.5 }) : "",
    points.map((p) => {
      const mine = p.c === records.context;
      const dot = svg("circle", { cx: sx(p.x), cy: sy(p.y), r: mine ? (p.i === records.branch ? 6 : 4) : 2.5,
        fill: p.ref ? "#6b46c1" : mine ? "#2f6fd6" : "#b9c3d0", opacity: mine ? 1 : 0.7, style: "cursor:pointer" },
      svg("title", {}, `${p.c} · ${axisLabel()}=${fmt(p.x, 4)} · rms ${fmt(p.y)}`));
      dot.addEventListener("click", () => { records.context = p.c; records.branch = p.i; renderKeyframes(); });
      return dot;
    })));
}
