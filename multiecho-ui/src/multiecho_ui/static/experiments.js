"use strict";
// Experiments: the designs in the workspace, and the editor that builds one (chain, declaration, methods).

const ROLES = ["discovery", "validation", "confirmation"];
const FACTOR_KINDS = ["parameter", "initial-condition", "label", "appearance", "prior", "range", "intent"];
const DISTRIBUTIONS = { fixed: ["value"], uniform: ["low", "high"], "signed-uniform": ["low", "high"],
  "log-uniform": ["low", "high"], choice: ["values"], cycle: ["values"] };
const KEYFRAME_MODES = ["free", "report", "tolerance", "identical"];
const TABS = [["chain", "1 · Chain"], ["declaration", "2 · Declaration"], ["methods", "3 · Methods"], ["json", "JSON"]];

const editor = { name: null, doc: null, dirty: false, tab: "chain", check: null };
// Form helpers report edits through these hooks: the editor marks the design unsaved; the run page re-renders itself.
const hooks = { changed: () => dirty(), render: () => renderEditor() };

/* ------------------------------------------------------------- list page */
const LIFECYCLE = [
  ["Build", "Compose a chain of worlds from the plugin catalog and declare the experiment: question, intervention values, contexts and draws, all fixed before anything is measured."],
  ["Measure discovery", "Every discovery context is measured at the reference and at each intervention value; a data-driven masker is calibrated and frozen here."],
  ["Freeze laws", "Fit one analysis method per task (forward: predict the response; backward: recover the intervention) on discovery responses only, then freeze it."],
  ["Test held-out", "Validation and confirmation contexts are measured only now, and each frozen law is judged against fixed controls."],
  ["Seal", "The report is sealed with its declaration, stages and outcomes, and checked by the 22 conformance checks."],
];

PAGES.experiments = {
  async show() {
    const root = document.getElementById("view-experiments");
    const [designs, runs] = await Promise.all([api("/api/designs"), api("/api/runs")]);
    const latest = {};
    for (const run of runs) if (!latest[run.design]) latest[run.design] = run;
    const name = el("input", { placeholder: "new-experiment", pattern: "[A-Za-z0-9][A-Za-z0-9._-]*" });
    const from = el("select", {}, el("option", { value: "" }, "a blank design"),
      designs.filter((d) => !d.error).map((d) => el("option", { value: d.name }, `a copy of ${d.name}`)));
    const create = async () => {
      try {
        const made = await post("/api/design/new", { name: name.value.trim(), from: from.value || null });
        editor.name = null;
        go("experiment", made.name);
      } catch (error) { notice(error.message); }
    };
    const rows = designs.length ? designs.map((d) => {
      const run = latest[d.name];
      return el("tr", { class: d.error ? "" : "clickable", onclick: d.error ? null : () => go("experiment", d.name) },
        el("td", {}, el("b", {}, d.name), d.error ? el("div", { class: "bad" }, d.error) : el("div", { class: "note" }, d.description || "")),
        el("td", {}, d.question || "—"),
        el("td", {}, (d.worlds || []).join(" → ")),
        el("td", {}, d.masker ? d.masker.split(":").pop() : "none"),
        el("td", {}, (d.tasks || []).join(", ")),
        el("td", {}, run ? el("span", {}, badge(run.stage), " ", Object.entries(run.outcomes || {}).map(([task, value]) =>
          el("span", { class: "note" }, ` ${task}: ${String(value).replaceAll("_", " ")}`))) : el("span", { class: "note" }, "not run")),
        el("td", {}, el("button", { onclick: (e) => { e.stopPropagation(); startRun(d.name); }, disabled: Boolean(d.error) }, "Run"),
          " ", el("button", { class: "quiet", onclick: async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete the design ${d.name}? Its runs and reports stay.`)) return;
            try { await post("/api/design/delete", { name: d.name }); route(); } catch (error) { notice(error.message); }
          } }, "Delete")));
    }) : [emptyRow("No designs yet. Create one below.", 7)];
    mount(root, 
      el("h1", {}, "Experiments"),
      el("p", { class: "lead" }, "An experiment measures how worlds respond to a counterfactual intervention. ",
        "It is a chain of worlds (physical scenes, simulators, learned models) joined by interfaces and observed by an analyst, ",
        "plus a declaration that fixes everything before measuring. Build one from this workspace's plugins, check it, and run it."),
      el("div", { class: "lifecycle" }, LIFECYCLE.map(([title, text], i) =>
        el("div", { class: "step" }, el("span", { class: "stepnum" }, i + 1), el("h4", {}, title), el("p", {}, text)))),
      el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Design", "Question", "Chain", "Masker", "Tasks", "Last run", ""]
        .map((h) => el("th", {}, h)))), el("tbody", {}, rows)),
      el("div", { class: "card" }, el("h3", {}, "New experiment"),
        el("div", { class: "controls" }, el("label", {}, "Name ", name), el("label", {}, "Start from ", from),
          el("button", { class: "primary", onclick: create }, "Create"))));
  },
};

async function startRun(name) {
  try {
    const run = await post("/api/run/start", { design: name });
    go("run", run.id);
  } catch (error) { notice(`This design cannot run yet: ${error.message}`); }
}

/* ---------------------------------------------------------------- inputs */
function dirty() {
  editor.dirty = true;
  const pill = document.getElementById("editor-status");
  if (pill) { pill.textContent = "unsaved changes"; pill.className = "pill warn"; }
}

const numberOrNull = (text) => (text.trim() === "" ? null : Number(text));
const csv = (text) => text.split(",").map((s) => s.trim()).filter(Boolean);
const numbers = (text) => csv(text).map(Number);
const showList = (value) => (Array.isArray(value) ? value.join(", ") : value ?? "");

function input(obj, key, { type = "text", parse = (v) => v, show = (v) => v ?? "", placeholder = "", rerender = false, size } = {}) {
  return el("input", { type, value: show(obj[key]), placeholder, step: type === "number" ? "any" : null, size,
    oninput: (e) => { const value = parse(e.target.value); if (value === undefined) delete obj[key]; else obj[key] = value; hooks.changed(); },
    onchange: rerender ? () => hooks.render() : null });
}

function area(obj, key, rows = 2) {
  return el("textarea", { rows, oninput: (e) => { obj[key] = e.target.value; hooks.changed(); } }, obj[key] ?? "");
}

function choose(obj, key, options, { onchange, blank } = {}) {
  return el("select", { onchange: (e) => { obj[key] = e.target.value || null; hooks.changed(); if (onchange) onchange(e.target.value); hooks.render(); } },
    blank !== undefined ? el("option", { value: "" }, blank) : "",
    options.map((option) => { const [value, label] = Array.isArray(option) ? option : [option, option];
      return el("option", { value, selected: obj[key] === value }, label); }));
}

function tick(obj, key, label) {
  return el("label", { class: "tick" }, el("input", { type: "checkbox", checked: Boolean(obj[key]),
    onchange: (e) => { obj[key] = e.target.checked; hooks.changed(); } }), " ", label);
}

const field = (label, control, help) => el("label", { class: "field" }, el("span", {}, label), control, help ? el("small", {}, help) : "");
const fields = (...items) => el("div", { class: "fields" }, items);

function rowsEditor(list, columns, make, title) {
  const body = list.map((row, index) => el("tr", {}, columns.map(([key, , options]) => el("td", {},
    options?.select ? choose(row, key, options.select) : input(row, key, options || {})),
  el("td", {}, el("button", { class: "quiet", title: "Remove", onclick: () => { list.splice(index, 1); hooks.changed(); hooks.render(); } }, "×")))));
  return el("div", { class: "rows" }, el("table", { class: "mini" },
    el("thead", {}, el("tr", {}, columns.map(([, label]) => el("th", {}, label)), el("th"))),
    el("tbody", {}, body.length ? body : emptyRow("none", columns.length + 1))),
  el("button", { class: "quiet", onclick: () => { list.push(make()); hooks.changed(); hooks.render(); } }, `+ ${title}`));
}

function valuesRows(holder, key, title, help) {
  const list = holder[key] || (holder[key] = []);
  return el("div", {}, el("div", { class: "rowlist" }, list.map((row, index) => el("span", { class: "chipinput" },
    el("input", { value: showList(row), size: 10, oninput: (e) => { list[index] = numbers(e.target.value); hooks.changed(); } }),
    el("button", { class: "quiet", title: "Remove", onclick: () => { list.splice(index, 1); hooks.changed(); hooks.render(); } }, "×"))),
  el("button", { class: "quiet", onclick: () => { list.push([]); hooks.changed(); hooks.render(); } }, `+ ${title}`)),
  help ? el("small", { class: "note" }, help) : "");
}

function paramsForm(entry, holder) {
  if (!entry) return el("p", { class: "note" }, "Choose a plugin.");
  holder.params = holder.params || {};
  const params = holder.params;
  if (!entry.parameters.length) return el("p", { class: "note" }, "No parameters.");
  return fields(entry.parameters.map((p) => {
    const help = [p.annotation, p.required ? "required" : ""].filter(Boolean).join(" · ");
    if (p.type === "auto") return field(p.name, el("input", { disabled: true, value: "set by the workbench" }), help);
    if (p.type === "fixed") return field(p.name, el("input", { disabled: true, value: p.default === null ? "default" : String(p.default) }),
      "not editable here");
    if (p.type === "bool") {
      return field(p.name, el("input", { type: "checkbox", checked: params[p.name] ?? p.default,
        onchange: (e) => { params[p.name] = e.target.checked; hooks.changed(); } }), help);
    }
    const parse = { int: (v) => (v.trim() === "" ? undefined : Number(v)), float: (v) => (v.trim() === "" ? undefined : Number(v)),
      numbers: (v) => (v.trim() === "" ? undefined : numbers(v)) }[p.type] || ((v) => (v === "" ? undefined : v));
    return field(p.name, input(params, p.name, { type: ["int", "float"].includes(p.type) ? "number" : "text", parse, show: showList,
      placeholder: p.type === "selection" ? "e.g. 0-63, 70" : p.default === null ? "" : showList(p.default) }), help);
  }));
}

function pluginPicker(holder, kind, label) {
  const entry = plugin(holder.plugin);
  const options = plugins(kind).map((p) => [p.id, `${p.name}${p.builtin ? " (framework)" : ""} — ${p.module}`]);
  return el("div", {}, field(label, choose(holder, "plugin", options, { blank: "— choose —", onchange: () => { holder.params = {}; } })),
    entry && entry.summary ? el("p", { class: "note" }, entry.summary) : "", paramsForm(entry, holder));
}

/* ---------------------------------------------------------------- chain */
function scopeForm(scope) {
  scope.channels = scope.channels || [];
  const region = scope.region;
  return el("div", {},
    fields(field("Scope id", input(scope, "id")),
      field("Channels", input(scope, "channels", { parse: csv, show: showList, placeholder: "video, truth" }), "comma-separated"),
      field("Time window", input(scope, "window", { parse: (v) => (v.trim() ? numbers(v) : null), show: showList, placeholder: "start, stop" }),
        "frame indices [start, stop) of a channel"),
      field("Clip", input(scope, "clip", { parse: (v) => (v.trim() ? numbers(v) : null), show: showList, placeholder: "t0, t1" }),
        "seconds of a video channel"),
      field("Keys", input(scope, "keys", { parse: (v) => (v.trim() ? csv(v) : null), show: showList }), "statements of a text channel"),
      field("Coordinates", input(scope, "coordinates", { parse: (v) => (v.trim() ? numbers(v) : null), show: showList }),
        "indices of a tensor's last axis")),
    tick({ get on() { return Boolean(scope.region); }, set on(v) { scope.region = v ? { label: "region", shape: [16, 16], rows: [0, 16], columns: [0, 16] } : null; renderEditor(); } },
      "on", "Observe a box region of the frames"),
    region ? fields(field("Region label", input(region, "label")),
      field("Frame shape", input(region, "shape", { parse: numbers, show: showList }), "height, width"),
      field("Rows", input(region, "rows", { parse: numbers, show: showList }), "start, stop"),
      field("Columns", input(region, "columns", { parse: numbers, show: showList }), "start, stop")) : "");
}

function defaultInterface(type, index) {
  if (type === "bridge") return { type, id: `bridge-${index + 1}`, channel: "parameters", maps: [], constants: [] };
  if (type === "plugin") return { type, id: `translator-${index + 1}`, plugin: null, params: {} };
  return { type: "scope", id: `interface-${index + 1}`, scope: { id: `scope-${index + 1}`, channels: [] } };
}

function worldTitle(world) {
  if (world.type === "physical") return `Physical scene · ${world.id || "?"}`;
  const entry = plugin(world.plugin);
  return entry ? `${entry.name}` : "Plugin world";
}

function worldCard(chain, index) {
  const world = chain.worlds[index], count = chain.worlds.length;
  const move = (delta) => {
    const target = index + delta;
    [chain.worlds[index], chain.worlds[target]] = [chain.worlds[target], chain.worlds[index]];
    if (chain.handle_world === index) chain.handle_world = target; else if (chain.handle_world === target) chain.handle_world = index;
    dirty(); renderEditor();
  };
  const remove = () => {
    chain.worlds.splice(index, 1);
    chain.interfaces.splice(Math.max(0, index - 1), 1);
    chain.handle_world = Math.min(chain.handle_world || 0, chain.worlds.length - 1);
    dirty(); renderEditor();
  };
  let body;
  if (world.type === "physical") {
    world.handle = world.handle || { id: "handle", semantics: "", axes: [] };
    body = el("div", {}, fields(field("World id", input(world, "id")), field("Revision", input(world, "revision")),
      field("Description", input(world, "description"))),
    el("h5", {}, "Handle: the axes an intervention sets"),
    fields(field("Handle id", input(world.handle, "id")), field("Meaning", input(world.handle, "semantics"))),
    rowsEditor(world.handle.axes, [["name", "Axis"], ["unit", "Unit"], ["lower", "Lower", { type: "number", parse: numberOrNull }],
      ["upper", "Upper", { type: "number", parse: numberOrNull }]], () => ({ name: "", unit: "1", lower: 0, upper: 1 }), "axis"));
  } else {
    body = pluginPicker(world, "world", "Plugin");
  }
  return el("div", { class: `world ${world.type}` },
    el("div", { class: "world-head" }, el("span", { class: "stepnum" }, index + 1), el("b", {}, worldTitle(world)),
      chain.handle_world === index ? el("span", { class: "pill ok" }, "takes the interventions") : "",
      el("span", { class: "spacer" }),
      el("button", { class: "quiet", disabled: index === 0, onclick: () => move(-1), title: "Move up" }, "↑"),
      el("button", { class: "quiet", disabled: index === count - 1, onclick: () => move(1), title: "Move down" }, "↓"),
      el("button", { class: "quiet", disabled: count === 1, onclick: remove, title: "Remove" }, "Remove")),
    body);
}

function interfaceCard(chain, index) {
  const spec = chain.interfaces[index];
  let body;
  if (spec.type === "bridge") {
    spec.maps = spec.maps || []; spec.constants = spec.constants || [];
    body = el("div", {}, fields(field("Interface id", input(spec, "id")), field("Channel", input(spec, "channel"), "the parameters channel the next world accepts")),
      el("h5", {}, "Parameter maps: statement of this world → parameter of the next"),
      rowsEditor(spec.maps, [["source", "From"], ["target", "To"], ["scale", "Scale", { type: "number", parse: numberOrNull, placeholder: "1" }],
        ["offset", "Offset", { type: "number", parse: numberOrNull, placeholder: "0" }], ["unit", "Unit"]], () => ({ source: "", target: "", scale: 1, offset: 0, unit: "" }), "map"),
      el("h5", {}, "Constants added for the next world"),
      rowsEditor(spec.constants, [["key", "Key"], ["value", "Value", { parse: (v) => (v.trim() !== "" && !Number.isNaN(Number(v)) ? Number(v) : v) }], ["unit", "Unit"]],
        () => ({ key: "", value: 0, unit: "" }), "constant"));
  } else if (spec.type === "plugin") {
    body = el("div", {}, fields(field("Interface id", input(spec, "id"))), pluginPicker(spec, "interface", "Translator"));
  } else {
    spec.scope = spec.scope || { id: "scope", channels: [] };
    body = el("div", {}, fields(field("Interface id", input(spec, "id"))), scopeForm(spec.scope));
  }
  return el("div", { class: "interface" },
    el("div", { class: "world-head" }, el("span", { class: "arrow" }, "↓"), el("b", {}, "Interface"),
      el("select", { onchange: (e) => { chain.interfaces[index] = defaultInterface(e.target.value, index); dirty(); renderEditor(); } },
        [["scope", "Scope: pass selected channels"], ["bridge", "Parameter bridge"], ["plugin", "Translator plugin"]].map(([value, label]) =>
          el("option", { value, selected: spec.type === value }, label)))),
    body);
}

function renderChain(doc) {
  const chain = doc.chain = doc.chain || {};
  chain.worlds = chain.worlds || []; chain.interfaces = chain.interfaces || [];
  chain.analyst = chain.analyst || { id: "analyst", scope: { id: "analyst-scope", channels: [] }, masker: null };
  chain.observation = chain.observation || { id: "observation", semantics: "", unit: "1" };
  const analyst = chain.analyst;
  const add = (world) => {
    if (chain.worlds.length) chain.interfaces.push(defaultInterface("scope", chain.interfaces.length));
    chain.worlds.push(world); dirty(); renderEditor();
  };
  const items = [];
  chain.worlds.forEach((_, index) => { items.push(worldCard(chain, index)); if (index < chain.worlds.length - 1) items.push(interfaceCard(chain, index)); });
  const masker = analyst.masker;
  return el("div", {},
    el("p", { class: "note" }, "Worlds run in order: each emits an artifact, and the interface below it passes part of it to the next world. ",
      "The analyst observes the last world through a scope. Use “Check design” to trace one branch through the chain."),
    field("Chain id", input(chain, "id")),
    el("div", { class: "chain" }, items.length ? items : el("p", { class: "empty" }, "No worlds yet.")),
    el("div", { class: "controls" },
      el("button", { onclick: () => add({ type: "physical", id: "scene", revision: "v1", description: "", handle: { id: "handle", semantics: "", axes: [] } }) }, "+ Physical scene"),
      el("button", { onclick: () => add({ type: "plugin", plugin: null, params: {} }) }, "+ Plugin world"),
      field("World taking the interventions", el("select", { onchange: (e) => { chain.handle_world = Number(e.target.value); dirty(); renderEditor(); } },
        chain.worlds.map((world, index) => el("option", { value: index, selected: (chain.handle_world || 0) === index }, `${index + 1}. ${worldTitle(world)}`))))),
    el("div", { class: "world analyst" },
      el("div", { class: "world-head" }, el("span", { class: "arrow" }, "↓"), el("b", {}, "Analyst: what is observed and fitted")),
      fields(field("Interface id", input(analyst, "id"))), scopeForm(analyst.scope),
      el("h5", {}, "Masker (optional): keep only some locations of the observed channel"),
      field("Masker", el("select", { onchange: (e) => { analyst.masker = e.target.value ? { plugin: e.target.value, params: {}, width: 1 } : null; dirty(); renderEditor(); } },
        el("option", { value: "" }, "none: the full observation"),
        plugins("masker").map((p) => el("option", { value: p.id, selected: masker && masker.plugin === p.id }, p.name)))),
      masker ? el("div", {}, plugin(masker.plugin) ? el("p", { class: "note" }, plugin(masker.plugin).summary) : "",
        field("Values per location", input(masker, "width", { type: "number", parse: (v) => Number(v) || 1 }),
          "the observed vector is split into locations of this many consecutive values (1 = each value; e.g. channels per token)"),
        paramsForm(plugin(masker.plugin), masker)) : ""),
    el("div", { class: "world" }, el("div", { class: "world-head" }, el("b", {}, "Observation")),
      fields(field("Observation id", input(chain.observation, "id")), field("Meaning", input(chain.observation, "semantics")),
        field("Unit", input(chain.observation, "unit")))),
    el("div", { class: "world" }, el("div", { class: "world-head" }, el("b", {}, "Stimulus delivered to a learned world"),
      tick({ get on() { return Boolean(chain.stimulus); }, set on(v) { chain.stimulus = v ? { frames: 16, interval: 0.0625, height: 64, width: 64, channel: "video", observe_after: 1 } : null; renderEditor(); } }, "on", "declared")),
    chain.stimulus ? fields(field("Frames", input(chain.stimulus, "frames", { type: "number", parse: Number })),
      field("Interval (s)", input(chain.stimulus, "interval", { type: "number", parse: Number })),
      field("Height", input(chain.stimulus, "height", { type: "number", parse: Number })),
      field("Width", input(chain.stimulus, "width", { type: "number", parse: Number })),
      field("Channel", input(chain.stimulus, "channel")),
      field("Observe after", input(chain.stimulus, "observe_after", { type: "number", parse: Number }), "frames after the keyframe")) :
      el("p", { class: "note" }, "No stimulus: the chain has no learned world, or the stimulus is not declared (keyframe policies then cannot capture it).")));
}

/* ----------------------------------------------------------- declaration */
function factorRow(factors, index) {
  const factor = factors[index];
  factor.distribution = factor.distribution || { type: "fixed", value: 0 };
  const dist = factor.distribution;
  const earlier = factors.slice(0, index).map((f) => f.key).filter(Boolean);
  const distFields = (DISTRIBUTIONS[dist.type] || []).map((key) => field(key,
    key === "values" ? input(dist, key, { parse: (v) => csv(v).map((s) => (Number.isNaN(Number(s)) ? s : Number(s))), show: showList })
      : key === "value" ? input(dist, key, { parse: (v) => (v.trim() !== "" && !Number.isNaN(Number(v)) ? Number(v) : v) })
        : input(dist, key, { type: "number", parse: numberOrNull })));
  return el("div", { class: "factor" },
    fields(field("Key", input(factor, "key")), field("Unit", input(factor, "unit")),
      field("Kind", choose(factor, "kind", FACTOR_KINDS)),
      field("Distribution", el("select", { onchange: (e) => { factor.distribution = { type: e.target.value }; dirty(); renderEditor(); } },
        Object.keys(DISTRIBUTIONS).map((type) => el("option", { value: type, selected: dist.type === type }, type)))),
      ...distFields,
      field("Only when", el("select", { onchange: (e) => { factor.when = e.target.value ? { key: e.target.value, equals: "", otherwise: 0 } : null; dirty(); renderEditor(); } },
        el("option", { value: "" }, "always"), earlier.map((key) => el("option", { value: key, selected: factor.when && factor.when.key === key }, `${key} equals …`)))),
      factor.when ? field("equals", input(factor.when, "equals")) : "",
      factor.when ? field("otherwise", input(factor.when, "otherwise", { parse: (v) => (v.trim() !== "" && !Number.isNaN(Number(v)) ? Number(v) : v) })) : ""),
    el("button", { class: "quiet", onclick: () => { factors.splice(index, 1); dirty(); renderEditor(); } }, "Remove factor"));
}

function renderDeclaration(doc) {
  const d = doc.design = doc.design || {};
  d.contexts = d.contexts || { seed: 0, counts: {}, factors: [], conditions: [] };
  d.contexts.counts = d.contexts.counts || {}; d.contexts.factors = d.contexts.factors || []; d.contexts.conditions = d.contexts.conditions || [];
  d.statistics = d.statistics || {}; d.sweep = d.sweep || { values: [], draws: 1 }; d.keyframes = d.keyframes || { mode: "free" };
  d.tasks = d.tasks || ["forward"];
  const tasks = { get forward() { return d.tasks.includes("forward"); }, set forward(v) { setTask("forward", v); },
    get backward() { return d.tasks.includes("backward"); }, set backward(v) { setTask("backward", v); } };
  function setTask(task, on) { d.tasks = ["forward", "backward"].filter((t) => (t === task ? on : d.tasks.includes(t))); renderEditor(); }
  const labels = d.contexts.factors.filter((f) => f.key);
  const conditionTicks = labels.map((f) => {
    const holder = { get on() { return d.contexts.conditions.includes(f.key); },
      set on(v) { d.contexts.conditions = v ? [...d.contexts.conditions, f.key] : d.contexts.conditions.filter((k) => k !== f.key); } };
    return tick(holder, "on", f.key);
  });
  return el("div", {},
    el("h3", {}, "Question"),
    fields(field("Experiment id", input(d, "experiment_id")), field("What the intervention sets", input(d, "handle")),
      field("Observation time", input(d, "observation_time")), field("Independence unit", input(d, "independence_unit"))),
    field("Question", area(d, "question")),
    el("h3", {}, "Intervention values"),
    el("p", { class: "note" }, "Each context is measured at the reference value and at every calibration and query value (one number per handle axis, comma-separated). ",
      "In held-out contexts a frozen law sees the reference and calibration responses and must predict the queries."),
    fields(field("Horizon (steps)", input(d, "steps", { type: "number", parse: Number }), "shared by every branch"),
      field("Reference", input(d, "reference", { parse: numbers, show: showList }))),
    field("Calibration values", valuesRows(d, "calibration", "calibration value")),
    field("Query values", valuesRows(d, "queries", "query value")),
    el("h3", {}, "Contexts"),
    el("p", { class: "note" }, "Contexts are scenes that differ in factors other than the intervention. They are drawn from the factors below with one random stream from the seed; ",
      "discovery contexts are used for fitting, validation and confirmation contexts only for testing."),
    fields(field("Seed", input(d.contexts, "seed", { type: "number", parse: Number })),
      ...ROLES.map((role) => field(`${role} contexts`, input(d.contexts.counts, role, { type: "number", parse: Number })))),
    el("div", { class: "factors" }, d.contexts.factors.map((_, i) => factorRow(d.contexts.factors, i))),
    el("button", { class: "quiet", onclick: () => { d.contexts.factors.push({ key: "", unit: "", kind: "parameter", distribution: { type: "uniform", low: 0, high: 1 } }); dirty(); renderEditor(); } }, "+ factor"),
    labels.length ? field("Conditions (outcomes are also reported per value of these factors)", el("div", { class: "rowlist" }, conditionTicks)) : "",
    el("h3", {}, "Tasks, draws and statistics"),
    el("div", { class: "controls" }, tick(tasks, "forward", "forward: predict the response"), tick(tasks, "backward", "backward: recover the intervention")),
    fields(field("Draws per context", input(d, "draws", { type: "number", parse: Number }), "1 = one fixed realization; more = context draws shared by every branch"),
      field("Determinism repeats", input(d, "repeats", { type: "number", parse: Number })),
      field("Minimum improvement", input(d.statistics, "minimum_improvement", { type: "number", parse: Number }), "declared margin a law must beat a control by"),
      field("Equivalence margin", input(d.statistics, "equivalence_margin", { type: "number", parse: numberOrNull }), "optional"),
      field("Confidence", input(d.statistics, "confidence", { type: "number", parse: Number })),
      field("Bootstrap resamples", input(d.statistics, "resamples", { type: "number", parse: Number })),
      field("Statistics seed", input(d.statistics, "seed", { type: "number", parse: Number }))),
    el("h3", {}, "Sweep (for maskers that need one)"),
    el("p", { class: "note" }, "A data-driven masker sees only a sweep at these values, disjoint from the evaluation; declare it exactly when the masker needs a sweep."),
    field("Sweep values", valuesRows(d.sweep, "values", "sweep value")),
    field("Sweep draws", input(d.sweep, "draws", { type: "number", parse: Number })),
    el("h3", {}, "Keyframes"),
    fields(field("Policy", choose(d.keyframes, "mode", KEYFRAME_MODES), "how close the keyframes of a context's branches must be"),
      field("Warn above", input(d.keyframes, "warn", { type: "number", parse: numberOrNull })),
      field("Tolerance", input(d.keyframes, "tolerance", { type: "number", parse: numberOrNull }), "required by the 'tolerance' policy"),
      field("Kept frames", input(d, "keep_frames", { parse: (v) => numbers(v).map(Math.trunc), show: showList, placeholder: "0, -1" }),
        "stimulus frames kept for the keyframe gallery")),
    el("h3", {}, "Study lock"),
    field("Study", choose(d, "study", state.studies.filter((s) => !s.error).map((s) => [s.name, `${s.study_id} (${s.name})`]), { blank: "none" }),
      "a lock refuses this declaration before anything runs if a locked field differs"));
}

function renderMethods(doc) {
  doc.methods = doc.methods || {};
  const tasks = (doc.design && doc.design.tasks) || ["forward"];
  return el("div", {}, el("p", { class: "note" }, "Methods are fitted on discovery responses after measuring, then frozen. ",
    "These are the planned choices for “Run every stage”; on the run page each can be chosen again before freezing."),
  tasks.map((task) => { doc.methods[task] = doc.methods[task] || { plugin: null, params: {} };
    return el("div", { class: "world" }, el("div", { class: "world-head" }, el("b", {}, task === "forward" ? "Forward: predict the response" : "Backward: recover the intervention")),
      pluginPicker(doc.methods[task], task, "Method")); }));
}

function renderJson(doc) {
  const text = el("textarea", { class: "json", rows: 30, spellcheck: "false" }, JSON.stringify(doc, null, 1));
  return el("div", {}, el("p", { class: "note" }, "The design document (multiecho-ui.design/0.1). Edit and apply to change anything the forms do not cover."),
    text, el("div", { class: "controls" }, el("button", { onclick: () => {
      try { editor.doc = JSON.parse(text.value); dirty(); renderEditor(); notice("Applied.", "ok"); } catch (error) { notice(`Not valid JSON: ${error.message}`); }
    } }, "Apply JSON")));
}

/* ----------------------------------------------------------------- check */
function renderCheck(result) {
  if (!result) return el("p", { class: "note" }, "Check the design to build the chain, trace one branch through it and seal the declaration, without measuring anything.");
  if (!result.ok) return el("div", { class: "banner bad" }, el("div", {}, `Cannot build: ${result.error.where}`), el("div", { class: "detail" }, result.error.message));
  const chain = result.chain, decl = result.declaration;
  const diagram = el("div", { class: "diagram" }, chain.worlds.map((w, i) => [
    el("div", { class: `node ${w.kind}` }, el("div", { class: "kind" }, w.kind), el("b", {}, w.world_id),
      el("div", { class: "note" }, `accepts ${w.accepts.join(", ") || "—"}`), el("div", { class: "note" }, `emits ${w.emits.join(", ")}`),
      w.handle ? el("div", { class: "pill ok" }, `handle: ${w.handle.axes.map((a) => `${a.name} [${a.lower}, ${a.upper}] ${a.unit}`).join("; ")}`) : "",
      w.contract ? el("div", { class: "note" }, `history ${w.contract.history}, horizon ${w.contract.horizon}`) : ""),
    i < chain.interfaces.length ? el("div", { class: "edge" }, "→", el("small", {}, chain.interfaces[i].interface_id)) : ""]),
  el("div", { class: "edge" }, "→", el("small", {}, "analyst")),
  el("div", { class: "node analyst" }, el("div", { class: "kind" }, "analyst"), el("b", {}, chain.analyst.scope),
    el("div", { class: "note" }, `${chain.analyst.channels.join(", ")} → ${chain.observation.dims} values`),
    chain.analyst.masker ? el("div", { class: "pill" }, `masker ${chain.analyst.masker.masker_id}: ${chain.analyst.masker.locations} locations × ${chain.analyst.masker.width}`) : ""));
  const trace = el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Step", "Channels"].map((h) => el("th", {}, h)))),
    el("tbody", {}, result.trace.map((row) => el("tr", {}, el("td", {}, row.label || row.source),
      el("td", {}, row.channels.map((c) => el("div", {}, el("code", {}, c.name), ` ${c.modality} [${c.shape.join(" × ")}]`,
        c.statements ? el("span", { class: "note" }, " ", Object.entries(c.statements).map(([k, v]) => `${k}=${typeof v === "number" ? fmt(v, 3) : v}`).join(", ")) : "")))))));
  return el("div", {},
    el("div", { class: "banner ok" }, `The design builds. Chain ${digest(chain.digest)} · declaration ${digest(decl.digest)}`),
    diagram,
    el("div", { class: "stats" }, ROLES.map((role) => el("div", { class: "stat" }, el("span", {}, `${role} contexts`), el("strong", {}, decl.contexts[role]))),
      el("div", { class: "stat" }, el("span", {}, "branches per context"), el("strong", {}, decl.branches_per_context)),
      el("div", { class: "stat" }, el("span", {}, "draws"), el("strong", {}, `${decl.draws} (${decl.coupling})`)),
      el("div", { class: "stat" }, el("span", {}, "observed values"), el("strong", {}, chain.observation.dims))),
    el("p", { class: "note" }, `Tasks: ${decl.tasks.join(", ")} · conditions: ${Object.entries(decl.conditions).map(([k, v]) => `${k} (${v.join(", ")})`).join("; ") || "none"}`,
      ` · sweep values: ${decl.sweep} · study: ${decl.study || "none"}`),
    el("h4", {}, "One discovery branch at the reference value, traced through the chain"),
    el("p", { class: "note" }, "Every artifact and message of the branch. A masker is not applied here: it is calibrated on discovery when the run measures."),
    trace,
    el("h4", {}, "First contexts"),
    el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "Context"), el("th", {}, "Statements"))),
      el("tbody", {}, result.contexts.map((c) => el("tr", {}, el("td", {}, c.context_id),
        el("td", {}, Object.entries(c.statements).map(([k, v]) => `${k} = ${typeof v === "number" ? fmt(v, 3) : v}`).join(" · ")))))));
}

/* ---------------------------------------------------------------- editor */
async function saveDesign() {
  await post("/api/design/save", { name: editor.name, document: editor.doc });
  editor.dirty = false;
  const pill = document.getElementById("editor-status");
  if (pill) { pill.textContent = "saved"; pill.className = "pill ok"; }
}

async function checkDesign() {
  const panel = document.getElementById("check-panel");
  mount(panel, el("p", { class: "note" }, "Building the chain and tracing one branch…"));
  try {
    editor.check = await post("/api/design/validate", { document: editor.doc });
  } catch (error) {
    editor.check = { ok: false, error: { where: "request", message: error.message } };
  }
  mount(panel, renderCheck(editor.check));
}

function renderEditor() {
  const root = document.getElementById("view-experiment"), doc = editor.doc;
  if (!doc) return;
  const scrollY = window.scrollY;
  const body = { chain: renderChain, declaration: renderDeclaration, methods: renderMethods, json: renderJson }[editor.tab](doc);
  mount(root, 
    el("div", { class: "title-row" }, el("h1", {}, doc.name || editor.name),
      el("span", { id: "editor-status", class: `pill ${editor.dirty ? "warn" : "ok"}` }, editor.dirty ? "unsaved changes" : "saved"),
      el("span", { class: "spacer" }),
      el("button", { onclick: async () => { try { await saveDesign(); } catch (error) { notice(error.message); } } }, "Save"),
      el("button", { onclick: checkDesign }, "Check design"),
      el("button", { class: "primary", onclick: async () => { try { if (editor.dirty) await saveDesign(); await startRun(editor.name); } catch (error) { notice(error.message); } } }, "Run")),
    field("Description", input(doc, "description")),
    el("div", { class: "editor" },
      el("div", {}, el("div", { class: "subnav" }, TABS.map(([key, label]) => el("button", { class: editor.tab === key ? "active" : "",
        onclick: () => { editor.tab = key; renderEditor(); } }, label))), el("div", { class: "card" }, body)),
      el("aside", { class: "card", id: "check-panel" }, renderCheck(editor.check))));
  window.scrollTo(0, scrollY);
}

PAGES.experiment = {
  async show([name]) {
    if (!name) { go("experiments"); return; }
    hooks.changed = () => dirty();
    hooks.render = () => renderEditor();
    state.studies = await api("/api/studies");
    if (editor.name !== name || !editor.doc) {
      editor.doc = await api(`/api/design?name=${encodeURIComponent(name)}`);
      Object.assign(editor, { name, dirty: false, check: null, tab: "chain" });
    }
    renderEditor();
  },
};

window.addEventListener("beforeunload", (event) => { if (editor.dirty) { event.preventDefault(); event.returnValue = ""; } });
