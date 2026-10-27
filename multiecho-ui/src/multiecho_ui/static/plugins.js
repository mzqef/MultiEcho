"use strict";
// Plugins: the catalog an experiment is built from (worlds, interfaces, maskers, analysis methods).

const KIND_TEXT = {
  world: ["Worlds", "Engines, samplers and learned models. A world receives, evolves and emits; physical scenes are described in the editor itself."],
  interface: ["Interfaces", "Translators between two worlds. Scopes and parameter bridges are built in the editor."],
  masker: ["Maskers", "Select whole locations of the analyst's observation, frozen on discovery. Framework maskers are always available."],
  forward: ["Forward methods", "Fit on discovery responses; the frozen law predicts a context's response to a queried value."],
  backward: ["Backward methods", "Fit on discovery responses; the frozen law recovers the intervention from a response."],
};

PAGES.plugins = {
  async show() {
    const catalog = await loadCatalog(true);
    const root = document.getElementById("view-plugins");
    const errors = Object.entries(catalog.errors || {});
    mount(root, 
      el("h1", {}, "Plugins"),
      el("p", { class: "lead" }, "The building blocks of this workspace's experiments. Plugin modules are listed in ",
        el("code", {}, "workspace.json"), " (", el("code", {}, '"plugins"'), " and, if needed, ", el("code", {}, '"paths"'), "); each defines ",
        el("code", {}, "MULTIECHO_PLUGINS"), ", a sequence of ", el("code", {}, "(kind, factory)"), " pairs. Restart the workbench after changing plugin code."),
      el("p", { class: "note" }, `Modules: ${catalog.modules.join(", ") || "none (only the framework's maskers)"}`),
      errors.length ? el("div", { class: "banner bad" }, errors.map(([module, error]) => el("div", {}, `${module}: ${error}`))) : "",
      Object.entries(KIND_TEXT).map(([kind, [title, text]]) => {
        const entries = catalog.plugins.filter((p) => p.kind === kind);
        return el("div", {}, el("h2", {}, title), el("p", { class: "note" }, text),
          entries.length ? el("div", { class: "plugin-grid" }, entries.map((p) => el("div", { class: "card plugin" },
            el("div", { class: "title-row" }, el("b", {}, p.name), p.builtin ? el("span", { class: "pill" }, "framework") : ""),
            el("div", { class: "note" }, el("code", {}, p.id)),
            p.summary ? el("p", {}, p.summary) : "",
            p.parameters.length ? el("table", { class: "mini" }, el("tr", {}, ["Parameter", "Kind", "Default"].map((h) => el("th", {}, h))),
              p.parameters.map((q) => el("tr", {}, el("td", {}, q.name, q.required ? " *" : ""),
                el("td", {}, q.type === "auto" ? "set by the workbench" : q.type === "fixed" ? "not editable" : q.type),
                el("td", {}, q.default === null ? "—" : short(q.default, 40))))) : el("p", { class: "note" }, "No parameters."),
            p.doc && p.doc !== p.summary ? el("details", {}, el("summary", {}, "Documentation"), el("pre", {}, p.doc)) : "")))
            : el("p", { class: "empty" }, "None in this workspace."));
      }));
  },
};
