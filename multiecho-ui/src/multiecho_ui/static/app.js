"use strict";
// Core: DOM helpers, API calls, formatting, routing and start-up. Page modules register in PAGES.

const PAGES = {};
const state = { info: null, catalog: null, studies: [] };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "value" && (tag === "input" || tag === "textarea" || tag === "select")) node.value = value;
    else if (key === "checked") node.checked = Boolean(value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === "") continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Replace a node's children; arrays are flattened and empty values skipped, as in `el`. */
function mount(node, ...children) {
  node.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false && c !== "")
    .map((c) => (c instanceof Node ? c : document.createTextNode(String(c)))));
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === "") continue;
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

async function post(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

const missing = (x) => x === null || x === undefined || (typeof x === "number" && Number.isNaN(x));
const fmt = (x, digits = 3) => (missing(x) ? "—"
  : Math.abs(x) >= 1e4 || (Math.abs(x) > 0 && Math.abs(x) < 1e-3) ? Number(x).toExponential(2) : Number(x).toFixed(digits));
const short = (value, n = 90) => { const text = JSON.stringify(value); return text && text.length > n ? `${text.slice(0, n - 3)}...` : text; };
const digest = (text, n = 12) => (text ? `${String(text).slice(0, n)}…` : "—");
const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

function badge(status) {
  const kind = { supported: "ok", stable_law: "ok", sealed: "ok", frozen: "ok", done: "ok", identical: "ok", within_tolerance: "ok",
    regime_specific: "warn", not_supported: "bad", unidentified_in_scope: "bad", inconclusive: "warn",
    differs: "bad", exceeds_tolerance: "bad", not_captured: "warn" }[status] || "";
  return el("span", { class: `pill ${kind}` }, String(status ?? "—").replaceAll("_", " "));
}

function notice(message, kind = "bad") {
  const host = document.getElementById("notice");
  const box = el("div", { class: `banner ${kind}` }, message, el("button", { class: "close", onclick: () => box.remove() }, "×"));
  host.replaceChildren(box);
  if (kind === "ok") setTimeout(() => box.remove(), 4000);
}

function emptyRow(text, span) {
  return el("tr", {}, el("td", { colspan: span, class: "empty" }, text));
}

/* ---------------------------------------------------------------- routing */
function go(...parts) {
  const hash = "#" + parts.filter((part) => part !== undefined && part !== null && part !== "").map(encodeURIComponent).join("/");
  if (location.hash === hash) route(); else location.hash = hash;
}

function route() {
  const parts = location.hash.slice(1).split("/").filter(Boolean).map(decodeURIComponent);
  const name = PAGES[parts[0]] ? parts[0] : "experiments";
  const nav = { experiment: "experiments", run: "runs" }[name] || name;
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === nav));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  Object.values(PAGES).forEach((page) => page.leave && page !== PAGES[name] && page.leave());
  document.getElementById("notice").replaceChildren();
  Promise.resolve(PAGES[name].show(parts.slice(1))).catch((error) => {
    console.error(error);
    notice(error.message);
  });
}

async function loadCatalog(force = false) {
  if (!state.catalog || force) state.catalog = await api("/api/catalog");
  return state.catalog;
}

const plugins = (kind) => (state.catalog ? state.catalog.plugins.filter((p) => p.kind === kind) : []);
const plugin = (id) => (state.catalog ? state.catalog.plugins.find((p) => p.id === id) : null);

async function boot() {
  document.querySelectorAll("#nav button").forEach((b) => b.addEventListener("click", () => go(b.dataset.view)));
  try {
    state.info = await api("/api/workspace");
    document.getElementById("workspace").textContent = `${state.info.title} · ${state.info.root}`;
    await loadCatalog();
    const errors = Object.entries(state.info.catalog_errors || {});
    window.addEventListener("hashchange", route);
    route();
    if (errors.length) notice(`Plugin modules failed to load: ${errors.map(([m, e]) => `${m} (${e})`).join("; ")}`, "warn");
  } catch (error) {
    notice(`The workbench could not start: ${error.message}`);
  }
}

window.addEventListener("DOMContentLoaded", boot);
