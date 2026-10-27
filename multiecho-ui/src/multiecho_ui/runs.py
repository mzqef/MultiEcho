"""Experiment runs: one design measured, frozen, tested and sealed in the order the protocol enforces.

A run holds a live `Experiment`. Its stages are jobs on a background thread (one job at a time in a workspace):

    measure   discovery contexts (the masker, if any, is calibrated and frozen here)
    freeze    one analysis method per declared task, fitted on discovery responses only, then frozen
    test      validation and confirmation contexts, measured only after every task is frozen
    seal      the sealed report, its conformance checks, and the keyframe gallery

The framework refuses any other order; the run reports what it refuses. After each stage a compact summary
(responses, laws, evaluations, per-query diagnostics) is written to `runs/<id>.json`. A run lives in memory:
after a restart its record remains, and a new run of the same design reproduces it (measurement is seeded
by the declaration).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from multiecho_framework import DataRole, Experiment, dumps_report, save_gallery
from multiecho_framework.analysis import split_rows, views
from multiecho_framework.conformance import check_report

from .catalog import Catalog, CatalogError
from .design import Built, DesignError, build

RUN_SCHEMA = "multiecho-ui.run/0.1"
STAGES = ("declared", "measured", "frozen", "tested", "sealed")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


class RunError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _r(value, digits: int = 6):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _rms(array) -> float:
    array = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(array ** 2))) if array.size else 0.0


def _plain(value):
    return json.loads(json.dumps(value, default=lambda item: item.value if hasattr(item, "value") else str(item)))


def _kind(pair_id: str) -> str:
    return "calibration" if pair_id.rsplit("/", 1)[-1].startswith("c") else "query"


class Run:
    def __init__(self, run_id: str, design_name: str, document: dict, built: Built):
        self.run_id, self.design_name, self.document, self.built = run_id, design_name, document, built
        self.experiment = Experiment(built.declaration, built.chain, keep_frames=built.keep_frames)
        self.lock = threading.RLock()
        self.record = {
            "schema": RUN_SCHEMA, "id": run_id, "design": design_name, "created_at": _now(),
            "design_digest": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
            "experiment_id": built.declaration.experiment_id, "declaration_digest": built.declaration.digest,
            "chain_digest": built.chain.describe().digest, "tasks": list(built.declaration.tasks),
            "stage": "declared", "busy": None, "progress": None, "error": None, "live": True,
            "log": [{"at": _now(), "stage": "declared", "note": "declaration sealed; nothing measured yet"}],
            "design_summary": built.summary["declaration"], "chain": built.summary["chain"],
            "methods": document.get("methods") or {},
            "discovery": None, "laws": {}, "tests": None, "report": None,
        }
        self.basis: dict | None = None   # discovery principal directions, for projecting held-out responses

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.record))

    def note(self, stage: str, note: str, seconds: float | None = None) -> None:
        with self.lock:
            self.record["log"].append({"at": _now(), "stage": stage, "note": note,
                                       **({"seconds": round(seconds, 3)} if seconds is not None else {})})

    def allowed(self) -> dict:
        """Which actions the protocol admits now (the framework enforces the same order)."""
        record, tasks = self.record, set(self.record["tasks"])
        measured = record["discovery"] is not None
        frozen = set(record["laws"]) == tasks
        return {"measure": record["live"] and not measured,
                "freeze": record["live"] and measured and not frozen,
                "test": record["live"] and frozen and record["tests"] is None,
                "seal": record["live"] and record["tests"] is not None and record["report"] is None}

    # ---------------------------------------------------------------- counting
    def _count_pairs(self, phase: str, total: int):
        runner = self.experiment.runner
        original = runner.run
        state = {"done": 0}

        def counted(request):
            result = original(request)
            state["done"] += 1
            with self.lock:
                self.record["progress"] = {"phase": phase, "done": state["done"], "total": total}
            return result

        runner.run = counted
        return lambda: setattr(runner, "run", original)

    def _pairs(self, role: DataRole) -> int:
        declaration = self.built.declaration
        return len(declaration.contexts_for(role)) * (len(declaration.calibration) + len(declaration.interventions))

    # ----------------------------------------------------------------- stages
    def measure(self) -> None:
        if not self.allowed()["measure"]:
            raise RunError("Discovery is already measured")
        restore = self._count_pairs("measuring discovery pairs", self._pairs(DataRole.DISCOVERY))
        start = time.perf_counter()
        try:
            dataset = self.experiment.measure(DataRole.DISCOVERY)
        finally:
            restore()
        summary = self._discovery(dataset)
        with self.lock:
            self.record["discovery"] = summary
            self.record["stage"] = "measured"
        self.note("measured", f"{len(dataset.results)} discovery pairs; determinism "
                  f"{'identical' if summary['determinism'].get('identical') else 'VIOLATED'}", time.perf_counter() - start)

    def freeze(self, task: str, method: dict, catalog: Catalog) -> None:
        if not self.allowed()["freeze"]:
            raise RunError("Freeze laws after measuring discovery, before testing")
        if task not in self.record["tasks"]:
            raise RunError(f"Task {task!r} was not declared")
        if task in self.record["laws"]:
            raise RunError(f"A {task} law is already frozen; a frozen law cannot be refitted")
        start = time.perf_counter()
        fitter = catalog.build(method["plugin"], task, method.get("params"))
        law = fitter.fit(self.experiment.responses())
        digest = self.experiment.freeze(law, task)
        spec = law.describe()
        with self.lock:
            self.record["laws"][task] = {
                "method": {"plugin": method["plugin"], "params": method.get("params") or {}},
                "law_id": spec.law_id, "revision": spec.revision, "hypothesis": spec.hypothesis,
                "validity_domain": spec.validity_domain, "fitted_on": len(spec.fitted_on), "digest": digest}
            if set(self.record["laws"]) == set(self.record["tasks"]):
                self.record["stage"] = "frozen"
        self.note("frozen", f"{task}: {spec.law_id} ({method['plugin'].split(':')[-1]})", time.perf_counter() - start)

    def test(self) -> None:
        if not self.allowed()["test"]:
            raise RunError("Test held-out contexts after every declared task is frozen")
        declaration = self.built.declaration
        total = sum(self._pairs(role) for role in declaration.held_out_roles)
        restore = self._count_pairs("measuring held-out pairs", total)
        start = time.perf_counter()
        try:
            evaluations = self.experiment.test()
        finally:
            restore()
        summary = self._tests(evaluations)
        with self.lock:
            self.record["tests"] = summary
            self.record["stage"] = "tested"
        self.note("tested", ", ".join(f"{role.value} {len(declaration.contexts_for(role))} contexts"
                                      for role in declaration.held_out_roles), time.perf_counter() - start)

    def seal(self, workspace_root: Path) -> None:
        if not self.allowed()["seal"]:
            raise RunError("Seal the report after testing")
        start = time.perf_counter()
        report = self.experiment.report()
        reports = workspace_root / "reports"
        reports.mkdir(exist_ok=True)
        name = f"{self.run_id}.json"
        (reports / name).write_text(dumps_report(report), encoding="utf-8")
        gallery = None
        if self.experiment.tap is not None and self.experiment.tap.keep:
            folder = workspace_root / "keyframes"
            folder.mkdir(exist_ok=True)
            save_gallery(folder / f"{self.run_id}.npz", self.experiment.keyframe_gallery())
            gallery = f"{self.run_id}.npz"
        checks = check_report(report)
        summary = {
            "name": name, "gallery": gallery, "seal": report["seal"],
            "checks": {"passed": sum(c.passed for c in checks), "total": len(checks),
                       "failed": [{"id": c.check_id, "title": c.title, "detail": c.detail} for c in checks if not c.passed]},
            "outcomes": {task: {"value": analysis["outcome"]["value"], "meaning": analysis["outcome"].get("meaning"),
                                "holds_in": analysis["outcome"].get("holds_in") or {},
                                "claim": (analysis.get("claims") or [{}])[0].get("statement")}
                         for task, analysis in report["analyses"].items()},
            "response": _plain(report.get("response")),
            "keyframes": None if not report.get("keyframes") else {
                key: report["keyframes"].get(key) for key in ("verdict", "status", "metric", "summary", "warnings")},
            "masking": _plain(report.get("masking")),
        }
        with self.lock:
            self.record["report"] = summary
            self.record["stage"] = "sealed"
        self.note("sealed", f"report {name}; {summary['checks']['passed']}/{summary['checks']['total']} checks",
                  time.perf_counter() - start)

    # ---------------------------------------------------------------- summaries
    def _discovery(self, dataset) -> dict:
        declaration = self.built.declaration
        conditions = {dimension.name: dict(dimension.assignments) for dimension in declaration.conditions}
        effects = np.stack([row.mean_effect for row in dataset.results])
        centred = effects - effects.mean(axis=0)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        energy = singular ** 2
        share = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
        components = min(2, vt.shape[0])
        scores = centred @ vt[:components].T if components else np.zeros((len(effects), 0))
        self.basis = {"mean": effects.mean(axis=0), "vectors": vt[:components]}
        contexts: dict[str, dict] = {}
        for index, row in enumerate(dataset.results):
            context = row.request.context
            entry = contexts.setdefault(context.context_id, {
                "context_id": context.context_id,
                "conditions": {name: values.get(context.context_id) for name, values in conditions.items()},
                "baseline_rms": _r(_rms(row.baseline.mean(axis=0))), "branches": []})
            entry["branches"].append({
                "kind": _kind(row.request.pair_id), "values": [float(v) for v in row.request.intervention.values],
                "effect_rms": _r(_rms(row.mean_effect)),
                "draw_spread": _r(float(np.sqrt(np.mean(np.var(row.intervention - row.baseline, axis=0))))
                                  if row.intervention.shape[0] > 1 else 0.0),
                "pc": [_r(v) for v in scores[index]]})
        for entry in contexts.values():
            entry["branches"].sort(key=lambda branch: branch["values"])
        masking = self.experiment.masking
        keyframes = self.experiment.keyframe_section()
        return {
            "pairs": len(dataset.results), "dims": int(effects.shape[1]),
            "reference": [float(v) for v in declaration.reference.values],
            "axes": [axis.name for axis in declaration.world.intervention_space.axes],
            "effect_rms": _r(_rms(effects)), "explained": [_r(v) for v in share[:components]],
            "contexts": list(contexts.values()), "determinism": _plain(self.experiment.determinism),
            "masking": None if masking is None else _plain({"frozen": masking.get("frozen"),
                                                             "coverage": masking.get("coverage")}),
            "keyframes": None if keyframes is None else _plain({key: keyframes.get(key) for key in
                                                                ("verdict", "status", "metric", "summary", "warnings")}),
        }

    def _tests(self, evaluations) -> dict:
        declaration, experiment = self.built.declaration, self.experiment
        out = {"tasks": {}, "diagnostics": {}}
        for task, roles in evaluations.items():
            out["tasks"][task.value] = {
                role.value: {"descriptive_r2": _r(report.descriptive_r2), "group_weighted_mse": _r(report.group_weighted_mse),
                             "queries": len(report.unit_ids), "target": report.target,
                             "comparisons": [{"comparator": c.comparator, "advantage": _r(c.mean_error_advantage),
                                              "interval": None if c.interval is None else [_r(v) for v in c.interval],
                                              "status": c.status.value, "groups": c.groups, "confidence": c.confidence}
                                             for c in report.comparisons]}
                for role, report in roles.items()}
            out["tasks"][task.value]["regimes"] = _plain(experiment.regimes.get(task, {}))
        laws = {task.value: law for task, law in experiment.laws.items()}
        for role in declaration.held_out_roles:
            dataset = experiment.datasets[role]
            view_map, (_, queries) = views(dataset), split_rows(dataset)
            rows = []
            for context_id, indices in queries.items():
                view = view_map[context_id]
                for index in indices:
                    row = dataset.results[index]
                    truth = [float(v) for v in row.request.intervention.values]
                    item = {"context_id": context_id, "values": truth, "effect_rms": _r(_rms(row.mean_effect))}
                    if "forward" in laws:
                        predicted = np.asarray(laws["forward"].predict(view, row.request.intervention), dtype=np.float64)
                        item["forward"] = {"error_rms": _r(_rms(predicted - row.mean_effect)),
                                           "predicted_rms": _r(_rms(predicted)),
                                           "observed_pc": self._project(row.mean_effect),
                                           "predicted_pc": self._project(predicted)}
                    if "backward" in laws:
                        item["backward"] = {"recovered": [_r(v) for v in laws["backward"].recover(view, row.mean_effect)]}
                    rows.append(item)
            out["diagnostics"][role.value] = rows
        return out

    def _project(self, vector) -> list:
        if self.basis is None or not len(self.basis["vectors"]):
            return []
        return [_r(v) for v in (np.asarray(vector, dtype=np.float64) - self.basis["mean"]) @ self.basis["vectors"].T]


class Runs:
    """The runs of a workspace: live runs in memory, every run's record in `runs/`."""

    def __init__(self, root: Path, catalog: Catalog, study=None):
        self.root, self.catalog, self.study = root, catalog, study
        self.live: dict[str, Run] = {}
        self.job: threading.Thread | None = None
        self.active: str | None = None   # the run whose stage is running; cleared together with its `busy`
        self.lock = threading.Lock()

    @property
    def folder(self) -> Path:
        return self.root / "runs"

    def _save(self, run: Run) -> None:
        self.folder.mkdir(exist_ok=True)
        snapshot = run.snapshot()
        (self.folder / f"{run.run_id}.json").write_text(json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")

    def records(self) -> list[dict]:
        out = {}
        if self.folder.is_dir():
            for path in sorted(self.folder.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if record.get("schema") == RUN_SCHEMA:
                    out[record["id"]] = record
        for run_id, run in self.live.items():
            out[run_id] = run.snapshot()
        rows = []
        for record in out.values():
            live = record["id"] in self.live
            rows.append({key: record.get(key) for key in ("id", "design", "experiment_id", "created_at", "stage", "busy",
                                                          "error")}
                        | {"live": live, "report": (record.get("report") or {}).get("name"),
                           "outcomes": {task: value.get("value") for task, value in
                                        ((record.get("report") or {}).get("outcomes") or {}).items()}})
        return sorted(rows, key=lambda row: row["created_at"] or "", reverse=True)

    def get(self, run_id: str) -> dict:
        if run_id in self.live:
            run = self.live[run_id]
            return {**run.snapshot(), "allowed": run.allowed()}
        path = self.folder / f"{run_id}.json"
        if not NAME.fullmatch(run_id) or not path.is_file():
            raise KeyError(run_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["live"] = False
        if record.get("busy"):
            record["busy"] = None
            record["error"] = record.get("error") or "interrupted: the workbench stopped during this stage"
        return {**record, "allowed": {"measure": False, "freeze": False, "test": False, "seal": False}}

    def start(self, design_name: str, document: dict) -> str:
        built = build(document, self.catalog, self.study)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = re.sub(r"[^A-Za-z0-9._-]", "-", design_name)[:80] or "run"
        run_id, suffix = f"{base}-{stamp}", 1
        while run_id in self.live or (self.folder / f"{run_id}.json").exists():
            suffix += 1
            run_id = f"{base}-{stamp}-{suffix}"
        run = Run(run_id, design_name, document, built)
        self.live[run_id] = run
        self._save(run)
        return run_id

    def act(self, run_id: str, action: str, payload: dict | None = None) -> None:
        """Start one stage (or all remaining stages) as the workspace's single background job."""
        run = self.live.get(run_id)
        if run is None:
            raise RunError("This run is not live (the workbench restarted); start a new run of its design")
        payload = payload or {}
        steps = self._steps(run, action, payload)
        with self.lock:
            if self.active is not None:
                raise RunError("Another stage is running; wait for it to finish")
            self.active = run_id
            with run.lock:
                run.record["busy"], run.record["error"], run.record["progress"] = action, None, None
            self.job = threading.Thread(target=self._work, args=(run, steps), daemon=True, name=f"run:{run_id}")
            self.job.start()

    def _steps(self, run: Run, action: str, payload: dict) -> list:
        allowed = run.allowed()
        if action in ("measure", "test"):
            if not allowed[action]:
                raise RunError({"measure": "Discovery is already measured",
                                "test": "Test held-out contexts after every declared task is frozen, once"}[action])
            return [getattr(run, action)]
        if action == "seal":
            if not allowed["seal"]:
                raise RunError("Seal the report after testing, once")
            return [lambda: run.seal(self.root)]
        if action == "freeze":
            task, method = payload.get("task"), payload.get("method") or {}
            if not allowed["freeze"]:
                raise RunError("Freeze laws after measuring discovery, before testing")
            if task not in run.record["tasks"]:
                raise RunError(f"Task {task!r} was not declared")
            if task in run.record["laws"]:
                raise RunError(f"A {task} law is already frozen; a frozen law cannot be refitted")
            self.catalog.entry(method.get("plugin", ""), task)
            return [lambda: run.freeze(task, method, self.catalog)]
        if action == "all":
            methods = payload.get("methods") or run.record["methods"]
            steps = []
            if allowed["measure"]:
                steps.append(run.measure)
            for task in run.record["tasks"]:
                if task in run.record["laws"]:
                    continue
                method = methods.get(task)
                if not method or not method.get("plugin"):
                    raise RunError(f"Choose a {task} method before running every stage")
                self.catalog.entry(method["plugin"], task)
                steps.append(lambda task=task, method=method: run.freeze(task, method, self.catalog))
            if run.record["tests"] is None:
                steps.append(run.test)
            if run.record["report"] is None:
                steps.append(lambda: run.seal(self.root))
            return steps
        raise RunError(f"Unknown action {action!r}")

    def _work(self, run: Run, steps: list) -> None:
        try:
            for step in steps:
                step()
                self._save(run)
        except (RunError, CatalogError, DesignError, PermissionError, ValueError, RuntimeError, TypeError) as error:
            with run.lock:
                run.record["error"] = f"{type(error).__name__}: {error}"
            run.note("error", run.record["error"])
        except Exception as error:  # noqa: BLE001 - a plugin failure is reported in the run, not lost
            with run.lock:
                run.record["error"] = f"{type(error).__name__}: {error}"
                run.record["traceback"] = traceback.format_exc(limit=8)
            run.note("error", run.record["error"])
        finally:
            with self.lock, run.lock:
                run.record["busy"], run.record["progress"] = None, None
                self.active = None
            self._save(run)

    def wait(self, timeout: float = 60.0) -> None:
        """Block until the current job finishes (tests and scripts)."""
        job = self.job
        if job is not None:
            job.join(timeout)

    def discard(self, run_id: str) -> None:
        """Drop a live run from memory (its record and any sealed report stay)."""
        with self.lock:
            if self.active == run_id:
                raise RunError("This run is busy")
            self.live.pop(run_id, None)
