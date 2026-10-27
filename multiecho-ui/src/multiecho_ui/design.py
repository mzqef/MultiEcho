"""Experiment designs: one JSON document that builds a framework `Chain` and `Declaration`.

Schema `multiecho-ui.design/0.1` (see the README for every field):

    chain    worlds in order (a physical scene described here, or catalog worlds), one interface between each
             pair of consecutive worlds (a parameter bridge, a scope, or a catalog translator), the world taking
             the interventions, the analyst scope (optionally masked by a catalog masker), the observation and
             the stimulus delivered to a learned world.
    design   the declaration: question, handle, reference / calibration / query values, contexts generated from
             declared factors, tasks, draws, statistics, sweep, keyframe policy, kept frames and study lock.
    methods  the analysis method (catalog plugin) planned for each task; chosen again when a law is frozen.

`build` returns the chain, the declaration and a description of both, including one branch traced through
the chain (the analyst's view before any masker, which is calibrated on discovery when the experiment runs).
Every error names the part of the design it belongs to.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from multiecho_framework import (
    Axis,
    Chain,
    ConditionDimension,
    DataRole,
    Declaration,
    Grouping,
    Intervention,
    KeyframeAlignment,
    MaskedScope,
    ObservationSpec,
    ParameterBridge,
    ParameterMap,
    PhysicalScene,
    Region,
    Scope,
    ScopeInterface,
    SeedMode,
    Space,
    Statement,
    StimulusSpec,
    scene_context,
)

from .catalog import Catalog, CatalogError

SCHEMA = "multiecho-ui.design/0.1"
ROLES = (DataRole.DISCOVERY, DataRole.VALIDATION, DataRole.CONFIRMATION)
DISTRIBUTIONS = ("fixed", "uniform", "signed-uniform", "log-uniform", "choice", "cycle")


class DesignError(ValueError):
    """A design that cannot be built; `where` names the part of the design."""

    def __init__(self, where: str, message: str):
        super().__init__(f"{where}: {message}")
        self.where, self.message = where, message


@dataclass
class Built:
    chain: Chain
    declaration: Declaration
    keep_frames: tuple[int, ...]
    summary: dict


class _At:
    """Re-raise any error inside the block as a DesignError located at `where`."""

    def __init__(self, where: str):
        self.where = where

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is None or isinstance(error, DesignError):
            return False
        if isinstance(error, KeyError):
            raise DesignError(self.where, f"missing field {error}") from error
        if isinstance(error, Exception):
            raise DesignError(self.where, str(error) or type(error).__name__) from error
        return False


def _tuple(values, cast=float) -> tuple:
    if values is None:
        return ()
    if not isinstance(values, list | tuple):
        raise ValueError(f"expected a list, not {values!r}")
    return tuple(cast(value) for value in values)


def _pair(values, cast=float):
    if values in (None, [], ""):
        return None
    items = _tuple(values, cast)
    if len(items) != 2:
        raise ValueError(f"expected two numbers, not {values!r}")
    return items


def _number(value, name: str, cast=float):
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{name} is a number")
    return cast(value)


# ------------------------------------------------------------------------------------------- chain parts
def scope_of(spec: dict) -> Scope:
    region = spec.get("region")
    if region:
        region = Region.box(region.get("label") or "region", _pair(region["shape"], int), _pair(region["rows"], int),
                            _pair(region["columns"], int))
    return Scope(spec["id"], _tuple(spec.get("channels"), str), window=_pair(spec.get("window"), int),
                 region=region or None, coordinates=_tuple(spec["coordinates"], int) if spec.get("coordinates") else None,
                 keys=_tuple(spec["keys"], str) if spec.get("keys") else None, clip=_pair(spec.get("clip")))


def physical_scene(spec: dict) -> PhysicalScene:
    handle = spec.get("handle")
    space = None
    if handle:
        axes = tuple(Axis(axis["name"], axis.get("unit", ""), _number(axis["lower"], "lower"),
                          _number(axis["upper"], "upper")) for axis in handle.get("axes", []))
        space = Space(handle["id"], axes, handle.get("semantics", ""))
    return PhysicalScene(spec["id"], spec.get("revision") or "v1", space, description=spec.get("description", ""))


def world_of(spec: dict, catalog: Catalog):
    kind = spec.get("type")
    if kind == "physical":
        return physical_scene(spec)
    if kind == "plugin":
        return catalog.build(spec["plugin"], "world", spec.get("params"))
    raise ValueError(f"a world is 'physical' or 'plugin', not {kind!r}")


def interface_of(spec: dict, source: str, target: str, catalog: Catalog):
    kind = spec.get("type")
    if kind == "bridge":
        maps = tuple(ParameterMap(item["source"], item["target"], float(item.get("scale", 1.0)),
                                  float(item.get("offset", 0.0)), item.get("unit", "")) for item in spec.get("maps", []))
        constants = tuple(Statement(item["key"], item["value"], item.get("unit", ""), item.get("kind") or "parameter")
                          for item in spec.get("constants", []))
        return ParameterBridge(spec["id"], source, target, maps, constants, spec.get("channel") or "parameters")
    if kind == "scope":
        return ScopeInterface(spec["id"], source, target, scope_of(spec["scope"]))
    if kind == "plugin":
        return catalog.build(spec["plugin"], "interface", spec.get("params"), source=source, target=target,
                             interface_id=spec["id"])
    raise ValueError(f"an interface is 'bridge', 'scope' or 'plugin', not {kind!r}")


def stimulus_of(spec: dict | None) -> StimulusSpec | None:
    if not spec:
        return None
    return StimulusSpec(int(spec["frames"]), float(spec["interval"]), int(spec["height"]), int(spec["width"]),
                        spec.get("channel") or "video", int(spec.get("observe_after", 1)))


# ------------------------------------------------------------------------------------------- contexts
def _sample(distribution: dict, rng: np.random.Generator, index: int):
    kind = distribution.get("type")
    if kind == "fixed":
        value = distribution["value"]
        return value if isinstance(value, str) else float(value)
    if kind in ("uniform", "signed-uniform", "log-uniform"):
        low, high = float(distribution["low"]), float(distribution["high"])
        if not low <= high:
            raise ValueError("low must not exceed high")
        if kind == "log-uniform":
            if low <= 0:
                raise ValueError("a log-uniform range is positive")
            value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            value = float(rng.uniform(low, high))
            if kind == "signed-uniform":
                value *= float(rng.choice([-1.0, 1.0]))
        return round(value, 6)
    if kind in ("choice", "cycle"):
        values = list(distribution.get("values") or [])
        if not values:
            raise ValueError(f"a {kind} needs values")
        value = values[int(rng.integers(len(values)))] if kind == "choice" else values[index % len(values)]
        return value if isinstance(value, str) else float(value)
    raise ValueError(f"the distribution is one of {DISTRIBUTIONS}, not {kind!r}")


def contexts_of(spec: dict) -> tuple[tuple, tuple]:
    """Contexts per role from declared factors (one random stream from `seed`), and the condition dimensions."""
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    counts, factors = spec.get("counts") or {}, spec.get("factors") or []
    keys = [factor["key"] for factor in factors]
    if len(set(keys)) != len(keys) or not all(keys):
        raise ValueError("factor keys are unique and nonempty")
    contexts, values = [], {}
    for role in ROLES:
        count = int(counts.get(role.value, 0))
        if count < 0:
            raise ValueError(f"the {role.value} count is not negative")
        for index in range(count):
            context_id = f"{role.value}-{index:02d}"
            sampled: dict[str, object] = {}
            for position, factor in enumerate(factors):
                with _At(f"design.contexts.factors[{position}] ({factor.get('key')})"):
                    value = _sample(factor.get("distribution") or {}, rng, index)
                    gate = factor.get("when")
                    if gate:
                        if gate.get("key") not in sampled:
                            raise ValueError("'when' refers to an earlier factor")
                        if str(sampled[gate["key"]]) != str(gate.get("equals")):
                            other = gate.get("otherwise", 0.0)
                            value = other if isinstance(other, str) else float(other)
                    sampled[factor["key"]] = value
            statements = tuple(Statement(factor["key"], sampled[factor["key"]], factor.get("unit", ""),
                                         factor.get("kind") or "parameter") for factor in factors)
            contexts.append(scene_context(context_id, statements, role))
            values[context_id] = sampled
    conditions = []
    for name in spec.get("conditions") or []:
        if name not in keys:
            raise ValueError(f"condition {name!r} is not a declared factor")
        conditions.append(ConditionDimension(name, tuple((context_id, str(row[name])) for context_id, row in values.items())))
    return tuple(contexts), tuple(conditions)


def interventions(rows, steps: int, where: str) -> tuple[Intervention, ...]:
    with _At(where):
        return tuple(Intervention(_tuple(row if isinstance(row, list | tuple) else [row]), steps) for row in rows or [])


# ------------------------------------------------------------------------------------------- summaries
def _shape(channel) -> list[int]:
    return list(getattr(channel.data, "shape", [len(channel.data)]))


def describe_trace(trace) -> list[dict]:
    rows = []
    for artifact in trace:
        rows.append({"source": artifact.source.split("@")[0], "via": artifact.via.split(":")[0] if artifact.via else "",
                     "scope": artifact.scope.scope_id if artifact.scope else None,
                     "channels": [{"name": channel.name, "modality": channel.modality.value, "shape": _shape(channel),
                                   **({"statements": {s.key: s.value for s in channel.data}}
                                      if channel.modality.value == "text" and len(channel.data) <= 12 else {})}
                                  for channel in artifact.channels]})
    return rows


def describe_world(world) -> dict:
    card = world.describe()
    handle = card.handle
    return {"world_id": card.world_id, "revision": card.revision, "kind": card.kind.value, "accepts": list(card.accepts),
            "emits": list(card.emits), "stochastic": card.stochastic, "description": card.description,
            "handle": None if handle is None else {"id": handle.space_id, "semantics": handle.semantics,
                                                    "axes": [asdict(axis) for axis in handle.axes]},
            "contract": None if card.contract is None else asdict(card.contract)}


def _plain(value):
    return json.loads(json.dumps(value, default=lambda item: item.value if hasattr(item, "value") else str(item)))


# ------------------------------------------------------------------------------------------- build
def _labels(chain: Chain) -> list[str]:
    """What each artifact of a branch is, in the order `Chain.apply` records them."""
    ids = [world.describe().world_id for world in chain.worlds]
    labels = []
    for index, world_id in enumerate(ids):
        labels.append(f"{world_id} emits")
        if index < len(chain.interfaces):
            labels.append(f"{chain.interfaces[index].describe().interface_id} passes to {ids[index + 1]}")
    return [*labels, "joined for the analyst (context identity added)",
            f"the analyst sees (scope {chain.observer.scope.scope_id})"]


def _trace(chain: Chain, context, reference: Intervention):
    root = chain.open(context, 0)
    branch = chain.fork(root)
    try:
        state = chain.apply(branch, reference, 0)
        vector = np.asarray(chain.observer.scope.vector(state.final), dtype=np.float64)
        rows, labels = describe_trace(chain.trace(state)), _labels(chain)
        if len(rows) == len(labels):
            for row, label in zip(rows, labels, strict=True):
                row["label"] = label
        return rows, int(vector.size)
    finally:
        chain.close(branch)
        chain.close(root)


def build(document: dict, catalog: Catalog, study=None) -> Built:
    """Build the chain and declaration of a design; `study(name)` returns a StudyLock from the workspace."""
    if document.get("schema") != SCHEMA:
        raise DesignError("schema", f"expected {SCHEMA}")
    chain_spec, design = document.get("chain") or {}, document.get("design") or {}

    worlds = []
    for index, spec in enumerate(chain_spec.get("worlds") or []):
        with _At(f"chain.worlds[{index}]"):
            worlds.append(world_of(spec, catalog))
    if len(worlds) < 1:
        raise DesignError("chain.worlds", "declare at least one world")
    ids = [world.describe().world_id for world in worlds]
    if len(set(ids)) != len(ids):
        raise DesignError("chain.worlds", f"world identities repeat: {ids}")
    specs = chain_spec.get("interfaces") or []
    if len(specs) != len(worlds) - 1:
        raise DesignError("chain.interfaces", f"{len(worlds)} worlds need {len(worlds) - 1} interfaces, not {len(specs)}")
    interfaces = []
    for index, spec in enumerate(specs):
        with _At(f"chain.interfaces[{index}]"):
            interfaces.append(interface_of(spec, ids[index], ids[index + 1], catalog))
    with _At("chain.handle_world"):
        handle = int(chain_spec.get("handle_world", 0))
        if not 0 <= handle < len(worlds) or worlds[handle].describe().handle is None:
            raise ValueError("the world taking interventions must declare a handle (e.g. the physical scene's axes)")
    analyst = chain_spec.get("analyst") or {}
    with _At("chain.analyst.scope"):
        plain_scope = scope_of(analyst["scope"])
    with _At("chain.stimulus"):
        stimulus = stimulus_of(chain_spec.get("stimulus"))
    observation = chain_spec.get("observation") or {}
    chain_id = chain_spec.get("id") or "chain"

    def make_chain(scope, dims: int) -> Chain:
        return Chain(chain_id, tuple(worlds), tuple(interfaces),
                     ScopeInterface(analyst.get("id") or "analyst", ids[-1], "analyst", scope),
                     ObservationSpec(observation.get("id") or plain_scope.scope_id, (dims,),
                                     observation.get("semantics") or "analyst observation", observation.get("unit") or "1"),
                     handle=handle, stimulus=stimulus)

    with _At("chain"):
        provisional = make_chain(plain_scope, 1)
    with _At("design.steps"):
        steps = int(design.get("steps", 1))
    reference = interventions([design.get("reference")], steps, "design.reference")[0] \
        if design.get("reference") is not None else None
    if reference is None:
        raise DesignError("design.reference", "declare the reference value")
    with _At("design.contexts"):
        contexts, conditions = contexts_of(design.get("contexts") or {})
    discovery = [context for context in contexts if context.role is DataRole.DISCOVERY]
    if not discovery:
        raise DesignError("design.contexts.counts", "declare at least one discovery context")
    with _At("trace (one discovery branch at the reference value)"):
        trace, dims = _trace(provisional, discovery[0], reference)
    traced = dims

    masker_spec, masker_info = analyst.get("masker"), None
    scope = plain_scope
    if masker_spec:
        with _At("chain.analyst.masker"):
            if len(plain_scope.channels) != 1:
                raise ValueError("a masked analyst scope observes exactly one channel")
            width = int(masker_spec.get("width") or 1)
            if width < 1 or dims % width:
                raise ValueError(f"the location width {width} must divide the {dims} observed values")
            grouping = Grouping.blocks(dims // width, width)
            masker = catalog.build(masker_spec["plugin"], "masker", masker_spec.get("params"), grouping=grouping)
            fields = {key: getattr(plain_scope, key) for key in ("window", "region", "coordinates", "keys", "clip")}
            scope = MaskedScope(plain_scope.scope_id, plain_scope.channels, **fields, masker=masker,
                                observed=plain_scope.channels[0])
            output = int(masker.output_dims(dims))
            card = masker.describe()
            masker_info = {"masker_id": card.masker_id, "kind": card.kind, "needs": card.needs,
                           "locations": grouping.groups, "width": width, "output_dims": output,
                           "description": card.description}
            dims = output
    with _At("chain"):
        chain = make_chain(scope, dims)

    with _At("design.calibration"):
        calibration = interventions(design.get("calibration"), steps, "design.calibration")
    queries = interventions(design.get("queries"), steps, "design.queries")
    sweep_spec = design.get("sweep") or {}
    sweep = interventions(sweep_spec.get("values"), steps, "design.sweep.values")
    statistics = design.get("statistics") or {}
    keyframes = design.get("keyframes") or {}
    lock = None
    if design.get("study"):
        with _At("design.study"):
            if study is None:
                raise ValueError("no study locks are available")
            lock = study(design["study"])
    with _At("design"):
        draws = int(design.get("draws", 1))
        declaration = Declaration(
            design.get("experiment_id") or document.get("name") or "experiment", design.get("question") or "",
            chain.describe(), chain.record(), design.get("handle") or "", design.get("observation_time") or "",
            reference, queries, contexts, design.get("independence_unit") or "", calibration=calibration,
            tasks=tuple(design.get("tasks") or ("forward",)),
            coupling=SeedMode.FIXED if draws == 1 else SeedMode.CONTEXT, draws=draws,
            repeats=int(design.get("repeats", 2)), conditions=conditions,
            minimum_improvement=float(statistics.get("minimum_improvement", 0.0)),
            equivalence_margin=None if statistics.get("equivalence_margin") in (None, "") else
            float(statistics["equivalence_margin"]),
            confidence=float(statistics.get("confidence", 0.95)), resamples=int(statistics.get("resamples", 1999)),
            seed=int(statistics.get("seed", 0)), stimulus=stimulus, sweep=sweep,
            sweep_draws=int(sweep_spec.get("draws", 1)),
            keyframes=KeyframeAlignment(keyframes.get("mode") or "report", keyframes.get("warn"),
                                        keyframes.get("tolerance")),
            study=lock)
    with _At("design.keep_frames"):
        keep = _tuple(design.get("keep_frames"), int)

    counts = {role.value: len(declaration.contexts_for(role)) for role in ROLES}
    branches = len(calibration) + len(queries)
    summary = {
        "chain": {"id": chain_id, "digest": chain.describe().digest, "revision": chain.describe().revision,
                  "worlds": [describe_world(world) for world in worlds],
                  "interfaces": [_plain({"interface_id": item.describe().interface_id, "kind": item.describe().kind,
                                         "description": item.describe().description}) for item in interfaces],
                  "handle_world": handle, "analyst": {"scope": plain_scope.scope_id, "channels": list(plain_scope.channels),
                                                      "masker": masker_info},
                  "observation": {"dims": dims, "traced_dims": traced},
                  "stimulus": None if stimulus is None else asdict(stimulus)},
        "trace": trace,
        "declaration": {"experiment_id": declaration.experiment_id, "digest": declaration.digest, "contexts": counts,
                        "branches_per_context": branches, "pairs": {role: count * branches for role, count in counts.items()},
                        "draws": draws, "coupling": declaration.coupling.value, "tasks": list(declaration.tasks),
                        "conditions": {dimension.name: sorted(set(dimension.values)) for dimension in conditions},
                        "sweep": len(sweep), "study": lock.study_id if lock else None,
                        "handle": _plain([asdict(axis) for axis in declaration.world.intervention_space.axes])},
        "contexts": [{"context_id": context.context_id, "role": context.role.value,
                      "statements": {key: value for key, value in _context_values(context).items()}}
                     for context in contexts[:6]],
    }
    return Built(chain, declaration, keep, summary)


def _context_values(context) -> dict:
    """The statements of an inline (parameter-only) context; empty for contexts with recorded media."""
    if not context.payload_ref.startswith("inline:"):
        return {}
    payload = json.loads(context.payload_ref.removeprefix("inline:"))
    return {item["key"]: item["value"] for item in payload} if isinstance(payload, list) else {}


def validate(document: dict, catalog: Catalog, study=None) -> dict:
    """The summary of a design that builds, or the first error with its location."""
    try:
        built = build(document, catalog, study)
    except DesignError as error:
        return {"ok": False, "error": {"where": error.where, "message": error.message}}
    except CatalogError as error:
        return {"ok": False, "error": {"where": "catalog", "message": str(error)}}
    return {"ok": True, **built.summary}


def blank(name: str) -> dict:
    """A minimal design to start from: a physical scene with one axis, observed directly."""
    return copy.deepcopy({
        "schema": SCHEMA, "name": name, "description": "",
        "chain": {"id": f"{name}-chain", "handle_world": 0,
                  "worlds": [{"type": "physical", "id": "scene", "revision": "v1", "description": "",
                              "handle": {"id": "handle", "semantics": "the value the intervention sets", "axes": [
                                  {"name": "x", "unit": "1", "lower": 0.0, "upper": 1.0}]}}],
                  "interfaces": [],
                  "analyst": {"id": "analyst", "scope": {"id": "scene", "channels": ["scene"], "keys": ["x"]},
                              "masker": None},
                  "observation": {"id": "scene", "semantics": "the scene's statements", "unit": "1"},
                  "stimulus": None},
        "design": {"experiment_id": name, "question": "How does the observation respond to x?",
                   "handle": "x set in the scene", "observation_time": "after the intervention",
                   "independence_unit": "one context", "reference": [0.5], "calibration": [[0.25], [0.75]],
                   "queries": [[0.4], [0.6]], "steps": 1,
                   "contexts": {"seed": 0, "counts": {"discovery": 8, "validation": 4, "confirmation": 4},
                                "factors": [{"key": "offset", "unit": "1", "kind": "parameter",
                                             "distribution": {"type": "uniform", "low": 0.0, "high": 1.0}}],
                                "conditions": []},
                   "tasks": ["forward"], "draws": 1, "repeats": 2,
                   "statistics": {"minimum_improvement": 0.0, "equivalence_margin": None, "confidence": 0.95,
                                  "resamples": 1999, "seed": 0},
                   "sweep": {"values": [], "draws": 1}, "keyframes": {"mode": "free", "warn": None, "tolerance": None},
                   "keep_frames": [], "study": None},
        "methods": {},
    })


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
