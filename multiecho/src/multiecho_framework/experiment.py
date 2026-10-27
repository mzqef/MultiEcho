"""A stage-enforcing experiment session over a chain, producing a sealed protocol report.

Stages (PROTOCOL.md section 7): declare -> [sweep discovery -> freeze masker -> assign
instances] -> measure discovery -> check determinism -> freeze each declared task ->
[sweep held-out -> assign instances] -> measure and test held-out -> report.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from ._version import __version__
from .analysis import CONTROLS, Task, response_set, split_rows, views
from .chain import Chain
from .contracts import DataRole, PairRequest, fingerprint
from .datasets import PairDataset
from .evaluation import EvaluationPlan, EvaluationReport, EvaluationTarget, evaluate_predictions
from .execution import PairRunner, WorldAdapter
from .keyframes import KeyframeGallery, not_captured, summarize
from .masking import MaskedScope, MaskInput, Sweep, array_digest, coverage
from .protocol import (
    OUTCOME_MEANING,
    PROTOCOL_VERSION,
    Claim,
    Declaration,
    Outcome,
    _plain,
    adjudicate,
    regime_verdict,
)
from .serialization import dumps_dataset
from .worlds import WorldKind

REPORT_SCHEMA = "multiecho-report/0.3"
RESPONSE_METHOD = ("Descriptive only; no triviality judgment. Response RMS of held-out query responses; draw spread: "
                   "RMS standard error of each pair's effect over context draws.")
TARGETS = {Task.FORWARD: EvaluationTarget.MEAN_RESPONSE, Task.BACKWARD: EvaluationTarget.INTERVENTION_DELTA}


def _seed(declaration: Declaration, *parts: object) -> int:
    key = json.dumps([declaration.digest, *parts], sort_keys=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def context_seeds(declaration: Declaration, context_id: str) -> tuple[int, ...]:
    """Hard control: every branch of a context uses these draw seeds."""
    return tuple(_seed(declaration, context_id, "draw", draw) for draw in range(declaration.draws))


def sweep_seeds(declaration: Declaration, context_id: str) -> tuple[int, ...]:
    return tuple(_seed(declaration, context_id, "sweep", draw) for draw in range(declaration.sweep_draws))


class _KeyframeTap:
    """The measured chain, unchanged, recording the stimulus each branch delivered to its first learned world.

    It has the chain's identity and delegates every call; it only reads
    `Chain.delivered` after `apply`. Frames listed in `keep` are retained for a gallery.
    """

    def __init__(self, chain, learned: str, channel: str, keep: tuple[int, ...]):
        self.chain, self.learned, self.channel, self.keep = chain, learned, channel, keep
        self.entries: list[dict] = []
        self._seen: set[tuple] = set()

    def describe(self):
        return self.chain.describe()

    def open(self, context, seed):
        return self.chain.open(context, seed)

    def fork(self, state):
        return self.chain.fork(state)

    def observe(self, state):
        return self.chain.observe(state)

    def close(self, state):
        self.chain.close(state)

    def apply(self, state, intervention, seed):
        state = self.chain.apply(state, intervention, seed)
        key = (state.context.context_id, intervention.values, intervention.steps, seed)
        if key not in self._seen:
            self._seen.add(key)
            frames = np.asarray(self.chain.delivered(state, self.learned).channel(self.channel).data)
            self.entries.append({"context_id": state.context.context_id, "values": intervention.values, "seed": seed,
                                 "frame": np.array(frames[-1], dtype=np.float32),
                                 "kept": np.array(frames[list(self.keep)], dtype=np.float32) if self.keep else None})
        return state


class Experiment:
    """Declare -> measure discovery -> freeze each declared task -> test held-out stages -> report.

    Held-out contexts (including their sweeps) are measured only after every declared
    task is frozen; a frozen law cannot be replaced; the measured world's identity
    cannot change. A masked analyst scope is calibrated on discovery sweeps only.
    """

    def __init__(self, declaration: Declaration, subject: WorldAdapter, *, keep_frames: tuple[int, ...] = ()):
        """`keep_frames` (stimulus frame indices, e.g. (0, -2, -1)) are retained for `keyframe_gallery`."""
        if subject.describe().digest != declaration.world.digest:
            raise ValueError("Measured world identity differs from the declared world")
        self.declaration, self.subject = declaration, subject
        self.tap = self._keyframe_tap(tuple(keep_frames))
        self.runner = PairRunner(self.tap if self.tap is not None else subject)
        self.tasks = tuple(Task(task) for task in declaration.tasks)
        self.datasets: dict[DataRole, PairDataset] = {}
        self.laws: dict[Task, object] = {}
        self.determinism: dict | None = None
        self.evaluations: dict[Task, dict[DataRole, EvaluationReport]] = {task: {} for task in self.tasks}
        self.regimes: dict[Task, dict[str, dict[str, dict[str, dict[str, str]]]]] = {task: {} for task in self.tasks}
        self.reference: dict | None = None
        observer = getattr(subject, "observer", None)
        self.scope = observer.scope if observer is not None and isinstance(observer.scope, MaskedScope) else None
        if (self.scope is not None) != ("masker" in declaration.chain.get("observer", {}).get("scope", {})):
            raise ValueError("A declared masker requires a chain whose analyst scope is that MaskedScope")
        self.masking = {"instances": {}, "coverage": {}, "frozen": None} if self.scope is not None else None
        evaluation = {seed for context in declaration.contexts for seed in context_seeds(declaration, context.context_id)}
        swept = {seed for context in declaration.contexts for seed in sweep_seeds(declaration, context.context_id)}
        if declaration.sweep and evaluation & swept:
            raise ValueError("Sweep seeds collide with evaluation seeds; change the declaration seed")
        self.stages = [{"stage": "declared", "role": None, "task": None, "digest": declaration.digest,
                        "world": declaration.world.digest}]

    def _log(self, stage: str, role: DataRole | None, digest: str, task: Task | None = None) -> None:
        self.stages.append({"stage": stage, "role": role.value if role else None,
                            "task": task.value if task else None, "digest": digest,
                            "world": self.runner.spec.digest})

    def _keyframe_tap(self, keep: tuple[int, ...]) -> _KeyframeTap | None:
        """A tap on the chain's delivered stimulus, or None (with `keyframe_reason`) when none can be placed."""
        declaration, subject = self.declaration, self.subject
        self.keyframe_reason = ""
        if declaration.stimulus is None:
            self.keyframe_reason = "no stimulus is declared"
        elif declaration.keyframes.mode == "free" and not keep:
            self.keyframe_reason = "the keyframe policy is free"
        elif not isinstance(subject, Chain):
            self.keyframe_reason = "the measured world is not a Chain, so its delivered stimulus is not visible"
        else:
            learned = [w.describe().world_id for w in subject.worlds if w.describe().kind is WorldKind.LEARNED]
            if learned:
                frames = declaration.stimulus.frames
                if any(not -frames <= index < frames for index in keep):
                    raise ValueError(f"Kept frame indices must lie within the {frames}-frame stimulus")
                return _KeyframeTap(subject, learned[0], declaration.stimulus.channel, keep)
            self.keyframe_reason = "the chain has no learned world"
        return None

    def keyframe_section(self) -> dict | None:
        """The report's keyframe section: differences of every branch's keyframe to its reference branch."""
        declaration = self.declaration
        if declaration.stimulus is None:
            return None
        if self.tap is None:
            return not_captured(declaration.keyframes, self.keyframe_reason)
        roles = {context.context_id: context.role.value for context in declaration.contexts}
        measured = {context.context_id for role in self.datasets for context in declaration.contexts_for(role)}
        entries = [entry for entry in self.tap.entries if entry["context_id"] in measured]
        if declaration.keyframes.mode == "free":
            return not_captured(declaration.keyframes, "the keyframe policy is free")
        return summarize(declaration.keyframes, declaration.reference.values, entries, roles)

    def keyframe_gallery(self) -> KeyframeGallery:
        """The kept stimulus frames of every measured branch (construct with `keep_frames`)."""
        if self.tap is None or not self.tap.keep:
            raise ValueError("Construct the experiment with keep_frames to collect a keyframe gallery")
        declaration, entries = self.declaration, self.tap.entries
        if not entries:
            raise ValueError("No branch has been measured yet")
        frames = declaration.stimulus.frames
        labels = tuple(f"frame {index % frames}" + (" (keyframe)" if index % frames == frames - 1 else "")
                       for index in self.tap.keep)
        roles = {context.context_id: context.role.value for context in declaration.contexts}
        kept = np.stack([entry["kept"] for entry in entries])
        if kept.ndim == 4:
            kept = kept[..., None]
        return KeyframeGallery(
            kept, labels, tuple(e["context_id"] for e in entries), tuple(roles[e["context_id"]] for e in entries),
            np.array([e["values"] for e in entries], dtype=np.float64), tuple(e["seed"] for e in entries),
            tuple(tuple(e["values"]) == tuple(declaration.reference.values) for e in entries),
            {"experiment_id": declaration.experiment_id, "declaration_digest": declaration.digest,
             "handle": declaration.handle, "policy": declaration.keyframes.record(),
             "reference": list(declaration.reference.values),
             "study": declaration.study.study_id if declaration.study else None})

    def _requests(self, role: DataRole) -> tuple[PairRequest, ...]:
        declaration, requests = self.declaration, []
        for context in declaration.contexts_for(role):
            seeds = context_seeds(declaration, context.context_id)
            for kind, branches in (("c", declaration.calibration), ("q", declaration.interventions)):
                for index, action in enumerate(branches):
                    requests.append(PairRequest(f"{context.context_id}/{kind}{index}", context, declaration.reference,
                                                action, declaration.coupling, seeds))
        return tuple(requests)

    def _sweep(self, role: DataRole) -> list[MaskInput]:
        """Run the declared sweep on every context of a role, through the chain, unmasked."""
        declaration, subject, scope = self.declaration, self.subject, self.scope
        names = getattr(subject, "reference_channels", frozenset())
        inputs, digests = [], []
        scope.begin_sweep()
        try:
            for context in declaration.contexts_for(role):
                seeds, outputs, references = sweep_seeds(declaration, context.context_id), [], {}
                for seed in seeds:
                    row = []
                    for action in declaration.sweep:
                        root = subject.open(context, seed)
                        branch = subject.fork(root)
                        try:
                            state = subject.apply(branch, action, seed)
                            row.append(np.asarray(state.final.channel(scope.observed).data, dtype=np.float64))
                            if not references:
                                joined = state.trace[-2]
                                references = {c.name: c.data for c in joined.channels if c.name in names}
                        finally:
                            subject.close(branch)
                            subject.close(root)
                    outputs.append(row)
                sweep = Sweep(declaration.sweep, seeds, np.asarray(outputs))
                inputs.append(MaskInput(context.context_id, sweep, references))
                digests.append(array_digest(sweep.outputs))
        finally:
            scope.end_sweep()
        self._log("swept", role, fingerprint(digests))
        return inputs

    def _assign(self, role: DataRole, inputs: list[MaskInput]) -> None:
        scope, masking = self.scope, self.masking
        frozen = scope.frozen
        card = frozen.describe()
        values = []
        for item in inputs:
            instance = frozen.fit(item)
            masking["instances"][item.context_id] = scope.assign(item.context_id, instance)
            values.append(coverage(card, frozen, instance))
        masking["coverage"][role.value] = float(np.mean(values))
        self._log("masked", role, fingerprint({c.context_id: masking["instances"][c.context_id]
                                               for c in self.declaration.contexts_for(role)}))

    def _prepare_masks(self, role: DataRole) -> None:
        if self.scope is None:
            return
        inputs = self._sweep(role) if self.declaration.sweep else [
            MaskInput(context.context_id) for context in self.declaration.contexts_for(role)]
        if role is DataRole.DISCOVERY:
            frozen = self.scope.masker.calibrate(inputs)
            self.scope.freeze(frozen)
            self.masking["frozen"] = frozen.describe().record()
            self._log("masker-frozen", None, frozen.describe().digest)
        self._assign(role, inputs)

    def measure(self, role: DataRole) -> PairDataset:
        if role in self.datasets:
            return self.datasets[role]
        if role is not DataRole.DISCOVERY and set(self.laws) != set(self.tasks):
            raise PermissionError("Held-out contexts are measured only after every declared task is frozen")
        if not self.declaration.contexts_for(role):
            raise ValueError(f"No declared contexts for role {role.value}")
        self._prepare_masks(role)
        dataset = PairDataset(tuple(self.runner.run(request) for request in self._requests(role)))
        self.datasets[role] = dataset
        self._log("measured", role, hashlib.sha256(dumps_dataset(dataset).encode()).hexdigest())
        if role is DataRole.DISCOVERY:
            self.determinism = self._check_determinism()
            self._log("checked", role, fingerprint(self.determinism))
        return dataset

    def _check_determinism(self) -> dict:
        """Hard control: the reference branch of a context, repeated with its own seed, is bit-identical."""
        declaration, subject = self.declaration, self.subject
        largest, identical = 0.0, True
        for context in declaration.contexts_for(DataRole.DISCOVERY):
            seed = context_seeds(declaration, context.context_id)[0]
            runs = []
            for _ in range(declaration.repeats):
                root = subject.open(context, seed)
                branch = subject.fork(root)
                try:
                    state = subject.apply(branch, declaration.reference, seed)
                    runs.append(np.asarray(subject.observe(state), dtype=np.float64))
                finally:
                    subject.close(branch)
                    subject.close(root)
            for other in runs[1:]:
                identical &= bool(np.array_equal(runs[0], other))
                largest = max(largest, float(np.max(np.abs(runs[0] - other))))
        return {"identical": identical, "max_abs_difference": largest, "repeats": declaration.repeats,
                "contexts": len(declaration.contexts_for(DataRole.DISCOVERY))}

    def responses(self):
        """Labeled discovery responses for fitting forward and backward methods."""
        if DataRole.DISCOVERY not in self.datasets:
            raise PermissionError("Measure discovery contexts before fitting")
        return response_set(self.datasets[DataRole.DISCOVERY])

    def freeze(self, law: object, task: Task | str = Task.FORWARD) -> str:
        task = Task(task)
        if task not in self.tasks:
            raise PermissionError(f"Task {task.value} was not declared")
        if task in self.laws:
            raise PermissionError(f"A {task.value} law is already frozen; refitting after freeze is not permitted")
        if DataRole.DISCOVERY not in self.datasets:
            raise PermissionError("Measure discovery contexts before freezing a law")
        spec = law.describe()
        if spec.world.digest != self.declaration.world.digest:
            raise ValueError("The law belongs to a different world revision")
        discovery = {context.digest for context in self.declaration.contexts_for(DataRole.DISCOVERY)}
        if not spec.fitted_on or not {context.digest for context in spec.fitted_on} <= discovery:
            raise PermissionError("A frozen law must record fitting contexts drawn from discovery only")
        self.laws[task] = law
        self._log("frozen", None, spec.digest, task)
        return spec.digest

    @staticmethod
    def _queries(dataset: PairDataset) -> tuple[PairDataset, list[int]]:
        _, queries = split_rows(dataset)
        index = [i for rows in queries.values() for i in rows]
        return PairDataset(tuple(dataset.results[i] for i in index)), index

    def _predict(self, task: Task, dataset: PairDataset) -> np.ndarray:
        law, spec = self.laws[task], self.laws[task].describe()
        view_map, world = views(dataset), self.declaration.world
        queries, _ = self._queries(dataset)
        output = []
        for row in queries.results:
            view = view_map[row.request.context.context_id]
            if task is Task.FORWARD:
                output.append(world.observation.validate(np.asarray(law.predict(view, row.request.intervention))))
            else:
                values = tuple(float(value) for value in law.recover(view, row.mean_effect))
                world.intervention_space.validate(values)
                output.append(np.subtract(values, self.declaration.reference.values))
            if law.describe().digest != spec.digest:
                raise RuntimeError("A frozen law changed during prediction")
        return np.stack(output)

    def _controls(self, task: Task, queries: PairDataset) -> dict[str, np.ndarray]:
        discovery, _ = self._queries(self.datasets[DataRole.DISCOVERY])
        if task is Task.FORWARD:
            by_action: dict[tuple, list[np.ndarray]] = {}
            for row in discovery.results:
                by_action.setdefault(row.request.intervention.values, []).append(row.mean_effect)
            means = {key: np.mean(values, axis=0) for key, values in by_action.items()}
            return {"zero-response": np.zeros_like(queries.mean_effects),
                    "discovery-mean-response": np.stack([means[row.request.intervention.values]
                                                         for row in queries.results])}
        mean = discovery.intervention_deltas.mean(axis=0)
        return {"reference-value": np.zeros_like(queries.intervention_deltas),
                "discovery-mean-value": np.repeat(mean[None, :], len(queries.results), axis=0)}

    def _plan(self, task: Task, role: DataRole, suffix: str = "") -> EvaluationPlan:
        declaration = self.declaration
        return EvaluationPlan(f"{declaration.experiment_id}/{task.value}/{role.value}{suffix}", TARGETS[task], role,
                              declaration.minimum_improvement, declaration.equivalence_margin,
                              declaration.confidence, declaration.resamples, declaration.seed)

    def test(self) -> dict[Task, dict[DataRole, EvaluationReport]]:
        if set(self.laws) != set(self.tasks):
            raise PermissionError("Freeze every declared task before testing held-out stages")
        discovery, _ = self._queries(self.datasets[DataRole.DISCOVERY])
        for role in self.declaration.held_out_roles:
            dataset = self.measure(role)
            queries, _ = self._queries(dataset)
            for task in self.tasks:
                if role in self.evaluations[task]:
                    continue
                predictions, controls = self._predict(task, dataset), self._controls(task, queries)
                report = evaluate_predictions(queries, predictions, controls, self._plan(task, role), fitted_on=discovery)
                self.evaluations[task][role] = report
                self._log("tested", role, fingerprint(_plain(asdict(report))), task)
                for dimension in self.declaration.conditions:
                    for value in dimension.values:
                        index = [i for i, row in enumerate(queries.results)
                                 if dimension.value(row.request.context.context_id) == value]
                        if not index:
                            continue
                        sub = evaluate_predictions(
                            PairDataset(tuple(queries.results[i] for i in index)), predictions[index],
                            {name: values[index] for name, values in controls.items()},
                            self._plan(task, role, f"/{dimension.name}={value}"), fitted_on=discovery,
                        )
                        self.regimes[task].setdefault(dimension.name, {}).setdefault(value, {})[role.value] = {
                            comparison.comparator: comparison.status.value for comparison in sub.comparisons}
        return self.evaluations

    def compare_reference(self, reference: object) -> dict:
        """Second-layer comparison of measured responses with a reference world's forward law."""
        if not any(self.evaluations.values()):
            raise PermissionError("Compare with a reference only after held-out testing")
        tolerance = self.declaration.reference_tolerance
        if tolerance is None:
            raise ValueError("Declare a reference world and tolerance before comparing")
        errors = {}
        for role in self.declaration.held_out_roles:
            dataset = self.datasets[role]
            queries, _ = self._queries(dataset)
            view_map = views(dataset)
            predicted = np.stack([np.asarray(reference.predict(view_map[row.request.context.context_id],
                                                               row.request.intervention), dtype=float)
                                  for row in queries.results])
            scale = float(np.sqrt(np.mean(queries.mean_effects ** 2)))
            error = float(np.sqrt(np.mean((predicted - queries.mean_effects) ** 2)))
            errors[role.value] = {"relative_rms_error": error / scale if scale else None, "absolute_rms_error": error}
        defined = [row["relative_rms_error"] for row in errors.values() if row["relative_rms_error"] is not None]
        status = "not_compared" if not defined else ("agrees" if max(defined) <= tolerance else "deviates")
        self.reference = {"world": self.declaration.reference_world, "law": reference.describe().law_id,
                          "tolerance": tolerance, "stages": errors, "status": status}
        self._log("referenced", None, fingerprint(self.reference))
        return self.reference

    def report(self) -> dict:
        roles = set(self.declaration.held_out_roles)
        if not self.laws or any(set(self.evaluations[task]) != roles for task in self.tasks):
            raise PermissionError("Test every declared task on every held-out stage before reporting")
        return build_report(self)


def _analysis(experiment: Experiment, task: Task, undefined: dict[str, str]) -> dict:
    spec = experiment.laws[task].describe()
    evaluations, stage_status = {}, {}
    for role, report in experiment.evaluations[task].items():
        comparisons = {}
        for comparison in report.comparisons:
            comparisons[comparison.comparator] = {
                "advantage": comparison.mean_error_advantage,
                "interval": list(comparison.interval) if comparison.interval else None,
                "status": comparison.status.value, "groups": comparison.groups, "confidence": comparison.confidence,
            }
            if comparison.interval is None:
                undefined[f"analyses.{task.value}.evaluations.{role.value}.controls.{comparison.comparator}.interval"] = \
                    "fewer than two exchangeability groups"
        evaluations[role.value] = {"group_weighted_mse": report.group_weighted_mse,
                                   "descriptive_r2": report.descriptive_r2, "controls": comparisons,
                                   "queries": len(report.unit_ids)}
        if report.descriptive_r2 is None:
            undefined[f"analyses.{task.value}.evaluations.{role.value}.descriptive_r2"] = "zero total target variance"
        stage_status[role.value] = {name: row["status"] for name, row in comparisons.items()}
    regimes = {name: {value: {"stages": stages, "status": regime_verdict(
        [status for controls in stages.values() for status in controls.values()])}
        for value, stages in values.items()} for name, values in experiment.regimes[task].items()}
    regime_status = {name: {value: row["status"] for value, row in values.items()} for name, values in regimes.items()}
    outcome, holds = adjudicate(stage_status, regime_status)
    return {
        "candidate": {"law_id": spec.law_id, "revision": spec.revision, "digest": spec.digest,
                      "hypothesis": spec.hypothesis, "validity_domain": spec.validity_domain,
                      "fitted_on": sorted(context.context_id for context in spec.fitted_on)},
        "controls": CONTROLS[task],
        "evaluations": evaluations,
        "regimes": regimes,
        "outcome": {"value": outcome.value, "meaning": OUTCOME_MEANING[outcome],
                    "holds_in": {key: list(value) for key, value in holds.items()}},
        "claims": [_plain(asdict(_claim(experiment, task, outcome, holds)))],
    }


def _response(experiment: Experiment, undefined: dict[str, str]) -> dict:
    declaration = experiment.declaration
    rms, spread, ratio = {}, {}, {}
    for role in declaration.held_out_roles:
        queries, _ = Experiment._queries(experiment.datasets[role])
        key = role.value
        rms[key] = float(np.sqrt(np.mean(queries.mean_effects ** 2)))
        effects = np.stack([row.intervention - row.baseline for row in queries.results])     # (pairs, draws, ...)
        if effects.shape[1] < 2:
            spread[key] = ratio[key] = None
            undefined[f"response.draw_spread_rms.{key}"] = "a single context draw has no spread"
            undefined[f"response.response_to_draw_spread.{key}"] = "a single context draw has no spread"
            continue
        sem = effects.var(axis=1, ddof=1) / effects.shape[1]
        spread[key] = float(np.sqrt(np.mean(sem)))
        ratio[key] = rms[key] / spread[key] if spread[key] > 0 else None
        if ratio[key] is None:
            undefined[f"response.response_to_draw_spread.{key}"] = "zero draw spread (identical draws)"
    return {"method": RESPONSE_METHOD, "response_rms": rms, "draw_spread_rms": spread,
            "response_to_draw_spread": ratio, "determinism": experiment.determinism}


def build_report(experiment: Experiment) -> dict:
    declaration, undefined = experiment.declaration, {}
    response = _response(experiment, undefined)
    masking = None
    if experiment.masking is not None:
        masking = {"rule": experiment.scope.rule, "frozen": experiment.masking["frozen"],
                   "instances": dict(sorted(experiment.masking["instances"].items())),
                   "coverage": experiment.masking["coverage"],
                   "sweep": {"values": [list(action.values) for action in declaration.sweep],
                             "draws": declaration.sweep_draws}}
    record = {
        "schema": REPORT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "implementation": {"name": "multiecho-framework", "version": __version__},
        "declaration": {**declaration.as_record(), "digest": declaration.digest},
        "stages": [{"index": index, **event} for index, event in enumerate(experiment.stages)],
        "response": response,
        "masking": masking,
        "keyframes": experiment.keyframe_section(),
        "analyses": {task.value: _analysis(experiment, task, undefined) for task in experiment.tasks},
        "reference": experiment.reference,
        "undefined": undefined,
        "artifacts": {event["role"]: event["digest"] for event in experiment.stages if event["stage"] == "measured"},
    }
    return seal(record)


def _claim(experiment: Experiment, task: Task, outcome: Outcome, holds: dict[str, tuple[str, ...]]) -> Claim:
    declaration, spec = experiment.declaration, experiment.laws[task].describe()
    world, space, chain = declaration.world, declaration.world.intervention_space, declaration.chain
    positive = outcome in (Outcome.STABLE_LAW, Outcome.REGIME_SPECIFIC)
    # Positive claims cover only condition values that were individually supported.
    regime = ({name: list(values) for name, values in holds.items()} if positive else
              {dimension.name: list(dimension.values) for dimension in declaration.conditions})
    domain = spec.validity_domain if positive else "declared scope"
    if regime:
        domain += "; conditions " + "; ".join(f"{name} in {{{', '.join(values)}}}" for name, values in regime.items())
    verb = "predicts the response of" if task is Task.FORWARD else "recovers the intervention from the response of"
    statement = {
        Outcome.STABLE_LAW: f"{spec.law_id} {verb} {world.world_id} to {space.space_id}",
        Outcome.REGIME_SPECIFIC: f"{spec.law_id} {verb} {world.world_id} to {space.space_id} only in the listed conditions",
        Outcome.UNIDENTIFIED: f"Nothing is identified for {world.world_id} under {space.space_id} by {spec.law_id} ({task.value}) in this scope",
    }[outcome]
    return Claim(
        f"{declaration.experiment_id}/{task.value}/{outcome.value}", task.value, outcome, statement,
        {"world_id": world.world_id, "revision": world.revision, "access": world.access.value,
         "chain": [f"{card['kind']}:{card['world_id']}" for card in chain["worlds"]]},
        {"space_id": world.observation.space_id, "semantics": world.observation.semantics,
         "unit": world.observation.unit, "time": declaration.observation_time, "scope": chain["observer"]["scope"]},
        declaration.handle,
        {"space_id": space.space_id, "axes": [axis.name for axis in space.axes],
         "reference": list(declaration.reference.values), "queried": [list(a.values) for a in declaration.interventions],
         "calibration": [list(a.values) for a in declaration.calibration], "steps": declaration.reference.steps},
        {role.value: len(declaration.contexts_for(role)) for role in (DataRole.DISCOVERY, *declaration.held_out_roles)}
        | {"independence_unit": declaration.independence_unit},
        regime,
        {"candidate": spec.law_id, "revision": spec.revision, "controls": sorted(CONTROLS[task]),
         "response": RESPONSE_METHOD},
        domain,
        tuple(role.value for role in declaration.held_out_roles),
    )


def seal(record: dict) -> dict:
    body = {key: value for key, value in record.items() if key != "seal"}
    return {**body, "seal": fingerprint(body)}


def dumps_report(report: dict) -> str:
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"


def loads_report(text: str) -> dict:
    report = json.loads(text)
    if not isinstance(report, dict) or report.get("seal") != fingerprint({k: v for k, v in report.items() if k != "seal"}):
        raise ValueError("Report seal does not match its content")
    return report
