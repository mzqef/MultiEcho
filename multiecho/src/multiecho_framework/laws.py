"""Frozen candidate tests on independent trajectories and unobserved query proposals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Protocol

import numpy as np

from .contracts import Context, DataRole, Intervention, WorldSpec, fingerprint
from .datasets import PairDataset, validate_training_roles
from .evaluation import EvaluationPlan, EvidenceStatus, grouped_values
from .execution import PairResult, TraceResult


@dataclass(frozen=True)
class LawSpec:
    law_id: str
    revision: str
    world: WorldSpec
    hypothesis: str
    validity_domain: str
    fitted_on: tuple[Context, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.law_id, self.revision, self.hypothesis, self.validity_domain)):
            raise ValueError("A law needs an identity, revision, hypothesis, and validity domain")
        if not isinstance(self.fitted_on, tuple) or any(
            context.role not in {DataRole.TRAIN, DataRole.DISCOVERY, DataRole.CALIBRATION}
            for context in self.fitted_on
        ):
            raise ValueError("Frozen laws cannot be fitted on validation or confirmation contexts")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class LawQuery:
    context: Context
    baseline: Intervention
    intervention: Intervention

    @classmethod
    def from_result(cls, result: PairResult) -> LawQuery:
        return cls(result.request.context, result.request.baseline, result.request.intervention)


class ResponseLaw(Protocol):
    def describe(self) -> LawSpec: ...

    def in_domain(self, query: LawQuery) -> bool: ...

    def predict(self, query: LawQuery, baseline_mean: np.ndarray) -> np.ndarray: ...


def require_unused_contexts(spec: LawSpec, contexts: tuple[Context, ...]) -> None:
    for field in ("context_id", "payload_digest", "exchangeability_group"):
        used = {getattr(context, field) for context in spec.fitted_on}
        if used & {getattr(context, field) for context in contexts}:
            raise ValueError(f"Candidate fitting/evaluation leakage in {field}")


def predict_responses(candidate: ResponseLaw, dataset: PairDataset) -> np.ndarray:
    spec = candidate.describe()
    if spec.world.digest != dataset.world.digest:
        raise ValueError("Candidate world or observation identity mismatch")
    contexts = tuple(row.request.context for row in dataset.results)
    require_unused_contexts(spec, contexts)
    predictions = []
    for row in dataset.results:
        query = LawQuery.from_result(row)
        if not candidate.in_domain(query):
            raise ValueError(f"Query outside candidate validity domain: {row.request.pair_id}")
        baseline = spec.world.observation.validate(row.baseline.mean(axis=0))
        predicted = candidate.predict(query, baseline)
        predictions.append(spec.world.observation.validate(predicted))
        if candidate.describe().digest != spec.digest:
            raise RuntimeError("Candidate changed during frozen response prediction")
    return np.stack(predictions)


class TrajectoryLaw(Protocol):
    def describe(self) -> LawSpec: ...

    def in_domain(self, context: Context, actions: tuple[Intervention, ...]) -> bool: ...

    def predict(self, initial: np.ndarray, actions: tuple[Intervention, ...], context: Context) -> np.ndarray: ...


class ResidualLaw(Protocol):
    def describe(self) -> LawSpec: ...

    def in_domain(self, trace: TraceResult) -> bool: ...

    def residual(self, trace: TraceResult) -> np.ndarray: ...


@dataclass(frozen=True)
class LawTestPlan:
    evaluation: EvaluationPlan
    tolerance: float
    residual_semantics: str
    residual_unit: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("A positive absolute residual tolerance is required")
        if not self.residual_semantics or not self.residual_unit:
            raise ValueError("Residual meaning and units must be declared")


@dataclass(frozen=True)
class LawTestReport:
    law_digest: str
    world_digest: str
    evaluation_id: str
    role: DataRole
    residual_semantics: str
    residual_unit: str
    tolerance: float
    mean_group_residual: float
    interval: tuple[float, float] | None
    status: EvidenceStatus
    groups: int
    trace_ids: tuple[str, ...]
    per_trace_residual: tuple[float, ...]
    coverage: str
    confidence: float
    claim_boundary: str = "Declared candidate, finite access/domain, and group-mean tolerance only; no reference-physics or universal-law claim"


def _check_traces(spec: LawSpec, traces: tuple[TraceResult, ...], plan: LawTestPlan) -> None:
    if not traces or len({trace.request.trace_id for trace in traces}) != len(traces):
        raise ValueError("Provide nonempty traces with unique execution identities")
    if any(trace.world.digest != spec.world.digest for trace in traces):
        raise ValueError("Law and trace world/observation identities must match")
    if any(trace.request.context.role is not plan.evaluation.role for trace in traces):
        raise ValueError("Law test role does not match trajectory roles")
    require_unused_contexts(spec, tuple(trace.request.context for trace in traces))
    contexts: dict[str, str] = {}
    for trace in traces:
        context = trace.request.context
        if contexts.setdefault(context.context_id, context.digest) != context.digest:
            raise ValueError("A law-test context identity changed")


def _residual_report(
    spec: LawSpec, traces: tuple[TraceResult, ...], residuals: list[float], plan: LawTestPlan,
    coverage: str, family_size: int = 1,
) -> LawTestReport:
    groups = tuple(trace.request.context.exchangeability_group for trace in traces)
    values = grouped_values(np.asarray(residuals), groups)
    confidence = 1 - (1 - plan.evaluation.confidence) / family_size
    status = EvidenceStatus.UNRESOLVED
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(plan.evaluation.seed)
        draws = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(plan.evaluation.resamples)]
        tail = (1 - confidence) / 2
        lower, upper = (float(value) for value in np.quantile(draws, [tail, 1 - tail]))
        interval = (lower, upper)
        if upper < plan.tolerance:
            status = EvidenceStatus.SUPPORTED
        elif lower > plan.tolerance:
            status = EvidenceStatus.CONTRADICTED
    return LawTestReport(
        spec.digest, spec.world.digest, plan.evaluation.evaluation_id, plan.evaluation.role,
        plan.residual_semantics, plan.residual_unit, plan.tolerance, float(values.mean()),
        interval, status, len(values), tuple(trace.request.trace_id for trace in traces),
        tuple(residuals), coverage, confidence,
    )


def test_residual_law(
    candidate: ResidualLaw, traces: tuple[TraceResult, ...], plan: LawTestPlan,
) -> LawTestReport:
    spec = candidate.describe()
    _check_traces(spec, traces, plan)
    residuals = []
    for trace in traces:
        if not candidate.in_domain(trace):
            raise ValueError(f"Trace outside residual-law domain: {trace.request.trace_id}")
        values = np.asarray(candidate.residual(trace), dtype=float)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Law residuals must be finite and nonempty")
        residuals.append(float(np.max(np.abs(values))))
        if candidate.describe().digest != spec.digest:
            raise RuntimeError("Candidate changed during residual-law testing")
    return _residual_report(spec, traces, residuals, plan, "Per-trace maximum absolute declared residual")


def test_trajectory_law(
    candidate: TrajectoryLaw, traces: tuple[TraceResult, ...], plan: LawTestPlan,
) -> tuple[LawTestReport, ...]:
    spec = candidate.describe()
    _check_traces(spec, traces, plan)
    horizons = {len(trace.request.interventions) for trace in traces}
    if len(horizons) != 1:
        raise ValueError("A trajectory-law test needs a common declared horizon")
    horizon = next(iter(horizons))
    residuals = [[] for _ in range(horizon)]
    for trace in traces:
        if not candidate.in_domain(trace.request.context, trace.request.interventions):
            raise ValueError(f"Trace outside trajectory-law domain: {trace.request.trace_id}")
        initial = spec.world.observation.validate(trace.observations[0])
        predicted = np.asarray(candidate.predict(initial, trace.request.interventions, trace.request.context), dtype=float)
        truth = trace.observations[1:]
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError("Trajectory predictions must match all future observations")
        errors = np.sqrt(np.mean((truth - predicted) ** 2, axis=tuple(range(1, truth.ndim))))
        for values, error in zip(residuals, errors, strict=True):
            values.append(float(error))
        if candidate.describe().digest != spec.digest:
            raise RuntimeError("Candidate changed during trajectory prediction")
    return tuple(_residual_report(spec, traces, values, plan, f"Observed rollout RMSE at action boundary {index + 1}", horizon)
                 for index, values in enumerate(residuals))


def test_regimes(
    candidate: ResidualLaw, regimes: dict[str, tuple[TraceResult, ...]], plan: LawTestPlan,
) -> dict[str, LawTestReport]:
    if not regimes or any(not name or not traces for name, traces in regimes.items()):
        raise ValueError("Named, nonempty, predeclared regimes are required")
    confidence = 1 - (1 - plan.evaluation.confidence) / len(regimes)
    adjusted = replace(plan, evaluation=replace(plan.evaluation, confidence=confidence))
    return {name: test_residual_law(candidate, traces, adjusted) for name, traces in regimes.items()}


@dataclass(frozen=True)
class PathPair:
    pair_id: str
    left: TraceResult
    right: TraceResult
    equivalence_claim: str
    randomness_statement: str

    def __post_init__(self) -> None:
        if not self.pair_id or not self.equivalence_claim or not self.randomness_statement:
            raise ValueError("Path pairs need an identity, a declared equivalence, and a random-coupling statement")
        if self.left.request.trace_id == self.right.request.trace_id:
            raise ValueError("Path comparison requires separately executed traces")
        if self.left.world.digest != self.right.world.digest:
            raise ValueError("Equivalent paths must belong to one frozen world")
        if self.left.request.context.digest != self.right.request.context.digest:
            raise ValueError("Equivalent paths must start from the same context identity")
        if self.left.request.context_seed != self.right.request.context_seed:
            raise ValueError("Equivalent paths must use the same initial-context seed")
        if sum(action.steps for action in self.left.request.interventions) != sum(action.steps for action in self.right.request.interventions):
            raise ValueError("Equivalent paths must end at the same declared total horizon")
        if not np.array_equal(self.left.observations[0], self.right.observations[0]):
            raise ValueError("Equivalent paths must have identical initial observations")


class PathLaw(Protocol):
    def describe(self) -> LawSpec: ...

    def in_domain(self, pair: PathPair) -> bool: ...

    def endpoint(self, observation: np.ndarray, context: Context) -> np.ndarray: ...


def test_path_law(
    candidate: PathLaw, pairs: tuple[PathPair, ...], plan: LawTestPlan,
) -> LawTestReport:
    if not pairs or len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("Path-law testing requires unique, nonempty path-pair identities")
    traces = tuple(trace for pair in pairs for trace in (pair.left, pair.right))
    spec = candidate.describe()
    _check_traces(spec, traces, plan)
    residuals = []
    for pair in pairs:
        if not candidate.in_domain(pair):
            raise ValueError(f"Path pair outside the declared equivalence domain: {pair.pair_id}")
        endpoints = [np.asarray(candidate.endpoint(trace.observations[-1].copy(), trace.request.context), dtype=float)
                     for trace in (pair.left, pair.right)]
        if not endpoints[0].size or endpoints[0].shape != endpoints[1].shape or not all(np.isfinite(value).all() for value in endpoints):
            raise ValueError("Path-law endpoint observations must have matching finite shapes")
        residuals.append(float(np.sqrt(np.mean((endpoints[0] - endpoints[1]) ** 2))))
        if candidate.describe().digest != spec.digest:
            raise RuntimeError("Candidate changed during independent path testing")
    report = _residual_report(spec, tuple(pair.left for pair in pairs), residuals, plan,
                              "RMSE between independently executed path endpoints after the frozen observation map")
    return replace(report, trace_ids=tuple(trace.request.trace_id for trace in traces))


@dataclass(frozen=True)
class ProposedQuery:
    proposal_id: str
    world_digest: str
    context_digest: str
    baseline: Intervention
    intervention: Intervention
    candidate_digests: tuple[str, ...]
    disagreement_rms: float
    selection_role: DataRole
    rationale: str = "Largest declared candidate disagreement among supplied unobserved interventions; not confirmation evidence"


def propose_disagreement(
    candidates: tuple[ResponseLaw, ...], world: WorldSpec, context: Context,
    baseline: Intervention, baseline_observation: np.ndarray, interventions: tuple[Intervention, ...],
) -> ProposedQuery:
    if context.role is not DataRole.DISCOVERY or len(candidates) < 2 or not interventions:
        raise ValueError("Active discovery needs at least two candidates and unobserved choices on discovery contexts")
    specs = tuple(candidate.describe() for candidate in candidates)
    if any(spec.world.digest != world.digest for spec in specs) or len({spec.digest for spec in specs}) != len(specs):
        raise ValueError("Distinct frozen candidates must share the query world")
    world.intervention_space.validate(baseline.values)
    source = world.observation.validate(baseline_observation)
    scored = []
    for action in interventions:
        world.intervention_space.validate(action.values)
        if action.steps != baseline.steps:
            raise ValueError("Active query branches must share the observation horizon")
        query = LawQuery(context, baseline, action)
        if not all(candidate.in_domain(query) for candidate in candidates):
            raise ValueError("Proposed query falls outside a candidate domain")
        predicted = np.stack([world.observation.validate(candidate.predict(query, source.copy())) for candidate in candidates])
        if tuple(candidate.describe().digest for candidate in candidates) != tuple(spec.digest for spec in specs):
            raise RuntimeError("Candidate changed during active query selection")
        scored.append(float(np.sqrt(np.mean((predicted - predicted.mean(axis=0)) ** 2))))
    index = int(np.argmax(scored))
    digests = tuple(spec.digest for spec in specs)
    identity = fingerprint({"world": world.digest, "context": context.digest, "baseline": asdict(baseline),
                            "interventions": [asdict(action) for action in interventions], "candidates": digests,
                            "selected": index, "disagreement": scored[index]})
    return ProposedQuery(identity, world.digest, context.digest, baseline, interventions[index], digests, scored[index], context.role)


class LawFitter(Protocol):
    def fit(self, dataset: PairDataset) -> ResponseLaw: ...


def fit_response_law(fitter: LawFitter, dataset: PairDataset) -> ResponseLaw:
    validate_training_roles(dataset, allow_discovery=True)
    candidate = fitter.fit(dataset)
    spec = candidate.describe()
    if spec.world.digest != dataset.world.digest:
        raise ValueError("Fitted law must retain the training world identity")
    required = {row.request.context.digest for row in dataset.results}
    if not required <= {context.digest for context in spec.fitted_on}:
        raise ValueError("Fitted law must record every fitting context")
    return candidate