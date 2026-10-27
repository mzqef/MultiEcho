"""Target-specific metrics with exchangeability-group uncertainty, never a world score."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .contracts import DataRole, SeedMode
from .datasets import PairDataset, require_disjoint, validate_training_roles


class EvaluationTarget(StrEnum):
    MEAN_RESPONSE = "mean_response"
    INTERVENTION_DELTA = "intervention_delta"
    BRANCH_DISTRIBUTIONS = "branch_distributions"
    RECEIVER_OBSERVATION = "receiver_observation"
    MESSAGE_AGREEMENT = "message_agreement"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    EQUIVALENT = "equivalent"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class EvaluationPlan:
    evaluation_id: str
    target: EvaluationTarget
    role: DataRole
    minimum_improvement: float
    equivalence_margin: float | None = None
    confidence: float = 0.95
    resamples: int = 1999
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.evaluation_id or not isinstance(self.target, EvaluationTarget):
            raise ValueError("An evaluation identity and target are required")
        if not isinstance(self.role, DataRole) or self.role is DataRole.TRAIN:
            raise ValueError("Evaluation needs a non-training data role")
        if not np.isfinite(self.minimum_improvement) or self.minimum_improvement < 0:
            raise ValueError("Minimum improvement must be finite and nonnegative")
        if self.equivalence_margin is not None and (
            not np.isfinite(self.equivalence_margin) or self.equivalence_margin <= 0
        ):
            raise ValueError("Equivalence requires a positive declared margin")
        if (not 0 < self.confidence < 1 or type(self.resamples) is not int or self.resamples < 99
            or type(self.seed) is not int or self.seed < 0):
            raise ValueError("Invalid confidence, resampling budget, or seed")


@dataclass(frozen=True)
class Comparison:
    comparator: str
    mean_error_advantage: float
    interval: tuple[float, float] | None
    status: EvidenceStatus
    groups: int
    confidence: float


@dataclass(frozen=True)
class EvaluationReport:
    evaluation_id: str
    world_digest: str
    observation_digest: str
    target: str
    role: DataRole
    group_weighted_mse: float
    descriptive_r2: float | None
    comparisons: tuple[Comparison, ...]
    unit_ids: tuple[str, ...]
    uncertainty_method: str = "equal-group bootstrap; Bonferroni across named comparators"
    claim_boundary: str = "Frozen sensor and declared target only; not native causal use or absent information"


def grouped_values(values: np.ndarray, groups: tuple[str, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) != len(groups) or not len(array) or not np.isfinite(array).all():
        raise ValueError("Finite scalar observations and matching group IDs are required")
    if any(not group for group in groups):
        raise ValueError("Every observation requires an exchangeability group")
    return np.array([array[np.array(groups) == group].mean() for group in sorted(set(groups))])


def summarize_advantage(
    values: np.ndarray,
    groups: tuple[str, ...],
    plan: EvaluationPlan,
    name: str,
    family_size: int = 1,
) -> Comparison:
    if not name or family_size < 1:
        raise ValueError("Comparison name and positive family size are required")
    estimates = grouped_values(values, groups)
    confidence = 1 - (1 - plan.confidence) / family_size
    if len(estimates) < 2:
        return Comparison(name, float(estimates.mean()), None, EvidenceStatus.UNRESOLVED, len(estimates), confidence)
    rng = np.random.default_rng(plan.seed)
    bootstrap = np.array([
        rng.choice(estimates, size=len(estimates), replace=True).mean()
        for _ in range(plan.resamples)
    ])
    tail = (1 - confidence) / 2
    lower, upper = (float(value) for value in np.quantile(bootstrap, [tail, 1 - tail]))
    margin = plan.equivalence_margin
    status = EvidenceStatus.UNRESOLVED
    if margin is not None and -margin < lower <= upper < margin:
        status = EvidenceStatus.EQUIVALENT
    elif lower > plan.minimum_improvement:
        status = EvidenceStatus.SUPPORTED
    elif upper < -plan.minimum_improvement:
        status = EvidenceStatus.CONTRADICTED
    return Comparison(name, float(estimates.mean()), (lower, upper), status, len(estimates), confidence)


def evaluate_predictions(
    dataset: PairDataset,
    predictions: np.ndarray,
    controls: dict[str, np.ndarray],
    plan: EvaluationPlan,
    *,
    fitted_on: PairDataset | None = None,
) -> EvaluationReport:
    if plan.target not in (EvaluationTarget.MEAN_RESPONSE, EvaluationTarget.INTERVENTION_DELTA):
        raise ValueError("Point evaluation requires a mean-response or intervention-delta target")
    if dataset.roles != {plan.role}:
        raise ValueError("Dataset roles do not match the frozen evaluation plan")
    if fitted_on is not None:
        validate_training_roles(fitted_on, allow_discovery=True)
        require_disjoint(fitted_on, dataset)
    truth = dataset.mean_effects if plan.target is EvaluationTarget.MEAN_RESPONSE else dataset.intervention_deltas
    predicted = np.asarray(predictions, dtype=float)
    if predicted.shape != truth.shape or not np.isfinite(predicted).all():
        raise ValueError("Predictions must match the target shape and contain finite values")
    if not controls or any(not name for name in controls):
        raise ValueError("At least one named, task-matched comparator is required")
    reduce_axes = tuple(range(1, truth.ndim))
    errors = np.mean((truth - predicted) ** 2, axis=reduce_axes)
    comparisons = []
    for name, values in sorted(controls.items()):
        values = np.asarray(values, dtype=float)
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(f"Comparator target mismatch: {name}")
        control_errors = np.mean((truth - values) ** 2, axis=reduce_axes)
        comparisons.append(summarize_advantage(control_errors - errors, dataset.groups, plan, name, len(controls)))
    residual = float(np.sum((truth - predicted) ** 2))
    total = float(np.sum((truth - truth.mean(axis=0)) ** 2))
    return EvaluationReport(
        plan.evaluation_id, dataset.world.digest, dataset.world.observation.digest,
        plan.target.value, plan.role, float(grouped_values(errors, dataset.groups).mean()),
        1 - residual / total if total > 0 else None, tuple(comparisons),
        tuple(row.request.pair_id for row in dataset.results),
    )


def energy_score(samples: np.ndarray, outcomes: np.ndarray) -> float:
    """Empirical proper energy score; coordinates and units must already be aligned."""
    draws = np.asarray(samples, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if draws.ndim < 2 or observed.ndim != draws.ndim or draws.shape[1:] != observed.shape[1:]:
        raise ValueError("Sample and outcome distributions must share observation shape")
    if not len(draws) or not len(observed) or not np.isfinite(draws).all() or not np.isfinite(observed).all():
        raise ValueError("Finite nonempty sample distributions are required")
    draws = draws.reshape(len(draws), -1)
    observed = observed.reshape(len(observed), -1)
    cross = sum(np.linalg.norm(observed - draw, axis=1).mean() for draw in draws) / len(draws)
    within = sum(np.linalg.norm(draws - draw, axis=1).mean() for draw in draws) / len(draws)
    return float(cross - within / 2)


@dataclass(frozen=True)
class BranchPrediction:
    baseline: tuple[np.ndarray, ...]
    intervention: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, tuple) or not isinstance(self.intervention, tuple):
            raise ValueError("Distribution predictions must be immutable tuples of sample arrays")
        if not self.baseline or len(self.baseline) != len(self.intervention):
            raise ValueError("Both predicted branches must cover the same nonempty set of pairs")
        for field in ("baseline", "intervention"):
            frozen = []
            for value in getattr(self, field):
                array = np.array(value, dtype=float, copy=True)
                if array.ndim < 2 or not len(array) or not np.isfinite(array).all():
                    raise ValueError("Each prediction needs nonempty finite draws and observation dimensions")
                array.setflags(write=False)
                frozen.append(array)
            object.__setattr__(self, field, tuple(frozen))


@dataclass(frozen=True)
class DistributionReport:
    evaluation_id: str
    world_digest: str
    observation_digest: str
    role: DataRole
    group_weighted_energy_score: float
    comparisons: tuple[Comparison, ...]
    unit_ids: tuple[str, ...]
    claim_boundary: str = "Empirical branch distributions in declared coordinates; no canonical pathwise coupling or universal law claim"


def evaluate_distributions(
    dataset: PairDataset, predictions: BranchPrediction, controls: dict[str, BranchPrediction],
    plan: EvaluationPlan, *, fitted_on: PairDataset | None = None,
) -> DistributionReport:
    if plan.target is not EvaluationTarget.BRANCH_DISTRIBUTIONS or dataset.roles != {plan.role}:
        raise ValueError("Distribution evaluation requires matching target and dataset roles")
    if any(row.request.mode is SeedMode.FIXED for row in dataset.results):
        raise ValueError("One fixed realization cannot provide a population distribution test")
    if not controls or any(not name for name in controls):
        raise ValueError("Distribution evaluation requires named task-matched controls")
    if fitted_on is not None:
        validate_training_roles(fitted_on, allow_discovery=True)
        require_disjoint(fitted_on, dataset)

    def errors(predicted: BranchPrediction) -> np.ndarray:
        if len(predicted.baseline) != len(dataset.results):
            raise ValueError("Predicted distributions do not cover the dataset")
        return np.array([
            (energy_score(baseline, row.baseline) + energy_score(intervention, row.intervention)) / 2
            for row, baseline, intervention in zip(
                dataset.results, predicted.baseline, predicted.intervention, strict=True,
            )
        ])

    predicted_errors = errors(predictions)
    comparisons = tuple(summarize_advantage(errors(control) - predicted_errors, dataset.groups, plan,
                                             name, len(controls)) for name, control in sorted(controls.items()))
    return DistributionReport(plan.evaluation_id, dataset.world.digest, dataset.world.observation.digest,
                               plan.role, float(grouped_values(predicted_errors, dataset.groups).mean()),
                               comparisons, tuple(row.request.pair_id for row in dataset.results))