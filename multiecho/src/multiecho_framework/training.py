"""Training-data exchange and reference losses, separate from frozen-world evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from .contracts import SeedMode, WorldSpec
from .datasets import PairDataset, require_disjoint, validate_training_roles
from .evaluation import energy_score, grouped_values


class TrainingTarget(StrEnum):
    MEAN_RESPONSE = "mean_response"
    PAIRED_RESPONSE = "paired_response"
    BRANCH_DISTRIBUTIONS = "branch_distributions"


@dataclass(frozen=True)
class TrainingBatch:
    world: WorldSpec
    pair_ids: tuple[str, ...]
    context_refs: tuple[str, ...]
    context_digests: tuple[str, ...]
    exchangeability_groups: tuple[str, ...]
    baseline_values: np.ndarray
    intervention_values: np.ndarray
    horizons: tuple[int, ...]
    seed_modes: tuple[SeedMode, ...]
    baseline_seeds: tuple[tuple[int, ...], ...]
    intervention_seeds: tuple[tuple[int, ...], ...]
    baseline_samples: tuple[np.ndarray, ...]
    intervention_samples: tuple[np.ndarray, ...]
    target: TrainingTarget
    responses: tuple[np.ndarray, ...] | None
    interpretation: str

    def __post_init__(self) -> None:
        size = len(self.pair_ids)
        if not size or len(set(self.pair_ids)) != size or not isinstance(self.target, TrainingTarget):
            raise ValueError("A training batch needs unique pairs and a typed target")
        for field in ("pair_ids", "context_refs", "context_digests", "exchangeability_groups", "horizons",
                      "seed_modes", "baseline_seeds", "intervention_seeds", "baseline_samples", "intervention_samples"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or len(values) != size:
                raise ValueError(f"Training batch field length mismatch: {field}")
        for field in ("baseline_values", "intervention_values"):
            array = np.array(getattr(self, field), dtype=float, copy=True)
            if array.shape != (size, len(self.world.intervention_space.axes)):
                raise ValueError("Training intervention coordinates do not match the world")
            for values in array:
                self.world.intervention_space.validate(tuple(float(value) for value in values))
            array.setflags(write=False)
            object.__setattr__(self, field, array)
        for field in ("baseline_samples", "intervention_samples", "responses"):
            arrays = getattr(self, field)
            if arrays is None:
                continue
            if not isinstance(arrays, tuple) or len(arrays) != size:
                raise ValueError("Training response counts must match pair identities")
            frozen = []
            for array in arrays:
                value = np.array(array, dtype=float, copy=True)
                if not np.isfinite(value).all() or not value.size:
                    raise ValueError("Training arrays must be finite and nonempty")
                value.setflags(write=False)
                frozen.append(value)
            object.__setattr__(self, field, tuple(frozen))
        for index, mode in enumerate(self.seed_modes):
            if not isinstance(mode, SeedMode) or type(self.horizons[index]) is not int or self.horizons[index] < 1:
                raise ValueError("Training draws and horizons must be explicitly typed")
            for field, seeds in (("baseline_samples", self.baseline_seeds[index]),
                                 ("intervention_samples", self.intervention_seeds[index])):
                samples = getattr(self, field)[index]
                if samples.shape != (len(seeds), *self.world.observation.shape):
                    raise ValueError("Training samples must match seeds and observation shape")
            if self.target is TrainingTarget.PAIRED_RESPONSE and mode is SeedMode.INDEPENDENT:
                raise ValueError("Independent draws cannot provide paired response targets")
        if (self.target is TrainingTarget.BRANCH_DISTRIBUTIONS) != (self.responses is None):
            raise ValueError("Distribution targets retain both branches instead of response differences")
        if self.responses is not None:
            for index, response in enumerate(self.responses):
                expected = self.world.observation.shape
                if self.target is TrainingTarget.PAIRED_RESPONSE:
                    expected = (len(self.baseline_seeds[index]), *expected)
                if response.shape != expected:
                    raise ValueError("Response target shape does not match its training objective")


def export_training(
    dataset: PairDataset, target: TrainingTarget, *, evaluation: PairDataset | None = None,
    allow_discovery: bool = False,
) -> TrainingBatch:
    validate_training_roles(dataset, allow_discovery=allow_discovery)
    if not isinstance(target, TrainingTarget):
        raise ValueError("Declare a TrainingTarget rather than inferring an objective")
    if evaluation is not None:
        require_disjoint(dataset, evaluation)
    if target is TrainingTarget.PAIRED_RESPONSE:
        responses = tuple(row.paired_effects for row in dataset.results)
        interpretation = "Seed-coupled pathwise responses; pairing is not a seed-invariant response law"
    elif target is TrainingTarget.MEAN_RESPONSE:
        responses = tuple(row.mean_effect for row in dataset.results)
        interpretation = "Empirical mean contrasts; cannot identify interventions that only change spread or other moments"
    else:
        if any(row.request.mode is SeedMode.FIXED for row in dataset.results):
            raise ValueError("Branch-distribution training requires multiple declared draws")
        responses = None
        interpretation = "Separate empirical branch distributions; does not assume a canonical sample-to-sample coupling"
    rows = dataset.results
    return TrainingBatch(
        dataset.world,
        tuple(row.request.pair_id for row in rows),
        tuple(row.request.context.payload_ref for row in rows),
        tuple(row.request.context.payload_digest for row in rows),
        dataset.groups,
        np.array([row.request.baseline.values for row in rows]),
        np.array([row.request.intervention.values for row in rows]),
        tuple(row.request.baseline.steps for row in rows),
        tuple(row.request.mode for row in rows),
        tuple(row.request.seeds for row in rows),
        tuple(row.request.independent_seeds if row.request.mode is SeedMode.INDEPENDENT else row.request.seeds for row in rows),
        tuple(row.baseline for row in rows), tuple(row.intervention for row in rows),
        target, responses, interpretation,
    )


def response_mse(
    baseline_prediction: np.ndarray, intervention_prediction: np.ndarray, response_target: np.ndarray,
    *, group_ids: tuple[str, ...],
) -> float:
    baseline, intervention, truth = (
        np.asarray(value, dtype=float)
        for value in (baseline_prediction, intervention_prediction, response_target)
    )
    if baseline.shape != intervention.shape or baseline.shape != truth.shape or truth.ndim < 2:
        raise ValueError("Point predictions and response targets must share batch and observation dimensions")
    if not all(np.isfinite(value).all() for value in (baseline, intervention, truth)):
        raise ValueError("Loss arguments must contain finite values")
    residual = np.mean(((intervention - baseline) - truth) ** 2, axis=tuple(range(1, truth.ndim)))
    return float(grouped_values(residual, group_ids).mean())


def branch_energy_loss(
    baseline_predictions: tuple[np.ndarray, ...], intervention_predictions: tuple[np.ndarray, ...],
    batch: TrainingBatch,
) -> float:
    if batch.target is not TrainingTarget.BRANCH_DISTRIBUTIONS:
        raise ValueError("Branch energy scoring requires distributional training targets")
    if len(baseline_predictions) != len(batch.pair_ids) or len(intervention_predictions) != len(batch.pair_ids):
        raise ValueError("Prediction distributions must cover every training pair")
    losses = [
        (energy_score(baseline, baseline_truth) + energy_score(intervention, intervention_truth)) / 2
        for baseline, intervention, baseline_truth, intervention_truth in zip(
            baseline_predictions, intervention_predictions, batch.baseline_samples, batch.intervention_samples, strict=True,
        )
    ]
    return float(grouped_values(np.asarray(losses), batch.exchangeability_groups).mean())


class TrainingConsumer(Protocol):
    def consume(self, batch: TrainingBatch) -> object: ...