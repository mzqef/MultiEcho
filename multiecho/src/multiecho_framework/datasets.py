"""Validated collections with explicit context and exchangeability identities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import DataRole, WorldSpec
from .execution import PairResult


@dataclass(frozen=True)
class PairDataset:
    results: tuple[PairResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or not self.results:
            raise ValueError("A dataset needs an immutable nonempty tuple of results")
        if len({row.request.pair_id for row in self.results}) != len(self.results):
            raise ValueError("Pair IDs must be unique within a dataset")
        worlds = {row.world.digest for row in self.results}
        if len(worlds) != 1:
            raise ValueError("One dataset must use one frozen world and observation contract")
        contexts: dict[str, str] = {}
        groups: dict[str, DataRole] = {}
        for row in self.results:
            context = row.request.context
            if contexts.setdefault(context.context_id, context.digest) != context.digest:
                raise ValueError("Context identity changed within the dataset")
            if groups.setdefault(context.exchangeability_group, context.role) != context.role:
                raise ValueError("An exchangeability group cannot straddle data roles")

    @property
    def world(self) -> WorldSpec:
        return self.results[0].world

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(row.request.context.exchangeability_group for row in self.results)

    @property
    def roles(self) -> frozenset[DataRole]:
        return frozenset(row.request.context.role for row in self.results)

    @property
    def mean_effects(self) -> np.ndarray:
        return np.stack([row.mean_effect for row in self.results])

    @property
    def intervention_deltas(self) -> np.ndarray:
        return np.array([
            np.subtract(row.request.intervention.values, row.request.baseline.values)
            for row in self.results
        ])

    def select(self, role: DataRole) -> PairDataset:
        return PairDataset(tuple(row for row in self.results if row.request.context.role is role))


def require_disjoint(training: PairDataset, evaluation: PairDataset) -> None:
    for attribute in ("context_id", "payload_digest", "exchangeability_group"):
        left = {getattr(row.request.context, attribute) for row in training.results}
        right = {getattr(row.request.context, attribute) for row in evaluation.results}
        if left & right:
            raise ValueError(f"Training/evaluation leakage in {attribute}")


def validate_training_roles(dataset: PairDataset, *, allow_discovery: bool = False) -> None:
    allowed = {DataRole.TRAIN, DataRole.DISCOVERY} if allow_discovery else {DataRole.TRAIN}
    if not dataset.roles <= allowed:
        raise ValueError("Training export refuses calibration, validation, or confirmation data")