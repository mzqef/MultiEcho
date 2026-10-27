"""Analysis tasks on measured responses: forward and backward fitting.

The framework defines what a method may see and the statistics that judge it. It
ships no estimator: forward and backward methods are plugins.

- Forward: predict the response to an intervention in a context.
- Backward: recover the intervention from a response in a context.

In every stage a frozen law sees a `ContextView`: the context, the reference
observation and the declared labeled calibration branches of that same context.
Query outcomes are visible only in discovery, for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from .contracts import Context, DataRole, Intervention, WorldSpec
from .datasets import PairDataset
from .laws import LawSpec


class Task(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


CONTROLS = {
    Task.FORWARD: {
        "zero-response": "Predicts no response; the unchanged reference",
        "discovery-mean-response": "Discovery mean response at the same intervention, ignoring context",
    },
    Task.BACKWARD: {
        "reference-value": "Recovers the reference value for every response",
        "discovery-mean-value": "Recovers the discovery mean of the queried values for every response",
    },
}


@dataclass(frozen=True)
class Labeled:
    intervention: Intervention
    response: np.ndarray

    def __post_init__(self) -> None:
        array = np.array(self.response, dtype=float, copy=True)
        if not array.size or not np.isfinite(array).all():
            raise ValueError("A labeled response must be finite and nonempty")
        array.setflags(write=False)
        object.__setattr__(self, "response", array)


@dataclass(frozen=True)
class ContextView:
    """What a frozen law may see in any context."""

    context: Context
    reference: Intervention
    baseline: np.ndarray
    calibration: tuple[Labeled, ...]

    def __post_init__(self) -> None:
        array = np.array(self.baseline, dtype=float, copy=True)
        if not np.isfinite(array).all():
            raise ValueError("The reference observation must be finite")
        array.setflags(write=False)
        object.__setattr__(self, "baseline", array)


@dataclass(frozen=True)
class ResponseSet:
    """Discovery responses for fitting: one view and its labeled queries per context."""

    world: WorldSpec
    views: tuple[ContextView, ...]
    queries: tuple[tuple[Labeled, ...], ...]

    def __post_init__(self) -> None:
        if not self.views or len(self.views) != len(self.queries):
            raise ValueError("Every discovery view needs its labeled queries")
        if any(view.context.role is not DataRole.DISCOVERY for view in self.views):
            raise ValueError("Methods are fitted on discovery contexts only")

    @property
    def contexts(self) -> tuple[Context, ...]:
        return tuple(view.context for view in self.views)


class ForwardLaw(Protocol):
    def describe(self) -> LawSpec: ...

    def predict(self, view: ContextView, intervention: Intervention) -> np.ndarray: ...


class BackwardLaw(Protocol):
    def describe(self) -> LawSpec: ...

    def recover(self, view: ContextView, response: np.ndarray) -> tuple[float, ...]: ...


class ForwardMethod(Protocol):
    def fit(self, data: ResponseSet) -> ForwardLaw: ...


class BackwardMethod(Protocol):
    def fit(self, data: ResponseSet) -> BackwardLaw: ...


def split_rows(dataset: PairDataset) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Row indices of calibration (`/c`) and query (`/q`) branches, per context."""
    calibration: dict[str, list[int]] = {}
    queries: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.results):
        kind = row.request.pair_id.rsplit("/", 1)[-1][:1]
        target = calibration if kind == "c" else queries if kind == "q" else None
        if target is None:
            raise ValueError(f"Unrecognized branch identity: {row.request.pair_id}")
        target.setdefault(row.request.context.context_id, []).append(index)
    return calibration, queries


def views(dataset: PairDataset) -> dict[str, ContextView]:
    calibration, queries = split_rows(dataset)
    output = {}
    for context_id, indices in queries.items():
        rows = [dataset.results[index] for index in indices + calibration.get(context_id, [])]
        baseline = np.mean([row.baseline.mean(axis=0) for row in rows], axis=0)
        labeled = tuple(Labeled(dataset.results[index].request.intervention, dataset.results[index].mean_effect)
                        for index in calibration.get(context_id, []))
        output[context_id] = ContextView(rows[0].request.context, rows[0].request.baseline, baseline, labeled)
    return output


def response_set(dataset: PairDataset) -> ResponseSet:
    view_map, (_, queries) = views(dataset), split_rows(dataset)
    ordered = sorted(view_map)
    return ResponseSet(dataset.world, tuple(view_map[key] for key in ordered), tuple(
        tuple(Labeled(dataset.results[index].request.intervention, dataset.results[index].mean_effect)
              for index in queries[key]) for key in ordered))
