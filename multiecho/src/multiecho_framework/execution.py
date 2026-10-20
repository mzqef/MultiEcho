"""Isolated execution of paired branches through an external frozen-world adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import Context, Intervention, PairRequest, SeedMode, WorldSpec


@runtime_checkable
class WorldAdapter(Protocol):
    def describe(self) -> WorldSpec: ...

    def open(self, context: Context, seed: int) -> object: ...

    def fork(self, state: object) -> object: ...

    def apply(self, state: object, intervention: Intervention, seed: int) -> object: ...

    def observe(self, state: object) -> np.ndarray: ...

    def close(self, state: object) -> None: ...


@dataclass(frozen=True)
class PairResult:
    world: WorldSpec
    request: PairRequest
    baseline: np.ndarray
    intervention: np.ndarray

    def __post_init__(self) -> None:
        for action in (self.request.baseline, self.request.intervention):
            self.world.intervention_space.validate(action.values)
        expected = (len(self.request.seeds), *self.world.observation.shape)
        for field in ("baseline", "intervention"):
            value = np.array(getattr(self, field), dtype=np.float64, copy=True)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError("PairResult shape or finite-value contract violated")
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    @property
    def mean_effect(self) -> np.ndarray:
        return self.intervention.mean(axis=0) - self.baseline.mean(axis=0)

    @property
    def paired_effects(self) -> np.ndarray:
        if self.request.mode is SeedMode.INDEPENDENT:
            raise ValueError("Independent draws do not identify a paired pathwise effect")
        return self.intervention - self.baseline


class PairRunner:
    def __init__(self, adapter: WorldAdapter):
        self.adapter = adapter
        self.spec = adapter.describe()

    def _unchanged(self) -> None:
        if self.adapter.describe().digest != self.spec.digest:
            raise RuntimeError("World revision changed during frozen execution")

    def _branch(self, state: object, action: Intervention, seed: int) -> np.ndarray:
        branch = self.adapter.fork(state)
        if branch is state:
            raise ValueError("fork must return an isolated state handle")
        output = branch
        try:
            output = self.adapter.apply(branch, action, seed)
            if output is state:
                output = branch
                raise ValueError("apply must not return the shared root state")
            self._unchanged()
            return self.spec.observation.validate(self.adapter.observe(output))
        finally:
            if output is not branch:
                self.adapter.close(output)
            self.adapter.close(branch)

    def run(self, request: PairRequest) -> PairResult:
        self._unchanged()
        self.spec.intervention_space.validate(request.baseline.values)
        self.spec.intervention_space.validate(request.intervention.values)
        baseline, intervention = [], []
        for index, seed in enumerate(request.seeds):
            state = self.adapter.open(request.context, seed)
            try:
                baseline.append(self._branch(state, request.baseline, seed))
                if request.mode is SeedMode.INDEPENDENT:
                    branch_seed = request.independent_seeds[index]
                    separate = self.adapter.open(request.context, branch_seed)
                    try:
                        intervention.append(self._branch(separate, request.intervention, branch_seed))
                    finally:
                        self.adapter.close(separate)
                else:
                    intervention.append(self._branch(state, request.intervention, seed))
                self._unchanged()
            finally:
                self.adapter.close(state)
        return PairResult(self.spec, request, np.stack(baseline), np.stack(intervention))


@dataclass(frozen=True)
class SequenceRequest:
    trace_id: str
    context: Context
    interventions: tuple[Intervention, ...]
    context_seed: int
    step_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.trace_id or not isinstance(self.interventions, tuple) or not self.interventions:
            raise ValueError("A trace needs an identity and a nonempty immutable action sequence")
        if not isinstance(self.step_seeds, tuple) or len(self.step_seeds) != len(self.interventions):
            raise ValueError("Declare one step seed per sequence intervention")
        if any(type(seed) is not int or seed < 0 for seed in (self.context_seed, *self.step_seeds)):
            raise ValueError("Sequence seeds must be nonnegative integers")


@dataclass(frozen=True)
class TraceResult:
    world: WorldSpec
    request: SequenceRequest
    observations: np.ndarray

    def __post_init__(self) -> None:
        if not self.world.supports_sequence:
            raise ValueError("World does not declare sequential intervention access")
        for action in self.request.interventions:
            self.world.intervention_space.validate(action.values)
        array = np.array(self.observations, dtype=float, copy=True)
        expected = (len(self.request.interventions) + 1, *self.world.observation.shape)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError("Trace must contain the initial and all subsequent observations")
        array.setflags(write=False)
        object.__setattr__(self, "observations", array)


class SequenceRunner:
    def __init__(self, adapter: WorldAdapter):
        self.adapter = adapter
        self.spec = adapter.describe()
        if not self.spec.supports_sequence:
            raise ValueError("World does not declare sequential intervention access")

    def _unchanged(self) -> None:
        if self.adapter.describe().digest != self.spec.digest:
            raise RuntimeError("World revision changed during frozen sequence execution")

    def run(self, request: SequenceRequest) -> TraceResult:
        self._unchanged()
        for action in request.interventions:
            self.spec.intervention_space.validate(action.values)
        root = self.adapter.open(request.context, request.context_seed)
        handles = [root]
        try:
            state = self.adapter.fork(root)
            if state is root:
                raise ValueError("Sequence execution needs an isolated state handle")
            handles.append(state)
            observations = [self.spec.observation.validate(self.adapter.observe(state))]
            self._unchanged()
            for action, seed in zip(request.interventions, request.step_seeds, strict=True):
                state = self.adapter.apply(state, action, seed)
                if state is root:
                    raise ValueError("Sequence apply must not return the shared root state")
                if not any(state is handle for handle in handles):
                    handles.append(state)
                observations.append(self.spec.observation.validate(self.adapter.observe(state)))
                self._unchanged()
            return TraceResult(self.spec, request, np.stack(observations))
        finally:
            for handle in reversed(handles):
                self.adapter.close(handle)