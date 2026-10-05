"""Normative objects of the MultiEcho measurement protocol (see PROTOCOL.md).

A `Declaration` is the preregistered experimental unit. `adjudicate` maps sealed
evidence to exactly one outcome of the protocol's outcome space. "No law" is not an
outcome: a response that cannot be described within the declared scope is
`UNIDENTIFIED`, a statement about the instrument and scope rather than the world.
The protocol makes no triviality judgment, neither as a gate nor as an outcome:
reports carry descriptive response statistics, and whether a response is trivial is
analysed case by case for the specific question (R-8.1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from .contracts import (
    HARD_CONTROL_COUPLINGS,
    Context,
    DataRole,
    Intervention,
    SeedMode,
    StimulusSpec,
    WorldSpec,
    fingerprint,
)
from .evaluation import EvidenceStatus
from .keyframes import KeyframeAlignment
from .study import StudyLock

PROTOCOL_VERSION = "0.3"
HELD_OUT_ROLES = (DataRole.VALIDATION, DataRole.CONFIRMATION)
DECLARED_ROLES = frozenset({DataRole.DISCOVERY, *HELD_OUT_ROLES})
TASKS = ("forward", "backward")


class Outcome(StrEnum):
    STABLE_LAW = "stable_law"
    REGIME_SPECIFIC = "regime_specific"
    UNIDENTIFIED = "unidentified_in_scope"


OUTCOME_MEANING = {
    Outcome.STABLE_LAW: "A frozen candidate beats every declared control in every held-out stage and declared condition",
    Outcome.REGIME_SPECIFIC: "A frozen candidate holds only on a subset of declared conditions; the claim is restricted to that subset",
    Outcome.UNIDENTIFIED: "The frozen candidate does not beat the declared controls; nothing is identified in this scope. Whether the response itself is trivial is a case-by-case analysis of the reported response statistics, not an outcome",
}


@dataclass(frozen=True)
class ConditionDimension:
    """A declared nuisance or condition family, measured rather than assumed away."""

    name: str
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.assignments, tuple) or not self.assignments:
            raise ValueError("A condition dimension needs a name and context assignments")
        if any(not isinstance(row, tuple) or len(row) != 2 or not all(row) for row in self.assignments):
            raise ValueError("Condition assignments are nonempty (context_id, value) pairs")
        if len({context for context, _ in self.assignments}) != len(self.assignments):
            raise ValueError("A context has exactly one value per condition dimension")

    def value(self, context_id: str) -> str:
        return dict(self.assignments)[context_id]

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(sorted({value for _, value in self.assignments}))


@dataclass(frozen=True)
class Declaration:
    """The sealed experimental unit: everything that must be fixed before any measurement."""

    experiment_id: str
    question: str
    world: WorldSpec
    chain: dict
    handle: str
    observation_time: str
    reference: Intervention
    interventions: tuple[Intervention, ...]
    contexts: tuple[Context, ...]
    independence_unit: str
    calibration: tuple[Intervention, ...] = ()
    tasks: tuple[str, ...] = ("forward",)
    coupling: SeedMode = SeedMode.FIXED
    draws: int = 1
    repeats: int = 2
    conditions: tuple[ConditionDimension, ...] = ()
    minimum_improvement: float = 0.0
    equivalence_margin: float | None = None
    confidence: float = 0.95
    resamples: int = 1999
    seed: int = 0
    reference_world: str = ""
    reference_tolerance: float | None = None
    stimulus: StimulusSpec | None = None
    sweep: tuple[Intervention, ...] = ()
    sweep_draws: int = 1
    keyframes: KeyframeAlignment = KeyframeAlignment()
    study: StudyLock | None = None
    def __post_init__(self) -> None:
        text = (self.experiment_id, self.question, self.handle, self.observation_time, self.independence_unit)
        if not all(isinstance(value, str) and value for value in text):
            raise ValueError("Experiment identity, question, handle, observation time and independence unit are required")
        if not isinstance(self.coupling, SeedMode) or type(self.draws) is not int or self.draws < 1:
            raise ValueError("Declare an explicit random coupling and a positive draw count")
        if self.coupling not in HARD_CONTROL_COUPLINGS:
            raise ValueError("Hard control: every branch of a context shares each draw (fixed or context coupling)")
        if (self.coupling is SeedMode.FIXED) != (self.draws == 1):
            raise ValueError("A fixed realization uses exactly one draw; context draws need several")
        if type(self.repeats) is not int or self.repeats < 2:
            raise ValueError("At least two repeat runs are required to check determinism")
        if not isinstance(self.interventions, tuple) or not self.interventions:
            raise ValueError("Declare an immutable, nonempty tuple of queried interventions")
        if not isinstance(self.chain, dict) or self.chain.get("chain_id") != self.world.world_id:
            raise ValueError("Declare the chain record of the measured world")
        if fingerprint(self.chain) != self.world.revision:
            raise ValueError("The measured world revision must seal its chain record")
        if not isinstance(self.tasks, tuple) or not self.tasks or not set(self.tasks) <= set(TASKS) \
                or len(set(self.tasks)) != len(self.tasks):
            raise ValueError(f"Declare analysis tasks from {TASKS}")
        if not isinstance(self.calibration, tuple):
            raise ValueError("Calibration branches are an immutable tuple")
        space = self.world.intervention_space
        space.validate(self.reference.values)
        branches = (*self.calibration, *self.interventions)
        for action in branches:
            space.validate(action.values)
            if action.steps != self.reference.steps:
                raise ValueError("Every branch shares the reference horizon")
        if self.reference in branches or len(set(branches)) != len(branches):
            raise ValueError("Calibration and queried branches are unique and differ from the reference")
        self._check_chain_contracts(branches)
        self._check_contexts()
        for dimension in self.conditions:
            if {context for context, _ in dimension.assignments} != {context.context_id for context in self.contexts}:
                raise ValueError(f"Condition dimension must cover every declared context: {dimension.name}")
        if len({dimension.name for dimension in self.conditions}) != len(self.conditions):
            raise ValueError("Condition dimension names must be unique")
        if not np.isfinite(self.minimum_improvement) or self.minimum_improvement < 0:
            raise ValueError("The improvement margin must be nonnegative")
        if self.reference_tolerance is not None and not (self.reference_world and self.reference_tolerance > 0):
            raise ValueError("A reference tolerance needs a named reference world and a positive value")
        if not isinstance(self.keyframes, KeyframeAlignment):
            raise ValueError("The keyframe alignment policy is a KeyframeAlignment")
        if self.keyframes.requires_capture and self.stimulus is None:
            raise ValueError(f"A {self.keyframes.mode!r} keyframe policy needs a declared stimulus")
        if self.study is not None:
            if not isinstance(self.study, StudyLock):
                raise ValueError("The study is a StudyLock")
            self.study.require(self)

    def _check_chain_contracts(self, branches: tuple[Intervention, ...]) -> None:
        declared = self.chain.get("stimulus")
        stimulus = asdict(self.stimulus) if isinstance(self.stimulus, StimulusSpec) else None
        if self.stimulus is not None and stimulus is None:
            raise ValueError("The stimulus is declared as a StimulusSpec")
        if declared != stimulus:
            raise ValueError("The declared stimulus must equal the stimulus of the measured chain")
        masker = self.chain.get("observer", {}).get("scope", {}).get("masker")
        if not isinstance(self.sweep, tuple) or type(self.sweep_draws) is not int or self.sweep_draws < 1:
            raise ValueError("The sweep is an immutable tuple of interventions with a positive draw count")
        needs_sweep = masker is not None and masker.get("needs") == "sweep"
        if needs_sweep != bool(self.sweep):
            raise ValueError("Declare sweep values exactly when the analyst scope's masker needs a sweep")
        if len(self.sweep) == 1 or len(set(self.sweep)) != len(self.sweep):
            raise ValueError("A sweep has at least two distinct values")
        for action in self.sweep:
            self.world.intervention_space.validate(action.values)
            if action.steps != self.reference.steps:
                raise ValueError("Sweep values share the reference horizon")
            if action == self.reference or action in branches:
                raise ValueError("Sweep values must be disjoint from the reference, calibration and queried values")

    def _check_contexts(self) -> None:
        if not isinstance(self.contexts, tuple) or not self.contexts:
            raise ValueError("Declare an immutable, nonempty tuple of contexts")
        roles = {context.role for context in self.contexts}
        if not roles <= DECLARED_ROLES or DataRole.DISCOVERY not in roles or not roles & set(HELD_OUT_ROLES):
            raise ValueError("Contexts need discovery plus at least one held-out (validation/confirmation) role")
        if len({context.context_id for context in self.contexts}) != len(self.contexts):
            raise ValueError("Context identities must be unique")
        for field in ("payload_digest", "exchangeability_group"):
            owners: dict[str, DataRole] = {}
            for context in self.contexts:
                if owners.setdefault(getattr(context, field), context.role) is not context.role:
                    raise ValueError(f"Discovery and held-out contexts share a {field}")

    def contexts_for(self, role: DataRole) -> tuple[Context, ...]:
        return tuple(context for context in self.contexts if context.role is role)

    @property
    def held_out_roles(self) -> tuple[DataRole, ...]:
        return tuple(role for role in HELD_OUT_ROLES if self.contexts_for(role))

    def as_record(self) -> dict:
        return _plain(asdict(self))

    @property
    def digest(self) -> str:
        return fingerprint(self.as_record())


@dataclass(frozen=True)
class Claim:
    """A scoped statement: every claim names what, where, how and within which domain."""

    claim_id: str
    task: str
    outcome: Outcome
    statement: str
    subject: dict
    interface: dict
    handle: str
    intervention: dict
    contexts: dict
    regime: dict
    instrument: dict
    validity_domain: str
    stages: tuple[str, ...]


def adjudicate(
    stage_status: dict[str, dict[str, str]],
    regime_status: dict[str, dict[str, str]],
) -> tuple[Outcome, dict[str, tuple[str, ...]]]:
    """Deterministic reference adjudication rule (PROTOCOL.md Appendix B).

    `stage_status[role][control]` and `regime_status[dimension][value]` hold
    `EvidenceStatus` values. Returns the outcome and the condition values where it
    holds. No triviality judgment is made (R-8.1).
    """
    if not stage_status:
        raise ValueError("Adjudication needs control evidence for every held-out stage")
    supported = EvidenceStatus.SUPPORTED.value
    holds = {name: tuple(value for value, status in values.items() if status == supported)
             for name, values in regime_status.items()}
    failed = any(status == EvidenceStatus.CONTRADICTED.value
                 for values in regime_status.values() for status in values.values())
    every_stage = all(controls and all(status == supported for status in controls.values())
                      for controls in stage_status.values())
    if every_stage:
        return (Outcome.REGIME_SPECIFIC if failed else Outcome.STABLE_LAW), holds
    if any(holds.values()):
        return Outcome.REGIME_SPECIFIC, holds
    return Outcome.UNIDENTIFIED, {}


def regime_verdict(statuses: list[str]) -> str:
    """Combine per-stage, per-control statuses for one condition value."""
    if any(status == EvidenceStatus.CONTRADICTED.value for status in statuses):
        return EvidenceStatus.CONTRADICTED.value
    if statuses and all(status == EvidenceStatus.SUPPORTED.value for status in statuses):
        return EvidenceStatus.SUPPORTED.value
    return EvidenceStatus.UNRESOLVED.value


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value
