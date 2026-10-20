"""World-to-world channels: the four channel tests and a world-agnostic channel record (experimental).

A channel is a frozen translator from world A to world B for one kind of content
(scene, object, law or mechanism). Any two worlds that implement the `World`
protocol can be joined; the framework knows nothing about engines, models or
physics. A channel is judged by four tests:

1. consistency -- translated messages agree with B's own artifacts (`evaluate_translation`);
2. utility -- B does better with the correct message than with none or a wrong one
   (`evaluate_communication`);
3. commutation -- evolving in A then translating agrees with translating then evolving in B;
4. round trip -- what A -> B -> A loses.

Every test compares against named controls with the shared `EvaluationPlan`, so the
improvement that counts as support is declared by the study, never by the framework.
The channel outcome only aggregates per-test evidence; a failed or partial channel is
relative to the translator revision and the declared plans, not a statement that no
channel between the two worlds exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .artifacts import Artifact, Scope
from .communication import CommunicationReport, TranslationReport
from .contracts import DataRole, fingerprint
from .evaluation import (
    Comparison,
    EvaluationPlan,
    EvaluationTarget,
    EvidenceStatus,
    grouped_values,
    summarize_advantage,
)
from .interfaces import Interface
from .worlds import WorldCard, WorldKind


class ContentKind(StrEnum):
    SCENE = "scene"
    OBJECT = "object"
    LAW = "law"
    MECHANISM = "mechanism"


class ChannelTest(StrEnum):
    CONSISTENCY = "consistency"
    UTILITY = "utility"
    COMMUTATION = "commutation"
    ROUND_TRIP = "round_trip"


class ChannelOutcome(StrEnum):
    ESTABLISHED = "established"
    PARTIAL = "partial"
    FAILED = "failed"


FIT_ROLES = frozenset({DataRole.TRAIN, DataRole.DISCOVERY, DataRole.CALIBRATION})
TEST_ROLES = frozenset({DataRole.VALIDATION, DataRole.CONFIRMATION})


def _scoped_matrix(scope: Scope, artifacts: list[Artifact], via: str) -> np.ndarray:
    return np.stack([scope.vector(scope.apply(artifact, via=via)) for artifact in artifacts])


def _compare(errors: np.ndarray, truth: np.ndarray, controls: dict[str, np.ndarray], groups: tuple[str, ...],
             plan: EvaluationPlan, label: str) -> tuple[Comparison, ...]:
    if not controls or any(not name for name in controls):
        raise ValueError(f"{label} needs at least one named control")
    comparisons = []
    for name, values in sorted(controls.items()):
        values = np.asarray(values, dtype=float)
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(f"{label} control does not match the scoped target: {name}")
        comparisons.append(summarize_advantage(np.mean((truth - values) ** 2, axis=1) - errors, groups, plan,
                                               name, len(controls)))
    return tuple(comparisons)


def _check_plan(plan: EvaluationPlan, label: str) -> None:
    if plan.target is not EvaluationTarget.MESSAGE_AGREEMENT:
        raise ValueError(f"{label} is a message-agreement evaluation")
    if plan.role not in TEST_ROLES:
        raise ValueError(f"{label} is evaluated only on validation or confirmation data")


@dataclass(frozen=True)
class CommutationCase:
    """One state of A seen along both paths of the commuting square.

    - `source_later`: A's own artifact after A evolved the state;
    - `evolved_then_translated`: the translator's message for `source_later`;
    - `message`: the translator's message for A's artifact before evolution;
    - `translated_then_evolved`: B's artifact after it received `message` and evolved.
    """

    case_id: str
    source_later: Artifact
    evolved_then_translated: Artifact
    message: Artifact
    translated_then_evolved: Artifact
    exchangeability_group: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.exchangeability_group:
            raise ValueError("A commutation case needs an identity and an exchangeability group")
        if self.evolved_then_translated.parent != self.source_later.digest:
            raise ValueError(f"Case {self.case_id}: the translated path must derive from A's evolved artifact")


@dataclass(frozen=True)
class CommutationReport:
    evaluation_id: str
    role: DataRole
    interface_digest: str
    receiver_digest: str
    scope_digest: str
    group_weighted_mse: float
    comparisons: tuple[Comparison, ...]
    case_ids: tuple[str, ...]
    claim_boundary: str = ("Agreement of evolve-then-translate with translate-then-evolve under a declared scope of the "
                           "receiver; relative to this translator revision")


def evaluate_commutation(
    interface: Interface, receiver: WorldCard, cases: tuple[CommutationCase, ...], scope: Scope,
    controls: dict[str, np.ndarray], plan: EvaluationPlan,
) -> CommutationReport:
    """Commuting-square test: T(evolve_A(x)) against evolve_B(T(x)), on held-out cases."""
    _check_plan(plan, "Commutation")
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Commutation needs uniquely identified cases")
    digest = interface.describe().digest
    for case in cases:
        if case.evolved_then_translated.via != digest or case.message.via != digest:
            raise ValueError(f"Case {case.case_id}: both messages must come from the interface under test")
        if case.translated_then_evolved.source != receiver.source:
            raise ValueError(f"Case {case.case_id}: the evolved path must be emitted by the declared receiver")
    translated = _scoped_matrix(scope, [case.evolved_then_translated for case in cases], digest)
    evolved = _scoped_matrix(scope, [case.translated_then_evolved for case in cases], digest)
    groups = tuple(case.exchangeability_group for case in cases)
    errors = np.mean((translated - evolved) ** 2, axis=1)
    comparisons = _compare(errors, translated, controls, groups, plan, "Commutation")
    return CommutationReport(plan.evaluation_id, plan.role, digest, receiver.digest, scope.digest,
                             float(grouped_values(errors, groups).mean()), comparisons,
                             tuple(case.case_id for case in cases))


@dataclass(frozen=True)
class RoundTripCase:
    """A -> B -> A: `intermediate` is the forward message, `returned` the backward message."""

    case_id: str
    original: Artifact
    intermediate: Artifact
    returned: Artifact
    exchangeability_group: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.exchangeability_group:
            raise ValueError("A round-trip case needs an identity and an exchangeability group")
        if self.intermediate.parent != self.original.digest or self.returned.parent != self.intermediate.digest:
            raise ValueError(f"Case {self.case_id}: the round trip must chain original -> intermediate -> returned")


@dataclass(frozen=True)
class RoundTripReport:
    evaluation_id: str
    role: DataRole
    forward_digest: str
    backward_digest: str
    scope_digest: str
    group_weighted_mse: float
    loss_profile: np.ndarray = field(repr=False)
    comparisons: tuple[Comparison, ...] = ()
    case_ids: tuple[str, ...] = ()
    claim_boundary: str = ("What the declared forward and backward translators lose, per scoped coordinate of the "
                           "source world; relative to these translator revisions")

    def __post_init__(self) -> None:
        profile = np.array(self.loss_profile, dtype=float, copy=True)
        if profile.ndim != 1 or not np.isfinite(profile).all():
            raise ValueError("A loss profile is one finite value per scoped coordinate")
        profile.setflags(write=False)
        object.__setattr__(self, "loss_profile", profile)


def evaluate_round_trip(
    forward: Interface, backward: Interface, cases: tuple[RoundTripCase, ...], scope: Scope,
    controls: dict[str, np.ndarray], plan: EvaluationPlan,
) -> RoundTripReport:
    """Round-trip test: compare A's artifact with what returns from B, and report the per-coordinate loss."""
    _check_plan(plan, "Round trip")
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("A round trip needs uniquely identified cases")
    forward_digest, backward_digest = forward.describe().digest, backward.describe().digest
    for case in cases:
        if case.intermediate.via != forward_digest or case.returned.via != backward_digest:
            raise ValueError(f"Case {case.case_id}: messages must come from the declared forward and backward interfaces")
    original = _scoped_matrix(scope, [case.original for case in cases], backward_digest)
    returned = _scoped_matrix(scope, [case.returned for case in cases], backward_digest)
    groups = tuple(case.exchangeability_group for case in cases)
    squared = (original - returned) ** 2
    errors = squared.mean(axis=1)
    profile = np.stack([grouped_values(squared[:, index], groups) for index in range(squared.shape[1])]).mean(axis=1)
    comparisons = _compare(errors, original, controls, groups, plan, "Round trip")
    return RoundTripReport(plan.evaluation_id, plan.role, forward_digest, backward_digest, scope.digest,
                           float(grouped_values(errors, groups).mean()), profile, comparisons,
                           tuple(case.case_id for case in cases))


@dataclass(frozen=True)
class ChannelEvidence:
    """The comparisons one channel test produced on held-out data."""

    test: ChannelTest
    evaluation_id: str
    role: DataRole
    interface_digest: str
    comparisons: tuple[Comparison, ...]
    return_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.test, ChannelTest) or not self.evaluation_id or not self.interface_digest:
            raise ValueError("Channel evidence needs a test, an evaluation identity and the translator digest")
        if (self.test is ChannelTest.ROUND_TRIP) != bool(self.return_digest):
            raise ValueError("Round-trip evidence, and only round-trip evidence, names its return translator")
        if self.role not in TEST_ROLES:
            raise ValueError("Channel evidence comes only from validation or confirmation data")
        if not isinstance(self.comparisons, tuple) or not self.comparisons:
            raise ValueError("Channel evidence needs at least one comparison against a control")

    @property
    def status(self) -> EvidenceStatus:
        statuses = {comparison.status for comparison in self.comparisons}
        if statuses == {EvidenceStatus.SUPPORTED}:
            return EvidenceStatus.SUPPORTED
        for status in (EvidenceStatus.CONTRADICTED, EvidenceStatus.EQUIVALENT):
            if status in statuses:
                return status
        return EvidenceStatus.UNRESOLVED

    @classmethod
    def from_translation(cls, report: TranslationReport, plan: EvaluationPlan) -> ChannelEvidence:
        _require_plan(report.evaluation_id, plan)
        return cls(ChannelTest.CONSISTENCY, report.evaluation_id, plan.role, report.interface_digest, report.comparisons)

    @classmethod
    def from_utility(cls, report: CommunicationReport, plan: EvaluationPlan, interface_digest: str) -> ChannelEvidence:
        _require_plan(report.evaluation_id, plan)
        return cls(ChannelTest.UTILITY, report.evaluation_id, plan.role, interface_digest, report.comparisons)

    @classmethod
    def from_commutation(cls, report: CommutationReport) -> ChannelEvidence:
        return cls(ChannelTest.COMMUTATION, report.evaluation_id, report.role, report.interface_digest,
                   report.comparisons)

    @classmethod
    def from_round_trip(cls, report: RoundTripReport) -> ChannelEvidence:
        return cls(ChannelTest.ROUND_TRIP, report.evaluation_id, report.role, report.forward_digest,
                   report.comparisons, report.backward_digest)


def _require_plan(evaluation_id: str, plan: EvaluationPlan) -> None:
    if evaluation_id != plan.evaluation_id:
        raise ValueError("Evidence must be paired with the plan that produced it")


@dataclass(frozen=True)
class ChannelRecord:
    """One frozen translator from world A to world B for one kind of content, with its test evidence.

    `outcome` is derived, never declared: established only when all four tests ran and
    each was supported under its plan; failed when none was supported; partial otherwise,
    with the unsupported and missing tests named.
    """

    channel_id: str
    content: ContentKind
    source: str
    source_kind: WorldKind
    target: str
    target_kind: WorldKind
    translator: str
    fitted_on: tuple[DataRole, ...]
    evidence: tuple[ChannelEvidence, ...]
    return_translator: str = ""
    content_schema: str = ""
    claim_boundary: str = ("Relative to this translator revision, the declared evaluation plans and the plugins used; "
                           "a failed or partial channel is not evidence that no channel exists")

    def __post_init__(self) -> None:
        if not self.channel_id or not isinstance(self.content, ContentKind):
            raise ValueError("A channel needs an identity and a declared content kind")
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("A channel joins two distinct world revisions")
        if not isinstance(self.source_kind, WorldKind) or not isinstance(self.target_kind, WorldKind):
            raise ValueError("Declare the kinds of both worlds")
        if not self.translator:
            raise ValueError("A channel needs a frozen translator digest")
        if (not isinstance(self.fitted_on, tuple) or not self.fitted_on
                or not set(self.fitted_on) <= FIT_ROLES):
            raise ValueError("Translators are fitted only on train, discovery or calibration data")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("A channel record needs evidence from at least one test")
        tests = [item.test for item in self.evidence]
        if len(set(tests)) != len(tests):
            raise ValueError("Each channel test appears at most once; run separate records for separate plans")
        for item in self.evidence:
            if item.interface_digest != self.translator:
                raise ValueError(f"{item.test} evidence was not produced by the declared translator")
            if item.return_digest and item.return_digest != self.return_translator:
                raise ValueError("Round-trip evidence was not produced by the declared return translator")

    @property
    def direction(self) -> str:
        return f"{self.source_kind.value}->{self.target_kind.value}"

    @property
    def statuses(self) -> dict[ChannelTest, EvidenceStatus]:
        return {item.test: item.status for item in self.evidence}

    @property
    def not_run(self) -> tuple[ChannelTest, ...]:
        ran = {item.test for item in self.evidence}
        return tuple(test for test in ChannelTest if test not in ran)

    @property
    def unsupported(self) -> tuple[ChannelTest, ...]:
        return tuple(test for test, status in self.statuses.items() if status is not EvidenceStatus.SUPPORTED)

    @property
    def outcome(self) -> ChannelOutcome:
        supported = [test for test, status in self.statuses.items() if status is EvidenceStatus.SUPPORTED]
        if not supported:
            return ChannelOutcome.FAILED
        if len(supported) == len(ChannelTest):
            return ChannelOutcome.ESTABLISHED
        return ChannelOutcome.PARTIAL

    def record(self) -> dict:
        return {
            "channel_id": self.channel_id, "content": self.content.value, "content_schema": self.content_schema,
            "source": self.source, "source_kind": self.source_kind.value,
            "target": self.target, "target_kind": self.target_kind.value, "direction": self.direction,
            "translator": self.translator, "return_translator": self.return_translator,
            "fitted_on": [role.value for role in self.fitted_on],
            "evidence": [{"test": item.test.value, "evaluation_id": item.evaluation_id, "role": item.role.value,
                          "interface": item.interface_digest, "return_interface": item.return_digest,
                          "status": item.status.value,
                          "comparisons": [{"comparator": comparison.comparator,
                                           "advantage": comparison.mean_error_advantage,
                                           "interval": list(comparison.interval) if comparison.interval else None,
                                           "status": comparison.status.value, "groups": comparison.groups,
                                           "confidence": comparison.confidence}
                                          for comparison in item.comparisons]}
                         for item in self.evidence],
            "outcome": self.outcome.value,
            "unsupported": [test.value for test in self.unsupported],
            "not_run": [test.value for test in self.not_run],
            "claim_boundary": self.claim_boundary,
        }

    @property
    def digest(self) -> str:
        return fingerprint(self.record())
