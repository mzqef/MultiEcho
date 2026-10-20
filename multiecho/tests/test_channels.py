from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from multiecho_framework import (
    Artifact,
    Channel,
    ChannelEvidence,
    ChannelOutcome,
    ChannelRecord,
    ChannelTest,
    CommutationCase,
    Comparison,
    ContentKind,
    DataRole,
    EvaluationPlan,
    EvaluationTarget,
    EvidenceStatus,
    InterfaceSpec,
    Modality,
    RoundTripCase,
    Scope,
    WorldCard,
    WorldKind,
    evaluate_commutation,
    evaluate_round_trip,
)

A_SOURCE = "world-a@0000000000000000"
B = WorldCard("world-b", "r1", WorldKind.SIMULATION, accepts=("state",), emits=("state",))
SCOPE = Scope("state", ("state",))


def artifact(source: str, values, *, parent: str = "", via: str = "") -> Artifact:
    return Artifact(source, (Channel("state", Modality.TENSOR, np.asarray(values, dtype=float)),),
                    parent=parent, via=via)


@dataclass(frozen=True)
class LinearTranslator:
    """A frozen linear map between the state spaces of two toy worlds."""

    interface_id: str
    matrix: tuple[tuple[float, ...], ...]
    source: str
    target: str

    def describe(self) -> InterfaceSpec:
        return InterfaceSpec(self.interface_id, repr(self.matrix), self.source, self.target, "translator",
                             "linear state map")

    def transmit(self, item: Artifact) -> Artifact:
        mapped = np.asarray(self.matrix) @ item.channel("state").data
        return artifact(item.source, mapped, parent=item.digest, via=self.describe().digest)


# World A rotates its 2-D state; world B represents the same state with axes swapped and scaled.
EVOLVE_A = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
EMBED = np.array([[0.0, 2.0], [1.0, 0.0]])
EVOLVE_B = EMBED @ EVOLVE_A @ np.linalg.inv(EMBED)


def plan(evaluation_id: str, role: DataRole = DataRole.VALIDATION) -> EvaluationPlan:
    return EvaluationPlan(evaluation_id, EvaluationTarget.MESSAGE_AGREEMENT, role, minimum_improvement=0.0)


def states(count: int = 24, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, 2))


def commutation_cases(translator: LinearTranslator, evolve_b: np.ndarray) -> tuple[CommutationCase, ...]:
    cases = []
    for index, state in enumerate(states()):
        later = artifact(A_SOURCE, EVOLVE_A @ state)
        message = translator.transmit(artifact(A_SOURCE, state))
        evolved = artifact(B.source, evolve_b @ message.channel("state").data)
        cases.append(CommutationCase(f"c{index}", later, translator.transmit(later), message, evolved, f"g{index}"))
    return tuple(cases)


def shuffled(cases, attribute: str) -> np.ndarray:
    values = np.stack([getattr(case, attribute).channel("state").data for case in cases])
    return np.roll(values, 1, axis=0)


def test_commutation_supports_an_exact_translator_and_contradicts_reversed_dynamics():
    translator = LinearTranslator("embed", tuple(map(tuple, EMBED)), "world-a", "world-b")
    cases = commutation_cases(translator, EVOLVE_B)
    report = evaluate_commutation(translator, B, cases, SCOPE,
                                  {"mismatched": shuffled(cases, "translated_then_evolved")}, plan("comm"))
    assert report.group_weighted_mse < 1e-20
    assert report.comparisons[0].status is EvidenceStatus.SUPPORTED

    # The verdict is relative to the declared control: B running the reversed dynamics does worse than chance pairing.
    reversed_cases = commutation_cases(translator, -EVOLVE_B)
    report = evaluate_commutation(translator, B, reversed_cases, SCOPE,
                                  {"mismatched": shuffled(reversed_cases, "translated_then_evolved")},
                                  plan("comm-reversed"))
    assert report.comparisons[0].status is EvidenceStatus.CONTRADICTED


def test_commutation_rejects_foreign_messages_receivers_and_roles():
    translator = LinearTranslator("embed", tuple(map(tuple, EMBED)), "world-a", "world-b")
    other = LinearTranslator("other", ((1.0, 0.0), (0.0, 1.0)), "world-a", "world-b")
    cases = commutation_cases(translator, EVOLVE_B)
    control = {"mismatched": shuffled(cases, "translated_then_evolved")}
    with pytest.raises(ValueError, match="interface under test"):
        evaluate_commutation(other, B, cases, SCOPE, control, plan("comm"))
    stranger = WorldCard("world-c", "r1", WorldKind.SIMULATION, emits=("state",))
    with pytest.raises(ValueError, match="declared receiver"):
        evaluate_commutation(translator, stranger, cases, SCOPE, control, plan("comm"))
    with pytest.raises(ValueError, match="validation or confirmation"):
        evaluate_commutation(translator, B, cases, SCOPE, control, plan("comm", DataRole.DISCOVERY))
    with pytest.raises(ValueError, match="named control"):
        evaluate_commutation(translator, B, cases, SCOPE, {}, plan("comm"))
    first = cases[0]
    with pytest.raises(ValueError, match="evolved artifact"):
        CommutationCase("bad", artifact(A_SOURCE, [9.0, 9.0]), first.evolved_then_translated, first.message,
                        first.translated_then_evolved, "g")


def round_trip_cases(forward: LinearTranslator, backward: LinearTranslator) -> tuple[RoundTripCase, ...]:
    cases = []
    for index, state in enumerate(states(seed=5)):
        original = artifact(A_SOURCE, state)
        intermediate = forward.transmit(original)
        cases.append(RoundTripCase(f"r{index}", original, intermediate, backward.transmit(intermediate), f"g{index}"))
    return tuple(cases)


def test_round_trip_reports_what_is_lost_per_coordinate():
    forward = LinearTranslator("embed", tuple(map(tuple, EMBED)), "world-a", "world-b")
    exact = LinearTranslator("unembed", tuple(map(tuple, np.linalg.inv(EMBED))), "world-b", "world-a")
    lossy = LinearTranslator("drop-second", ((0.0, 1.0), (0.0, 0.0)), "world-b", "world-a")

    cases = round_trip_cases(forward, exact)
    report = evaluate_round_trip(forward, exact, cases, SCOPE, {"mismatched": shuffled(cases, "original")},
                                 plan("rt"))
    assert np.allclose(report.loss_profile, 0.0, atol=1e-20)
    assert report.comparisons[0].status is EvidenceStatus.SUPPORTED

    cases = round_trip_cases(forward, lossy)
    report = evaluate_round_trip(forward, lossy, cases, SCOPE, {"mismatched": shuffled(cases, "original")},
                                 plan("rt-lossy"))
    assert report.loss_profile[0] < 1e-20 < report.loss_profile[1]
    with pytest.raises(ValueError, match="declared forward and backward"):
        evaluate_round_trip(forward, exact, cases, SCOPE, {"mismatched": shuffled(cases, "original")}, plan("rt"))


def comparison(status: EvidenceStatus) -> tuple[Comparison, ...]:
    return (Comparison("control", 1.0, (0.5, 1.5), status, 8, 0.95),)


def evidence(test: ChannelTest, status=EvidenceStatus.SUPPORTED, digest="T", role=DataRole.VALIDATION):
    return ChannelEvidence(test, f"e-{test.value}", role, digest, comparison(status),
                           "R" if test is ChannelTest.ROUND_TRIP else "")


def record(*items: ChannelEvidence, **overrides) -> ChannelRecord:
    fields = dict(channel_id="ch", content=ContentKind.LAW, source=A_SOURCE, source_kind=WorldKind.SIMULATION,
                  target=B.source, target_kind=WorldKind.LEARNED, translator="T", return_translator="R",
                  fitted_on=(DataRole.DISCOVERY,), evidence=items)
    fields.update(overrides)
    return ChannelRecord(**fields)


def test_channel_outcome_is_derived_from_evidence_without_framework_thresholds():
    full = record(*(evidence(test) for test in ChannelTest))
    assert full.outcome is ChannelOutcome.ESTABLISHED and not full.unsupported and not full.not_run
    assert full.direction == "simulation->learned"

    partial = record(evidence(ChannelTest.CONSISTENCY), evidence(ChannelTest.COMMUTATION, EvidenceStatus.CONTRADICTED))
    assert partial.outcome is ChannelOutcome.PARTIAL
    assert partial.unsupported == (ChannelTest.COMMUTATION,)
    assert set(partial.not_run) == {ChannelTest.UTILITY, ChannelTest.ROUND_TRIP}

    failed = record(evidence(ChannelTest.UTILITY, EvidenceStatus.UNRESOLVED))
    assert failed.outcome is ChannelOutcome.FAILED
    assert "not evidence that no channel exists" in failed.record()["claim_boundary"]
    assert failed.digest == record(evidence(ChannelTest.UTILITY, EvidenceStatus.UNRESOLVED)).digest


def test_channel_record_enforces_roles_identity_and_provenance():
    with pytest.raises(ValueError, match="fitted only"):
        record(evidence(ChannelTest.CONSISTENCY), fitted_on=(DataRole.VALIDATION,))
    with pytest.raises(ValueError, match="declared translator"):
        record(evidence(ChannelTest.CONSISTENCY, digest="other"))
    with pytest.raises(ValueError, match="return translator"):
        record(evidence(ChannelTest.ROUND_TRIP), return_translator="other")
    with pytest.raises(ValueError, match="at most once"):
        record(evidence(ChannelTest.UTILITY), evidence(ChannelTest.UTILITY))
    with pytest.raises(ValueError, match="distinct world"):
        record(evidence(ChannelTest.UTILITY), target=A_SOURCE)
    with pytest.raises(ValueError, match="validation or confirmation"):
        evidence(ChannelTest.UTILITY, role=DataRole.DISCOVERY)
