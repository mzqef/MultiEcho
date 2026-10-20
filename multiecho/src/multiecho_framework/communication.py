"""Communication between worlds, as three separate tests.

1. Interface: a declared protocol turns world A's artifact into a message
   (masking, clipping, parameter mapping, or a plugin translator).
2. Translation: on semantically aligned queries, do messages agree with the
   receiver's own artifacts under the receiver's scope?
3. Utility: does world B do better with the correct message than with no message
   or a matched wrong message, on isolated forks of the same state?

An experiment is the special case where the last receiver is an analyst.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import Artifact, Scope
from .contracts import Context, Intervention, SeedMode
from .evaluation import (
    Comparison,
    EvaluationPlan,
    EvaluationTarget,
    grouped_values,
    summarize_advantage,
)
from .interfaces import Interface, ScopeInterface
from .worlds import World, WorldCard, require_accepted


@dataclass(frozen=True)
class Alignment:
    alignment_id: str
    message: Artifact
    target: Artifact
    exchangeability_group: str
    correspondence: str

    def __post_init__(self) -> None:
        if not self.alignment_id or not self.exchangeability_group or not self.correspondence:
            raise ValueError("An alignment needs an identity, an independent group and a declared correspondence")


@dataclass(frozen=True)
class TranslationReport:
    evaluation_id: str
    interface_digest: str
    scope_digest: str
    group_weighted_mse: float
    comparisons: tuple[Comparison, ...]
    alignment_ids: tuple[str, ...]
    claim_boundary: str = "Agreement of messages with the receiver's own artifacts under a declared scope; not use of the message"


def evaluate_translation(
    interface: Interface, alignments: tuple[Alignment, ...], scope: Scope,
    controls: dict[str, np.ndarray], plan: EvaluationPlan,
) -> TranslationReport:
    if plan.target is not EvaluationTarget.MESSAGE_AGREEMENT or not alignments or not controls:
        raise ValueError("Translation tests need aligned queries, named controls and a message-agreement target")
    if len({row.alignment_id for row in alignments}) != len(alignments) or any(not name for name in controls):
        raise ValueError("Alignment identities must be unique and controls named")
    digest = interface.describe().digest
    predicted, truth = [], []
    for row in alignments:
        if row.message.via != digest:
            raise ValueError(f"Alignment {row.alignment_id} was not produced by the interface under test")
        predicted.append(scope.vector(scope.apply(row.message, via=digest)))
        truth.append(scope.vector(scope.apply(row.target, via=digest)))
    predicted, truth = np.stack(predicted), np.stack(truth)
    groups = tuple(row.exchangeability_group for row in alignments)
    errors = np.mean((truth - predicted) ** 2, axis=1)
    comparisons = []
    for name, values in sorted(controls.items()):
        values = np.asarray(values, dtype=float)
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(f"Translation control does not match the scoped target: {name}")
        comparisons.append(summarize_advantage(np.mean((truth - values) ** 2, axis=1) - errors, groups, plan,
                                               name, len(controls)))
    return TranslationReport(plan.evaluation_id, digest, scope.digest, float(grouped_values(errors, groups).mean()),
                             tuple(comparisons), tuple(row.alignment_id for row in alignments))


@dataclass(frozen=True)
class CommunicationRequest:
    trial_id: str
    context: Context
    message: Artifact
    wrong_message: Artifact
    wrong_message_rationale: str
    seed_mode: SeedMode
    seeds: tuple[int, ...]
    continuation: Intervention | None = None

    def __post_init__(self) -> None:
        if not self.trial_id or not self.wrong_message_rationale:
            raise ValueError("Trial identity and a wrong-message rationale are required")
        if self.seed_mode not in (SeedMode.FIXED, SeedMode.PAIRED):
            raise ValueError("Communication controls use explicit fixed or paired receiver randomness")
        if not isinstance(self.seeds, tuple) or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Receiver seeds must be immutable, nonempty and unique")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Receiver seeds must be nonnegative integers")
        if (self.seed_mode is SeedMode.FIXED) != (len(self.seeds) == 1):
            raise ValueError("Receiver seed count does not match its coupling mode")
        if self.message.via != self.wrong_message.via or not self.message.via:
            raise ValueError("Correct and wrong messages must come through the same declared interface")
        if self.message.digest == self.wrong_message.digest:
            raise ValueError("A wrong-message control cannot be the correct message")
        shape = [(channel.name, channel.modality, np.shape(channel.data)) for channel in self.message.channels]
        if shape != [(channel.name, channel.modality, np.shape(channel.data)) for channel in self.wrong_message.channels]:
            raise ValueError("A wrong-message control must match the channels and budget of the correct message")


@dataclass(frozen=True)
class CommunicationResult:
    receiver: WorldCard
    request: CommunicationRequest
    without_message: np.ndarray
    with_message: np.ndarray
    with_wrong_message: np.ndarray

    def __post_init__(self) -> None:
        shapes = {np.shape(getattr(self, name)) for name in ("without_message", "with_message", "with_wrong_message")}
        if len(shapes) != 1 or next(iter(shapes))[0] != len(self.request.seeds):
            raise ValueError("Every branch needs one observation per receiver seed")
        for name in ("without_message", "with_message", "with_wrong_message"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("Communication outputs must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class CommunicationRunner:
    """Runs no-message, correct-message and wrong-message branches on forks of one receiver state."""

    def __init__(self, receiver: World, observer: ScopeInterface):
        self.receiver, self.observer = receiver, observer
        self.card = receiver.describe()
        if observer.source != self.card.world_id:
            raise ValueError("The observer must observe the receiving world")

    def _branch(self, root: object, message: Artifact | None, request: CommunicationRequest, seed: int) -> np.ndarray:
        branch = self.receiver.fork(root)
        if branch is root:
            raise ValueError("Message branches must fork isolated state handles")
        handles = [branch]
        try:
            if message is not None:
                require_accepted(self.card, message)
            state = self.receiver.receive(branch, message, seed)
            handles.append(state)
            if request.continuation is not None:
                state = self.receiver.apply(state, request.continuation, seed)
                handles.append(state)
            if any(handle is root for handle in handles):
                raise ValueError("A receiver must not return the shared root state")
            observed = self.observer.transmit(self.receiver.emit(state))
            if self.receiver.describe().digest != self.card.digest:
                raise RuntimeError("Receiver revision changed during a communication trial")
            return self.observer.scope.vector(observed)
        finally:
            closed: set[int] = set()
            for handle in reversed(handles):
                if id(handle) not in closed:
                    closed.add(id(handle))
                    self.receiver.close(handle)

    def run(self, request: CommunicationRequest) -> CommunicationResult:
        outputs: dict[str, list[np.ndarray]] = {"without_message": [], "with_message": [], "with_wrong_message": []}
        for seed in request.seeds:
            root = self.receiver.open(request.context, seed)
            try:
                for name, message in (("without_message", None), ("with_message", request.message),
                                      ("with_wrong_message", request.wrong_message)):
                    outputs[name].append(self._branch(root, message, request, seed))
            finally:
                self.receiver.close(root)
        return CommunicationResult(self.card, request, **{name: np.stack(values) for name, values in outputs.items()})


@dataclass(frozen=True)
class CommunicationReport:
    evaluation_id: str
    receiver_digest: str
    message_digests: tuple[str, ...]
    target_semantics: str
    group_weighted_mse: float
    comparisons: tuple[Comparison, ...]
    claim_boundary: str = "Receiver task utility relative to no and wrong messages; not universal semantic alignment"


def evaluate_communication(
    results: tuple[CommunicationResult, ...], targets: np.ndarray, plan: EvaluationPlan, *, target_semantics: str,
) -> CommunicationReport:
    if not results or not target_semantics or plan.target is not EvaluationTarget.RECEIVER_OBSERVATION:
        raise ValueError("Communication evaluation needs trials and a declared receiver-observation target")
    receiver = results[0].receiver
    if len({result.request.trial_id for result in results}) != len(results):
        raise ValueError("Communication trial identities must be unique")
    if any(result.receiver.digest != receiver.digest or result.request.context.role is not plan.role for result in results):
        raise ValueError("All trials must share the receiver and the evaluation role")
    correct = np.stack([result.with_message.mean(axis=0) for result in results])
    truth = np.asarray(targets, dtype=float)
    if truth.shape != correct.shape or not np.isfinite(truth).all():
        raise ValueError("Receiver targets must match the observed receiver vectors")
    groups = tuple(result.request.context.exchangeability_group for result in results)
    errors = np.mean((truth - correct) ** 2, axis=1)
    comparisons = []
    for name in ("without_message", "with_wrong_message"):
        control = np.stack([getattr(result, name).mean(axis=0) for result in results])
        comparisons.append(summarize_advantage(np.mean((truth - control) ** 2, axis=1) - errors, groups, plan, name, 2))
    return CommunicationReport(plan.evaluation_id, receiver.digest,
                               tuple(result.request.message.digest for result in results), target_semantics,
                               float(grouped_values(errors, groups).mean()), tuple(comparisons))
