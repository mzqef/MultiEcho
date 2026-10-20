"""World-general law channel (docs/protocol-0.4-draft.md, channels): a frozen shared-profile law of world A sent to world B.

The message is the law's shared profiles; the receiver calibrates them on its own observations at the law's
calibration values and predicts the rest. Consistency is judged on held-out data of B against two controls
(no law, a profile-reversed wrong law). The study declares the evaluation levels and the world kinds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..artifacts import Artifact, Channel, Modality, Scope
from ..channels import ChannelEvidence, ChannelRecord, ContentKind
from ..communication import Alignment, evaluate_translation
from ..contracts import DataRole
from ..evaluation import EvaluationPlan, EvaluationTarget
from ..interfaces import InterfaceSpec
from ..worlds import WorldKind
from .block import ResponseBlock
from .families import SharedProfile


@dataclass(frozen=True)
class ProfileTransfer:
    """Law channel: world A's frozen shared profiles, calibrated in world B at A's calibration values."""

    law: SharedProfile
    source: str
    target: str

    def describe(self) -> InterfaceSpec:
        return InterfaceSpec("law-transfer:shared-profile", self.law.code_digest, self.source, self.target,
                             "law-translator", "Shared profiles of the source world, calibrated in the target world")

    def transmit(self, artifact: Artifact) -> Artifact:
        calibration = artifact.channel("calibration").data
        queries = artifact.channel("queries").data
        profiles = self.law.profiles(np.asarray(self.law.calibration_values))
        directions = np.linalg.solve(profiles, calibration)
        prediction = self.law.profiles(queries) @ directions
        return Artifact(artifact.source, (Channel("effect", Modality.TENSOR, prediction.reshape(1, -1)),),
                        parent=artifact.digest, via=self.describe().digest)


def _shuffled(law: SharedProfile) -> SharedProfile:
    """Wrong-message control: the same profile coefficients with the knot order reversed."""
    parameters = dict(law.parameters)
    parameters["profile"] = np.asarray(law.parameters["profile"])[::-1].copy()
    return replace(law, parameters=parameters, variant=law.variant + "-reversed")


def law_channel(source_law: SharedProfile, source_world: str, target_block: ResponseBlock, target_world: str, *,
                source_kind: WorldKind, target_kind: WorldKind, minimum_improvement: float, resamples: int,
                seed: int):
    interface = ProfileTransfer(source_law, source_world, target_world)
    wrong = ProfileTransfer(_shuffled(source_law), source_world, target_world)
    scope = Scope("effect", ("effect",))
    alignments, zero, wrong_predictions = [], [], []
    for unit in target_block.units:
        mask = ~np.isin(unit.values, source_law.calibration_values)
        request = Artifact(target_world, (
            Channel("calibration", Modality.TENSOR, unit.effects[unit.index(source_law.calibration_values)]),
            Channel("queries", Modality.TENSOR, unit.values[mask])))
        message = interface.transmit(request)
        target = Artifact(target_world, (Channel("effect", Modality.TENSOR, unit.effects[mask].reshape(1, -1)),))
        alignments.append(Alignment(f"{unit.context}/r{unit.replicate}", message, target, unit.context,
                                    "same context, same axis values, effects relative to the same baseline"))
        zero.append(np.zeros(unit.effects[mask].size))
        try:
            wrong_predictions.append(wrong.transmit(request).channel("effect").data.reshape(-1))
        except np.linalg.LinAlgError:
            wrong_predictions.append(np.zeros(unit.effects[mask].size))
    plan = EvaluationPlan(f"consistency:{source_law.code_digest[:12]}", EvaluationTarget.MESSAGE_AGREEMENT,
                          target_block.role, minimum_improvement=minimum_improvement,
                          resamples=resamples, seed=seed)
    report = evaluate_translation(interface, tuple(alignments), scope,
                                  {"no-law": np.stack(zero), "reversed-profile": np.stack(wrong_predictions)}, plan)
    evidence = ChannelEvidence.from_translation(report, plan)
    record = ChannelRecord(f"law:{source_world[:12]}->{target_world[:12]}:{target_block.scope.axis}",
                           ContentKind.LAW, source_world, WorldKind(source_kind), target_world, WorldKind(target_kind),
                           interface.describe().digest, (DataRole.DISCOVERY,), (evidence,),
                           content_schema="multiecho.law-core/shared-profile")
    return record, report
