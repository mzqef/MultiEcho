"""World-general law channel (docs/protocol-0.4-draft.md, channels): a frozen shared-profile law of world A sent to world B.

The message is the law's shared profiles; the receiver calibrates them on its own observations at the law's
calibration values and predicts the rest. Profiles are pinned to zero at the receiver's baseline (the request
declares it when it differs from the law's), and a receiver that lacks a calibration value because it is its
own baseline calibrates on the values it has. Consistency is judged on held-out data of B against two controls
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
        names = {channel.name for channel in artifact.channels}
        anchor = float(np.asarray(artifact.channel("anchor").data).reshape(-1)[0]) if "anchor" in names else None
        values = (np.asarray(artifact.channel("calibration_at").data, dtype=float).reshape(-1) if "calibration_at" in names
                  else np.asarray(self.law.calibration_values))
        profiles = self.law.profiles(values, anchor)
        directions = (np.linalg.solve(profiles, calibration) if profiles.shape[0] == profiles.shape[1]
                      else np.linalg.lstsq(profiles, calibration, rcond=None)[0])
        prediction = self.law.profiles(queries, anchor) @ directions
        return Artifact(artifact.source, (Channel("effect", Modality.TENSOR, prediction.reshape(1, -1)),),
                        parent=artifact.digest, via=self.describe().digest)


def _shuffled(law: SharedProfile) -> SharedProfile:
    """Wrong-message control: the same profile coefficients with the knot order reversed."""
    parameters = dict(law.parameters)
    parameters["profile"] = np.asarray(law.parameters["profile"])[::-1].copy()
    return replace(law, parameters=parameters, variant=law.variant + "-reversed")


def _calibration_rows(law: SharedProfile, unit, anchor: float) -> np.ndarray:
    """The receiver's observations at the law's calibration values; only its own baseline may be absent."""
    rows = np.zeros(len(unit.values), dtype=bool)
    for value in law.calibration_values:
        hits = np.flatnonzero(np.isclose(unit.values, value, rtol=0, atol=1e-12))
        if hits.size:
            rows[hits[0]] = True
        elif not np.isclose(value, anchor, rtol=0, atol=1e-12):
            raise ValueError(f"Unit {unit.context}/r{unit.replicate} lacks calibration value {value!r}")
    return rows


def law_transfer(source_law: SharedProfile, source: str, target_block: ResponseBlock, target: str, *,
                 minimum_improvement: float, resamples: int, seed: int):
    """Consistency of a frozen shared-profile law on held-out units of another block, against the no-law and
    profile-reversed controls. Used across worlds (`law_channel`) and, within one world, across observation
    conditions (a reuse reading, not a channel). Queries are the uncalibrated values every receiving unit has,
    so per-unit baselines that remove a grid value from some units keep the comparison aligned."""
    interface = ProfileTransfer(source_law, source, target)
    wrong = ProfileTransfer(_shuffled(source_law), source, target)
    scope = Scope("effect", ("effect",))
    shared = set.intersection(*(set(np.round(unit.values, 12).tolist()) for unit in target_block.units))
    alignments, zero, wrong_predictions = [], [], []
    for unit in target_block.units:
        anchor = target_block.anchor(unit)
        calibrated = _calibration_rows(source_law, unit, anchor)
        mask = ~calibrated & np.isin(np.round(unit.values, 12), sorted(shared))
        channels = [Channel("calibration", Modality.TENSOR, unit.effects[calibrated]),
                    Channel("queries", Modality.TENSOR, unit.values[mask])]
        if source_law.baseline is None or not np.isclose(anchor, source_law.baseline, rtol=0, atol=1e-12):
            channels.append(Channel("anchor", Modality.TENSOR, np.array([anchor])))
        if calibrated.sum() != len(source_law.calibration_values):
            channels.append(Channel("calibration_at", Modality.TENSOR, unit.values[calibrated]))
        request = Artifact(target, tuple(channels))
        message = interface.transmit(request)
        observed = Artifact(target, (Channel("effect", Modality.TENSOR, unit.effects[mask].reshape(1, -1)),))
        alignments.append(Alignment(f"{unit.context}/r{unit.replicate}", message, observed, unit.context,
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
    return interface, plan, report


def law_channel(source_law: SharedProfile, source_world: str, target_block: ResponseBlock, target_world: str, *,
                source_kind: WorldKind, target_kind: WorldKind, minimum_improvement: float, resamples: int,
                seed: int):
    interface, plan, report = law_transfer(source_law, source_world, target_block, target_world,
                                           minimum_improvement=minimum_improvement, resamples=resamples, seed=seed)
    evidence = ChannelEvidence.from_translation(report, plan)
    record = ChannelRecord(f"law:{source_world[:12]}->{target_world[:12]}:{target_block.scope.axis}",
                           ContentKind.LAW, source_world, WorldKind(source_kind), target_world, WorldKind(target_kind),
                           interface.describe().digest, (DataRole.DISCOVERY,), (evidence,),
                           content_schema="multiecho.law-core/shared-profile")
    return record, report
