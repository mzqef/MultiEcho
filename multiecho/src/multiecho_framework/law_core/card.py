"""Law cards (PROTOCOL R-20; docs/protocol-0.4-draft.md): a law as a scoped, sealed reading."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..contracts import DataRole
from ..law_cards import LawCard
from .block import LawScope, ResponseBlock
from .code import CodeSpec, NoiseModel, delta_l
from .families import FittedLaw


@dataclass(frozen=True)
class DiscoverySummary:
    """What a law card needs from the discovery block, so a frozen fit can issue cards without it."""

    scope: LawScope
    digest: str
    dimension: int
    values: tuple[float, ...]
    means: np.ndarray = field(repr=False)

    @classmethod
    def of(cls, block: ResponseBlock | DiscoverySummary) -> DiscoverySummary:
        if isinstance(block, DiscoverySummary):
            return block
        if block.role is not DataRole.DISCOVERY:
            raise ValueError("Cards come from a discovery fit and a held-out evaluation")
        reference: dict[float, list[np.ndarray]] = {}
        for unit in block.units:
            for value, effect in zip(unit.values, unit.effects, strict=True):
                reference.setdefault(round(float(value), 12), []).append(effect)
        values = tuple(sorted(reference))
        means = np.stack([np.mean(reference[value], axis=0) for value in values])
        return cls(block.scope, block.digest, block.dimension, values, means)


def forward_r2(law: FittedLaw, heldout: ResponseBlock, discovery: ResponseBlock | DiscoverySummary) -> float | None:
    """Held-out forward R² on uncalibrated points, relative to the discovery mean effect at each value."""
    summary = DiscoverySummary.of(discovery)
    reference = dict(zip(summary.values, summary.means, strict=True))
    residual = total = 0.0
    for unit in heldout.units:
        mask = ~law.calibration_mask(unit)
        if not mask.any():
            continue
        prediction = law.predict(law.calibrate(unit), unit.values[mask])
        for value, observed, predicted in zip(unit.values[mask], unit.effects[mask], prediction, strict=True):
            mean = reference.get(round(float(value), 12), np.zeros_like(observed))
            residual += float(((observed - predicted) ** 2).sum())
            total += float(((observed - mean) ** 2).sum())
    return None if total == 0 else 1.0 - residual / total



def law_card(law: FittedLaw, scope: LawScope, discovery: ResponseBlock | DiscoverySummary, heldout: ResponseBlock,
             noise: NoiseModel, code: CodeSpec, *, confidence: float, resamples: int, seed: int,
             backward: dict | None = None, cv_delta_l: float | None = None) -> LawCard:
    discovery = DiscoverySummary.of(discovery)
    if heldout.role not in {DataRole.VALIDATION, DataRole.CONFIRMATION}:
        raise ValueError("Cards come from a discovery fit and a held-out evaluation")
    if discovery.scope != scope or heldout.scope != scope:
        raise ValueError("Discovery, held-out data and card share one scope")
    gain = delta_l(law, heldout, noise)
    interval = gain.interval(confidence=confidence, resamples=resamples, seed=seed)
    found = gain.value > 0 and law.family >= 0
    plugin_set = scope.plugin_set or "unversioned"
    statement = (f"{law.form} on axis {scope.axis} of world {scope.world[:16]} (input family "
                 f"{scope.observation}), ΔL = {gain.value:.1f} bits on {heldout.role.value} contexts"
                 if found else f"No law with ΔL > 0 found under plugin set {plugin_set} for this scope")
    dimension = discovery.dimension
    return LawCard(
        law_id=f"me2:family{law.family}:{law.variant}@{law.code_digest[:12]}" if law.family >= 0 else "me2:zero",
        family=law.family, form=law.form, scope=asdict(scope),
        status="found" if found else "not_found", statement=statement,
        parameters={"shared": law.k, "per_context": len(law.calibration_values) * dimension, "per_time": 0,
                    "calibration_values": list(law.calibration_values)},
        reuse={"context": {"role": heldout.role.value, "contexts": len(heldout.contexts),
                           "delta_l": gain.value, "interval": list(interval), "confidence": confidence,
                           "forward_r2": forward_r2(law, heldout, discovery), "cv_delta_l": cv_delta_l,
                           "backward": backward},
               "cross_world": []},
        delta_l={"value": gain.value, "data_gain_bits": gain.data_gain_bits, "law_bits": gain.law_bits,
                 "k": gain.k, "n": gain.n, "points_coded": gain.points_coded,
                 "calibration_points": gain.calibration_points, "code": asdict(code), "noise": noise.digest},
        coordinates=(scope.coordinate,),
        readout_dependence={"readout": scope.readout, "mask": scope.mask_reading, "reduce": scope.reduce_reading},
        evidence={"discovery": discovery.digest, "heldout": heldout.digest, "law": law.code_digest})
