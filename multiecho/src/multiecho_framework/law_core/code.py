"""Declared two-part codes and the compression gain ΔL (codes N1 and N2).

ΔL = L(D | zero response) − [L(law) + L(D | law)], with D = every non-baseline observation of
the held-out units. Calibration observations are coded as zero response and the law costs
(k/2)·log2 n + L_struct bits. ΔL is only ever compared between candidate laws of one world on
one data set; it is never a score of the world.

N1 codes residuals under a Gaussian with per-dimension paired-effect noise frozen on discovery (needs
replicates). N2 needs none: residual variances are estimated by maximum likelihood per context and per
declared group of dimensions, in both codes, so the variance parameters cancel and a context contributes
Σ_g (n_g / 2)·log2(RSS0_g / RSS_g) bits. Gains stay additive over contexts, so cross-validation, the
context bootstrap and the indistinguishability test apply unchanged. Fits under N2 weight dimensions equally.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..contracts import DataRole, fingerprint
from .block import ResponseBlock

LN2 = math.log(2.0)
SIGMA_SOURCES = (
    "paired-replicates:discovery",          # F1: same value, resampled appearance/noise (default)
    "observation-pairs:discovery:effect=2x",  # repeated observations of one value; effect variance = 2x
)
ML_SOURCES = ("ml:pooled", "ml:per-location")   # N2: one residual variance per context, or per context and location
CODES = {"N1-gauss-known": SIGMA_SOURCES, "N2-gauss-ml": ML_SOURCES}


@dataclass(frozen=True)
class CodeSpec:
    noise: str = "N1-gauss-known"
    sigma_source: str = "paired-replicates:discovery"
    quantum_rel: float = 0.01
    parameter_n: str = "scalars"
    k_count: str = "effective"

    def __post_init__(self) -> None:
        if self.sigma_source not in CODES.get(self.noise, ()):
            raise ValueError(f"Declared codes and their variance sources: {CODES}")
        if not 0 < self.quantum_rel < 1 or self.parameter_n not in {"scalars", "vectors"}:
            raise ValueError("Invalid quantization or parameter-count convention")
        if self.k_count not in {"effective", "nominal"}:
            raise ValueError("k is counted as effective or nominal degrees of freedom")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class NoiseModel:
    """N1: per-dimension paired-effect variance, estimated on discovery only and then frozen.
    N2: unit fitting weights, the group of every dimension and the declared quantization floor (residual
    variances are estimated per context when a law is coded)."""

    sigma2: np.ndarray = field(repr=False)
    pairs: int = 0
    source: str = ""
    code: str = "N1"
    groups: np.ndarray | None = field(default=None, repr=False)
    floor: float = 0.0

    def __post_init__(self) -> None:
        sigma2 = np.array(self.sigma2, dtype=float, copy=True)
        if sigma2.ndim != 1 or not (sigma2 > 0).all() or not np.isfinite(sigma2).all():
            raise ValueError("Noise variances are positive and finite")
        sigma2.setflags(write=False)
        object.__setattr__(self, "sigma2", sigma2)
        if self.code not in {"N1", "N2"}:
            raise ValueError("The noise model belongs to code N1 or N2")
        if self.code == "N2":
            groups = np.array(self.groups, dtype=np.int64, copy=True)
            if groups.shape != sigma2.shape or groups.min() < 0 or not np.isfinite(self.floor) or self.floor <= 0:
                raise ValueError("N2 declares a group for every dimension and a positive quantization floor")
            groups.setflags(write=False)
            object.__setattr__(self, "groups", groups)
        elif self.groups is not None:
            raise ValueError("Only N2 groups dimensions")

    @classmethod
    def for_ml(cls, block: ResponseBlock, code: CodeSpec, groups=None) -> NoiseModel:
        """N2 on discovery: `groups` labels the dimensions by location for `ml:per-location`. The floor is
        (q²/12) per scalar with q = quantum_rel × the RMS discovery effect."""
        if code.noise != "N2-gauss-ml":
            raise ValueError("Declare the N2 code")
        if block.role is not DataRole.DISCOVERY:
            raise ValueError("The noise model is frozen on discovery only")
        if code.sigma_source == "ml:pooled":
            labels = np.zeros(block.dimension, dtype=np.int64)
        else:
            if groups is None:
                raise ValueError("Per-location variances need the location of every dimension")
            labels = np.unique(np.asarray(groups), return_inverse=True)[1].astype(np.int64)
        mean_square = float(np.mean([np.mean(unit.effects ** 2) for unit in block.units]))
        quantum = code.quantum_rel * math.sqrt(mean_square) if mean_square > 0 else 1.0
        return cls(np.ones(block.dimension), 0, f"{code.sigma_source}:{block.digest}", "N2", labels,
                   max(quantum ** 2 / 12, np.finfo(float).tiny))

    @classmethod
    def from_replicates(cls, block: ResponseBlock, code: CodeSpec) -> NoiseModel:
        if block.role is not DataRole.DISCOVERY:
            raise ValueError("The noise model is frozen on discovery only")
        total, count = np.zeros(block.dimension), 0
        by_context: dict[str, list] = {}
        for unit in block.units:
            by_context.setdefault(unit.context, []).append(unit)
        for units in by_context.values():
            for first, second in zip(units, units[1:], strict=False):
                shared = np.intersect1d(first.values, second.values)
                if not shared.size:
                    continue
                difference = first.effects[first.index(shared)] - second.effects[second.index(shared)]
                total += (difference ** 2 / 2).sum(axis=0)
                count += shared.size
        if not count:
            raise ValueError("Paired-effect noise needs at least one context with two replicates")
        return cls._floored(total / count, count, block.digest, code)

    @classmethod
    def from_observation_pairs(cls, observations: np.ndarray, code: CodeSpec, *, source: str) -> NoiseModel:
        """Observations of shape (contexts, repeats, values, D) at fixed values; effect variance = 2x.

        Used when only repeated observations (not resampled baselines) exist; the factor two assumes the
        branch and the baseline observation carry independent noise of the same size. The source must
        be discovery data and is declared in the code specification.
        """
        if code.sigma_source != "observation-pairs:discovery:effect=2x":
            raise ValueError("Declare the observation-pair noise source in the code specification")
        data = np.asarray(observations, dtype=float)
        if data.ndim != 4 or data.shape[1] < 2:
            raise ValueError("Observation pairs need (contexts, repeats >= 2, values, D)")
        total, count = np.zeros(data.shape[-1]), 0
        for first in range(data.shape[1]):
            for second in range(first + 1, data.shape[1]):
                difference = data[:, first] - data[:, second]
                total += (difference ** 2 / 2).sum(axis=(0, 1))
                count += difference.shape[0] * difference.shape[1]
        return cls._floored(2.0 * total / count, count, source, code)

    @classmethod
    def _floored(cls, sigma2, count, source, code):
        positive = np.sqrt(sigma2[sigma2 > 0])
        quantum = code.quantum_rel * float(np.median(positive)) if positive.size else 1.0
        floor = max(quantum ** 2 / 12, np.finfo(float).tiny)
        return cls(np.maximum(sigma2, floor), count, source)

    @property
    def digest(self) -> str:
        record = {"sigma2": np.round(self.sigma2, 15).tolist(), "pairs": self.pairs, "source": self.source}
        if self.code == "N2":
            record |= {"code": "N2", "groups": fingerprint(self.groups.tolist()), "floor": self.floor}
        return fingerprint(record)


@dataclass(frozen=True)
class DeltaL:
    value: float
    data_gain_bits: float
    law_bits: float
    k: float
    n: int
    per_context: dict = field(repr=False)
    points_coded: int = 0
    calibration_points: int = 0
    code: str = ""

    def interval(self, *, confidence: float, resamples: int, seed: int) -> tuple[float, float]:
        """Cluster bootstrap over held-out contexts; confidence and resampling budget are the caller's."""
        gains = np.array(list(self.per_context.values()), dtype=float)
        if gains.size < 2:
            return (self.value, self.value)
        rng = np.random.default_rng(seed)
        draws = rng.choice(gains, size=(resamples, gains.size), replace=True).sum(axis=1) - self.law_bits
        tail = (1 - confidence) / 2
        lower, upper = np.quantile(draws, [tail, 1 - tail])
        return (float(lower), float(upper))


def data_gain(law, block: ResponseBlock, noise: NoiseModel) -> tuple[dict[str, float], int, int]:
    """Per-context bits saved by the law on the block's uncalibrated observations."""
    if noise.code == "N2":
        return _ml_gain(law, block, noise)
    weights = 1.0 / (2.0 * noise.sigma2 * LN2)
    per_context: dict[str, float] = {}
    coded = calibrated = 0
    for unit in block.units:
        state = law.calibrate(unit)
        mask = ~law.calibration_mask(unit)
        y = unit.effects[mask]
        prediction = law.predict(state, unit.values[mask]) if mask.any() else np.zeros_like(y)
        gain = float(((y ** 2 - (y - prediction) ** 2) * weights).sum())
        per_context[unit.context] = per_context.get(unit.context, 0.0) + gain
        coded += int(mask.sum())
        calibrated += int((~mask).sum())
    return per_context, coded, calibrated


def _ml_gain(law, block: ResponseBlock, noise: NoiseModel) -> tuple[dict[str, float], int, int]:
    """N2: Σ_g (n_g/2)·log2(RSS0_g/RSS_g) per context, the calibration observations coded as zero response
    in both codes; residual sums are floored at n_g times the declared quantization variance."""
    labels = noise.groups
    count = int(labels.max()) + 1
    width = np.bincount(labels, minlength=count)
    sums: dict[str, list[np.ndarray]] = {}
    coded = calibrated = 0
    for unit in block.units:
        state = law.calibrate(unit)
        mask = ~law.calibration_mask(unit)
        prediction = np.zeros_like(unit.effects)
        if mask.any():
            prediction[mask] = law.predict(state, unit.values[mask])
        entry = sums.setdefault(unit.context, [np.zeros(count), np.zeros(count), np.zeros(count)])
        entry[0] += np.bincount(labels, weights=(unit.effects ** 2).sum(axis=0), minlength=count)
        entry[1] += np.bincount(labels, weights=((unit.effects - prediction) ** 2).sum(axis=0), minlength=count)
        entry[2] += width * len(unit.values)
        coded += int(mask.sum())
        calibrated += int((~mask).sum())
    per_context = {}
    for context, (zero, law_rss, n) in sums.items():
        used = n > 0
        floor = n[used] * noise.floor
        ratio = np.maximum(zero[used], floor) / np.maximum(law_rss[used], floor)
        per_context[context] = float((n[used] / 2 * np.log2(ratio)).sum())
    return per_context, coded, calibrated


def law_bits(law) -> float:
    return 0.5 * law.k * math.log2(max(law.n_fit, 2)) + law.struct_bits


def delta_l(law, block: ResponseBlock, noise: NoiseModel) -> DeltaL:
    """ΔL of a frozen law on held-out units; calibration observations contribute zero by construction."""
    if block.role is DataRole.DISCOVERY:
        raise ValueError("ΔL is evaluated on held-out units; use cross-validated ΔL on discovery")
    per_context, coded, calibrated = data_gain(law, block, noise)
    cost = law_bits(law)
    gain = float(sum(per_context.values()))
    return DeltaL(gain - cost, gain, cost, float(law.k), int(law.n_fit), per_context, coded, calibrated,
                  law.code_digest)
