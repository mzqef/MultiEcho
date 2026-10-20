"""Backward laws: read the axis value back from an observed effect, or say it is not identifiable.

The confidence set is a cross-fitted conformal set on discovery (Barber et al. 2021,
arXiv:1905.02928); a value is reported only when the set is one interval no wider than the
declared tolerance. The miscoverage level and the tolerance are the caller's declarations.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .block import ResponseBlock
from .code import NoiseModel


@dataclass(frozen=True)
class BackwardResult:
    estimate: float | None
    confidence_set: tuple[tuple[float, float], ...]
    identifiable: bool
    reason: str


def fold_groups(contexts: tuple[str, ...], folds: int) -> list[tuple[str, ...]]:
    count = min(folds, len(contexts))
    if count < 2:
        raise ValueError("Cross-fitting needs at least two contexts")
    return [tuple(contexts[i::count]) for i in range(count)]


def conformal_threshold(fit: Callable[[ResponseBlock], object], block: ResponseBlock, noise: NoiseModel, *,
                        alpha: float, folds: int) -> float:
    """Cross-fitted conformal quantile of the weighted residual at the true value."""
    if not 0 < alpha < 1:
        raise ValueError("Miscoverage level must lie in (0, 1)")
    scores = []
    for held in fold_groups(block.contexts, folds):
        law = fit(block.subset(set(block.contexts) - set(held)))
        for unit in block.subset(held).units:
            mask = np.ones(len(unit.values), dtype=bool)
            if law.calibration_values:
                mask[unit.index(law.calibration_values)] = False
            if not mask.any():
                continue
            residual = unit.effects[mask] - law.predict(law.calibrate(unit), unit.values[mask])
            scores.extend(((residual ** 2) / noise.sigma2).sum(axis=1).tolist())
    if not scores:
        raise ValueError("No uncalibrated discovery observations to score")
    scores = np.sort(np.asarray(scores))
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(scores[min(rank, len(scores)) - 1])


def recover(law, state, effect, noise: NoiseModel, *, threshold: float, tolerance: float,
            points: int = 2001) -> BackwardResult:
    """Grid search plus bounded refinement; ``tolerance`` is half the largest admissible set width."""
    if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Declare a positive conformal threshold and identifiability tolerance")
    effect = np.asarray(effect, dtype=float)
    grid = np.linspace(*law.domain, points)
    scores = (((law.predict(state, grid) - effect) ** 2) / noise.sigma2).sum(axis=1)
    inside = scores <= threshold
    intervals = []
    start = None
    for index, flag in enumerate(inside):
        if flag and start is None:
            start = index
        if (not flag or index == len(grid) - 1) and start is not None:
            stop = index if flag else index - 1
            intervals.append((float(grid[start]), float(grid[stop])))
            start = None
    if not intervals:
        return BackwardResult(None, (), False, "outside-model")
    if len(intervals) > 1:
        return BackwardResult(None, tuple(intervals), False, "non-injective")
    if intervals[0][1] - intervals[0][0] > 2 * tolerance:
        return BackwardResult(None, tuple(intervals), False, "flat")
    best = int(np.argmin(scores))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    refined = golden_section(lambda u: float(((((law.predict(state, [u])[0] - effect) ** 2) / noise.sigma2)).sum()),
                             left, right, tolerance=1e-9 * max(1.0, abs(right - left)))
    return BackwardResult(refined, tuple(intervals), True, "")


def golden_section(function, left: float, right: float, *, tolerance: float) -> float:
    """Minimum of a unimodal function on [left, right] (bounded golden-section search)."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(left), float(right)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = function(c), function(d)
    while b - a > tolerance:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = function(d)
    return (a + b) / 2.0
