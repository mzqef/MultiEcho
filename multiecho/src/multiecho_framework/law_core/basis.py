"""Cubic B-spline profile basis through (u0, 0), with an integrated curvature penalty."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bspline_design(knots: np.ndarray, x: np.ndarray, *, degree: int = 3, derivative: int = 0) -> np.ndarray:
    """All B-spline basis functions (or their derivatives) at x; shape (len(x), len(knots) − degree − 1).

    Cox–de Boor recursion on the knot interval containing x, clamped to the first/last interval, so values
    outside the knot span extrapolate the end polynomials (as `scipy.interpolate.BSpline(extrapolate=True)`).
    """
    knots = np.asarray(knots, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    count = len(knots) - degree - 1
    interval = np.clip(np.searchsorted(knots, x, side="right") - 1, degree, count - 1)

    def ratio(numerator, denominator):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), 0.0)

    def basis(order: int, nu: int) -> np.ndarray:
        if order == 0:
            values = np.zeros((len(x), len(knots) - 1))
            values[np.arange(len(x)), interval] = 1.0
            return values
        lower = basis(order - 1, nu - 1 if nu else 0)
        left, right = knots[:-(order + 1)], knots[order:-1]
        left_next, right_next = knots[1:len(knots) - order], knots[order + 1:]
        if nu:
            return order * (ratio(lower[:, :-1], (right - left)[None]) - ratio(lower[:, 1:], (right_next - left_next)[None]))
        return (ratio((x[:, None] - left[None]) * lower[:, :-1], (right - left)[None])
                + ratio((right_next[None] - x[:, None]) * lower[:, 1:], (right_next - left_next)[None]))

    return basis(degree, derivative)


@dataclass(frozen=True)
class ProfileBasis:
    """ψ(u) = β(u) − β(u0); knots at the base-grid points; one column dropped (the β sum to one)."""

    domain: tuple[float, float]
    knots_at: np.ndarray = field(repr=False)
    baseline: float = 0.0

    def __post_init__(self) -> None:
        lower, upper = self.domain
        interior = np.array(sorted({float(v) for v in self.knots_at if lower < v < upper}), dtype=float)
        knots = np.concatenate([[lower] * 4, interior, [upper] * 4])
        object.__setattr__(self, "knots_at", interior)
        object.__setattr__(self, "_knots", knots)
        anchor = bspline_design(knots, np.array([self.baseline]))[0]
        object.__setattr__(self, "_anchor", anchor)
        nodes, weights = np.polynomial.legendre.leggauss(8)
        edges = np.concatenate([[lower], interior, [upper]])
        penalty = np.zeros((self.size, self.size))
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            x = (right - left) / 2 * nodes + (right + left) / 2
            values = bspline_design(knots, x, derivative=2)[:, :-1]
            penalty += (values * (weights * (right - left) / 2)[:, None]).T @ values
        object.__setattr__(self, "penalty", penalty)

    @property
    def size(self) -> int:
        return len(self.knots_at) + 3

    def __call__(self, values, anchor: float | None = None) -> np.ndarray:
        """ψ(u) = β(u) − β(anchor) with the declared baseline as default anchor.

        The coefficient space does not depend on the anchor: coefficients c describe g(u) = Σ c_j β_j(u),
        and the basis anchored at a evaluates g(u) − g(a). Units with their own baselines therefore share
        one profile and differ only in where it is pinned to zero.
        """
        values = np.atleast_1d(np.asarray(values, dtype=float))
        offset = self._anchor if anchor is None else bspline_design(self._knots, np.array([float(anchor)]))[0]
        return (bspline_design(self._knots, values) - offset)[:, :-1]
