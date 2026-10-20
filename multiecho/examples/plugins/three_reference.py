"""Three-reference estimator plugins: PL, linear and kernel (the MultiEcho paper's local estimators).

Each context supplies three labeled references: the reference observation at u0 and the
two calibration endpoints u- < u0 < u+. A coordinate c (the raw value, or its log for
positive scale parameters) is normalized to positions p = (c - c-) / (c+ - c-), so the
references sit at 0, m = (c0 - c-) / (c+ - c-) and 1. Every estimator predicts a response
as fixed weights over the three references, Y(p) = w(p) @ [Y-, Y0, Y+], with weights that
sum to one and reproduce p:

- `pl`: piecewise-linear interpolation between neighbouring references;
- `linear`: least-squares affine fit a + b p through the three references;
- `kernel`: Gaussian RBF interpolation with an affine tail (scipy `RBFInterpolator`,
  degree 1, zero smoothing). Its width epsilon is chosen on discovery contexts with the
  paper's loss: per-context squared error over query energy, averaged over contexts.

Forward predicts the effect w-(p) dY- + w+(p) dY+. Backward recovers the position on a
fine grid whose forward prediction is closest to the observed effect, then maps it back
to the intervention value. Both are interpolation laws: queries outside [u-, u+] are
outside their validity domain and raise. These are analysis-method plugins; the
framework ships no estimator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from multiecho_framework import ContextView, Intervention, LawSpec, ResponseSet
from multiecho_framework.contracts import fingerprint

METHODS = ("pl", "linear", "kernel")
EPSILONS = (0.3, 1.0, 3.0)
COORDINATES = ("raw", "log")
GRID = np.linspace(0.0, 1.0, 1001)


def weights(method: str, positions, middle: float = 0.5, epsilon: float = 1.0) -> np.ndarray:
    """(Q, 3) weights over [Y-, Y0, Y+] for references at normalized positions (0, middle, 1)."""
    positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    if not 0.0 < middle < 1.0:
        raise ValueError("The reference must lie strictly between the two calibration endpoints")
    if np.any(positions < -1e-9) or np.any(positions > 1 + 1e-9):
        raise ValueError("Three-reference laws interpolate between the calibration endpoints only")
    knots, identity = np.array([0.0, middle, 1.0]), np.eye(3)
    if method == "pl":
        out = np.column_stack([np.interp(positions, knots, identity[:, k]) for k in range(3)])
    elif method == "linear":
        out = np.column_stack([np.ones_like(positions), positions]) @ np.linalg.pinv(
            np.column_stack([np.ones(3), knots]))
    elif method == "kernel":
        from scipy.interpolate import RBFInterpolator

        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("The kernel width must be positive")
        out = RBFInterpolator(knots[:, None], identity, kernel="gaussian", epsilon=epsilon, degree=1,
                              smoothing=0.0)(positions[:, None])
    else:
        raise ValueError(f"Unknown three-reference method {method!r}; choose from {METHODS}")
    np.testing.assert_allclose(out.sum(axis=1), 1, atol=1e-9)
    np.testing.assert_allclose(out @ knots, positions, atol=1e-9)
    return out


def predict_effects(method: str, low: np.ndarray, high: np.ndarray, positions, middle: float = 0.5,
                    epsilon: float = 1.0) -> np.ndarray:
    """Effects at `positions` from endpoint effects: low, high (..., D) -> (..., Q, D)."""
    w = weights(method, positions, middle, epsilon)
    return w[:, 0, None] * np.asarray(low)[..., None, :] + w[:, 2, None] * np.asarray(high)[..., None, :]


def recover_positions(method: str, low: np.ndarray, high: np.ndarray, effects: np.ndarray, middle: float = 0.5,
                      epsilon: float = 1.0) -> np.ndarray:
    """Positions (C, Q) whose predicted effect is closest to observed effects (C, Q, D); low, high (C, D).

    A context whose endpoint effects are both zero recovers the reference position.
    """
    low, high, effects = (np.asarray(x, dtype=np.float64) for x in (low, high, effects))
    w = weights(method, GRID, middle, epsilon)
    a, b = w[:, 0], w[:, 2]
    ll, hh, lh = (np.einsum("cd,cd->c", x, y) for x, y in ((low, low), (high, high), (low, high)))
    ly, hy = np.einsum("cd,cqd->cq", low, effects), np.einsum("cd,cqd->cq", high, effects)
    model = a[None] ** 2 * ll[:, None] + 2 * a[None] * b[None] * lh[:, None] + b[None] ** 2 * hh[:, None]
    distance = model[:, None] - 2 * (ly[..., None] * a[None, None] + hy[..., None] * b[None, None])
    best = GRID[np.argmin(distance, axis=-1)]
    return np.where((ll + hh)[:, None] > 0, best, middle)


def selection_loss(method: str, low: np.ndarray, high: np.ndarray, positions: list[np.ndarray],
                   middles: list[float], truth: list[np.ndarray], epsilon: float) -> float:
    """The paper's loss: per-context SSE over query energy, averaged; exact zero responses cost zero."""
    losses = []
    for i in range(len(truth)):
        error = float(np.sum((predict_effects(method, low[i], high[i], positions[i], middles[i], epsilon)
                              - truth[i]) ** 2))
        energy = float(np.sum(truth[i] ** 2))
        losses.append(error / energy if energy > 0 else (0.0 if error == 0 else math.inf))
    return float(np.mean(losses))


def _coordinate(value: float, coordinate: str) -> float:
    if coordinate == "log":
        if value <= 0:
            raise ValueError("A log coordinate needs positive intervention values")
        return math.log(value)
    return float(value)


def _value(position_coordinate: float, coordinate: str) -> float:
    return math.exp(position_coordinate) if coordinate == "log" else float(position_coordinate)


@dataclass(frozen=True)
class _Geometry:
    low: np.ndarray
    high: np.ndarray
    lower: float
    width: float
    middle: float

    def position(self, value: float, coordinate: str) -> float:
        return (_coordinate(value, coordinate) - self.lower) / self.width


def _geometry(view: ContextView, coordinate: str) -> _Geometry:
    if len(view.reference.values) != 1 or len(view.calibration) != 2:
        raise ValueError("Three-reference laws need a one-dimensional handle and exactly two calibration branches")
    (first, second), u0 = sorted(view.calibration, key=lambda item: item.intervention.values[0]), view.reference.values[0]
    lower, upper = (_coordinate(item.intervention.values[0], coordinate) for item in (first, second))
    width = upper - lower
    if width <= 0:
        raise ValueError("Calibration values must be distinct")
    middle = (_coordinate(u0, coordinate) - lower) / width
    return _Geometry(np.asarray(first.response, dtype=np.float64), np.asarray(second.response, dtype=np.float64),
                     lower, width, middle)


def choose_epsilon(method: str, data: ResponseSet, coordinate: str) -> tuple[float | None, dict]:
    """Kernel width chosen on discovery contexts; None for methods without a width."""
    if method != "kernel":
        return None, {}
    lows, highs, positions, middles, truth = [], [], [], [], []
    for view, queries in zip(data.views, data.queries, strict=True):
        geometry = _geometry(view, coordinate)
        lows.append(geometry.low)
        highs.append(geometry.high)
        middles.append(geometry.middle)
        positions.append(np.array([geometry.position(q.intervention.values[0], coordinate) for q in queries]))
        truth.append(np.stack([q.response for q in queries]))
    risks = {e: selection_loss(method, np.stack(lows), np.stack(highs), positions, middles, truth, e) for e in EPSILONS}
    return min(EPSILONS, key=lambda e: (risks[e], e)), {str(k): v for k, v in risks.items()}


def _spec(task: str, method: str, coordinate: str, epsilon: float | None, risks: dict, data: ResponseSet) -> LawSpec:
    hypothesis = {
        "pl": "Responses vary piecewise-linearly between the three references of a context",
        "linear": "Responses vary affinely in the coordinate, least squares through the three references",
        "kernel": f"Responses follow a Gaussian RBF interpolant with affine tail (epsilon {epsilon}) of the references",
    }[method]
    return LawSpec(f"three-reference-{method}-{task}",
                   fingerprint({"method": method, "task": task, "coordinate": coordinate, "epsilon": epsilon,
                                "selection_risks": risks}),
                   data.world, hypothesis + f"; {coordinate} coordinate",
                   "Queries between the two calibration endpoints of each context", data.contexts)


@dataclass(frozen=True)
class ThreeReferenceForwardLaw:
    spec: LawSpec
    method: str
    coordinate: str
    epsilon: float

    def describe(self) -> LawSpec:
        return self.spec

    def predict(self, view: ContextView, intervention: Intervention) -> np.ndarray:
        geometry = _geometry(view, self.coordinate)
        position = geometry.position(intervention.values[0], self.coordinate)
        return predict_effects(self.method, geometry.low, geometry.high, [position], geometry.middle, self.epsilon)[0]


@dataclass(frozen=True)
class ThreeReferenceBackwardLaw:
    spec: LawSpec
    method: str
    coordinate: str
    epsilon: float

    def describe(self) -> LawSpec:
        return self.spec

    def recover(self, view: ContextView, response: np.ndarray) -> tuple[float, ...]:
        geometry = _geometry(view, self.coordinate)
        position = recover_positions(self.method, geometry.low[None], geometry.high[None],
                                     np.asarray(response, dtype=np.float64)[None, None], geometry.middle,
                                     self.epsilon)[0, 0]
        return (_value(geometry.lower + float(position) * geometry.width, self.coordinate),)


@dataclass(frozen=True)
class ThreeReferenceForward:
    """Forward method plugin: `method` in ('pl', 'linear', 'kernel'), `coordinate` in ('raw', 'log')."""

    method: str = "pl"
    coordinate: str = "raw"

    def __post_init__(self) -> None:
        if self.method not in METHODS or self.coordinate not in COORDINATES:
            raise ValueError(f"Choose a method from {METHODS} and a coordinate from {COORDINATES}")

    def fit(self, data: ResponseSet) -> ThreeReferenceForwardLaw:
        epsilon, risks = choose_epsilon(self.method, data, self.coordinate)
        return ThreeReferenceForwardLaw(_spec("forward", self.method, self.coordinate, epsilon, risks, data),
                                        self.method, self.coordinate, epsilon or 1.0)


@dataclass(frozen=True)
class ThreeReferenceBackward:
    """Backward method plugin; the kernel width is chosen on discovery with the forward selection loss."""

    method: str = "pl"
    coordinate: str = "raw"

    def __post_init__(self) -> None:
        if self.method not in METHODS or self.coordinate not in COORDINATES:
            raise ValueError(f"Choose a method from {METHODS} and a coordinate from {COORDINATES}")

    def fit(self, data: ResponseSet) -> ThreeReferenceBackwardLaw:
        epsilon, risks = choose_epsilon(self.method, data, self.coordinate)
        return ThreeReferenceBackwardLaw(_spec("backward", self.method, self.coordinate, epsilon, risks, data),
                                         self.method, self.coordinate, epsilon or 1.0)


__all__ = ["COORDINATES", "EPSILONS", "GRID", "METHODS", "ThreeReferenceBackward", "ThreeReferenceBackwardLaw",
           "ThreeReferenceForward", "ThreeReferenceForwardLaw", "choose_epsilon", "predict_effects",
           "recover_positions", "selection_loss", "weights"]
