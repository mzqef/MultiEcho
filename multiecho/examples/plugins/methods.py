"""Analysis-method plugins: generic in-context interpolation, not estimators of any paper.

Forward: piecewise-linear interpolation between the labeled calibration responses of
the same context. Backward: projection of a response onto that polyline. Neither
method learns parameters; `fit` only records the discovery contexts it was frozen on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiecho_framework import ContextView, Intervention, LawSpec, ResponseSet
from multiecho_framework.contracts import fingerprint


def _polyline(view: ContextView) -> tuple[np.ndarray, np.ndarray]:
    points = [(view.reference.values[0], np.zeros_like(view.calibration[0].response))]
    points += [(item.intervention.values[0], item.response) for item in view.calibration]
    points.sort(key=lambda point: point[0])
    return np.array([u for u, _ in points]), np.stack([response for _, response in points])


@dataclass(frozen=True)
class InterpolationLaw:
    spec: LawSpec

    def describe(self) -> LawSpec:
        return self.spec

    def predict(self, view: ContextView, intervention: Intervention) -> np.ndarray:
        values, responses = _polyline(view)
        u = float(np.clip(intervention.values[0], values[0], values[-1]))
        index = min(int(np.searchsorted(values, u, side="right")) - 1, len(values) - 2)
        t = (u - values[index]) / (values[index + 1] - values[index])
        return (1 - t) * responses[index] + t * responses[index + 1]


@dataclass(frozen=True)
class ProjectionLaw:
    spec: LawSpec

    def describe(self) -> LawSpec:
        return self.spec

    def recover(self, view: ContextView, response: np.ndarray) -> tuple[float, ...]:
        values, responses = _polyline(view)
        best, estimate = np.inf, values[0]
        for index in range(len(values) - 1):
            start, segment = responses[index], responses[index + 1] - responses[index]
            length = float(segment @ segment)
            t = 0.0 if length == 0 else float(np.clip((response - start) @ segment / length, 0, 1))
            distance = float(np.sum((start + t * segment - response) ** 2))
            if distance < best:
                best, estimate = distance, values[index] + t * (values[index + 1] - values[index])
        return (float(estimate),)


def _spec(law_id: str, data: ResponseSet, hypothesis: str) -> LawSpec:
    return LawSpec(law_id, fingerprint({"method": law_id}), data.world, hypothesis,
                   "Queries between the lowest and highest labeled branch of each context", data.contexts)


class InterpolationForward:
    def fit(self, data: ResponseSet) -> InterpolationLaw:
        return InterpolationLaw(_spec("in-context-interpolation", data,
                                      "Responses vary piecewise-linearly between labeled branches of a context"))


class ProjectionBackward:
    def fit(self, data: ResponseSet) -> ProjectionLaw:
        return ProjectionLaw(_spec("in-context-projection", data,
                                   "The intervention is read from the nearest point on the labeled response polyline"))
