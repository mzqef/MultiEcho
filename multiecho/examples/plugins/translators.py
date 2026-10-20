"""Translator plugin that needs a model: read spring stiffness from rendered video.

It tracks the blob, fits a second-order autoregression to the track, and converts
its oscillation frequency and decay into stiffness. The result is a text statement
a physical scene can receive. A real2sim translator would be the reverse, and a
plugin of the same kind.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from multiecho_framework import Artifact, Channel, InterfaceSpec, Modality, Statement
from multiecho_framework.contracts import fingerprint


@dataclass(frozen=True)
class StiffnessReadout:
    interface_id: str
    source: str
    target: str
    damping_prior: float = 0.3

    def describe(self) -> InterfaceSpec:
        return InterfaceSpec(self.interface_id, fingerprint({"model": "blob-track-ar2", "damping": self.damping_prior}),
                             self.source, self.target, "model-translator",
                             "Blob tracking and AR(2) frequency estimate converted to stiffness")

    def transmit(self, artifact: Artifact) -> Artifact:
        video = artifact.channel("video")
        frames = video.data[..., 0].sum(axis=1)
        weights = np.clip(frames - frames.min(axis=1, keepdims=True), 0, None)
        track = (weights * np.arange(frames.shape[1])).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)
        track = track - track.mean()
        design = np.stack([track[1:-1], track[:-2]], axis=1)
        (a, b), *_ = np.linalg.lstsq(design, track[2:], rcond=None)
        dt = 1 / video.rate
        roots = np.roots([1, -a, -b])
        omega = abs(np.angle(roots[0])) / dt
        stiffness = float(np.clip(omega**2 + (self.damping_prior / 2) ** 2, 1.0, 9.0))
        statement = Statement("stiffness", stiffness if math.isfinite(stiffness) else 1.0, "N/m", "parameter")
        return Artifact(artifact.source, (Channel("scene", Modality.TEXT, (statement,)),),
                        parent=artifact.digest, via=self.describe().digest)
