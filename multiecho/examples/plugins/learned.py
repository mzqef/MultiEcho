"""Learned-world plugin: a tiny frozen video model that adapts itself to the video interface.

It encodes each frame with fixed weights and predicts future latents by extrapolation.
It stands in for a real world model; the framework only sees the World protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiecho_framework import (
    Artifact,
    Channel,
    Context,
    Intervention,
    LearnedContract,
    Modality,
    WorldCard,
    WorldKind,
)
from multiecho_framework.contracts import fingerprint

LATENT, FUTURE, HISTORY = 8, 4, 16


@dataclass
class ModelState:
    frames: np.ndarray | None


class TinyVideoModel:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 1, (LATENT, 16 * 32)) / np.sqrt(16 * 32)
        self.card = WorldCard("tiny-video-model", fingerprint({"weights": self.weights.round(12).tolist()}),
                              WorldKind.LEARNED, accepts=("video",), emits=("latent",),
                              description="Frozen frame encoder with a constant-velocity latent predictor",
                              contract=LearnedContract(history=HISTORY, horizon=FUTURE, step=1, autoregressive=True,
                                                       adaptation="each frame flattened and encoded on its own",
                                                       precision="float64"))

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> ModelState:
        return ModelState(None)

    def fork(self, state: ModelState) -> ModelState:
        return ModelState(state.frames)

    def receive(self, state: ModelState, message: Artifact | None, seed: int) -> ModelState:
        if message is not None:
            state.frames = message.channel("video").data
        return state

    def apply(self, state: ModelState, intervention: Intervention, seed: int) -> ModelState:
        raise ValueError("This model is studied through its input history, not internal interventions")

    def emit(self, state: ModelState) -> Artifact:
        encoded = state.frames.reshape(len(state.frames), -1) @ self.weights.T
        velocity = encoded[-1] - encoded[-2]
        future = np.stack([encoded[-1] + step * velocity for step in range(1, FUTURE + 1)])
        return Artifact(self.card.source, (Channel("latent", Modality.TENSOR, np.concatenate([encoded, future]),
                                                   semantics="encoded input frames followed by predicted future latents"),))

    def close(self, state: ModelState) -> None:
        return None


class TinyFramePredictor:
    """A frozen next-frame predictor in pixel space: extrapolates the last two frames.

    Its output is one frame, so its observation has location structure: every pixel is
    one location for location-level masking.
    """

    def __init__(self, height: int = 16, width: int = 32):
        self.shape = (height, width)
        self.card = WorldCard("tiny-frame-predictor", fingerprint({"model": "linear-extrapolation", "shape": self.shape}),
                              WorldKind.LEARNED, accepts=("video",), emits=("frame",),
                              description="Predicts the next frame as 2 x last - previous",
                              contract=LearnedContract(history=HISTORY, horizon=1, step=1, autoregressive=False,
                                                       adaptation="frames used as delivered", precision="float64"))

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> ModelState:
        return ModelState(None)

    def fork(self, state: ModelState) -> ModelState:
        return ModelState(state.frames)

    def receive(self, state: ModelState, message: Artifact | None, seed: int) -> ModelState:
        if message is not None:
            state.frames = message.channel("video").data
        return state

    def apply(self, state: ModelState, intervention: Intervention, seed: int) -> ModelState:
        raise ValueError("This model is studied through its input history, not internal interventions")

    def emit(self, state: ModelState) -> Artifact:
        frame = (2 * state.frames[-1] - state.frames[-2])[..., 0]
        return Artifact(self.card.source, (Channel("frame", Modality.TENSOR, frame,
                                                   semantics="predicted next frame (height x width)"),))

    def close(self, state: ModelState) -> None:
        return None
