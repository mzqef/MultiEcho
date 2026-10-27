"""Simulation plugins: two engines for one physical scene, and a fast appearance sampler.

A physical scene can map to several simulations (different engines, or one engine
with different internal parameters). An effective simulator has two steps: an
engine generates a scene, then a fast visual sampler draws many scene variants
from it. Each draw is indexed by the seed, so paired branches share appearance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from multiecho_framework import (
    Artifact,
    Channel,
    Context,
    Intervention,
    Modality,
    Statement,
    WorldCard,
    WorldKind,
)
from multiecho_framework.contracts import fingerprint

HEIGHT, WIDTH, FRAMES = 16, 32, 16


def render(positions: np.ndarray) -> np.ndarray:
    """A bright blob moving along a horizontal track; one frame per position."""
    columns = np.arange(WIDTH)[None, :]
    centre = (np.clip(positions, -2, 2) + 2) / 4 * (WIDTH - 1)
    rows = np.exp(-0.5 * ((np.arange(HEIGHT) - HEIGHT / 2) / 2.0) ** 2)[:, None]
    frames = np.exp(-0.5 * ((columns - centre[:, None]) / 1.5) ** 2)[:, None, :] * rows[None]
    return frames[..., None]


@dataclass
class EngineState:
    parameters: dict[str, float]


class _OscillatorEngine:
    engine = "base"

    def __init__(self):
        self.card = WorldCard(f"{self.engine}-oscillator-engine", fingerprint({"engine": self.engine, "size": [FRAMES, HEIGHT, WIDTH]}),
                              WorldKind.SIMULATION, accepts=("parameters",), emits=("video", "truth"),
                              description=f"Unit mass on a damped spring, {self.engine} integration, rendered")

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> EngineState:
        return EngineState({})

    def fork(self, state: EngineState) -> EngineState:
        return EngineState(dict(state.parameters))

    def receive(self, state: EngineState, message: Artifact | None, seed: int) -> EngineState:
        if message is not None:
            state.parameters = {key: item.value for key, item in message.statements().items()}
        return state

    def apply(self, state: EngineState, intervention: Intervention, seed: int) -> EngineState:
        raise ValueError("Engines receive parameters through the bridge; they declare no handle")

    def trajectory(self, p: dict[str, float], times: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def emit(self, state: EngineState) -> Artifact:
        p = state.parameters
        times = np.linspace(0.0, p["steps"] * p["dt"], FRAMES)
        phase = self.trajectory(p, times)
        truth = (Statement("final_position", float(phase[-1, 0]), "m", "physical-quantity"),
                 Statement("final_velocity", float(phase[-1, 1]), "m/s", "physical-quantity"),
                 Statement("stiffness", p["k"], "N/m", "parameter"))
        return Artifact(self.card.source, (Channel("video", Modality.VIDEO, render(phase[:, 0]), rate=(FRAMES - 1) / times[-1],
                                                   semantics="grayscale track view"),
                                           Channel("truth", Modality.TEXT, truth)))

    def close(self, state: EngineState) -> None:
        return None


class AnalyticOscillatorEngine(_OscillatorEngine):
    engine = "analytic"

    def trajectory(self, p: dict[str, float], times: np.ndarray) -> np.ndarray:
        alpha = p["damping"] / 2
        omega = math.sqrt(p["k"] - alpha**2)
        x0, v0 = p["x0"], p["v0"]
        decay = np.exp(-alpha * times)
        position = decay * (x0 * np.cos(omega * times) + (v0 + alpha * x0) / omega * np.sin(omega * times))
        velocity = np.gradient(position, times)
        return np.stack([position, velocity], axis=1)


class EulerOscillatorEngine(_OscillatorEngine):
    engine = "semi-implicit-euler"

    def trajectory(self, p: dict[str, float], times: np.ndarray) -> np.ndarray:
        dt, x, v, out, t = 1e-3, p["x0"], p["v0"], [], 0.0
        for target in times:
            while t < target - 1e-12:
                v += dt * (-p["k"] * x - p["damping"] * v)
                x += dt * v
                t += dt
            out.append((x, v))
        return np.array(out)


@dataclass
class SamplerState:
    video: np.ndarray | None
    seed: int


class AppearanceSampler:
    """Second step of an effective simulator: fast seeded appearance variants of a rendered scene."""

    def __init__(self, gain: float = 0.05, offset: float = 0.02):
        self.gain, self.offset = gain, offset
        self.card = WorldCard("appearance-sampler", fingerprint({"gain": gain, "offset": offset}), WorldKind.SIMULATION,
                              accepts=("video", "truth"), emits=("video",), stochastic=True,
                              description="Seeded gain and background variation of rendered frames")

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> SamplerState:
        return SamplerState(None, seed)

    def fork(self, state: SamplerState) -> SamplerState:
        return SamplerState(state.video, state.seed)

    def receive(self, state: SamplerState, message: Artifact | None, seed: int) -> SamplerState:
        if message is not None:
            state.video = message.channel("video").data
        state.seed = seed
        return state

    def apply(self, state: SamplerState, intervention: Intervention, seed: int) -> SamplerState:
        raise ValueError("The sampler declares no handle")

    def emit(self, state: SamplerState) -> Artifact:
        rng = np.random.default_rng(state.seed)
        gain, offset = 1 + rng.uniform(-self.gain, self.gain), rng.uniform(0, self.offset)
        return Artifact(self.card.source, (Channel("video", Modality.VIDEO, gain * state.video + offset,
                                                   semantics="sampled appearance variant"),))

    def close(self, state: SamplerState) -> None:
        return None
