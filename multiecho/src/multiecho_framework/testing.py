"""Known-answer conformance doubles. They are not engines for users; they exist so the
protocol and its checks can be verified against analytic answers.

The positive control is a chain: physical scene -> parameter bridge -> analytic
oscillator double -> analyst scope. Two negative controls replace the double with a
world whose response is null or scrambled.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .analysis import ContextView, ResponseSet
from .artifacts import Artifact, Channel, Modality, Scope, Statement
from .chain import Chain
from .contracts import Axis, Context, DataRole, Intervention, ObservationSpec, Space, fingerprint
from .interfaces import ParameterBridge, ParameterMap, ScopeInterface
from .laws import LawSpec
from .protocol import ConditionDimension
from .worlds import (
    LearnedContract,
    PhysicalScene,
    WorldCard,
    WorldKind,
    scene_context,
    scene_statements,
)

TIME_STEP = 0.25
DAMPING = 0.3
STIFFNESS = Space("spring-stiffness", (Axis("stiffness", "N/m", 1.0, 9.0),),
                  "Spring stiffness acting on a unit mass for the horizon")


def transition(stiffness: float, damping: float, duration: float) -> np.ndarray:
    """Exact state map of x'' + damping x' + stiffness x = 0 over `duration` (underdamped)."""
    alpha = damping / 2
    if stiffness <= alpha**2:
        raise ValueError("The analytic oscillator requires an underdamped configuration")
    omega = math.sqrt(stiffness - alpha**2)
    cos, sin, decay = math.cos(omega * duration), math.sin(omega * duration), math.exp(-alpha * duration)
    return decay * np.array([[cos + alpha / omega * sin, sin / omega],
                             [-(omega + alpha**2 / omega) * sin, cos - alpha / omega * sin]])


@dataclass
class _Engine:
    parameters: dict[str, float]
    seed: int = 0


class OscillatorDouble:
    """Simulation double: an analytic damped oscillator emitting its phase state."""

    response = "analytic"

    def __init__(self, noise: float = 1e-4):
        self.noise = noise
        self.card = WorldCard(f"{self.response}-oscillator-double", fingerprint({"noise": noise, "kind": self.response}),
                              WorldKind.SIMULATION, accepts=("parameters",), emits=("state", "truth"),
                              stochastic=noise > 0, description="Conformance double")

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> _Engine:
        return _Engine({})

    def fork(self, state: _Engine) -> _Engine:
        return _Engine(dict(state.parameters), state.seed)

    def receive(self, state: _Engine, message: Artifact | None, seed: int) -> _Engine:
        if message is not None:
            state.parameters = {key: item.value for key, item in message.statements().items()}
        state.seed = seed
        return state

    def apply(self, state: _Engine, intervention: Intervention, seed: int) -> _Engine:
        raise ValueError("The oscillator double declares no intervention handle")

    def _phase(self, parameters: dict[str, float]) -> np.ndarray:
        step = transition(parameters["k"], parameters["damping"], parameters["steps"] * parameters["dt"])
        return step @ np.array([parameters["x0"], parameters["v0"]])

    def emit(self, state: _Engine) -> Artifact:
        phase = self._phase(state.parameters)
        # Paired branches share a seed and therefore one noise realization.
        noise = np.random.default_rng(state.seed).normal(0, self.noise, 2) if self.noise else 0
        truth = (Statement("position", float(phase[0]), "m"), Statement("velocity", float(phase[1]), "m/s"))
        return Artifact(self.card.source, (Channel("state", Modality.TENSOR, phase + noise, unit="m, m/s",
                                                   semantics="position and velocity at the end of the horizon"),
                                           Channel("truth", Modality.TEXT, truth)))

    def close(self, state: _Engine) -> None:
        return None


class NullDouble(OscillatorDouble):
    """Negative control: the emitted state never depends on the intervention."""

    response = "null"

    def __init__(self, noise: float = 1e-2):
        super().__init__(noise)

    def _phase(self, parameters: dict[str, float]) -> np.ndarray:
        return np.array([parameters["x0"], parameters["v0"]])


class ScrambledDouble(OscillatorDouble):
    """Negative control: a resolved response with no structure shared across queries."""

    response = "scrambled"

    def __init__(self):
        super().__init__(0.0)

    def _phase(self, parameters: dict[str, float]) -> np.ndarray:
        key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(key, "big")).normal(0, 1, 2)


def control_chain(simulator: OscillatorDouble | None = None) -> Chain:
    simulator = simulator or OscillatorDouble()
    scene = PhysicalScene("physical-scene", "parameters-only-v1", STIFFNESS,
                          description="Unit mass on a spring; parameters only")
    bridge = ParameterBridge(
        "scene-to-engine", scene.describe().world_id, simulator.describe().world_id,
        (ParameterMap("stiffness", "k"), ParameterMap("position", "x0"), ParameterMap("velocity", "v0"),
         ParameterMap("horizon", "steps")),
        (Statement("damping", DAMPING, "1/s"), Statement("dt", TIME_STEP, "s")),
    )
    observer = ScopeInterface("analyst", simulator.describe().world_id, "analyst", Scope("phase-state", ("state",)))
    return Chain(f"{simulator.response}-control-chain", (scene, simulator), (bridge,), observer,
                 ObservationSpec("phase-state", (2,), "Position and velocity at the end of the horizon", "m, m/s"))


def control_contexts(counts: dict[DataRole, int], seed: int = 0) -> tuple[Context, ...]:
    rng, contexts = np.random.default_rng(seed), []
    for role, count in counts.items():
        for index in range(count):
            moving = index % 2 == 1
            statements = (Statement("position", round(float(rng.uniform(-1, 1)), 12), "m"),
                          Statement("velocity", round(float(rng.uniform(-1, 1)), 12) if moving else 0.0, "m/s"),
                          Statement("intent", "How does the phase state respond to spring stiffness?", "", "intent"))
            contexts.append(scene_context(f"{role.value}-{index:02d}", statements, role))
    return tuple(contexts)


def release_condition(contexts: tuple[Context, ...]) -> ConditionDimension:
    return ConditionDimension("release", tuple(
        (context.context_id, "at-rest" if scene_statements(context)["velocity"].value == 0 else "moving")
        for context in contexts))


@dataclass(frozen=True)
class OscillatorLaw:
    """Forward law double: the linear-oscillator family with a damping constant."""

    spec: LawSpec
    damping: float

    def describe(self) -> LawSpec:
        return self.spec

    def predict(self, view: ContextView, intervention: Intervention) -> np.ndarray:
        duration = view.reference.steps * TIME_STEP
        propagate = transition(intervention.values[0], self.damping, duration)
        reference = transition(view.reference.values[0], self.damping, duration)
        return (propagate @ np.linalg.inv(reference) - np.eye(2)) @ view.baseline


@dataclass(frozen=True)
class OscillatorMethod:
    """Forward method double: fits the damping of the oscillator family on discovery only."""

    law_id: str = "linear-oscillator"

    def fit(self, data: ResponseSet) -> OscillatorLaw:
        draft = LawSpec("draft", "draft", data.world, "draft", "draft")

        def loss(damping: float) -> float:
            law = OscillatorLaw(draft, damping)
            return float(sum(np.sum((law.predict(view, row.intervention) - row.response) ** 2)
                             for view, rows in zip(data.views, data.queries, strict=True) for row in rows))

        grid = np.linspace(0.0, 1.9, 191)
        best = int(np.argmin([loss(value) for value in grid]))
        lower, upper = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        ratio = (math.sqrt(5) - 1) / 2
        for _ in range(60):
            left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
            lower, upper = (lower, right) if loss(left) < loss(right) else (left, upper)
        damping = round(float((lower + upper) / 2), 10)
        spec = LawSpec(self.law_id, fingerprint({"damping": damping, "dt": TIME_STEP}), data.world,
                       f"Unit-mass linear oscillator with damping {damping:.4f}",
                       "Declared stiffness box at the declared horizon", data.contexts)
        return OscillatorLaw(spec, damping)


# ---------------------------------------------------------------------------- learned-world doubles (protocol 0.3)

STIMULUS_FRAMES, FUTURE_FRAMES, TRACK_HEIGHT, TRACK_WIDTH = 8, 2, 6, 12
TRACK_DT = 0.25


def track_frames(positions: np.ndarray) -> np.ndarray:
    """A blob on a horizontal track, one frame per position: (T, H, W, 1)."""
    columns = np.arange(TRACK_WIDTH)[None, :]
    centre = (np.clip(positions, -2, 2) + 2) / 4 * (TRACK_WIDTH - 1)
    rows = np.exp(-0.5 * ((np.arange(TRACK_HEIGHT) - TRACK_HEIGHT / 2) / 1.5) ** 2)[:, None]
    return (np.exp(-0.5 * ((columns - centre[:, None]) / 1.2) ** 2)[:, None, :] * rows[None])[..., None]


class TrackDouble:
    """Simulation double: renders the oscillator's position as the stimulus and the true future."""

    def __init__(self):
        self.card = WorldCard("track-double", fingerprint({"frames": [STIMULUS_FRAMES, FUTURE_FRAMES], "dt": TRACK_DT}),
                              WorldKind.SIMULATION, accepts=("parameters",), emits=("video", "future"),
                              description="Conformance double: a damped oscillator rendered on a track")

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> _Engine:
        return _Engine({})

    def fork(self, state: _Engine) -> _Engine:
        return _Engine(dict(state.parameters), state.seed)

    def receive(self, state: _Engine, message: Artifact | None, seed: int) -> _Engine:
        if message is not None:
            state.parameters = {key: item.value for key, item in message.statements().items()}
        state.seed = seed
        return state

    def apply(self, state: _Engine, intervention: Intervention, seed: int) -> _Engine:
        raise ValueError("The track double declares no intervention handle")

    def frames(self, p: dict[str, float]) -> np.ndarray:
        start = np.array([p["x0"], p["v0"]])
        positions = [(transition(p["k"], p["damping"], i * TRACK_DT) @ start)[0] if i else start[0]
                     for i in range(STIMULUS_FRAMES + FUTURE_FRAMES)]
        return track_frames(np.array(positions))

    def emit(self, state: _Engine) -> Artifact:
        frames = self.frames(state.parameters)
        return Artifact(self.card.source, (
            Channel("video", Modality.VIDEO, frames[:STIMULUS_FRAMES], rate=1 / TRACK_DT, semantics="stimulus"),
            Channel("future", Modality.VIDEO, frames[STIMULUS_FRAMES:], rate=1 / TRACK_DT, semantics="true future")))

    def close(self, state: _Engine) -> None:
        return None


@dataclass
class _Frames:
    frames: np.ndarray | None = None
    extra: np.ndarray | None = None


class ExtrapolatorDouble:
    """Learned double: predicts the next frame by linear extrapolation of the last two."""

    world_id = "extrapolator-double"
    accepts: tuple[str, ...] = ("video",)

    def __init__(self):
        contract = LearnedContract(STIMULUS_FRAMES, 1, 1, False, "frames used as delivered", "float64")
        self.card = WorldCard(self.world_id, fingerprint({"model": self.world_id}), WorldKind.LEARNED,
                              accepts=self.accepts, emits=("prediction",), contract=contract,
                              description="Conformance double: next frame = 2 x last - previous")

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> _Frames:
        return _Frames()

    def fork(self, state: _Frames) -> _Frames:
        return _Frames(state.frames, state.extra)

    def receive(self, state: _Frames, message: Artifact | None, seed: int) -> _Frames:
        if message is not None:
            state.frames = message.channel("video").data
        return state

    def apply(self, state: _Frames, intervention: Intervention, seed: int) -> _Frames:
        raise ValueError("Learned doubles are studied through their input")

    def predict(self, state: _Frames) -> np.ndarray:
        return (2 * state.frames[-1] - state.frames[-2]).reshape(-1)

    def emit(self, state: _Frames) -> Artifact:
        return Artifact(self.card.source, (Channel("prediction", Modality.TENSOR, self.predict(state),
                                                   semantics="predicted next frame, flattened"),))

    def close(self, state: _Frames) -> None:
        return None


class PersistenceModelDouble(ExtrapolatorDouble):
    """Learned double: predicts that the keyframe persists."""

    world_id = "persistence-model-double"

    def predict(self, state: _Frames) -> np.ndarray:
        return state.frames[-1].reshape(-1)


PREDICTION_DIMS = TRACK_HEIGHT * TRACK_WIDTH
SWEEP = tuple(Intervention((value,), 4) for value in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5))


def learned_chain(model: ExtrapolatorDouble | None = None, masker=None, *, input_scope: Scope | None = None,
                  input_interface=None) -> Chain:
    """physical scene -> bridge -> track double -> stimulus interface -> learned double -> analyst."""
    from .chain import ReferenceRoute
    from .contracts import StimulusSpec
    from .masking import MaskedScope

    model = model or ExtrapolatorDouble()
    scene = PhysicalScene("physical-scene", "parameters-only-v1", STIFFNESS,
                          description="Unit mass on a spring; parameters only")
    track = TrackDouble()
    bridge = ParameterBridge(
        "scene-to-track", scene.describe().world_id, track.describe().world_id,
        (ParameterMap("stiffness", "k"), ParameterMap("position", "x0"), ParameterMap("velocity", "v0")),
        (Statement("damping", DAMPING, "1/s"),))
    to_model = input_interface or ScopeInterface("track-to-model", track.describe().world_id,
                                                 model.describe().world_id,
                                                 input_scope or Scope("model-input", ("video",)))
    if masker is None:
        scope = Scope("prediction", ("prediction",))
    else:
        scope = MaskedScope(f"masked-prediction:{masker.describe().masker_id}", ("prediction",), masker=masker,
                            observed="prediction")
    observer = ScopeInterface("analyst", model.describe().world_id, "analyst", scope)
    size = masker.output_dims(PREDICTION_DIMS) if masker is not None else PREDICTION_DIMS
    return Chain(f"track-to-{model.describe().world_id}", (scene, track, model), (bridge, to_model), observer,
                 ObservationSpec("prediction", (size,), "Predicted next frame after the keyframe"),
                 stimulus=StimulusSpec(STIMULUS_FRAMES, TRACK_DT, TRACK_HEIGHT, TRACK_WIDTH),
                 references=(ReferenceRoute(track.describe().world_id, Scope("reference", ("future",))),))


@dataclass(frozen=True)
class _InterpolationLaw:
    spec: LawSpec

    def describe(self) -> LawSpec:
        return self.spec

    def predict(self, view: ContextView, intervention: Intervention) -> np.ndarray:
        points = sorted([(view.reference.values[0], np.zeros_like(view.calibration[0].response))]
                        + [(item.intervention.values[0], item.response) for item in view.calibration], key=lambda p: p[0])
        values = np.array([u for u, _ in points])
        responses = np.stack([r for _, r in points])
        u = float(np.clip(intervention.values[0], values[0], values[-1]))
        index = min(int(np.searchsorted(values, u, side="right")) - 1, len(values) - 2)
        t = (u - values[index]) / (values[index + 1] - values[index])
        return (1 - t) * responses[index] + t * responses[index + 1]


@dataclass(frozen=True)
class InterpolationMethod:
    """Forward method double: piecewise-linear interpolation between the labeled branches of a context."""

    def fit(self, data: ResponseSet) -> _InterpolationLaw:
        return _InterpolationLaw(LawSpec("in-context-interpolation", fingerprint({"method": "pl"}), data.world,
                                         "Responses vary piecewise-linearly between labeled branches",
                                         "Queries between the labeled branches of each context", data.contexts))
