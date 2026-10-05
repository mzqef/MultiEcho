"""Worlds: anything that receives, evolves and emits. Physical, simulated and learned alike.

Every world implements one protocol, so any two can be joined by an interface.
The framework ships no engine and no model: those are plugins. It ships only the
physical world layer, which holds the valid data describing a real-world scene.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .artifacts import Artifact, Channel, Modality, Statement
from .contracts import Context, DataRole, Intervention, Space, fingerprint


class WorldKind(StrEnum):
    PHYSICAL = "physical"
    SIMULATION = "simulation"
    LEARNED = "learned"


@dataclass(frozen=True)
class LearnedContract:
    """What a learned world reads, what it emits, and how it adapts the protocol stimulus.

    - `history`: stimulus frames it reads; must equal the declared stimulus length (the
      whole stimulus, ending at the keyframe).
    - `horizon`: prediction steps it emits after the keyframe; 0 for a world that only
      represents its input (an encoder). The default observation is step 1.
    - `step`: stimulus frames covered by one prediction step.
    - `autoregressive`: steps beyond the first are rolled out from the world's own
      predictions only. Required when `horizon` > 1.
    - `adaptation`: how the world converts the protocol stimulus into its native input
      (resizing, tubelets, frame encoding, constant actions...). The conversion and its
      losses belong to the subject.
    - `precision`: numerical precision of inference.
    Emitted channels carry native outputs; analyst-side transforms belong to maskers.
    """

    history: int
    horizon: int
    step: int
    autoregressive: bool
    adaptation: str
    precision: str

    def __post_init__(self) -> None:
        if type(self.history) is not int or self.history < 1 or type(self.step) is not int or self.step < 1:
            raise ValueError("A learned contract reads a positive number of frames with a positive step")
        if type(self.horizon) is not int or self.horizon < 0 or type(self.autoregressive) is not bool:
            raise ValueError("Declare a nonnegative prediction horizon and an explicit autoregressive flag")
        if self.horizon > 1 and not self.autoregressive:
            raise ValueError("Steps beyond the first must be rolled out autoregressively from the world's own predictions")
        if not self.adaptation or not self.precision:
            raise ValueError("Declare how the stimulus is adapted and the numerical precision of inference")


@dataclass(frozen=True)
class WorldCard:
    """Frozen identity of one world. Any change to weights, engine, parameters or
    rendering must change the revision. Learned worlds declare a `LearnedContract`."""

    world_id: str
    revision: str
    kind: WorldKind
    accepts: tuple[str, ...] = ()
    emits: tuple[str, ...] = ()
    handle: Space | None = None
    stochastic: bool = False
    description: str = ""
    contract: LearnedContract | None = None

    def __post_init__(self) -> None:
        if not self.world_id or not self.revision or not isinstance(self.kind, WorldKind):
            raise ValueError("A world needs an identity, a revision and an explicit kind")
        if not isinstance(self.emits, tuple) or not self.emits or not isinstance(self.accepts, tuple):
            raise ValueError("Declare the channels a world accepts and the channels it emits")
        if type(self.stochastic) is not bool:
            raise ValueError("Stochasticity must be an explicit boolean")
        if (self.kind is WorldKind.LEARNED) != isinstance(self.contract, LearnedContract):
            raise ValueError("Learned worlds, and only learned worlds, declare a LearnedContract")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

    @property
    def source(self) -> str:
        return f"{self.world_id}@{self.digest[:16]}"


@runtime_checkable
class World(Protocol):
    def describe(self) -> WorldCard: ...

    def open(self, context: Context, seed: int) -> object: ...

    def fork(self, state: object) -> object: ...

    def receive(self, state: object, message: Artifact | None, seed: int) -> object: ...

    def apply(self, state: object, intervention: Intervention, seed: int) -> object: ...

    def emit(self, state: object) -> Artifact: ...

    def close(self, state: object) -> None: ...


def require_accepted(card: WorldCard, message: Artifact) -> None:
    unknown = {channel.name for channel in message.channels} - set(card.accepts)
    if unknown:
        raise ValueError(f"World {card.world_id} does not accept channels {sorted(unknown)}")


SceneLoader = Callable[[Context], tuple[dict[str, Statement], tuple[Channel, ...]]]
_INLINE = "inline:"


def scene_context(context_id: str, statements: tuple[Statement, ...], role: DataRole, group: str = "") -> Context:
    """A context whose payload is a parameter-only scene description carried inline."""
    payload = sorted((asdict(item) for item in statements), key=lambda item: item["key"])
    return Context(context_id, _INLINE + json.dumps(payload, sort_keys=True), fingerprint(payload),
                   role, group or context_id)


def load_inline_scene(context: Context) -> tuple[dict[str, Statement], tuple[Channel, ...]]:
    if not context.payload_ref.startswith(_INLINE):
        raise ValueError("The inline loader only opens inline scene payloads; supply a loader for recorded media")
    payload = json.loads(context.payload_ref.removeprefix(_INLINE))
    if fingerprint(payload) != context.payload_digest:
        raise ValueError("Context payload does not match its declared digest")
    return {item["key"]: Statement(**item) for item in payload}, ()


def scene_statements(context: Context) -> dict[str, Statement]:
    """The verified statements of an inline scene context."""
    return load_inline_scene(context)[0]


@dataclass
class SceneState:
    statements: dict[str, Statement]
    media: tuple[Channel, ...]
    received: list[str] = field(default_factory=list)


class PhysicalScene:
    """Physical world layer: the valid data describing a real-world scene.

    The data can be recorded video and audio plus text statements (intent, units,
    parameters, ranges). A `loader` supplies recorded media. With the default inline
    loader, a scene is described by parameters only. Interventions set the parameters
    named by the handle's axes. A physical scene can also receive: incoming text
    statements update the scene.
    """

    def __init__(self, world_id: str, revision: str, handle: Space | None = None, *,
                 loader: SceneLoader = load_inline_scene, media: tuple[str, ...] = (), description: str = ""):
        self.card = WorldCard(world_id, revision, WorldKind.PHYSICAL, accepts=("scene",), emits=("scene", *media),
                              handle=handle, description=description or "Physical scene description")
        self.loader = loader

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> SceneState:
        statements, media = self.loader(context)
        if tuple(channel.name for channel in media) != self.card.emits[1:]:
            raise ValueError("Loaded media channels differ from the declared ones")
        return SceneState(dict(statements), media)

    def fork(self, state: SceneState) -> SceneState:
        return SceneState(dict(state.statements), state.media, list(state.received))

    def receive(self, state: SceneState, message: Artifact | None, seed: int) -> SceneState:
        if message is not None:
            state.statements.update(message.statements())
            state.received.append(message.digest)
        return state

    def apply(self, state: SceneState, intervention: Intervention, seed: int) -> SceneState:
        if self.card.handle is None:
            raise ValueError(f"World {self.card.world_id} declares no intervention handle")
        self.card.handle.validate(intervention.values)
        for axis, value in zip(self.card.handle.axes, intervention.values, strict=True):
            state.statements[axis.name] = Statement(axis.name, float(value), axis.unit, "parameter")
        state.statements["horizon"] = Statement("horizon", float(intervention.steps), "steps", "horizon")
        return state

    def emit(self, state: SceneState) -> Artifact:
        text = Channel("scene", Modality.TEXT, tuple(sorted(state.statements.values(), key=lambda item: item.key)))
        return Artifact(self.card.source, (text, *state.media))

    def close(self, state: SceneState) -> None:
        return None
