"""An experiment is communication: worlds joined by interfaces, ending at an analyst.

    physical ─bridge→ simulation ─interface→ learned ─analyst scope→ observation
                          └──────────── reference route ────────────┘

A `Chain` runs every counterfactual branch through all worlds on isolated forks and
observes the final message under the analyst's explicit scope. It implements the
measured-world protocol (`WorldAdapter`), so runners, experiments, laws and
conformance checks apply to it unchanged.

Protocol 0.3 additions:
- Keyframe gate (R-2.6): a chain with learned worlds declares a `StimulusSpec`. Before
  a learned world receives, the chain checks that every video channel holds exactly
  the stimulus frames (ending at the keyframe) and that no reference channel is
  present. Masking inside the model is not a substitute: future frames never enter a
  learned world's input.
- Reference routes (R-4.4): data that must not reach a learned world (the true future,
  engine-derived supports) travels from a declared upstream world straight to the
  analyst, joined to the last world's emission together with the context identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .artifacts import Artifact, Channel, Modality, Scope, Statement
from .contracts import (
    AccessMode,
    Context,
    Intervention,
    ObservationSpec,
    StimulusSpec,
    WorldSpec,
    fingerprint,
)
from .interfaces import Interface, ScopeInterface
from .worlds import World, WorldKind, require_accepted

#: Name of the text channel carrying the context identity to the analyst (reserved).
CONTEXT_CHANNEL = "context_identity"


@dataclass(frozen=True)
class ReferenceRoute:
    """Channels of one upstream world's raw emission delivered to the analyst, bypassing later worlds."""

    source: str
    scope: Scope

    def record(self) -> dict:
        return {"source": self.source, "scope": self.scope.record()}


def stimulus_violations(message: Artifact, stimulus: StimulusSpec, reference_channels: frozenset[str]) -> list[str]:
    """Reasons a message may not be delivered to a learned world (empty if it may)."""
    problems = []
    names = {channel.name for channel in message.channels}
    if stimulus.channel not in names:
        problems.append(f"stimulus channel {stimulus.channel!r} missing")
    for channel in message.channels:
        if channel.name in reference_channels:
            problems.append(f"reference channel {channel.name!r} routed into a learned world")
        elif channel.modality is Modality.VIDEO:
            if len(channel.data) != stimulus.frames:
                problems.append(f"video channel {channel.name!r} has {len(channel.data)} frames; the stimulus has "
                                f"{stimulus.frames} ending at the keyframe")
            elif channel.name == stimulus.channel and channel.data.shape[1:3] != (stimulus.height, stimulus.width):
                problems.append(f"stimulus frames are {channel.data.shape[1:3]}, declared "
                                f"{(stimulus.height, stimulus.width)}")
    return problems


@dataclass
class ChainState:
    context: Context
    seed: int
    handles: list[object]
    owned: list[tuple[int, object]] = field(default_factory=list)
    final: Artifact | None = None
    trace: list[Artifact] = field(default_factory=list)
    delivered: dict[str, Artifact] = field(default_factory=dict)
    closed: bool = False


class Chain:
    def __init__(
        self, chain_id: str, worlds: tuple[World, ...], interfaces: tuple[Interface, ...],
        observer: ScopeInterface, observation: ObservationSpec, *, handle: int = 0,
        access: AccessMode = AccessMode.COMPILED_INPUT, stimulus: StimulusSpec | None = None,
        references: tuple[ReferenceRoute, ...] = (),
    ):
        if not chain_id or not isinstance(worlds, tuple) or not worlds or not isinstance(interfaces, tuple):
            raise ValueError("A chain needs an identity and an immutable tuple of worlds")
        if len(interfaces) != len(worlds) - 1:
            raise ValueError("Consecutive worlds are joined by exactly one interface each")
        cards = [world.describe() for world in worlds]
        ids = [card.world_id for card in cards]
        if len(set(ids)) != len(cards):
            raise ValueError("World identities within a chain must be unique")
        for index, interface in enumerate(interfaces):
            spec = interface.describe()
            if (spec.source, spec.target) != (cards[index].world_id, cards[index + 1].world_id):
                raise ValueError(f"Interface {spec.interface_id} does not join {cards[index].world_id} to {cards[index + 1].world_id}")
        if not isinstance(observer, ScopeInterface) or observer.source != cards[-1].world_id:
            raise ValueError("The analyst observes the last world through an explicit scope interface")
        if type(handle) is not int or not 0 <= handle < len(worlds) or cards[handle].handle is None:
            raise ValueError("The handle must index a world that declares an intervention space")
        if not isinstance(references, tuple) or any(not isinstance(route, ReferenceRoute) for route in references):
            raise ValueError("Reference routes are an immutable tuple of ReferenceRoute")
        reference_channels = frozenset(name for route in references for name in route.scope.channels)
        for route in references:
            if route.source not in ids:
                raise ValueError(f"Reference source {route.source!r} is not a world of this chain")
        if CONTEXT_CHANNEL in reference_channels or CONTEXT_CHANNEL in cards[-1].emits:
            raise ValueError(f"The channel name {CONTEXT_CHANNEL!r} is reserved for the context identity")
        if reference_channels & set(cards[-1].emits):
            raise ValueError("Reference channels must not share names with the last world's channels")
        learned = [card for card in cards if card.kind is WorldKind.LEARNED]
        if learned and not isinstance(stimulus, StimulusSpec):
            raise ValueError("A chain with learned worlds declares the protocol stimulus (StimulusSpec)")
        for card in learned:
            if card.contract.history != stimulus.frames:
                raise ValueError(f"Learned world {card.world_id!r} must read the whole {stimulus.frames}-frame stimulus")
            if stimulus.channel not in card.accepts:
                raise ValueError(f"Learned world {card.world_id!r} must accept the stimulus channel {stimulus.channel!r}")
            leaked = reference_channels & set(card.accepts)
            if leaked:
                raise ValueError(f"Learned world {card.world_id!r} accepts reference channels {sorted(leaked)}")
        self.chain_id, self.worlds, self.interfaces, self.observer = chain_id, worlds, interfaces, observer
        self.observation, self.handle, self.access = observation, handle, access
        self.stimulus, self.references, self.reference_channels = stimulus, references, reference_channels
        self._reference_index = {route.source: ids.index(route.source) for route in references}

    def record(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "worlds": [{**asdict(world.describe()), "digest": world.describe().digest} for world in self.worlds],
            "interfaces": [{**asdict(interface.describe()), "digest": interface.describe().digest}
                           for interface in self.interfaces],
            "observer": {**asdict(self.observer.describe()), "digest": self.observer.describe().digest,
                         "scope": self.observer.scope.record()},
            "handle": self.handle,
            "stimulus": asdict(self.stimulus) if self.stimulus else None,
            "references": [route.record() for route in self.references],
        }

    def describe(self) -> WorldSpec:
        cards = [world.describe() for world in self.worlds]
        return WorldSpec(self.chain_id, fingerprint(self.record()), cards[self.handle].handle,
                         self.observation, self.access, stochastic=any(card.stochastic for card in cards))

    def open(self, context: Context, seed: int) -> ChainState:
        handles = []
        try:
            for world in self.worlds:
                handles.append(world.open(context, seed))
        except Exception:
            for world, handle in zip(self.worlds, handles, strict=False):
                world.close(handle)
            raise
        return ChainState(context, seed, handles)

    def fork(self, state: ChainState) -> ChainState:
        return ChainState(state.context, state.seed,
                          [world.fork(handle) for world, handle in zip(self.worlds, state.handles, strict=True)])

    def apply(self, state: ChainState, intervention: Intervention, seed: int) -> ChainState:
        message: Artifact | None = None
        artifact: Artifact | None = None
        routed: list[Artifact] = []
        for index, world in enumerate(self.worlds):
            card = world.describe()
            if message is not None:
                require_accepted(card, message)
                if card.kind is WorldKind.LEARNED:
                    problems = stimulus_violations(message, self.stimulus, self.reference_channels)
                    if problems:
                        raise ValueError(f"Keyframe protocol violated at {card.world_id}: {'; '.join(problems)}")
                state.delivered[card.world_id] = message
            handle = world.receive(state.handles[index], message, seed)
            if index == self.handle:
                handle = world.apply(handle, intervention, seed)
            if handle is not state.handles[index]:
                state.owned.append((index, handle))
            state.handles[index] = handle
            artifact = world.emit(handle)
            state.trace.append(artifact)
            for route in self.references:
                if self._reference_index[route.source] == index:
                    routed.append(route.scope.apply(artifact, via=f"reference-route:{route.scope.digest}"))
            if index < len(self.interfaces):
                message = self.interfaces[index].transmit(artifact)
                state.trace.append(message)
        identity = Channel(CONTEXT_CHANNEL, Modality.TEXT,
                           (Statement("context_id", state.context.context_id, "", "identity"),),
                           semantics="the context of this branch (analyst-side bookkeeping)")
        joined = Artifact(artifact.source, (*artifact.channels, *(c for r in routed for c in r.channels), identity),
                          parent=artifact.digest, via="reference-join:" + fingerprint([r.digest for r in routed]))
        state.trace.append(joined)
        state.final = self.observer.transmit(joined)
        state.trace.append(state.final)
        return state

    def observe(self, state: ChainState) -> np.ndarray:
        if state.final is None:
            raise ValueError("Observe a branch only after it has run through the chain")
        return self.observation.validate(self.observer.scope.vector(state.final).reshape(self.observation.shape))

    def trace(self, state: ChainState) -> tuple[Artifact, ...]:
        """Every raw artifact and message of one branch, ending with the joined analyst input and the scoped view."""
        return tuple(state.trace)

    def emitted(self, state: ChainState, world_id: str) -> Artifact:
        """The raw artifact one member world emitted in this branch."""
        sources = {world.describe().world_id: world.describe().source for world in self.worlds}
        for artifact in state.trace:
            if not artifact.via and artifact.source == sources.get(world_id):
                return artifact
        raise ValueError(f"No artifact from world {world_id!r} in this branch")

    def delivered(self, state: ChainState, world_id: str) -> Artifact:
        """The message one member world received in this branch."""
        if world_id not in state.delivered:
            raise ValueError(f"World {world_id!r} received no message in this branch")
        return state.delivered[world_id]

    def close(self, state: ChainState) -> None:
        if state.closed:
            return
        state.closed = True
        closed: set[int] = set()
        for index, handle in [*reversed(state.owned), *enumerate(state.handles)]:
            if id(handle) not in closed:
                closed.add(id(handle))
                self.worlds[index].close(handle)
