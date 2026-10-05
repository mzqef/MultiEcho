"""Interfaces: the declared protocol between two worlds.

Masking, clipping and selection (`ScopeInterface`), parameter mapping
(`ParameterBridge`) and plugin translators (for example a real2sim model) all
implement one protocol. Every message records its parent artifact and the interface
that produced it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from .artifacts import Artifact, Channel, Modality, Scope, Statement
from .contracts import fingerprint


@dataclass(frozen=True)
class InterfaceSpec:
    interface_id: str
    revision: str
    source: str
    target: str
    kind: str
    description: str

    def __post_init__(self) -> None:
        if not all((self.interface_id, self.revision, self.source, self.target, self.kind, self.description)):
            raise ValueError("An interface needs an identity, revision, both endpoints, a kind and a description")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@runtime_checkable
class Interface(Protocol):
    def describe(self) -> InterfaceSpec: ...

    def transmit(self, artifact: Artifact) -> Artifact: ...


@dataclass(frozen=True)
class ScopeInterface:
    """Masking, clipping and selection under an explicit scope; the scope travels with the message."""

    interface_id: str
    source: str
    target: str
    scope: Scope
    description: str = "Masking, clipping and selection under an explicit observation scope"

    def describe(self) -> InterfaceSpec:
        return InterfaceSpec(self.interface_id, self.scope.digest, self.source, self.target, "scope", self.description)

    def transmit(self, artifact: Artifact) -> Artifact:
        return self.scope.apply(artifact, via=self.describe().digest)


@dataclass(frozen=True)
class ParameterMap:
    source_key: str
    target_key: str
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.source_key or not self.target_key or self.scale == 0:
            raise ValueError("A parameter map needs both keys and a nonzero scale")


@dataclass(frozen=True)
class ParameterBridge:
    """A translator that is only parameters: rename, rescale and complete text statements."""

    interface_id: str
    source: str
    target: str
    mappings: tuple[ParameterMap, ...]
    constants: tuple[Statement, ...] = ()
    channel: str = "parameters"
    description: str = "Parameter bridge: declared renaming, unit scaling and fixed engine constants"

    def __post_init__(self) -> None:
        targets = [item.target_key for item in self.mappings] + [item.key for item in self.constants]
        if not self.mappings or len(set(targets)) != len(targets):
            raise ValueError("A parameter bridge needs mappings with unique target keys")

    def describe(self) -> InterfaceSpec:
        revision = fingerprint({"mappings": [asdict(item) for item in self.mappings],
                                "constants": [asdict(item) for item in self.constants], "channel": self.channel})
        return InterfaceSpec(self.interface_id, revision, self.source, self.target, "parameter-bridge", self.description)

    def transmit(self, artifact: Artifact) -> Artifact:
        available = artifact.statements()
        mapped = []
        for item in self.mappings:
            if item.source_key not in available or not available[item.source_key].numeric:
                raise ValueError(f"Bridge input lacks numeric statement {item.source_key!r}")
            statement = available[item.source_key]
            mapped.append(Statement(item.target_key, statement.value * item.scale + item.offset,
                                    item.unit or statement.unit, statement.kind))
        statements = tuple(sorted((*mapped, *self.constants), key=lambda item: item.key))
        return Artifact(artifact.source, (Channel(self.channel, Modality.TEXT, statements),),
                        parent=artifact.digest, via=self.describe().digest)


@dataclass(frozen=True)
class ComposedInterface:
    """Interfaces applied in sequence, e.g. masking then a translator."""

    interface_id: str
    parts: tuple[Interface, ...]

    def __post_init__(self) -> None:
        if not self.interface_id or not isinstance(self.parts, tuple) or len(self.parts) < 2:
            raise ValueError("A composed interface needs an identity and at least two parts")

    def describe(self) -> InterfaceSpec:
        specs = [part.describe() for part in self.parts]
        return InterfaceSpec(self.interface_id, fingerprint([spec.digest for spec in specs]), specs[0].source,
                             specs[-1].target, "composed", " then ".join(spec.kind for spec in specs))

    def transmit(self, artifact: Artifact) -> Artifact:
        message = artifact
        for part in self.parts:
            message = part.transmit(message)
        return Artifact(message.source, message.channels, parent=artifact.digest, via=self.describe().digest,
                        scope=message.scope)
