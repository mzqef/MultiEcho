"""What flows between worlds: multimodal artifacts and explicit observation scopes."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from .contracts import fingerprint


class Modality(StrEnum):
    VIDEO = "video"    # (frames, height, width, colour channels)
    AUDIO = "audio"    # (samples, audio channels)
    TEXT = "text"      # statements of implicit quantities other modalities cannot carry
    TENSOR = "tensor"  # native numeric outputs, e.g. model latents or tokens


_ARRAY_RANK = {Modality.VIDEO: 4, Modality.AUDIO: 2}


@dataclass(frozen=True)
class Statement:
    """One implicit quantity expressed as text: a parameter, intent, law, range, ..."""

    key: str
    value: str | float
    unit: str = ""
    kind: str = "parameter"

    def __post_init__(self) -> None:
        if not self.key or not self.kind:
            raise ValueError("A statement needs a key and a kind")
        if isinstance(self.value, bool) or not isinstance(self.value, (str, int, float)):
            raise ValueError("Statement values are text or finite numbers")
        if not isinstance(self.value, str):
            if not np.isfinite(self.value):
                raise ValueError("Statement values are text or finite numbers")
            object.__setattr__(self, "value", float(self.value))

    @property
    def numeric(self) -> bool:
        return isinstance(self.value, float)


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True)
class Channel:
    """One modality stream of an artifact."""

    name: str
    modality: Modality
    data: np.ndarray | tuple[Statement, ...]
    rate: float | None = None
    unit: str = "1"
    semantics: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.modality, Modality):
            raise ValueError("A channel needs a name and an explicit modality")
        if self.rate is not None and not (np.isfinite(self.rate) and self.rate > 0):
            raise ValueError("A sampling rate must be positive and finite")
        if self.modality is Modality.TEXT:
            if not isinstance(self.data, tuple) or not all(isinstance(item, Statement) for item in self.data):
                raise ValueError("A text channel carries an immutable tuple of statements")
            if len({item.key for item in self.data}) != len(self.data):
                raise ValueError("Statement keys must be unique within a text channel")
            return
        array = np.array(self.data, dtype=np.float64, copy=True)
        rank = _ARRAY_RANK.get(self.modality)
        if (rank and array.ndim != rank) or array.ndim < 1 or not array.size or not np.isfinite(array).all():
            raise ValueError(f"{self.modality.value} channel {self.name!r} violates its shape or finite-value contract")
        array.setflags(write=False)
        object.__setattr__(self, "data", array)

    @property
    def statements(self) -> dict[str, Statement]:
        if self.modality is not Modality.TEXT:
            raise ValueError(f"Channel {self.name!r} is not a text channel")
        return {item.key: item for item in self.data}

    @property
    def digest(self) -> str:
        content = ([asdict(item) for item in self.data] if self.modality is Modality.TEXT
                   else {"shape": list(self.data.shape), "sha256": _array_digest(self.data)})
        return fingerprint({"name": self.name, "modality": self.modality.value, "rate": self.rate,
                            "unit": self.unit, "semantics": self.semantics, "content": content})


@dataclass(frozen=True)
class Region:
    """A spatial observation region for video channels (True = observed)."""

    label: str
    mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.array(self.mask, dtype=bool, copy=True)
        if not self.label or mask.ndim != 2 or not mask.any():
            raise ValueError("A region needs a label and a nonempty two-dimensional mask")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)

    @classmethod
    def box(cls, label: str, shape: tuple[int, int], rows: tuple[int, int], columns: tuple[int, int]) -> Region:
        mask = np.zeros(shape, dtype=bool)
        mask[rows[0]:rows[1], columns[0]:columns[1]] = True
        return cls(label, mask)

    def record(self) -> dict:
        return {"label": self.label, "shape": list(self.mask.shape), "observed": int(self.mask.sum()),
                "sha256": hashlib.sha256(np.packbits(self.mask).tobytes()).hexdigest()}


@dataclass(frozen=True)
class Scope:
    """What an observer is allowed to see: channels, time window, region, coordinates, statements, clipping.

    `apply` returns the scoped artifact (raw data outside the scope is zeroed or dropped,
    and the scope travels with it). `vector` flattens only in-scope values.
    """

    scope_id: str
    channels: tuple[str, ...]
    window: tuple[int, int] | None = None
    region: Region | None = None
    coordinates: tuple[int, ...] | None = None
    keys: tuple[str, ...] | None = None
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.scope_id or not isinstance(self.channels, tuple) or not self.channels:
            raise ValueError("A scope needs an identity and at least one channel")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("Scope channels must be unique")
        if self.window is not None and not (isinstance(self.window, tuple) and len(self.window) == 2
                                            and 0 <= self.window[0] < self.window[1]):
            raise ValueError("A time window is an ordered [start, stop) pair")
        if self.coordinates is not None and (not self.coordinates or len(set(self.coordinates)) != len(self.coordinates)
                                             or any(type(index) is not int or index < 0 for index in self.coordinates)):
            raise ValueError("Coordinates are unique nonnegative integers")
        if self.keys is not None and (not self.keys or len(set(self.keys)) != len(self.keys)):
            raise ValueError("Observed statement keys must be unique and nonempty")
        if self.clip is not None and not (np.isfinite(self.clip).all() and self.clip[0] < self.clip[1]):
            raise ValueError("Clipping bounds must be finite and ordered")

    def record(self) -> dict:
        return {"scope_id": self.scope_id, "channels": list(self.channels),
                "window": list(self.window) if self.window else None,
                "region": self.region.record() if self.region else None,
                "coordinates": list(self.coordinates) if self.coordinates else None,
                "keys": list(self.keys) if self.keys else None,
                "clip": list(self.clip) if self.clip else None}

    @property
    def digest(self) -> str:
        return fingerprint(self.record())

    def _scoped(self, channel: Channel) -> Channel:
        if channel.modality is Modality.TEXT:
            kept = tuple(item for item in channel.data if self.keys is None or item.key in self.keys)
            return Channel(channel.name, channel.modality, kept, channel.rate, channel.unit, channel.semantics)
        data = channel.data
        if self.window is not None:
            if self.window[1] > len(data):
                raise ValueError(f"Scope window exceeds channel {channel.name!r}")
            data = data[self.window[0]:self.window[1]]
        if self.region is not None and channel.modality is Modality.VIDEO:
            if self.region.mask.shape != data.shape[1:3]:
                raise ValueError(f"Region shape does not match video channel {channel.name!r}")
            data = data * self.region.mask[None, :, :, None]
        if self.coordinates is not None and channel.modality is Modality.TENSOR:
            flat = data.reshape(len(data), -1) if data.ndim > 1 else data.reshape(1, -1)
            if max(self.coordinates) >= flat.shape[1]:
                raise ValueError(f"Scope coordinates exceed channel {channel.name!r}")
            data = flat[:, list(self.coordinates)]
        if self.clip is not None:
            data = np.clip(data, *self.clip)
        return Channel(channel.name, channel.modality, data, channel.rate, channel.unit, channel.semantics)

    def apply(self, artifact: Artifact, *, via: str) -> Artifact:
        return Artifact(artifact.source, tuple(self._scoped(artifact.channel(name)) for name in self.channels),
                        parent=artifact.digest, via=via, scope=self)

    def vector(self, artifact: Artifact) -> np.ndarray:
        """Flattened in-scope numeric values of an artifact this scope was applied to."""
        if artifact.scope is None or artifact.scope.digest != self.digest:
            raise ValueError("Vectors are read only from artifacts carrying this exact scope")
        parts = []
        for name in self.channels:
            channel = artifact.channel(name)
            if channel.modality is Modality.TEXT:
                parts.append(np.array([item.value for item in sorted(channel.data, key=lambda s: s.key) if item.numeric]))
            elif self.region is not None and channel.modality is Modality.VIDEO:
                parts.append(channel.data[:, self.region.mask, :].reshape(-1))
            else:
                parts.append(channel.data.reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0)


@dataclass(frozen=True)
class Artifact:
    """A world's output, or a message derived from one by an interface."""

    source: str
    channels: tuple[Channel, ...]
    parent: str = ""
    via: str = ""
    scope: Scope | None = None

    def __post_init__(self) -> None:
        if not self.source or not isinstance(self.channels, tuple) or not self.channels:
            raise ValueError("An artifact needs a source and at least one channel")
        if len({channel.name for channel in self.channels}) != len(self.channels):
            raise ValueError("Channel names must be unique within an artifact")

    def channel(self, name: str) -> Channel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise ValueError(f"Artifact from {self.source} carries no channel {name!r}")

    def statements(self) -> dict[str, Statement]:
        merged: dict[str, Statement] = {}
        for channel in self.channels:
            if channel.modality is Modality.TEXT:
                merged.update(channel.statements)
        return merged

    @property
    def digest(self) -> str:
        return fingerprint({"source": self.source, "parent": self.parent, "via": self.via,
                            "scope": self.scope.digest if self.scope else None,
                            "channels": [channel.digest for channel in self.channels]})
