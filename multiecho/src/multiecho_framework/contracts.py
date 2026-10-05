"""Immutable, scene-independent identities and numerical observation contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AccessMode(StrEnum):
    COMPILED_INPUT = "compiled_input"
    NATIVE_ACTION = "native_action"
    INTERNAL_STATE = "internal_state"
    MESSAGE = "message"


class SeedMode(StrEnum):
    FIXED = "fixed_realization"
    CONTEXT = "context_draws"
    PAIRED = "paired_draws"
    INDEPENDENT = "independent_draws"


#: Couplings an experiment may declare: every branch of a context shares each draw (hard control).
HARD_CONTROL_COUPLINGS = frozenset({SeedMode.FIXED, SeedMode.CONTEXT})


class DataRole(StrEnum):
    DISCOVERY = "discovery"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"
    TRAIN = "train"


@dataclass(frozen=True)
class Axis:
    name: str
    unit: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("Axis name and unit are required (use '1' for dimensionless)")
        if not np.isfinite([self.lower, self.upper]).all() or self.lower > self.upper:
            raise ValueError("Axis bounds must be finite and ordered")


@dataclass(frozen=True)
class Space:
    space_id: str
    axes: tuple[Axis, ...]
    semantics: str

    def __post_init__(self) -> None:
        if not self.space_id or not self.semantics or not self.axes:
            raise ValueError("A space needs an identity, semantics, and axes")
        if not isinstance(self.axes, tuple) or len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("Axes must be an immutable tuple with unique names")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

    def validate(self, values: tuple[float, ...]) -> None:
        if not isinstance(values, tuple) or len(values) != len(self.axes):
            raise ValueError("Coordinate width does not match the declared space")
        for axis, value in zip(self.axes, values, strict=True):
            if not np.isfinite(value) or not axis.lower <= value <= axis.upper:
                raise ValueError(f"Coordinate outside domain: {axis.name}")


@dataclass(frozen=True)
class ObservationSpec:
    space_id: str
    shape: tuple[int, ...]
    semantics: str
    unit: str = "1"

    def __post_init__(self) -> None:
        if not self.space_id or not self.semantics or not self.unit:
            raise ValueError("Observation identity, semantics, and unit are required")
        if not isinstance(self.shape, tuple) or not self.shape or any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("Observation shape must have positive, fixed dimensions")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

    def validate(self, value: np.ndarray) -> np.ndarray:
        array = np.array(value, dtype=np.float64, copy=True)
        if array.shape != self.shape or not np.isfinite(array).all():
            raise ValueError("Observation shape or finite-value contract violated")
        array.setflags(write=False)
        return array


@dataclass(frozen=True)
class StimulusSpec:
    """The stimulus every learned world receives; a property of the protocol, never of a subject.

    `frames` frames sampled every `interval` seconds at `height` x `width` pixels are
    delivered on `channel`. The last frame is the keyframe: nothing after it is ever
    delivered to a learned world. The declared observation lies `observe_after` frames
    after the keyframe.
    """

    frames: int
    interval: float
    height: int
    width: int
    channel: str = "video"
    observe_after: int = 1

    def __post_init__(self) -> None:
        sizes = (self.frames, self.height, self.width, self.observe_after)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("Stimulus frames, height, width and observe_after are positive integers")
        if not isinstance(self.interval, (int, float)) or isinstance(self.interval, bool) \
                or not np.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("The stimulus frame interval is a positive number of seconds")
        if not self.channel:
            raise ValueError("Name the stimulus channel")

    @property
    def keyframe(self) -> int:
        return self.frames - 1

    @property
    def observation_time(self) -> float:
        """Seconds after the keyframe at which the declared observation lies."""
        return self.observe_after * float(self.interval)

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class WorldSpec:
    world_id: str
    revision: str
    intervention_space: Space
    observation: ObservationSpec
    access: AccessMode
    stochastic: bool = False
    supports_sequence: bool = False
    supports_messages: bool = False

    def __post_init__(self) -> None:
        if not self.world_id or not self.revision:
            raise ValueError("Frozen world identity and revision are required")
        if not isinstance(self.access, AccessMode):
            raise ValueError("Access must explicitly name an AccessMode")
        if any(type(value) is not bool for value in (self.stochastic, self.supports_sequence, self.supports_messages)):
            raise ValueError("World capabilities must be explicit booleans")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class Context:
    context_id: str
    payload_ref: str
    payload_digest: str
    role: DataRole
    exchangeability_group: str

    def __post_init__(self) -> None:
        if not all((self.context_id, self.payload_ref, self.payload_digest, self.exchangeability_group)):
            raise ValueError("Context identity, payload identity, and exchangeability group are required")
        if not isinstance(self.role, DataRole):
            raise ValueError("Context role must explicitly name a DataRole")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class Intervention:
    values: tuple[float, ...]
    steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or not self.values or type(self.steps) is not int or self.steps < 1:
            raise ValueError("Intervention needs immutable coordinates and positive integer steps")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in self.values):
            raise ValueError("Intervention coordinates must be finite real numbers")


@dataclass(frozen=True)
class PairRequest:
    pair_id: str
    context: Context
    baseline: Intervention
    intervention: Intervention
    mode: SeedMode
    seeds: tuple[int, ...]
    independent_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.pair_id or not isinstance(self.mode, SeedMode):
            raise ValueError("Pair identity and explicit random coupling are required")
        if not isinstance(self.seeds, tuple) or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide unique immutable draw seeds")
        if not isinstance(self.independent_seeds, tuple):
            raise ValueError("Independent draw seeds must be an immutable tuple")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds + self.independent_seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if self.mode is SeedMode.FIXED and len(self.seeds) != 1:
            raise ValueError("A fixed realization has exactly one draw")
        if self.mode in (SeedMode.PAIRED, SeedMode.INDEPENDENT) and len(self.seeds) < 2:
            raise ValueError("Distributional modes require multiple draws")
        if self.mode is SeedMode.INDEPENDENT:
            if len(self.independent_seeds) != len(self.seeds) or len(set(self.independent_seeds)) != len(self.seeds):
                raise ValueError("Independent branches require a separate unique seed set")
            if set(self.seeds) & set(self.independent_seeds):
                raise ValueError("Independent seed sets must not overlap")
        elif self.independent_seeds:
            raise ValueError("Only independent mode accepts a second seed set")
        if self.baseline.steps != self.intervention.steps:
            raise ValueError("Paired branches must have the same observation horizon")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))