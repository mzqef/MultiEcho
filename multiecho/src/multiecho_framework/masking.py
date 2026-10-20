"""Observation masking: analyst-side plugins that restrict or transform a subject's native output.

A masker is independent of the subject, the chain and the forward/backward methods.
It sees a counterfactual sweep (measured at declared values disjoint from the
evaluation) and, through reference routes, upstream data; it never sees evaluation
responses.

    describe()                 identity, kind and needed input (`MaskerCard`)
    calibrate(discovery)       freeze rule parameters on discovery contexts only
    fit(input)                 the mask instance of one context
    apply(instance, y)         the masked observation (fixed output size)

Selection is by location (R-9.7). A `Grouping` states which observation dimensions
form one location, for example the channels of one patch token or the colour values
of one pixel. Selection maskers keep or drop whole groups; they never select
individual channels. A vector with no location structure is a single group, so it is
kept or dropped as a whole. A masker only selects: it never projects, rescales or
combines dimensions (R-9.8). Dimensionality reduction is not masking; it is a
reserved stage of the analysis method (identity in protocol 0.3).

`MaskedScope` puts a masker into the analyst scope; `Experiment` runs the sweep,
freezes the masker and assigns one instance per context.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .artifacts import Artifact, Channel, Modality, Scope
from .chain import CONTEXT_CHANNEL
from .contracts import Intervention, fingerprint

KINDS = ("identity", "selection")
NEEDS = ("none", "sweep")


def array_digest(array: np.ndarray | None) -> str:
    if array is None:
        return "none"
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes() + str((array.dtype.str, array.shape)).encode()).hexdigest()


@dataclass(frozen=True)
class Grouping:
    """Observation dimensions grouped into locations; `labels[d]` is the group of dimension d."""

    grouping_id: str
    labels: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if not self.grouping_id or labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu":
            raise ValueError("A grouping needs an identity and one integer group label per dimension")
        groups = int(labels.max()) + 1
        if labels.min() < 0 or len(np.unique(labels)) != groups:
            raise ValueError("Group labels are 0..G-1 with no empty group")
        labels = labels.astype(np.int64, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def blocks(cls, groups: int, width: int, grouping_id: str = "") -> Grouping:
        """`groups` locations of `width` consecutive dimensions each (e.g. tokens x channels, flattened)."""
        return cls(grouping_id or f"blocks:{groups}x{width}", np.repeat(np.arange(groups), width))

    @classmethod
    def whole(cls, dims: int) -> Grouping:
        """No location structure: the whole vector is one group."""
        return cls(f"whole:{dims}", np.zeros(dims, dtype=np.int64))

    @property
    def dims(self) -> int:
        return len(self.labels)

    @property
    def groups(self) -> int:
        return int(self.labels.max()) + 1

    def record(self) -> dict:
        return {"grouping_id": self.grouping_id, "dims": self.dims, "groups": self.groups,
                "labels_digest": array_digest(self.labels)}

    def expand(self, selected: np.ndarray) -> np.ndarray:
        """(G,) group selection -> (D,) dimension mask."""
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != (self.groups,):
            raise ValueError("A group selection has one flag per group")
        return selected[self.labels]

    def sum(self, values: np.ndarray) -> np.ndarray:
        """(..., D) -> (..., G): sum of the dimensions of each group."""
        values = np.asarray(values, dtype=np.float64)
        rows = values.reshape(-1, values.shape[-1])
        out = np.stack([np.bincount(self.labels, weights=row, minlength=self.groups) for row in rows])
        return out.reshape(*values.shape[:-1], self.groups)

    def respects(self, mask: np.ndarray) -> bool:
        """True if a dimension mask keeps or drops every group whole."""
        mask = np.asarray(mask, dtype=bool)
        kept = np.bincount(self.labels, weights=mask, minlength=self.groups)
        sizes = np.bincount(self.labels, minlength=self.groups)
        return bool(np.all((kept == 0) | (kept == sizes)))


@dataclass(frozen=True)
class Sweep:
    """Observations of one context at declared sweep values: outputs[seed, value, dimension]."""

    values: tuple[Intervention, ...]
    seeds: tuple[int, ...]
    outputs: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        outputs = np.asarray(self.outputs, dtype=np.float64)
        if outputs.ndim != 3 or outputs.shape[:2] != (len(self.seeds), len(self.values)) or not np.isfinite(outputs).all():
            raise ValueError("Sweep outputs are finite (seeds, values, dimensions)")
        object.__setattr__(self, "outputs", outputs)


@dataclass(frozen=True)
class MaskInput:
    """Everything a masker may see for one context: its sweep and the reference-route data of that sweep."""

    context_id: str
    sweep: Sweep | None = None
    references: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class MaskerCard:
    masker_id: str
    kind: str
    needs: str
    parameters: dict
    description: str

    def __post_init__(self) -> None:
        if not self.masker_id or self.kind not in KINDS or self.needs not in NEEDS or not self.description:
            raise ValueError(f"A masker card needs an identity, a kind in {KINDS}, needs in {NEEDS} and a description")

    def record(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return fingerprint(self.record())


@runtime_checkable
class Masker(Protocol):
    def describe(self) -> MaskerCard: ...

    def calibrate(self, discovery: list[MaskInput]) -> Masker: ...

    def fit(self, data: MaskInput) -> np.ndarray | None: ...

    def apply(self, instance: np.ndarray | None, y: np.ndarray) -> np.ndarray: ...

    def output_dims(self, input_dims: int) -> int: ...


class NoMask:
    """The full native response."""

    def describe(self) -> MaskerCard:
        return MaskerCard("none", "identity", "none", {}, "the full native response")

    def calibrate(self, discovery: list[MaskInput]) -> NoMask:
        return self

    def fit(self, data: MaskInput) -> None:
        return None

    def apply(self, instance: None, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def output_dims(self, input_dims: int) -> int:
        return input_dims


class GroupSelection:
    """Base of selection maskers: an instance is a (G,) group flag; unselected groups are zeroed."""

    grouping: Grouping

    def calibrate(self, discovery: list[MaskInput]) -> GroupSelection:
        return self

    def apply(self, instance: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(self.grouping.expand(instance), np.asarray(y, dtype=np.float64), 0.0)

    def output_dims(self, input_dims: int) -> int:
        if input_dims != self.grouping.dims:
            raise ValueError("The grouping does not describe this observation")
        return input_dims


class ManualMask(GroupSelection):
    """A user-given set of locations, identical in every context."""

    def __init__(self, grouping: Grouping, selected: np.ndarray, label: str, description: str = "user-given locations"):
        self.grouping, self.label, self.text = grouping, label, description
        self.selected = np.asarray(selected, dtype=bool)
        grouping.expand(self.selected)

    def describe(self) -> MaskerCard:
        return MaskerCard(f"manual:{self.label}", "selection", "none",
                          {"grouping": self.grouping.record(), "selected_digest": array_digest(self.selected),
                           "selected_groups": int(self.selected.sum())}, self.text)

    def fit(self, data: MaskInput) -> np.ndarray:
        return self.selected


def _ratio(signal: np.ndarray, spread: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(spread > 0, signal / np.where(spread > 0, spread, 1), np.where(signal > 0, np.inf, 0.0))


def monotonic_scores(grouping: Grouping, outputs: np.ndarray) -> np.ndarray:
    """(G,) steadiness of each location along the sweep: |mean step| / standard error of the steps.

    Steps are the location's vectors of finite differences between consecutive sweep
    values (seed means first). Large when the location moves steadily one way along
    the axis; zero when it does not move. Pooling over the channels of a wide location
    weights every channel by its variation energy, so the score sits near 1 whenever
    most of that energy is not a steady trend; `MonotonicSNRMask` uses
    `steady_shares` instead. Kept as a diagnostic.
    """
    steps = np.diff(np.asarray(outputs, dtype=np.float64).mean(axis=0), axis=0)        # (K, D)
    count = len(steps)
    if count < 2:
        raise ValueError("A steadiness score needs at least three sweep values")
    mean = steps.mean(axis=0)
    signal = np.sqrt(grouping.sum(mean ** 2))
    spread = np.sqrt(grouping.sum(((steps - mean) ** 2).sum(axis=0)) / (count - 1) / count)
    return _ratio(signal, spread)


def channel_steadiness(outputs: np.ndarray) -> np.ndarray:
    """(D,) per-dimension |mean step| / standard error of the steps along the sweep (seed means first)."""
    steps = np.diff(np.asarray(outputs, dtype=np.float64).mean(axis=0), axis=0)        # (K, D)
    count = len(steps)
    if count < 2:
        raise ValueError("A steadiness score needs at least three sweep values")
    mean = steps.mean(axis=0)
    spread = np.sqrt(((steps - mean) ** 2).sum(axis=0) / (count - 1) / count)
    return _ratio(np.abs(mean), spread)


def steady_shares(grouping: Grouping, outputs: np.ndarray, channel_tau: float = 2.0) -> np.ndarray:
    """(G,) share of each location's sweep variation carried by its steady dimensions.

    A dimension is steady when `channel_steadiness` exceeds `channel_tau`. The share is
    energy-weighted: the steady dimensions' variance over the sweep values divided by
    the whole location's. It is 0 for a location that does not vary. The score belongs
    to the location; selection still keeps or drops the whole location (R-9.7).
    """
    mean_outputs = np.asarray(outputs, dtype=np.float64).mean(axis=0)                 # (K+1, D)
    energy = ((mean_outputs - mean_outputs.mean(axis=0)) ** 2).sum(axis=0)
    energy[np.ptp(mean_outputs, axis=0) == 0] = 0.0     # exactly constant: no averaging residue
    steady = channel_steadiness(outputs) > channel_tau
    total = grouping.sum(energy)
    return _ratio(grouping.sum(np.where(steady, energy, 0.0)), total)


def repeat_noise_scores(grouping: Grouping, outputs: np.ndarray) -> np.ndarray:
    """(G,) size of each location's sweep steps relative to their spread over sweep seeds."""
    outputs = np.asarray(outputs, dtype=np.float64)
    if outputs.shape[0] < 2:
        raise ValueError("Repeat noise needs a sweep with at least two seeds")
    steps = np.diff(outputs, axis=1)                                                     # (S, K, D)
    seeds = steps.shape[0]
    signal = np.sqrt(grouping.sum((steps.mean(axis=0) ** 2).mean(axis=0)))
    noise = np.sqrt(grouping.sum((steps.var(axis=0, ddof=1) / seeds).mean(axis=0)))
    return _ratio(signal, noise)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Chance-corrected Jaccard agreement of two selections; 0 when either is empty or full.

    The raw Jaccard index rewards empty selections (two empty sets agree perfectly) and
    grows with selection size by chance alone, so a threshold chosen by raw split-half
    agreement drifts to nothing or to everything. The expected index of independent
    selections of the same sizes is subtracted and the result rescaled to at most 1.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n, x, y = a.size, int(a.sum()), int(b.sum())
    if x == 0 or y == 0 or x == n or y == n:
        return 0.0
    observed = float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())
    overlap = x * y / n
    expected = overlap / (x + y - overlap)
    return (observed - expected) / (1 - expected)


class _ThresholdMask(GroupSelection):
    """Keep locations whose score exceeds tau; tau is frozen on discovery by chance-corrected split-half agreement.

    Exact ties go to the smaller tau: without evidence, locations are not discarded.
    """

    name = ""
    text = ""

    def __init__(self, grouping: Grouping, candidates: tuple[float, ...] = (2.0, 3.0, 5.0), tau: float | None = None):
        self.grouping, self.candidates, self.tau, self.stability = grouping, tuple(candidates), tau, None

    def score(self, outputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def halves(self, outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return outputs[:, 0::2], outputs[:, 1::2]

    def describe(self) -> MaskerCard:
        return MaskerCard(self.name, "selection", "sweep",
                          {"grouping": self.grouping.record(), "tau": self.tau, "candidates": list(self.candidates),
                           "stability": self.stability}, self.text)

    def calibrate(self, discovery: list[MaskInput]):
        scores = {}
        for tau in self.candidates:
            agreement = []
            for item in discovery:
                first, second = self.halves(item.sweep.outputs)
                agreement.append(_jaccard(self.score(first) > tau, self.score(second) > tau))
            scores[tau] = float(np.mean(agreement))
        best = max(self.candidates, key=lambda tau: (round(scores[tau], 9), -tau))
        frozen = type(self)(self.grouping, self.candidates, best)
        frozen.stability = {str(tau): value for tau, value in scores.items()}
        return frozen

    def fit(self, data: MaskInput) -> np.ndarray:
        if self.tau is None:
            raise ValueError(f"Calibrate the {self.name} masker on discovery before fitting instances")
        return self.score(data.sweep.outputs) > self.tau


class MonotonicSNRMask(GroupSelection):
    """Locations whose variation along the counterfactual axis is substantially steady (needs no repeat noise).

    A dimension is steady when its mean sweep step exceeds `channel_tau` times the
    standard error of its steps. A location is kept whole when steady dimensions carry
    more than `share` of its sweep variation energy (`steady_shares`). Both constants
    are declared; `calibrate` freezes them on discovery and records how much the rule
    keeps there (coverage, contexts left empty). It does not tune them: split-half
    agreement of a sweep cannot tell thresholds apart once chance agreement is removed,
    and raw agreement only rewards empty or full masks.
    """

    name = "monotonic-snr"
    text = ("locations where dimensions whose mean sweep step exceeds channel_tau x its standard error "
            "carry more than `share` of the sweep variation energy")

    def __init__(self, grouping: Grouping, channel_tau: float = 2.0, share: float = 0.1,
                 discovery: dict | None = None):
        if not channel_tau > 0 or not 0 <= share < 1:
            raise ValueError("channel_tau is positive and share lies in [0, 1)")
        self.grouping, self.channel_tau, self.share, self.discovery = grouping, float(channel_tau), float(share), discovery

    def describe(self) -> MaskerCard:
        return MaskerCard(self.name, "selection", "sweep",
                          {"grouping": self.grouping.record(), "channel_tau": self.channel_tau, "share": self.share,
                           "discovery": self.discovery}, self.text)

    def score(self, outputs: np.ndarray) -> np.ndarray:
        return steady_shares(self.grouping, outputs, self.channel_tau)

    def calibrate(self, discovery: list[MaskInput]) -> MonotonicSNRMask:
        if not discovery:
            raise ValueError("Calibration needs discovery sweeps")
        kept = [self.score(item.sweep.outputs) > self.share for item in discovery]
        summary = {"contexts": len(kept), "coverage": float(np.mean([k.mean() for k in kept])),
                   "empty_contexts": float(np.mean([not k.any() for k in kept]))}
        return type(self)(self.grouping, self.channel_tau, self.share, summary)

    def fit(self, data: MaskInput) -> np.ndarray:
        if self.discovery is None:
            raise ValueError(f"Calibrate the {self.name} masker on discovery before fitting instances")
        return self.score(data.sweep.outputs) > self.share


class RepeatNoiseSNRMask(_ThresholdMask):
    """Locations whose sweep steps exceed tau x their spread over sweep seeds (needs a stochastic sweep)."""

    name = "repeat-noise-snr"
    text = "locations whose sweep steps exceed tau x their spread over sweep seeds"

    def score(self, outputs: np.ndarray) -> np.ndarray:
        return repeat_noise_scores(self.grouping, outputs)


class EnergyMask(GroupSelection):
    """The fewest locations holding `fraction` of the sweep's variation energy."""

    def __init__(self, grouping: Grouping, fraction: float = 0.9):
        if not 0 < fraction <= 1:
            raise ValueError("The energy fraction lies in (0, 1]")
        self.grouping, self.fraction = grouping, fraction

    def describe(self) -> MaskerCard:
        return MaskerCard(f"energy:{self.fraction:g}", "selection", "sweep",
                          {"grouping": self.grouping.record(), "fraction": self.fraction},
                          "fewest locations holding the declared fraction of sweep variation energy")

    def fit(self, data: MaskInput) -> np.ndarray:
        outputs = data.sweep.outputs
        variance = np.var(outputs.mean(axis=0), axis=0)
        variance[np.ptp(outputs, axis=(0, 1)) == 0] = 0.0    # exactly constant: no averaging residue
        energy = self.grouping.sum(variance)
        selected = np.zeros(self.grouping.groups, dtype=bool)
        total = energy.sum()
        if total > 0:
            order = np.argsort(energy)[::-1]
            selected[order[:int(np.searchsorted(np.cumsum(energy[order]), self.fraction * total) + 1)]] = True
        return selected


def coverage(card: MaskerCard, masker: Masker, instance) -> float:
    """Fraction of observation dimensions a mask instance keeps."""
    if card.kind == "identity":
        return 1.0
    return float(masker.grouping.expand(instance).mean())


@dataclass(frozen=True)
class MaskedScope(Scope):
    """Analyst scope that applies a masker with one frozen instance per context.

    The scope record (and so the chain identity) seals the masker's rule card as
    declared. Calibrated parameters and instance digests are stage events of the
    experiment. During the sweep stage the scope passes the native output through.
    """

    masker: object = field(default=None, compare=False, repr=False)
    observed: str = ""
    rule: dict = field(default_factory=dict, compare=False, repr=False)
    state: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.masker, Masker) or not self.observed or self.channels != (self.observed,):
            raise ValueError("A masked scope observes exactly one native channel through a Masker")
        object.__setattr__(self, "rule", self.masker.describe().record())
        self.state.update({"phase": "measure", "frozen": None, "instances": {}})

    def record(self) -> dict:
        return {**super().record(), "masker": self.rule}

    @property
    def frozen(self) -> Masker | None:
        return self.state["frozen"]

    def begin_sweep(self) -> None:
        self.state["phase"] = "sweep"

    def end_sweep(self) -> None:
        self.state["phase"] = "measure"

    def freeze(self, masker: Masker) -> None:
        if self.state["frozen"] is not None:
            raise PermissionError("The masker is already frozen")
        if masker.describe().masker_id != self.rule["masker_id"]:
            raise ValueError("The frozen masker differs from the declared rule")
        self.state["frozen"] = masker

    def assign(self, context_id: str, instance) -> str:
        if self.state["frozen"] is None:
            raise PermissionError("Freeze the masker before assigning instances")
        if context_id in self.state["instances"]:
            raise PermissionError(f"Context {context_id!r} already has a mask instance")
        self.state["instances"][context_id] = instance
        return array_digest(None if instance is None else np.asarray(instance))

    def apply(self, artifact: Artifact, *, via: str) -> Artifact:
        y = np.asarray(artifact.channel(self.observed).data, dtype=np.float64).reshape(-1)
        if self.state["phase"] == "sweep":
            data = y
        else:
            context_id = artifact.channel(CONTEXT_CHANNEL).statements["context_id"].value
            if self.state["frozen"] is None or context_id not in self.state["instances"]:
                raise ValueError(f"No frozen mask instance for context {context_id!r}")
            data = np.asarray(self.state["frozen"].apply(self.state["instances"][context_id], y), dtype=np.float64)
        channel = Channel(self.observed, Modality.TENSOR, data,
                          semantics=f"{self.observed} under masker {self.rule['masker_id']}")
        return Artifact(artifact.source, (channel,), parent=artifact.digest, via=via, scope=self)


__all__ = ["EnergyMask", "GroupSelection", "Grouping", "ManualMask", "MaskInput", "MaskedScope", "Masker", "MaskerCard",
           "MonotonicSNRMask", "NoMask", "RepeatNoiseSNRMask", "Sweep", "array_digest", "channel_steadiness",
           "coverage", "monotonic_scores", "repeat_noise_scores", "steady_shares"]
