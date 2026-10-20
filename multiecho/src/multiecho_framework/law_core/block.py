"""Response blocks: the measured effects one law plugin reads (any world, any axis)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..contracts import DataRole, fingerprint
from ..law_cards import LawScope


@dataclass(frozen=True)
class Unit:
    """One observation unit (context, replicate): effects at natural-coordinate values, baseline excluded."""

    context: str
    replicate: int
    values: np.ndarray = field(repr=False)
    effects: np.ndarray = field(repr=False)
    origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        effects = np.array(self.effects, dtype=float, copy=True)
        if not self.context or type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("A unit needs a context identity and a nonnegative replicate")
        if values.ndim != 1 or effects.ndim != 2 or len(values) != len(effects) or not len(values):
            raise ValueError("A unit has J values and a (J, D) effect matrix")
        if len(np.unique(values)) != len(values) or not np.isfinite(values).all() or not np.isfinite(effects).all():
            raise ValueError("Unit values are unique and finite; effects are finite")
        origins = tuple(self.origins) or ("base-grid",) * len(values)
        if len(origins) != len(values):
            raise ValueError("One origin per value")
        order = np.argsort(values)
        values, effects = values[order], effects[order]
        values.setflags(write=False)
        effects.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "origins", tuple(origins[i] for i in order))

    def index(self, values) -> np.ndarray:
        result = []
        for value in values:
            hits = np.flatnonzero(np.isclose(self.values, value, rtol=0, atol=1e-12))
            if not hits.size:
                raise ValueError(f"Unit {self.context}/r{self.replicate} lacks value {value!r}")
            result.append(int(hits[0]))
        return np.asarray(result, dtype=int)


@dataclass(frozen=True)
class ResponseBlock:
    scope: LawScope
    role: DataRole
    baseline: float
    domain: tuple[float, float]
    base_grid: np.ndarray = field(repr=False)
    units: tuple[Unit, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, DataRole) or not isinstance(self.units, tuple) or not self.units:
            raise ValueError("A response block has a data role and at least one unit")
        grid = np.array(sorted({float(value) for value in self.base_grid}), dtype=float)
        lower, upper = (float(value) for value in self.domain)
        if not lower < upper or not lower <= self.baseline <= upper or grid.size < 2:
            raise ValueError("The baseline lies in the domain and the base grid has at least two values")
        if len({unit.effects.shape[1] for unit in self.units}) != 1:
            raise ValueError("All units share one response dimension")
        if any(np.isclose(unit.values, self.baseline, rtol=0, atol=1e-12).any() for unit in self.units):
            raise ValueError("Effects are relative to the baseline; the baseline is not a unit value")
        if len({(unit.context, unit.replicate) for unit in self.units}) != len(self.units):
            raise ValueError("Units are unique (context, replicate) pairs")
        if self.role is DataRole.DISCOVERY and any("confirmation-random" in unit.origins for unit in self.units):
            raise ValueError("Discovery blocks never contain confirmation positions")
        grid.setflags(write=False)
        object.__setattr__(self, "base_grid", grid)
        object.__setattr__(self, "domain", (lower, upper))

    @property
    def dimension(self) -> int:
        return self.units[0].effects.shape[1]

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(sorted({unit.context for unit in self.units}))

    def subset(self, contexts) -> ResponseBlock:
        chosen = set(contexts)
        units = tuple(unit for unit in self.units if unit.context in chosen)
        return ResponseBlock(self.scope, self.role, self.baseline, self.domain, self.base_grid, units)

    @property
    def digest(self) -> str:
        return fingerprint({
            "scope": asdict(self.scope), "role": self.role.value, "baseline": self.baseline,
            "domain": list(self.domain), "grid": self.base_grid.tolist(),
            "units": [[unit.context, unit.replicate, unit.values.tolist(), list(unit.origins),
                       fingerprint(np.round(unit.effects, 12).tolist())] for unit in self.units]})
