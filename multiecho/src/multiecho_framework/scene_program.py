"""Data contracts for reproducible scene families and their measurement grids."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from .artifacts import Statement
from .contracts import Context, DataRole, Intervention, StimulusSpec, fingerprint
from .worlds import scene_context, scene_statements

SCENE_PROGRAM_SCHEMA = "multiecho-scene-program/0.2"
_INTEGER_FIELDS = frozenset({
    "frames", "height", "width", "observe_after", "future_frames", "master_seed",
    "history", "horizon", "step", "render_limit", "r", "count", "index", "shape",
    "prediction", "frame",
})


def _canonicalize(value: object, key: str | None = None) -> object:
    if isinstance(value, dict):
        return {name: _canonicalize(item, name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        if key == "context_counts":
            return [[str(role), int(count)] for role, count in value]
        if key == "tiers":
            return [
                [name, [[str(role), int(count)] for role, count in role_counts]]
                for name, role_counts in value
            ]
        return [_canonicalize(item, key) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value if key in _INTEGER_FIELDS else float(value)
    if isinstance(value, float):
        return float(value)
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def uniform01(root: int, key: str, j: int = 0) -> float:
    """Return a stable SHA-256-derived 53-bit value in [0, 1)."""
    if type(root) is not int or not 0 <= root < 2**64 or not key or type(j) is not int or j < 0:
        raise ValueError("uniform01 requires a uint64 root, nonempty key, and nonnegative index")
    digest = hashlib.sha256(f"{root}|{key}|{j}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) * 2.0**-53


def _canonical_float(value: object, key: str | None = None) -> object:
    if isinstance(value, dict):
        return {name: _canonical_float(item, name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_float(item, key) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return value if key in _INTEGER_FIELDS else float(value)
    return value


@dataclass(frozen=True)
class ReplicateSeeds:
    appearance: int
    noise: int
    physics: int

    def __post_init__(self) -> None:
        if any(type(seed) is not int or not 0 <= seed < 2**64
               for seed in (self.appearance, self.noise, self.physics)):
            raise ValueError("Replicate seeds must be uint64 integers")


@dataclass(frozen=True)
class ReplicateDraw:
    r: int
    seeds: ReplicateSeeds
    statements: tuple[Statement, ...] = ()

    def __post_init__(self) -> None:
        if type(self.r) is not int or self.r < 0:
            raise ValueError("Replicate index must be a nonnegative integer")
        if not isinstance(self.seeds, ReplicateSeeds):
            raise TypeError("Declare replicate seeds")
        if not isinstance(self.statements, tuple) or not all(
            isinstance(statement, Statement) for statement in self.statements
        ):
            raise ValueError("Replicate statements must be an immutable tuple")
        if len({statement.key for statement in self.statements}) != len(self.statements):
            raise ValueError("Replicate statement keys must be unique")


class FactorKind(StrEnum):
    WORLD_CONSTANT = "world_constant"
    ENTITY_STATE = "entity_state"
    VISIBLE_PROPERTY = "visible_property"
    HIDDEN_PROPERTY = "hidden_property"
    RELATION = "relation"
    COMPLEXITY = "complexity"
    NUISANCE = "nuisance"


class DistributionKind(StrEnum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    CATEGORICAL = "categorical"


class SamplingLevel(StrEnum):
    CONTEXT = "context"
    REPLICATE = "replicate"


class NaturalCoordinate(StrEnum):
    LINEAR = "linear"
    LOG = "log"
    DISCRETE_COMPLEXITY = "discrete_complexity"
    PERIODIC = "periodic"


class TransitionKind(StrEnum):
    POINT = "point"
    BAND = "band"
    NONE = "none"


@dataclass(frozen=True)
class FactorDistribution:
    kind: DistributionKind
    values: tuple[float | str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DistributionKind) or not isinstance(self.values, tuple) or not self.values:
            raise ValueError("A factor distribution needs a kind and immutable values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, str))
            or (not isinstance(value, str) and not math.isfinite(value))
            for value in self.values
        ):
            raise ValueError("Distribution values must be finite numbers or strings")
        if self.kind is DistributionKind.FIXED and len(self.values) != 1:
            raise ValueError("A fixed distribution has exactly one value")
        if self.kind is DistributionKind.CATEGORICAL and len(set(self.values)) != len(self.values):
            raise ValueError("Categorical distribution values must be unique")
        if self.kind is DistributionKind.CATEGORICAL and any(not isinstance(value, str) for value in self.values):
            raise ValueError("Categorical distribution values must be strings")
        if self.kind in (DistributionKind.UNIFORM, DistributionKind.LOG_UNIFORM):
            if len(self.values) != 2 or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                for value in self.values
            ):
                raise ValueError("A continuous distribution needs two finite numeric bounds")
            lower, upper = self.values
            if lower >= upper or (self.kind is DistributionKind.LOG_UNIFORM and lower <= 0):
                raise ValueError("Continuous distribution bounds must be ordered and log bounds positive")
        if self.kind is not DistributionKind.CATEGORICAL:
            object.__setattr__(
                self, "values",
                tuple(float(value) if isinstance(value, (int, float)) else value for value in self.values),
            )

    def sample(self, rng: np.random.Generator) -> float | str:
        if self.kind is DistributionKind.FIXED:
            return self.values[0]
        if self.kind is DistributionKind.CATEGORICAL:
            return self.values[int(rng.integers(0, len(self.values)))]
        lower, upper = (float(value) for value in self.values)
        if self.kind is DistributionKind.LOG_UNIFORM:
            return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
        return float(rng.uniform(lower, upper))

    def sample_from(self, seed: int, key: str) -> float | str:
        if type(seed) is not int or not 0 <= seed < 2**64 or not isinstance(key, str) or not key:
            raise ValueError("Deterministic factor sampling needs a uint64 seed and nonempty key")
        u = uniform01(seed, key)
        if self.kind is DistributionKind.FIXED:
            return self.values[0]
        if self.kind is DistributionKind.CATEGORICAL:
            return self.values[min(int(u * len(self.values)), len(self.values) - 1)]
        lower, upper = (float(value) for value in self.values)
        if self.kind is DistributionKind.LOG_UNIFORM:
            return float(math.exp(math.log(lower) + (math.log(upper) - math.log(lower)) * u))
        return float(lower + (upper - lower) * u)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    unit: str
    kind: FactorKind
    distribution: FactorDistribution
    description: str
    visible_at_keyframe: bool
    level: SamplingLevel = SamplingLevel.CONTEXT

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.unit
            or not self.description
            or not isinstance(self.kind, FactorKind)
            or not isinstance(self.distribution, FactorDistribution)
            or not isinstance(self.level, SamplingLevel)
        ):
            raise ValueError("A factor needs a name, unit, type, and description")
        if type(self.visible_at_keyframe) is not bool:
            raise ValueError("Keyframe visibility must be explicit")


@dataclass(frozen=True)
class RelationSpec:
    relation_id: str
    factors: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if (
            not self.relation_id
            or not self.description
            or not isinstance(self.factors, tuple)
            or not self.factors
            or len(set(self.factors)) != len(self.factors)
        ):
            raise ValueError("A relation needs an identity, factors, and description")


@dataclass(frozen=True)
class RegimeSpec:
    regime_id: str
    lower: float
    upper: float
    description: str

    def __post_init__(self) -> None:
        if (
            not self.regime_id
            or not self.description
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (self.lower, self.upper)
            )
        ):
            raise ValueError("A regime needs an identity, description, and finite bounds")
        if self.lower > self.upper:
            raise ValueError("Regime bounds must be ordered")
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))


@dataclass(frozen=True)
class TransitionFeature:
    feature_id: str
    kind: TransitionKind
    location: tuple[float, ...]
    metric: str
    description: str
    coordinate: str = "raw"
    width: float | None = None
    required_window_s: float | None = None
    computed_on: str = "state"
    evidence: str = ""
    frame: int | None = None

    def __post_init__(self) -> None:
        expected = {
            TransitionKind.NONE: 0,
            TransitionKind.POINT: 1,
            TransitionKind.BAND: 2,
        }
        if not self.feature_id or not self.metric or not self.description or not isinstance(self.kind, TransitionKind):
            raise ValueError("A transition feature needs an identity, kind, metric, and description")
        if not isinstance(self.location, tuple) or len(self.location) != expected[self.kind]:
            raise ValueError("Transition location width does not match its kind")
        if any(not math.isfinite(value) for value in self.location):
            raise ValueError("Transition locations must be finite")
        if self.kind is TransitionKind.BAND and self.location[0] >= self.location[1]:
            raise ValueError("Transition band bounds must be ordered")
        if self.width is not None and (
            isinstance(self.width, bool) or not math.isfinite(self.width) or self.width <= 0
        ):
            raise ValueError("Transition width must be positive and finite")
        if self.required_window_s is not None and (
            isinstance(self.required_window_s, bool)
            or not math.isfinite(self.required_window_s)
            or self.required_window_s <= 0
        ):
            raise ValueError("Required observation window must be positive and finite")
        if self.computed_on not in {"state", "pixels", "field"}:
            raise ValueError("Transition metrics are computed on state, pixels, or field")
        if self.frame is not None and (type(self.frame) is not int or self.frame < 0):
            raise ValueError("A frame-located transition names a nonnegative observation frame index")
        object.__setattr__(self, "location", tuple(float(value) for value in self.location))
        if self.width is not None:
            object.__setattr__(self, "width", float(self.width))
        if self.required_window_s is not None:
            object.__setattr__(self, "required_window_s", float(self.required_window_s))


_REFERENCE_KINDS = frozenset({
    "segmentation", "state", "field", "support", "linear_radiance", "reset_residual", "events",
})


@dataclass(frozen=True)
class ReferenceChannelSpec:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    unit: str
    semantics: str

    def __post_init__(self) -> None:
        if not self.name or self.kind not in _REFERENCE_KINDS or not self.dtype or not self.semantics:
            raise ValueError("Reference channels need a name, supported kind, dtype, and semantics")
        if not isinstance(self.shape, tuple) or not self.shape or any(
            type(size) is not int or size <= 0 for size in self.shape
        ):
            raise ValueError("Reference channel shapes contain positive dimensions")
        try:
            np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError(f"Unsupported reference dtype: {self.dtype}") from error


@dataclass(frozen=True)
class ExactResetSpec:
    aligned: tuple[str, ...]
    method: str
    tolerance: float
    residual_channel: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.aligned, tuple)
            or not self.aligned
            or len(set(self.aligned)) != len(self.aligned)
            or self.method not in {"shooting", "event-shooting", "differentiable"}
            or not self.residual_channel
        ):
            raise ValueError("Exact reset declares aligned state, method, and residual channel")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("Exact-reset tolerance must be positive and finite")
        object.__setattr__(self, "tolerance", float(self.tolerance))


@dataclass(frozen=True)
class CombinationSpec:
    name: str
    expression: str
    unit: str = "1"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.expression or not self.unit:
            raise ValueError("Combination coordinates need a name, expression, and unit")

    def evaluate(self, values: Mapping[str, float]) -> float:
        import ast
        import operator

        functions = {
            "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
        }
        binary = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow,
        }
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as error:
            raise ValueError("Invalid combination expression") from error

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id == "pi":
                    return math.pi
                if node.id == "e":
                    return math.e
                value = values.get(node.id)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Undeclared or non-finite combination variable: {node.id}")
                return float(value)
            if isinstance(node, ast.BinOp) and type(node.op) in binary:
                return float(binary[type(node.op)](visit(node.left), visit(node.right)))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                if len(node.args) != 1 or node.keywords:
                    raise ValueError("Combination functions accept one positional argument")
                return float(functions[node.func.id](visit(node.args[0])))
            raise ValueError(f"Unsupported combination expression syntax: {type(node).__name__}")

        try:
            result = visit(tree)
        except (ArithmeticError, OverflowError) as error:
            raise ValueError("Combination expression is undefined for these values") from error
        if not math.isfinite(result):
            raise ValueError("Combination expression must produce a finite value")
        return result


@dataclass(frozen=True)
class ComplexitySpec:
    axis: str
    entity: str
    interacting: bool
    placement: str
    nested: bool = True
    linear_channel: str = ""
    render_limit: int = 0

    @property
    def additive_test(self) -> bool:
        """Superposition against single-entity responses needs nested entity sets.

        Non-nested complexity axes (particle counts, mode numbers, lattice sizes) support
        rank and saturation readings only.
        """
        return self.nested

    def __post_init__(self) -> None:
        if not self.axis or not self.entity or not self.placement:
            raise ValueError("Complexity specification needs axis, entity, and placement")
        if type(self.interacting) is not bool or type(self.nested) is not bool:
            raise ValueError("Complexity placements declare interaction and nesting explicitly")
        if type(self.render_limit) is not int or self.render_limit < 0:
            raise ValueError("Render limit must be a nonnegative integer")


@dataclass(frozen=True)
class WindowAcceptance:
    feature_id: str
    condition_id: str
    metric: str
    computed_on: str
    change: float
    noise: float
    within_window: bool
    passed: bool
    evidence: str

    def __post_init__(self) -> None:
        if not all((self.feature_id, self.condition_id, self.metric, self.evidence)):
            raise ValueError("Window acceptance needs feature, condition, metric, and evidence")
        if self.computed_on not in {"pixels", "field"}:
            raise ValueError("Window acceptance must be computed on pixels or a field, not analytic state")
        for name, value in (("change", self.change), ("noise", self.noise)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Window {name} must be finite and nonnegative")
        if type(self.within_window) is not bool or type(self.passed) is not bool:
            raise ValueError("Window verdict fields are booleans")


@dataclass(frozen=True)
class ReadinSpec:
    condition_id: str
    frames: tuple[int, ...]
    height: int
    width: int
    resize: str
    crop: str
    temporal_packing: str
    value_map: str
    prediction: tuple[int, ...]
    description: str

    def __post_init__(self) -> None:
        if not all((self.condition_id, self.resize, self.crop, self.temporal_packing,
                    self.value_map, self.description)):
            raise ValueError("Read-in specification fields must be explicit")
        if (
            not isinstance(self.frames, tuple)
            or not self.frames
            or any(type(frame) is not int or frame < 0 for frame in self.frames)
            or tuple(sorted(set(self.frames))) != self.frames
        ):
            raise ValueError("Read-in frames are unique, nonnegative, and strictly increasing")
        if type(self.height) is not int or self.height < 1 or type(self.width) is not int or self.width < 1:
            raise ValueError("Read-in dimensions must be positive integers")
        if not isinstance(self.prediction, tuple) or not self.prediction or any(
            type(frame) is not int or frame < 1 for frame in self.prediction
        ) or tuple(sorted(set(self.prediction))) != self.prediction:
            raise ValueError("Prediction offsets are positive, unique, and strictly increasing")


def validate_readin(condition: ObservationCondition, readin: ReadinSpec) -> None:
    if not isinstance(readin, ReadinSpec):
        raise TypeError("readin must be a ReadinSpec")
    if not isinstance(condition, ObservationCondition) or condition.condition_id != readin.condition_id:
        raise ValueError("Read-in condition does not match the declared observation condition")
    keyframe = condition.stimulus.frames - 1
    if readin.frames[-1] != keyframe or readin.frames[0] < 0 or readin.frames[-1] > keyframe:
        raise ValueError("Read-in must end at the stimulus keyframe and may not read future frames")
    if any(offset > condition.future_frames for offset in readin.prediction):
        raise ValueError("Prediction offsets must lie inside the declared reference future")
    packing, padded = readin.temporal_packing, len(readin.frames)
    if packing.startswith("prefix:repeat-first:"):
        prefix, _, packing = packing.removeprefix("prefix:repeat-first:").partition("+")
        if not prefix.isdigit() or int(prefix) < 1:
            raise ValueError("Prefix padding repeats the first read-in frame a positive number of times")
        padded += int(prefix)
        packing = packing or "frame"
    if packing.startswith("tubelet:"):
        suffix = packing.partition(":")[2]
        if not suffix.isdigit() or int(suffix) < 1 or padded % int(suffix):
            raise ValueError("Tubelet packing must declare a positive size dividing the read-in frame count")
    elif packing not in {"frame", "field-history"}:
        raise ValueError("Temporal packing must be frame, field-history, or tubelet:<positive integer>, "
                         "optionally after prefix:repeat-first:<m>+")


@dataclass(frozen=True)
class SceneAxis:
    name: str
    factor: str
    unit: str
    coordinate: NaturalCoordinate
    lower: float
    upper: float
    base_grid: tuple[float, ...]
    regimes: tuple[RegimeSpec, ...]
    transitions: tuple[TransitionFeature, ...]
    combination: str = ""
    baseline: float | None = None
    fixed: tuple[Statement, ...] = ()
    combinations: tuple[str, ...] = ()
    grid_rule: str = ""
    period: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.factor or not self.unit or not isinstance(self.coordinate, NaturalCoordinate):
            raise ValueError("A scene axis needs a name, factor, unit, and natural coordinate")
        if not np.isfinite([self.lower, self.upper]).all() or self.lower >= self.upper:
            raise ValueError("Scene axis bounds must be finite and increasing")
        if not isinstance(self.base_grid, tuple) or not self.base_grid or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.base_grid
        ):
            raise ValueError("A scene axis needs an immutable base grid")
        if not isinstance(self.regimes, tuple) or not all(
            isinstance(regime, RegimeSpec) for regime in self.regimes
        ):
            raise ValueError("Scene regimes must be an immutable tuple of regime specifications")
        if not isinstance(self.transitions, tuple) or not all(
            isinstance(transition, TransitionFeature) for transition in self.transitions
        ):
            raise ValueError("Scene transitions must be an immutable tuple of transition specifications")
        if tuple(sorted(set(self.base_grid))) != self.base_grid:
            raise ValueError("Base-grid values must be unique and increasing")
        if self.base_grid[0] < self.lower or self.base_grid[-1] > self.upper:
            raise ValueError("Base-grid values must lie inside the axis domain")
        if self.coordinate is NaturalCoordinate.LOG and self.lower <= 0:
            raise ValueError("A logarithmic axis needs a positive domain")
        if self.coordinate is NaturalCoordinate.PERIODIC:
            if (
                isinstance(self.period, bool)
                or not isinstance(self.period, (int, float))
                or not math.isfinite(self.period)
                or self.period <= 0
                or not math.isclose(self.upper - self.lower, self.period, rel_tol=1e-12)
            ):
                raise ValueError("Periodic axes need a positive period matching their domain width")
        elif self.period is not None:
            raise ValueError("Only periodic axes declare a period")
        if self.baseline is not None and (
            isinstance(self.baseline, bool)
            or not isinstance(self.baseline, (int, float))
            or not math.isfinite(self.baseline)
            or not self.lower <= self.baseline <= self.upper
        ):
            raise ValueError("Axis baseline must be finite and inside its domain")
        if not isinstance(self.fixed, tuple) or not all(
            isinstance(item, Statement) for item in self.fixed
        ) or len({item.key for item in self.fixed}) != len(self.fixed):
            raise ValueError("Axis fixed values must be unique immutable statements")
        if not isinstance(self.combinations, tuple) or len(set(self.combinations)) != len(self.combinations):
            raise ValueError("Axis combination names must be unique and immutable")
        if not isinstance(self.grid_rule, str):
            raise ValueError("Grid rule must be a string")
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))
        object.__setattr__(self, "base_grid", tuple(float(value) for value in self.base_grid))
        if self.baseline is not None:
            object.__setattr__(self, "baseline", float(self.baseline))
        if self.period is not None:
            object.__setattr__(self, "period", float(self.period))
        if self.coordinate is NaturalCoordinate.DISCRETE_COMPLEXITY:
            if not float(self.lower).is_integer() or not float(self.upper).is_integer():
                raise ValueError("A complexity axis needs integer domain bounds")
            if any(value <= 0 or not float(value).is_integer() for value in self.base_grid):
                raise ValueError("A complexity grid contains positive integers")
        for regime in self.regimes:
            if regime.lower < self.lower or regime.upper > self.upper:
                raise ValueError("Regimes must lie inside the axis domain")
        for transition in self.transitions:
            if any(value < self.lower or value > self.upper for value in transition.location):
                raise ValueError("Transitions must lie inside the axis domain")
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))
        object.__setattr__(self, "base_grid", tuple(float(value) for value in self.base_grid))
        if self.baseline is not None:
            object.__setattr__(self, "baseline", float(self.baseline))
        if self.period is not None:
            object.__setattr__(self, "period", float(self.period))

    def to_natural(self, value: float) -> float:
        if not self.lower <= value <= self.upper:
            raise ValueError(f"Value outside scene axis: {self.name}")
        if self.coordinate is NaturalCoordinate.LOG:
            return math.log(value)
        if self.coordinate is NaturalCoordinate.PERIODIC:
            return (value - self.lower) % self.period
        return value

    def from_natural(self, value: float) -> float:
        if self.coordinate is NaturalCoordinate.LOG:
            raw = math.exp(value)
        elif self.coordinate is NaturalCoordinate.PERIODIC:
            raw = self.lower + value % self.period
        else:
            raw = value
        if not self.lower <= raw <= self.upper:
            raise ValueError(f"Natural coordinate outside scene axis: {self.name}")
        return raw

    def grid_from_rule(self) -> tuple[float, ...]:
        if not self.grid_rule:
            raise ValueError(f"Axis {self.name!r} has no grid rule")
        kind, separator, expression = self.grid_rule.partition(":")
        if not separator or kind not in {"linear-equal", "log-equal"}:
            raise ValueError("Grid rule must use linear-equal: or log-equal:")
        try:
            count_text, plus, extras_text = expression.partition("+(")
            count = int(count_text)
            if plus and not extras_text.endswith(")"):
                raise ValueError
            extras = tuple(
                float(value.strip())
                for value in extras_text[:-1].split(",")
                if value.strip()
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid grid rule: {self.grid_rule}") from error
        if count < 2 or (plus and not extras):
            raise ValueError("Grid rules need at least two equal-spacing points; a +( ) list cannot be empty")
        if kind == "log-equal" and self.lower <= 0:
            raise ValueError("Log-equal grids require a positive lower bound")
        transform = math.log if kind == "log-equal" else lambda value: value
        inverse = math.exp if kind == "log-equal" else lambda value: value
        start, stop = transform(self.lower), transform(self.upper)
        base = [inverse(start + (stop - start) * index / (count - 1)) for index in range(count)]
        base[0], base[-1] = self.lower, self.upper
        grid_values = list(base)
        for value in extras:
            if not any(math.isclose(value, item, rel_tol=1e-12, abs_tol=0.0) for item in grid_values):
                grid_values.append(value)
        grid = tuple(sorted(grid_values))
        if any(value < self.lower or value > self.upper for value in grid):
            raise ValueError("Grid-rule values must lie inside the declared axis domain")
        return grid

    def grid_rule_matches(self, relative_tolerance: float = 1e-8) -> bool:
        generated = self.grid_from_rule()
        if len(generated) != len(self.base_grid):
            return False
        return all(
            abs(expected - actual) <= relative_tolerance * max(abs(expected), 1e-12)
            for expected, actual in zip(generated, self.base_grid, strict=True)
        )


@dataclass(frozen=True)
class ObservationCondition:
    condition_id: str
    stimulus: StimulusSpec
    future_frames: int
    keyframe_policy: str
    visibility: str
    viewpoint: str
    visualization: str
    observable_window: str
    reference_channels: tuple[ReferenceChannelSpec, ...] = ()
    exact_reset: ExactResetSpec | None = None
    window_start_s: float | None = None
    window_end_s: float | None = None
    exact_reset_axis: str | None = None

    def __post_init__(self) -> None:
        text = (
            self.condition_id,
            self.keyframe_policy,
            self.visibility,
            self.viewpoint,
            self.visualization,
            self.observable_window,
        )
        if (
            any(not value for value in text)
            or not isinstance(self.stimulus, StimulusSpec)
            or type(self.future_frames) is not int
            or self.future_frames < 1
        ):
            raise ValueError("Declare the complete observation condition and a positive reference future")
        if not isinstance(self.reference_channels, tuple) or not all(
            isinstance(item, ReferenceChannelSpec) for item in self.reference_channels
        ) or len({item.name for item in self.reference_channels}) != len(self.reference_channels):
            raise ValueError("Reference channel declarations must have unique names")
        if self.exact_reset is not None and not isinstance(self.exact_reset, ExactResetSpec):
            raise TypeError("exact_reset must be an ExactResetSpec")
        if (self.exact_reset is None) != (self.exact_reset_axis is None):
            raise ValueError("Exact reset declarations name the axis they align")
        for name, value in (("window_start_s", self.window_start_s), ("window_end_s", self.window_end_s)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.window_start_s is not None and self.window_end_s is not None and (
            self.window_start_s < 0 or self.window_start_s >= self.window_end_s
        ):
            raise ValueError("Observation windows must be nonnegative and increasing")


@dataclass(frozen=True)
class PhysicsRewrite:
    rewrite_id: str
    revision: str
    description: str
    changes: tuple[Statement, ...]
    expected_effect: str
    answer: tuple[Statement, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.rewrite_id, self.revision, self.description, self.expected_effect)):
            raise ValueError("A physics rewrite needs identity, revision, description, and expected effect")
        if not isinstance(self.changes, tuple) or not self.changes or not all(
            isinstance(change, Statement) for change in self.changes
        ):
            raise ValueError("A physics rewrite needs explicit changed statements")
        if not isinstance(self.answer, tuple) or not all(
            isinstance(item, Statement) for item in self.answer
        ) or len({item.key for item in self.answer}) != len(self.answer):
            raise ValueError("Rewrite answers must be unique immutable statements")


@dataclass(frozen=True)
class SupplementaryQuery:
    query_id: str
    plugin_id: str
    plugin_version: str
    axis: str
    interval: tuple[float, float]
    values: tuple[float, ...]
    rationale: str
    program: str = ""
    observation: str = ""
    role: DataRole = DataRole.DISCOVERY
    contexts: tuple[str, ...] = ()
    inputs_digest: str = ""
    plugin_set: str = ""

    def __post_init__(self) -> None:
        if not all((self.query_id, self.plugin_id, self.plugin_version, self.axis, self.rationale)):
            raise ValueError("A supplementary query needs plugin identity, axis, and rationale")
        if (
            not isinstance(self.interval, tuple)
            or len(self.interval) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in self.interval
            )
            or self.interval[0] >= self.interval[1]
        ):
            raise ValueError("A supplementary query needs a finite increasing interval")
        if not isinstance(self.values, tuple) or not self.values or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not self.interval[0] <= value <= self.interval[1]
            for value in self.values
        ):
            raise ValueError("Supplementary values must lie inside the requested interval")
        if len(set(self.values)) != len(self.values):
            raise ValueError("Supplementary query values must be unique")
        if not isinstance(self.role, DataRole):
            raise ValueError("Supplementary queries must declare a data role")
        if not isinstance(self.contexts, tuple) or len(set(self.contexts)) != len(self.contexts):
            raise ValueError("Supplementary-query contexts must be unique immutable identifiers")


@dataclass(frozen=True)
class SceneProgram:
    family_id: str
    revision: str
    factors: tuple[FactorSpec, ...]
    relations: tuple[RelationSpec, ...]
    axes: tuple[SceneAxis, ...]
    observation: ObservationCondition
    rewrites: tuple[PhysicsRewrite, ...]
    context_counts: tuple[tuple[DataRole, int], ...]
    seed_namespace: str
    master_seed: int
    reference: str
    extra_observations: tuple[ObservationCondition, ...] = ()
    tiers: tuple[tuple[str, tuple[tuple[DataRole, int], ...]], ...] = ()
    combinations: tuple[CombinationSpec, ...] = ()
    complexity: tuple[ComplexitySpec, ...] = ()
    evidence: tuple[str, ...] = ()
    schema: str = SCENE_PROGRAM_SCHEMA
    branch: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.branch, str):
            raise ValueError("The physical branch is a string label (empty when undeclared)")
        if not all((self.family_id, self.revision, self.seed_namespace, self.reference)):
            raise ValueError("A scene program needs frozen identity, seed namespace, and reference")
        if self.schema != SCENE_PROGRAM_SCHEMA:
            raise ValueError(f"Unsupported scene-program schema: {self.schema}")
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("The master seed must be a nonnegative integer")
        if (
            not isinstance(self.factors, tuple)
            or not self.factors
            or not all(isinstance(factor, FactorSpec) for factor in self.factors)
        ):
            raise ValueError("Scene factors must be nonempty and uniquely named")
        names = [factor.name for factor in self.factors]
        if len(names) != len(set(names)):
            raise ValueError("Scene factors must be nonempty and uniquely named")
        if (
            not isinstance(self.relations, tuple)
            or not all(isinstance(relation, RelationSpec) for relation in self.relations)
        ):
            raise ValueError("Scene relations must be an immutable tuple of relation specifications")
        if (
            not isinstance(self.axes, tuple)
            or not self.axes
            or not all(isinstance(axis, SceneAxis) for axis in self.axes)
        ):
            raise ValueError("Scene axes must be nonempty and uniquely named")
        axis_names = [axis.name for axis in self.axes]
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("Scene axes must be nonempty and uniquely named")
        if not isinstance(self.combinations, tuple) or not all(
            isinstance(item, CombinationSpec) for item in self.combinations
        ) or len({item.name for item in self.combinations}) != len(self.combinations):
            raise ValueError("Combination coordinates must have unique names")
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("Scene relation identities must be unique")
        if any(axis.factor not in names for axis in self.axes):
            raise ValueError("Every scene axis must refer to a declared factor")
        if any(
            next(factor for factor in self.factors if factor.name == axis.factor).level
            is not SamplingLevel.CONTEXT
            for axis in self.axes
        ):
            raise ValueError("Swept scene factors must be sampled at the context level")
        if "context_seed" in names:
            raise ValueError("'context_seed' is reserved for scene-context identity")
        if any(
            change.key not in names or change.key == axis.factor
            for axis in self.axes for change in axis.fixed
        ):
            raise ValueError("Fixed axis values must refer to declared non-swept factors")
        if any(
            combination not in {item.name for item in self.combinations}
            for axis in self.axes for combination in axis.combinations
        ):
            raise ValueError("Scene axes may reference only declared combinations")
        if any(any(name not in names for name in relation.factors) for relation in self.relations):
            raise ValueError("Every relation must refer to declared factors")
        if (
            not isinstance(self.context_counts, tuple)
            or not self.context_counts
            or not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], DataRole)
                and type(item[1]) is int
                and item[1] >= 1
                for item in self.context_counts
            )
        ):
            raise ValueError("Context counts need unique roles and positive counts")
        if (
            not isinstance(self.rewrites, tuple)
            or not self.rewrites
            or not all(isinstance(rewrite, PhysicsRewrite) for rewrite in self.rewrites)
        ):
            raise ValueError("Declare uniquely named rewritten-physics controls")
        roles = [role for role, _ in self.context_counts]
        if len(roles) != len(set(roles)):
            raise ValueError("Context counts need unique roles")
        rewrite_ids = [rewrite.rewrite_id for rewrite in self.rewrites]
        if len(rewrite_ids) != len(set(rewrite_ids)):
            raise ValueError("Declare uniquely named rewritten-physics controls")
        if any(
            len({change.key for change in rewrite.changes}) != len(rewrite.changes)
            for rewrite in self.rewrites
        ):
            raise ValueError("A physics rewrite cannot change the same statement more than once")
        if not isinstance(self.extra_observations, tuple) or not all(
            isinstance(item, ObservationCondition) for item in self.extra_observations
        ):
            raise ValueError("Additional observations must be immutable observation conditions")
        if len({item.condition_id for item in self.observations}) != len(self.observations):
            raise ValueError("Observation condition identities must be unique")
        channels_by_condition = {
            item.condition_id: {channel.name for channel in item.reference_channels}
            for item in self.observations
        }
        for condition in self.observations:
            if condition.exact_reset is not None:
                reset = condition.exact_reset
                reset_channels = channels_by_condition[condition.condition_id]
                if reset.residual_channel not in reset_channels:
                    raise ValueError("Exact-reset residual must be a declared reference channel")
                try:
                    reset_axis = self.axis(condition.exact_reset_axis)
                except KeyError as error:
                    raise ValueError("Exact reset refers to an undeclared axis") from error
                if next(
                    factor for factor in self.factors if factor.name == reset_axis.factor
                ).kind is not FactorKind.HIDDEN_PROPERTY:
                    raise ValueError("Exact reset may be used only with hidden-property axes")
        if not isinstance(self.complexity, tuple) or any(
            item.axis not in {axis.name for axis in self.axes} for item in self.complexity
        ):
            raise ValueError("Complexity declarations must refer to declared axes")
        if not isinstance(self.evidence, tuple) or any(not item for item in self.evidence):
            raise ValueError("Evidence references must be nonempty immutable strings")
        if not isinstance(self.tiers, tuple) or len(
            {name for name, _ in self.tiers}
        ) != len(self.tiers):
            raise ValueError("Sampling tiers must have unique immutable names")
        max_counts = dict(self.context_counts)
        for tier_name, role_counts in self.tiers:
            if not tier_name or not isinstance(role_counts, tuple):
                raise ValueError("Sampling tiers declare immutable role counts")
            if len({role for role, _ in role_counts}) != len(role_counts):
                raise ValueError(f"Sampling tier {tier_name!r} has duplicate roles")
            if any(
                not isinstance(role, DataRole)
                or type(count) is not int
                or count < 1
                or role not in max_counts
                or count > max_counts[role]
                for role, count in role_counts
            ):
                raise ValueError(f"Sampling tier {tier_name!r} exceeds its declared context budget")

    @property
    def digest(self) -> str:
        return fingerprint(self.to_record())

    def to_record(self) -> dict[str, object]:
        record = _canonicalize(_canonical_float(asdict(self)))
        # Optional fields added after schema 0.2 are omitted at their defaults, so earlier
        # programs keep their records and digests byte for byte.
        if record.get("branch") == "":
            del record["branch"]
        for axis in record["axes"]:
            for feature in axis["transitions"]:
                if feature.get("frame") is None:
                    feature.pop("frame", None)
        return record

    @classmethod
    def from_record(cls, record: object) -> SceneProgram:
        if not isinstance(record, dict) or record.get("schema") != SCENE_PROGRAM_SCHEMA:
            raise ValueError("Unsupported scene-program record schema")
        try:
            factors = tuple(
                FactorSpec(
                    item["name"],
                    item["unit"],
                    FactorKind(item["kind"]),
                    FactorDistribution(
                        DistributionKind(item["distribution"]["kind"]),
                        tuple(item["distribution"]["values"]),
                    ),
                    item["description"],
                    item["visible_at_keyframe"],
                    SamplingLevel(item.get("level", SamplingLevel.CONTEXT.value)),
                )
                for item in record["factors"]
            )
            axes = tuple(
                SceneAxis(
                    item["name"],
                    item["factor"],
                    item["unit"],
                    NaturalCoordinate(item["coordinate"]),
                    item["lower"],
                    item["upper"],
                    tuple(item["base_grid"]),
                    tuple(RegimeSpec(**regime) for regime in item["regimes"]),
                    tuple(TransitionFeature(
                        feature["feature_id"],
                        TransitionKind(feature["kind"]),
                        tuple(feature["location"]),
                        feature["metric"],
                        feature["description"],
                        feature.get("coordinate", "raw"),
                        feature.get("width"),
                        feature.get("required_window_s"),
                        feature.get("computed_on", "state"),
                        feature.get("evidence", ""),
                        feature.get("frame"),
                    ) for feature in item["transitions"]),
                    item.get("combination", ""),
                    item.get("baseline"),
                    tuple(Statement(**value) for value in item.get("fixed", ())),
                    tuple(item.get("combinations", ())),
                    item.get("grid_rule", ""),
                    item.get("period"),
                )
                for item in record["axes"]
            )
            stimulus = StimulusSpec(**record["observation"]["stimulus"])
            observation_data = dict(record["observation"])
            observation_data["stimulus"] = stimulus
            observation_data["reference_channels"] = tuple(
                ReferenceChannelSpec(**item)
                for item in observation_data.get("reference_channels", ())
            )
            if observation_data.get("exact_reset") is not None:
                observation_data["exact_reset"] = ExactResetSpec(**observation_data["exact_reset"])
            observation = ObservationCondition(**observation_data)
            extra_observations = []
            for item in record.get("extra_observations", ()):
                extra = dict(item)
                extra["stimulus"] = StimulusSpec(**extra["stimulus"])
                extra["reference_channels"] = tuple(
                    ReferenceChannelSpec(**value) for value in extra.get("reference_channels", ())
                )
                if extra.get("exact_reset") is not None:
                    extra["exact_reset"] = ExactResetSpec(**extra["exact_reset"])
                extra_observations.append(ObservationCondition(**extra))
            return cls(
                record["family_id"],
                record["revision"],
                factors,
                tuple(RelationSpec(**item) for item in record["relations"]),
                axes,
                observation,
                tuple(
                    PhysicsRewrite(
                        item["rewrite_id"],
                        item["revision"],
                        item["description"],
                        tuple(Statement(**change) for change in item["changes"]),
                        item["expected_effect"],
                        tuple(Statement(**answer) for answer in item.get("answer", ())),
                    )
                    for item in record["rewrites"]
                ),
                tuple((DataRole(role), count) for role, count in record["context_counts"]),
                record["seed_namespace"],
                record["master_seed"],
                record["reference"],
                tuple(extra_observations),
                tuple(
                    (name, tuple((DataRole(role), count) for role, count in counts))
                    for name, counts in record.get("tiers", ())
                ),
                tuple(CombinationSpec(**item) for item in record.get("combinations", ())),
                tuple(ComplexitySpec(**item) for item in record.get("complexity", ())),
                tuple(record.get("evidence", ())),
                record["schema"],
                record.get("branch", ""),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid scene-program record: {error}") from error

    def measurement_digest(self, axis: str, condition_id: str | None = None) -> str:
        selected_axis = self.axis(axis)
        condition = self.observation if condition_id is None else next(
            (item for item in self.observations if item.condition_id == condition_id), None
        )
        if condition is None:
            raise ValueError(f"Unknown observation condition: {condition_id}")
        axis_record = _canonicalize(asdict(selected_axis))
        axis_record = {
            key: axis_record[key]
            for key in (
                "name", "factor", "unit", "coordinate", "lower", "upper",
                "period", "baseline", "fixed",
            )
        }
        condition_record = _canonicalize(asdict(condition))
        return fingerprint({
            "family_id": self.family_id,
            "revision": self.revision,
            "factors": _canonicalize([asdict(item) for item in self.factors]),
            "relations": _canonicalize([asdict(item) for item in self.relations]),
            "seed_namespace": self.seed_namespace,
            "master_seed": self.master_seed,
            "axis": axis_record,
            "observation": condition_record,
        })

    @property
    def observations(self) -> tuple[ObservationCondition, ...]:
        return (self.observation, *self.extra_observations)

    def axis(self, name: str) -> SceneAxis:
        try:
            return next(axis for axis in self.axes if axis.name == name)
        except StopIteration as error:
            raise KeyError(f"Unknown scene axis: {name}") from error

    def derive_seed(self, purpose: str, axis: str, role: DataRole, index: int) -> int:
        if (
            not purpose
            or not isinstance(role, DataRole)
            or type(index) is not int
            or index < 0
        ):
            raise ValueError("Seed derivation needs purpose, axis, role, and nonnegative index")
        self.axis(axis)
        text = (
            f"{self.seed_namespace}|{self.master_seed}|{purpose}|{self.family_id}|"
            f"{axis}|{role.value}|{index}"
        )
        return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

    def _counts_for_tier(self, tier: str) -> dict[DataRole, int]:
        if not tier:
            raise ValueError("Sampling tier must be nonempty")
        tiers = dict(self.tiers)
        if tier in tiers:
            return dict(tiers[tier])
        if tier == "A":
            return dict(self.context_counts)
        raise ValueError(f"Unknown sampling tier: {tier}")

    def baseline_value(self, axis: str) -> float:
        specification = self.axis(axis)
        if specification.baseline is not None:
            return float(specification.baseline)
        factor = next(item for item in self.factors if item.name == specification.factor)
        if (
            factor.distribution.kind is DistributionKind.FIXED
            and isinstance(factor.distribution.values[0], (int, float))
            and not isinstance(factor.distribution.values[0], bool)
        ):
            return float(factor.distribution.values[0])
        raise ValueError(f"Axis {axis!r} needs an explicit numeric baseline")

    def context(self, axis: str, role: DataRole, index: int, tier: str = "A") -> Context:
        self.axis(axis)
        counts = self._counts_for_tier(tier)
        if type(index) is not int or role not in counts or not 0 <= index < counts[role]:
            raise ValueError("Context index is outside the declared role budget")
        context_id = f"{self.family_id}:{axis}:{role.value}:{index:02d}"
        context_seed = self.derive_seed("context", axis, role, index)
        specification = self.axis(axis)
        baseline = self.baseline_value(axis)
        statements = []
        for factor in self.factors:
            if factor.level is SamplingLevel.REPLICATE:
                continue
            value = (
                baseline if factor.name == specification.factor
                else factor.distribution.sample_from(context_seed, factor.name)
            )
            statements.append(Statement(factor.name, value, factor.unit, factor.kind.value))
        by_key = {item.key: item for item in statements}
        by_key.update({item.key: item for item in specification.fixed})
        statements = list(by_key.values())
        statements.append(Statement("context_seed", f"{context_seed:016x}", "", "identity"))
        return scene_context(context_id, tuple(statements), role, context_id)

    def context_at(
        self, axis: str, value: float, role: DataRole, index: int, tier: str = "A"
    ) -> Context:
        """Return a branch view; do not use it as an independent statistical context."""
        specification = self.axis(axis)
        value = float(value)
        specification.to_natural(value)
        if (
            specification.coordinate is NaturalCoordinate.DISCRETE_COMPLEXITY
            and not float(value).is_integer()
        ):
            raise ValueError(f"Discrete scene axis requires an integer value: {axis}")
        base = self.context(axis, role, index, tier)
        statements = dict(scene_statements(base))
        factor = next(item for item in self.factors if item.name == specification.factor)
        if factor.kind is FactorKind.COMPLEXITY and not float(value).is_integer():
            raise ValueError(f"Complexity factor requires an integer value: {specification.factor}")
        statements[factor.name] = Statement(factor.name, value, factor.unit, factor.kind.value)
        context_id = f"{base.context_id}:{value!r}"
        return scene_context(context_id, tuple(statements.values()), role, base.context_id)

    def replicate_draw(
        self, axis: str, role: DataRole, index: int, r: int, tier: str = "A"
    ) -> ReplicateDraw:
        self.axis(axis)
        counts = self._counts_for_tier(tier)
        if type(index) is not int or role not in counts or not 0 <= index < counts[role]:
            raise ValueError("Context index is outside the declared role budget")
        if type(r) is not int or r < 0:
            raise ValueError("Replicate index must be a nonnegative integer")
        appearance = self.derive_seed(f"appearance-r{r}", axis, role, index)
        seeds = ReplicateSeeds(
            appearance,
            self.derive_seed(f"noise-r{r}", axis, role, index),
            self.derive_seed(f"physics-r{r}", axis, role, index),
        )
        statements = tuple(
            Statement(
                factor.name,
                factor.distribution.sample_from(appearance, factor.name),
                factor.unit,
                factor.kind.value,
            )
            for factor in self.factors
            if factor.level is SamplingLevel.REPLICATE
        )
        return ReplicateDraw(r, seeds, statements)

    def derive_branch_seed(
        self, tag: str, axis: str, role: DataRole, index: int, r: int, value: float
    ) -> int:
        if not tag or type(r) is not int or r < 0 or not math.isfinite(value):
            raise ValueError("Branch seed needs a tag, nonnegative replicate, and finite value")
        return self.derive_seed(
            f"branch:{tag}:r{r}:{float(value).hex()}", axis, role, index
        )

    def branch_statements(
        self, context: Context, axis: str, value: float, rewrite_id: str | None = None,
        draw: ReplicateDraw | None = None,
    ) -> tuple[Statement, ...]:
        specification = self.axis(axis)
        specification.to_natural(value)
        statements = dict(scene_statements(context))
        factor = next(item for item in self.factors if item.name == specification.factor)
        statements[factor.name] = Statement(factor.name, value, factor.unit, factor.kind.value)
        statements.update({item.key: item for item in specification.fixed})
        if draw is not None:
            statements.update({item.key: item for item in draw.statements})
        if rewrite_id is not None:
            rewrite = next((item for item in self.rewrites if item.rewrite_id == rewrite_id), None)
            if rewrite is None:
                raise ValueError(f"Unknown physics rewrite: {rewrite_id}")
            statements.update({item.key: item for item in rewrite.changes})
        return tuple(statements[key] for key in sorted(statements))

    def confirmation_points(self, axis: str, count: int, tier: str = "A") -> tuple[float, ...]:
        specification = self.axis(axis)
        if type(count) is not int or count < 1 or not tier:
            raise ValueError("Confirmation sampling needs a positive count and tier")
        seed = self.derive_seed(f"confirmation-{tier}", axis, DataRole.CONFIRMATION, 0)
        natural_grid = np.array(sorted({
            specification.to_natural(value) for value in specification.base_grid
        }))
        if specification.grid_rule:
            _, _, rule = specification.grid_rule.partition(":")
            equal_count = int(rule.partition("+(")[0])
            minimum_spacing = (
                (specification.period if specification.coordinate is NaturalCoordinate.PERIODIC
                 else specification.to_natural(specification.upper) - specification.to_natural(specification.lower))
                / (equal_count - 1)
            )
        elif len(natural_grid) > 1:
            gaps = np.diff(natural_grid)
            if specification.coordinate is NaturalCoordinate.PERIODIC:
                gaps = np.append(gaps, specification.period - natural_grid[-1] + natural_grid[0])
            minimum_spacing = float(np.median(gaps))
        else:
            minimum_spacing = specification.to_natural(specification.upper) - specification.to_natural(
                specification.lower
            )
        exclusion_radius = 0.1 * minimum_spacing
        lower = specification.to_natural(specification.lower)
        upper = specification.to_natural(specification.upper)
        if specification.coordinate is NaturalCoordinate.DISCRETE_COMPLEXITY:
            candidates = [
                float(value)
                for value in range(math.ceil(specification.lower), math.floor(specification.upper) + 1)
                if self._grid_distance(specification, natural_grid, value) >= exclusion_radius
            ]
            if count > len(candidates):
                raise ValueError("Not enough confirmation positions remain outside the base-grid exclusion zones")
            chosen = sorted(
                candidates,
                key=lambda value: uniform01(seed, f"confirmation-discrete:{value}"),
            )[:count]
            return tuple(sorted(float(value) for value in chosen))

        accepted: list[float] = []
        max_attempts = max(1000, count * 1000)
        for attempt in range(max_attempts):
            if len(accepted) == count:
                break
            candidate = lower + (upper - lower) * uniform01(seed, "confirmation-position", attempt)
            if self._grid_distance(specification, natural_grid, candidate) < exclusion_radius:
                continue
            if any(
                self._grid_distance(
                    specification,
                    np.asarray([specification.to_natural(value)]),
                    candidate,
                ) < 1e-12
                for value in accepted
            ):
                continue
            accepted.append(specification.from_natural(candidate))
        if len(accepted) != count:
            raise ValueError("Could not sample enough confirmation positions outside the base grid")
        return tuple(sorted(accepted))

    @staticmethod
    def _grid_distance(axis: SceneAxis, grid: np.ndarray, value: float) -> float:
        differences = np.abs(grid - value)
        if axis.coordinate is NaturalCoordinate.PERIODIC:
            differences = np.minimum(differences, axis.period - differences)
        return float(np.min(differences))

    def validate_query(self, query: SupplementaryQuery) -> None:
        axis = self.axis(query.axis)
        if (
            query.interval[0] < axis.lower
            or query.interval[1] > axis.upper
            or any(not query.interval[0] <= value <= query.interval[1] for value in query.values)
        ):
            raise ValueError("Supplementary query lies outside the declared scene axis")
        if query.role is not DataRole.DISCOVERY:
            raise ValueError("Supplementary queries are permitted only on discovery contexts")
        if query.program and query.program not in {self.family_id, self.digest}:
            raise ValueError("Supplementary query program identity does not match this scene program")
        if query.observation and query.observation not in {
            item.condition_id for item in self.observations
        }:
            raise ValueError("Supplementary query observation is not declared by this program")
        if axis.coordinate is NaturalCoordinate.DISCRETE_COMPLEXITY and any(
            not float(value).is_integer() for value in (*query.interval, *query.values)
        ):
            raise ValueError("Supplementary values on a complexity axis must be integers")
        if any(
            math.isclose(float(value), self.baseline_value(query.axis), rel_tol=0.0, abs_tol=0.0)
            for value in query.values
        ):
            raise ValueError("Supplementary values may not equal the baseline")
        if query.contexts:
            declared_contexts = {
                self.context(query.axis, DataRole.DISCOVERY, index).context_id
                for index in range(self._counts_for_tier("A").get(DataRole.DISCOVERY, 0))
            }
            if not set(query.contexts) <= declared_contexts:
                raise ValueError("Supplementary query names a non-discovery or undeclared context")


def scene_space(program: SceneProgram, axis: str):
    from .contracts import Axis, Space

    specification = program.axis(axis)
    return Space(
        f"{program.family_id}:{axis}",
        (Axis(axis, specification.unit, specification.lower, specification.upper),),
        f"{specification.coordinate.value}; {specification.combination}",
    )


def scene_contexts(
    program: SceneProgram, axis: str, tier: str = "A"
) -> tuple[Context, ...]:
    counts = program._counts_for_tier(tier)
    role_order = (
        DataRole.TRAIN,
        DataRole.DISCOVERY,
        DataRole.CALIBRATION,
        DataRole.VALIDATION,
        DataRole.CONFIRMATION,
    )
    return tuple(
        program.context(axis, role, index, tier)
        for role in role_order
        if role in counts
        for index in range(counts[role])
    )


def scene_interventions(
    program: SceneProgram, axis: str, values: tuple[float, ...]
) -> tuple[Intervention, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("Scene interventions require immutable, nonempty values")
    specification = program.axis(axis)
    baseline = program.baseline_value(axis)
    if any(value == baseline for value in values):
        raise ValueError("Queried interventions must exclude the baseline")
    if len(set(values)) != len(values):
        raise ValueError("Scene intervention values must be unique")
    for value in values:
        specification.to_natural(value)
        if specification.coordinate is NaturalCoordinate.DISCRETE_COMPLEXITY and not float(value).is_integer():
            raise ValueError("Complexity interventions must be integers")
    return tuple(Intervention((float(value),)) for value in values)


def scene_declaration(
    program: SceneProgram,
    axis: str,
    condition_id: str,
    chain: object,
    *,
    tier: str = "A",
    values: tuple[float, ...] | None = None,
    draws: int = 1,
    tasks: tuple[str, ...] = ("forward",),
    conditions: tuple[object, ...] = (),
):
    from .protocol import Declaration

    condition = next(
        (item for item in program.observations if item.condition_id == condition_id), None
    )
    if condition is None:
        raise ValueError(f"Unknown observation condition: {condition_id}")
    world = chain.describe()
    record = chain.record()
    space = world.intervention_space
    if len(space.axes) != 1 or space.axes[0].name != axis:
        raise ValueError("Chain intervention handle must match the selected scene axis")
    baseline = program.baseline_value(axis)
    selected = values if values is not None else tuple(
        value for value in program.axis(axis).base_grid if value != baseline
    )
    interventions = scene_interventions(program, axis, selected)
    contexts = scene_contexts(program, axis, tier)
    if type(draws) is not int or draws < 1:
        raise ValueError("Scene declaration draws must be a positive integer")
    from .contracts import SeedMode

    coupling = SeedMode.FIXED if draws == 1 else SeedMode.CONTEXT
    stimulus_data = record.get("stimulus")
    stimulus = StimulusSpec(**stimulus_data) if stimulus_data else None
    return Declaration(
        experiment_id=f"{program.family_id}:{axis}:{condition_id}:{tier}",
        question=f"Measure responses over {axis} for {program.family_id}.",
        world=world,
        chain=record,
        handle=space.space_id,
        observation_time=(
            f"{condition.stimulus.observe_after} frame(s) after the keyframe; "
            f"future horizon={condition.future_frames}"
        ),
        reference=Intervention((baseline,)),
        interventions=interventions,
        contexts=contexts,
        independence_unit="context",
        tasks=tasks,
        coupling=coupling,
        draws=draws,
        conditions=conditions,
        stimulus=stimulus,
    )


class ReplayAdapter(Protocol):
    def describe(self) -> object: ...

    def open(self, context: Context, seed: int) -> object: ...

    def fork(self, state: object) -> object: ...

    def apply(self, state: object, intervention: Intervention, seed: int) -> object: ...

    def observe(self, state: object) -> np.ndarray: ...

    def close(self, state: object) -> None: ...


@dataclass(frozen=True)
class ReplayProbeResult:
    exact: bool
    maximum_absolute_difference: float
    first_digest: str
    second_digest: str


def probe_replay(
    adapter: ReplayAdapter,
    context: Context,
    intervention: Intervention,
    seed: int,
) -> ReplayProbeResult:
    """Run the same branch twice from separate roots and compare observations exactly."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Replay probing requires a nonnegative integer seed")

    def run() -> np.ndarray:
        root = adapter.open(context, seed)
        missing = object()
        branch = missing
        output = missing
        try:
            branch = adapter.fork(root)
            if branch is root:
                raise ValueError("Replay probing requires an isolated fork")
            output = adapter.apply(branch, intervention, seed)
            observed = np.asarray(adapter.observe(output))
            if observed.dtype.kind not in "iuf" or not observed.size or not np.isfinite(observed).all():
                raise ValueError("Replay observations must be nonempty finite real-valued arrays")
            return np.array(observed, copy=True)
        finally:
            if output is not missing and output is not branch and output is not root:
                adapter.close(output)
            if branch is not missing and branch is not root:
                adapter.close(branch)
            adapter.close(root)

    first = run()
    second = run()
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Replay observations must have equal shapes and finite values")
    delta = np.empty(first.shape, dtype=np.float64)
    np.subtract(first, second, out=delta, casting="unsafe")
    np.abs(delta, out=delta)
    difference = float(np.max(delta, initial=0.0))
    first_digest = hashlib.sha256(np.ascontiguousarray(first).view(np.uint8)).hexdigest()
    second_digest = hashlib.sha256(np.ascontiguousarray(second).view(np.uint8)).hexdigest()
    return ReplayProbeResult(
        bool(np.array_equal(first, second)),
        difference,
        first_digest,
        second_digest,
    )


def probe_isolation(
    adapter: ReplayAdapter,
    context: Context,
    baseline: Intervention,
    value: Intervention,
    seed: int,
) -> bool:
    """Check that another intervention cannot alter a fresh baseline branch."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Isolation probing requires a nonnegative integer seed")

    root = adapter.open(context, seed)

    def run(intervention: Intervention) -> np.ndarray:
        missing = object()
        branch = missing
        output = missing
        try:
            branch = adapter.fork(root)
            if branch is root:
                raise ValueError("Isolation probing requires an isolated fork")
            output = adapter.apply(branch, intervention, seed)
            observed = np.asarray(adapter.observe(output), dtype=np.float64)
            if not observed.size or not np.isfinite(observed).all():
                raise ValueError("Isolation observations must be nonempty and finite")
            return np.array(observed, copy=True)
        finally:
            if output is not missing and output is not branch and output is not root:
                adapter.close(output)
            if branch is not missing and branch is not root:
                adapter.close(branch)

    try:
        before = run(baseline)
        run(value)
        after = run(baseline)
        return before.shape == after.shape and np.array_equal(before, after)
    finally:
        adapter.close(root)


def dumps_program(program: SceneProgram) -> str:
    if not isinstance(program, SceneProgram):
        raise TypeError("dumps_program requires a SceneProgram")
    body = {
        "schema": SCENE_PROGRAM_SCHEMA,
        "kind": "scene-program",
        "payload": program.to_record(),
    }
    envelope = {**body, "sha256": fingerprint(body)}
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def loads_program(text: str) -> SceneProgram:
    if not isinstance(text, str):
        raise TypeError("loads_program requires JSON text")
    try:
        envelope = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Invalid scene-program JSON: {error}") from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema", "kind", "payload", "sha256"
    }:
        raise ValueError("Invalid scene-program envelope")
    if envelope["schema"] != SCENE_PROGRAM_SCHEMA or envelope["kind"] != "scene-program":
        raise ValueError("Unsupported scene-program schema or artifact kind")
    body = {key: envelope[key] for key in ("schema", "kind", "payload")}
    if fingerprint(body) != envelope["sha256"]:
        raise ValueError("Scene-program digest mismatch")
    return SceneProgram.from_record(envelope["payload"])
