"""Law families 0-2 and the zero-response law, for any world's response block.

Every law exposes the same interface: `calibration_values` (natural coordinates a new unit must
supply), `calibrate(unit)`, `predict(state, values)`, and its code length inputs `k`, `n_fit` and
`struct_bits`. Laws are fitted on discovery only and then frozen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..contracts import DataRole, fingerprint
from .basis import ProfileBasis
from .block import ResponseBlock, Unit
from .code import LN2, NoiseModel


class NotApplicable(ValueError):
    """The law family does not apply to this block (recorded as a reason, not as a finding)."""


def _require_discovery(block: ResponseBlock) -> None:
    if block.role is not DataRole.DISCOVERY:
        raise ValueError("Laws are fitted on discovery only")


def _n_scalars(block: ResponseBlock) -> int:
    return int(sum(unit.effects.size for unit in block.units))


@dataclass(frozen=True)
class FittedLaw:
    family: int
    variant: str
    form: str
    domain: tuple[float, float]
    baseline: float
    calibration_values: tuple[float, ...]
    k: float
    n_fit: int
    struct_bits: float
    parameters: dict = field(repr=False)

    @property
    def code_digest(self) -> str:
        return fingerprint({"family": self.family, "variant": self.variant, "k": self.k, "n": self.n_fit,
                            "struct": self.struct_bits, "calibration": list(self.calibration_values),
                            "parameters": {key: np.asarray(value).round(12).tolist() if isinstance(value, np.ndarray)
                                           else value for key, value in self.parameters.items()}})

    def refit(self, block: ResponseBlock, noise: NoiseModel) -> FittedLaw:
        """Refit on another discovery subset with this law's frozen structural choices."""
        return type(self).fit(block, noise)

    def calibrate(self, unit: Unit):
        raise NotImplementedError

    def predict(self, state, values) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class ZeroLaw(FittedLaw):
    """The reference code: no response. Its ΔL is zero by definition."""

    @classmethod
    def fit(cls, block: ResponseBlock, noise: NoiseModel) -> ZeroLaw:
        _require_discovery(block)
        return cls(-1, "zero", "Δ(u) = 0", block.domain, block.baseline, (), 0.0, _n_scalars(block), 0.0,
                   {"dimension": block.dimension})

    def calibrate(self, unit: Unit):
        return unit.effects.shape[1]

    def predict(self, state, values) -> np.ndarray:
        return np.zeros((len(np.atleast_1d(values)), state))


@dataclass(frozen=True)
class LocalInterpolation(FittedLaw):
    """Family 0, piecewise-linear three-reference law: baseline plus the two axis-end calibrations.

    Outside [u-, u+] the law makes no prediction (coded as zero response).
    """

    @classmethod
    def fit(cls, block: ResponseBlock, noise: NoiseModel) -> LocalInterpolation:
        _require_discovery(block)
        lower, upper = float(block.base_grid[0]), float(block.base_grid[-1])
        if not lower < block.baseline < upper:
            raise NotApplicable("Family 0 needs the baseline strictly between the two calibration values")
        return cls(0, "pl-axis", "Δ(u) = w(u)·Δ(u±), piecewise linear through (u-, Δ-), (u0, 0), (u+, Δ+)",
                   block.domain, block.baseline, (lower, upper), 0.0, _n_scalars(block), 0.0,
                   {"window": "axis"})

    def calibrate(self, unit: Unit):
        return unit.effects[unit.index(self.calibration_values)]

    def predict(self, state, values) -> np.ndarray:
        values = np.atleast_1d(np.asarray(values, dtype=float))
        (lower, upper), u0 = self.calibration_values, self.baseline
        below = np.where((values >= lower) & (values < u0), (u0 - values) / (u0 - lower), 0.0)
        above = np.where((values > u0) & (values <= upper), (values - u0) / (upper - u0), 0.0)
        return below[:, None] * state[0] + above[:, None] * state[1]


def _penalized(design: np.ndarray, penalty: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(design.T @ design + lam * penalty + 1e-12 * np.eye(len(penalty)), design.T)


def _lambda_grid(gram: np.ndarray, penalty: np.ndarray, count: int = 19) -> np.ndarray:
    scale = np.trace(gram) / max(np.trace(penalty), 1e-300)
    return scale * np.logspace(-8, 2, count)


@dataclass(frozen=True)
class DenseCurve(FittedLaw):
    """Family 1, one penalized spline per unit through all base-grid values: shape, not compression."""

    @classmethod
    def fit(cls, block: ResponseBlock, noise: NoiseModel) -> DenseCurve:
        _require_discovery(block)
        values = tuple(float(v) for v in block.base_grid if not np.isclose(v, block.baseline, rtol=0, atol=1e-12))
        basis = ProfileBasis(block.domain, block.base_grid, block.baseline)
        design = basis(np.array(values))
        if len(values) < 4:
            raise NotApplicable("Family 1 needs at least four calibration values")
        scores = []
        candidates = _lambda_grid(design.T @ design, basis.penalty)
        total = sum(unit.effects.size for unit in block.units)
        for lam in candidates:
            smoother = design @ _penalized(design, basis.penalty, lam)
            trace = float(np.trace(smoother))
            rss = 0.0
            for unit in block.units:
                y = unit.effects[unit.index(values)] / np.sqrt(noise.sigma2)
                rss += float(((y - smoother @ y) ** 2).sum())
            dof = total - block.dimension * len(block.units) * trace
            scores.append(np.inf if dof <= 0 else total * rss / dof ** 2)
        lam = float(candidates[int(np.argmin(scores))])
        return cls(1, "p-spline", "Δ_i(u) = ψ(u)·C_i, penalized cubic spline per unit through all grid values",
                   block.domain, block.baseline, values, 1.0, _n_scalars(block), 0.0,
                   {"lambda": lam, "knots": basis.knots_at})

    def _basis(self) -> ProfileBasis:
        return ProfileBasis(self.domain, np.asarray(self.parameters["knots"]), self.baseline)

    def calibrate(self, unit: Unit):
        basis = self._basis()
        design = basis(np.array(self.calibration_values))
        return _penalized(design, basis.penalty, self.parameters["lambda"]) @ unit.effects[unit.index(self.calibration_values)]

    def predict(self, state, values) -> np.ndarray:
        return self._basis()(values) @ state


@dataclass(frozen=True)
class SharedProfile(FittedLaw):
    """Family 2: Δ_i(u) ≈ Σ_k a_k(u)·b_{i,k}; profiles shared across units, directions per unit.

    A new unit is calibrated at K frozen values chosen on discovery, so the law needs K calibration
    observations per context. Two calibration designs: `d-optimal` (best-conditioned profile matrix) and
    `gain` (largest discovery data gain under the declared code; calibration observations are coded as
    zero response, so this design prices them). Both are candidates for selection.
    """

    DESIGNS = ("d-optimal", "gain")

    @classmethod
    def fit(cls, block: ResponseBlock, noise: NoiseModel, *, rank: int, lam: float | None = None,
            iterations: int = 500, tolerance: float = 1e-9, folds: int = 6,
            design: str = "d-optimal") -> SharedProfile:
        if design not in cls.DESIGNS:
            raise ValueError(f"Calibration design is one of {cls.DESIGNS}")
        _require_discovery(block)
        candidates = tuple(float(v) for v in block.base_grid
                           if not np.isclose(v, block.baseline, rtol=0, atol=1e-12))
        if type(rank) is not int or not 1 <= rank <= max(1, len(candidates) // 3):
            raise NotApplicable("Shared-profile rank must leave at least twice as many uncalibrated values")
        basis = ProfileBasis(block.domain, block.base_grid, block.baseline)
        if lam is None:
            lam = _select_lambda(block, noise, basis, rank, folds, iterations, tolerance)
        profile, iterations_used, objective = _als(block, noise, basis, rank, lam, iterations, tolerance)
        chosen = (_d_optimal(profile, basis, candidates, rank, block.baseline) if design == "d-optimal"
                  else _gain_design(profile, basis, candidates, rank, block, noise))
        gram = sum(basis(unit.values).T @ basis(unit.values) for unit in block.units)
        dof = float(np.trace(np.linalg.solve(gram + lam * basis.penalty, gram)))
        k = rank * max(dof - rank, 0.0) + 1.0
        struct = math.log2(math.comb(len(candidates), rank))
        variant = f"bspline-K{rank}" + ("" if design == "d-optimal" else f"-{design}")
        parameters = {"rank": rank, "lambda": float(lam), "profile": profile, "knots": basis.knots_at,
                      "iterations": iterations_used, "objective": objective, "dof": dof}
        if design != "d-optimal":
            parameters["design"] = design
        return cls(2, variant, f"Δ_i(u) = Σ_k a_k(u)·b_ik, K = {rank} shared profiles",
                   block.domain, block.baseline, chosen, k, _n_scalars(block), struct, parameters)

    def _basis(self) -> ProfileBasis:
        return ProfileBasis(self.domain, np.asarray(self.parameters["knots"]), self.baseline)

    def refit(self, block: ResponseBlock, noise: NoiseModel) -> SharedProfile:
        return SharedProfile.fit(block, noise, rank=self.parameters["rank"], lam=self.parameters["lambda"],
                                 design=self.parameters.get("design", "d-optimal"))

    def profiles(self, values) -> np.ndarray:
        return self._basis()(values) @ self.parameters["profile"]

    def calibrate(self, unit: Unit):
        matrix = self.profiles(np.array(self.calibration_values))
        observed = unit.effects[unit.index(self.calibration_values)]
        gram = matrix.T @ matrix
        if not np.trace(gram) > 0:
            return np.linalg.lstsq(matrix, observed, rcond=None)[0]   # zero profile: minimum-norm state
        if np.linalg.cond(matrix) > 1e4:
            return np.linalg.solve(gram + 1e-8 * np.trace(gram) * np.eye(len(matrix)), matrix.T @ observed)
        return np.linalg.solve(matrix, observed)

    def predict(self, state, values) -> np.ndarray:
        return self.profiles(values) @ state


def _als(block, noise, basis, rank, lam, iterations, tolerance):
    scale = 1.0 / np.sqrt(noise.sigma2)
    first = block.units[0].values
    if all(len(unit.values) == len(first) and np.allclose(unit.values, first, rtol=0, atol=1e-12)
           for unit in block.units):
        # With a shared value set the profile fit depends on the data only through the J x J Gram
        # matrix Σ_i Y_i Y_iᵀ, so one J x J square root replaces all units exactly.
        gram = sum((unit.effects * scale) @ (unit.effects * scale).T for unit in block.units)
        eigenvalues, vectors = np.linalg.eigh(gram)
        designs = [basis(first)]
        targets = [vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))]
    else:
        designs = [basis(unit.values) for unit in block.units]
        targets = [unit.effects * scale for unit in block.units]
    unit_values = [first] if len(designs) == 1 and len(block.units) > 1 else [unit.values for unit in block.units]
    union = np.unique(np.concatenate(unit_values))
    stacked = np.concatenate([np.stack([np.interp(union, values, column) for column in target.T], axis=1)
                              for values, target in zip(unit_values, targets, strict=True)], axis=1)
    left = np.linalg.svd(stacked, full_matrices=False)[0][:, :rank]
    union_design = basis(union)
    profile = _penalized(union_design, basis.penalty, lam) @ left
    fine = np.linspace(*block.domain, 401)
    profile = profile / np.maximum(np.linalg.norm(basis(fine) @ profile, axis=0) / math.sqrt(len(fine)), 1e-300)
    previous = np.inf
    used = 0
    for iteration in range(1, iterations + 1):
        used = iteration
        directions = [np.linalg.lstsq(design @ profile, target, rcond=None)[0]
                      for design, target in zip(designs, targets, strict=True)]
        normal = lam * np.kron(np.eye(rank), basis.penalty)
        rhs = np.zeros((basis.size, rank))
        for design, target, direction in zip(designs, targets, directions, strict=True):
            normal += np.kron(direction @ direction.T, design.T @ design)
            rhs += design.T @ target @ direction.T
        profile = np.linalg.solve(normal + 1e-12 * np.eye(len(normal)), rhs.reshape(-1, order="F"))
        profile = profile.reshape(basis.size, rank, order="F")
        objective = sum(float(((target - design @ profile @ direction) ** 2).sum())
                        for design, target, direction in zip(designs, targets, directions, strict=True))
        objective += lam * float(np.trace(profile.T @ basis.penalty @ profile))
        if abs(previous - objective) <= tolerance * max(abs(objective), 1e-300):
            break
        previous = objective
    return _canonical(profile, basis, designs, targets, block), used, objective


def _canonical(profile, basis, designs, targets, block):
    """Orthonormal profiles on the domain, ordered by direction energy, positive at the far end."""
    fine = np.linspace(*block.domain, 401)
    values = basis(fine) @ profile
    gram = values.T @ values / len(fine)
    root = np.linalg.cholesky(gram + 1e-14 * np.eye(len(gram)))
    whiten = np.linalg.inv(root).T
    profile = profile @ whiten
    energy = np.zeros((profile.shape[1],) * 2)
    for design, target in zip(designs, targets, strict=True):
        direction = np.linalg.lstsq(design @ profile, target, rcond=None)[0]
        energy += direction @ direction.T
    eigenvalues, vectors = np.linalg.eigh(energy)
    profile = profile @ vectors[:, np.argsort(eigenvalues)[::-1]]
    far = block.domain[0] if abs(block.domain[0] - block.baseline) > abs(block.domain[1] - block.baseline) else block.domain[1]
    signs = np.sign(basis(np.array([far])) @ profile)[0]
    return profile * np.where(signs == 0, 1.0, signs)


def _d_optimal(profile, basis, candidates, rank, baseline):
    values = basis(np.array(candidates)) @ profile

    def score(chosen):
        sign, logdet = np.linalg.slogdet(values[list(chosen)])
        return (logdet if sign != 0 else -np.inf, -sum(abs(candidates[i] - baseline) for i in chosen))

    chosen: list[int] = []
    for _ in range(rank):
        best = max((i for i in range(len(candidates)) if i not in chosen),
                   key=lambda i: _partial_volume(values, chosen + [i]))
        chosen.append(best)
    improved = True
    while improved:
        improved = False
        for position in range(rank):
            for candidate in range(len(candidates)):
                if candidate in chosen:
                    continue
                trial = chosen.copy()
                trial[position] = candidate
                if score(trial) > score(chosen):
                    chosen, improved = trial, True
    return tuple(sorted(candidates[i] for i in chosen))


def _gain_design(profile, basis, candidates, rank, block, noise):
    """Calibration values maximising the discovery data gain of the law under the declared code.

    With calibration set S, a unit's prediction at u is c_u·y_S with c_u = a(u)·A_S⁻¹ shared by all units,
    so the gain Σ_{u∉S} w·(y_u² − (y_u − c_u·y_S)²) = Σ_{u∉S} 2 c_u·G[S,u] − c_u G[S,S] c_uᵀ needs only
    the noise-weighted Gram matrix G of the units' effects at the candidate values.
    """
    candidates = np.array(candidates)
    weights = 1.0 / (2.0 * noise.sigma2 * LN2)
    gram = np.zeros((len(candidates),) * 2)
    for unit in block.units:
        rows = unit.effects[unit.index(candidates)]
        gram += (rows * weights) @ rows.T
    values = basis(candidates) @ profile

    def gain(chosen):
        matrix = values[list(chosen)]
        if np.linalg.matrix_rank(matrix) < rank:
            return -np.inf
        rest = [i for i in range(len(candidates)) if i not in chosen]
        coefficients = values[rest] @ np.linalg.inv(matrix)
        cross = gram[np.ix_(list(chosen), rest)]
        inner = gram[np.ix_(list(chosen), list(chosen))]
        return float(2 * np.einsum("uk,ku->", coefficients, cross)
                     - np.einsum("uk,kl,ul->", coefficients, inner, coefficients))

    # Start from the D-optimal set (well conditioned) and improve by single swaps.
    start = _d_optimal(profile, basis, tuple(candidates), rank, block.baseline)
    chosen = [int(np.flatnonzero(np.isclose(candidates, value, rtol=0, atol=1e-12))[0]) for value in start]
    best = gain(chosen)
    improved = True
    while improved:
        improved = False
        for position in range(rank):
            for candidate in range(len(candidates)):
                if candidate in chosen:
                    continue
                trial = chosen.copy()
                trial[position] = candidate
                score = gain(trial)
                if score > best + 1e-9 * max(1.0, abs(best)):
                    chosen, best, improved = trial, score, True
    return tuple(sorted(float(candidates[i]) for i in chosen))


def _partial_volume(values, chosen):
    rows = values[chosen]
    return (float(np.linalg.slogdet(rows @ rows.T)[1]), -len(chosen))


def _select_lambda(block, noise, basis, rank, folds, iterations, tolerance):
    contexts = block.contexts
    count = min(folds, len(contexts))
    if count < 2:
        return float(_lambda_grid(basis(block.base_grid).T @ basis(block.base_grid), basis.penalty, 7)[3])
    design = basis(block.base_grid)
    candidates = _lambda_grid(design.T @ design, basis.penalty, 7)
    groups = [contexts[i::count] for i in range(count)]
    scale = 1.0 / np.sqrt(noise.sigma2)
    errors = []
    for lam in candidates:
        total = 0.0
        for held in groups:
            train = block.subset(set(contexts) - set(held))
            try:
                law = SharedProfile.fit(train, noise, rank=rank, lam=float(lam), iterations=iterations,
                                        tolerance=tolerance)
            except (NotApplicable, np.linalg.LinAlgError):
                total = np.inf
                break
            for unit in block.subset(held).units:
                mask = np.ones(len(unit.values), dtype=bool)
                mask[unit.index(law.calibration_values)] = False
                residual = (unit.effects[mask] - law.predict(law.calibrate(unit), unit.values[mask])) * scale
                total += float((residual ** 2).sum())
        errors.append(total)
    return float(candidates[int(np.argmin(errors))])


def candidate_ranks(block: ResponseBlock, maximum: int = 6) -> tuple[int, ...]:
    count = sum(1 for v in block.base_grid if not np.isclose(v, block.baseline, rtol=0, atol=1e-12))
    return tuple(range(1, min(maximum, count // 3) + 1))


__all__ = ["DenseCurve", "FittedLaw", "LocalInterpolation", "NotApplicable", "SharedProfile", "ZeroLaw",
           "candidate_ranks"]


LAW_CLASSES = {cls.__name__: cls for cls in (ZeroLaw, LocalInterpolation, DenseCurve, SharedProfile)}


def law_state(law: FittedLaw) -> tuple[dict, dict[str, np.ndarray]]:
    """Split a fitted law into JSON state and named arrays (for a frozen plugin fit)."""
    if type(law).__name__ not in LAW_CLASSES:
        raise ValueError("Unknown law class")
    arrays = {key: np.asarray(value, dtype=float) for key, value in law.parameters.items()
              if isinstance(value, np.ndarray)}
    scalars = {key: value for key, value in law.parameters.items() if key not in arrays}
    return {"class": type(law).__name__, "family": law.family, "variant": law.variant, "form": law.form,
            "domain": list(law.domain), "baseline": law.baseline, "calibration_values": list(law.calibration_values),
            "k": law.k, "n_fit": law.n_fit, "struct_bits": law.struct_bits, "parameters": scalars,
            "arrays": sorted(arrays), "code_digest": law.code_digest}, arrays


def law_from_state(state: dict, arrays: dict) -> FittedLaw:
    """Inverse of `law_state`; the restored law has the same code digest."""
    parameters = dict(state["parameters"]) | {key: np.array(arrays[key], dtype=float) for key in state["arrays"]}
    law = LAW_CLASSES[state["class"]](
        int(state["family"]), state["variant"], state["form"], tuple(state["domain"]), float(state["baseline"]),
        tuple(float(value) for value in state["calibration_values"]), float(state["k"]), int(state["n_fit"]),
        float(state["struct_bits"]), parameters)
    if law.code_digest != state["code_digest"]:
        raise ValueError("Restored law differs from the frozen law")
    return law
