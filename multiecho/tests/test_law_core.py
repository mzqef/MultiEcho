"""Reference law core on known-answer data: planted laws of known rank, a pure-noise world and a trivial world.

The numeric tolerances here are engineering-test tolerances on constructed data (README §5
rule 12); they verify the implementation, not a capability level of the plugin on any world.
"""
from __future__ import annotations

import numpy as np
import pytest

from multiecho_framework import DataRole
from multiecho_framework.law_core import (
    CodeSpec,
    DenseCurve,
    LawScope,
    LocalInterpolation,
    NoiseModel,
    ResponseBlock,
    SharedProfile,
    Unit,
    ZeroLaw,
    conformal_threshold,
    data_gain,
    default_candidates,
    delta_l,
    law_card,
    recover,
    select,
)

SCOPE = LawScope("a" * 64, "b" * 64, "video", "u", "output", "linear", plugin_set="law-core-dev")
GRID = np.linspace(0.0, 1.0, 21)
BASELINE = 0.5
VALUES = GRID[~np.isclose(GRID, BASELINE)]


def planted(profiles, *, role, contexts, dimension=10, sigma=0.02, seed=0, replicates=2):
    """Δ_i(u) = Σ_k a_k(u)·b_ik + noise; directions differ per context, replicates differ only by noise."""
    rng = np.random.default_rng([seed, list(DataRole).index(role)])
    units = []
    for c in range(contexts):
        directions = rng.normal(size=(len(profiles), dimension))
        clean = sum(np.outer(profile(VALUES) - profile(np.array([BASELINE]))[0], direction)
                    for profile, direction in zip(profiles, directions, strict=True)) if profiles else \
            np.zeros((len(VALUES), dimension))
        for r in range(replicates):
            units.append(Unit(f"{role.value}:{c:02d}", r, VALUES, clean + rng.normal(scale=sigma, size=clean.shape)))
    return ResponseBlock(SCOPE, role, BASELINE, (0.0, 1.0), GRID, tuple(units))


def tanh_profile(center, width):
    return lambda u: np.tanh((np.asarray(u) - center) / width)


SMOOTH = (tanh_profile(0.7, 0.05), lambda u: (np.asarray(u) - 0.2) ** 2)


def world(profiles, **kwargs):
    discovery = planted(profiles, role=DataRole.DISCOVERY, contexts=12, **kwargs)
    validation = planted(profiles, role=DataRole.VALIDATION, contexts=8, **kwargs)
    noise = NoiseModel.from_replicates(discovery, CodeSpec())
    return discovery, validation, noise


def test_zero_law_and_calibration_points_cost_nothing():
    discovery, validation, noise = world(SMOOTH)
    assert delta_l(ZeroLaw.fit(discovery, noise), validation, noise).value == 0.0
    dense = DenseCurve.fit(discovery, noise)
    gains, coded, calibrated = data_gain(dense, validation, noise)
    assert coded == 0 and calibrated == len(VALUES) * len(validation.units) and sum(gains.values()) == 0.0
    assert delta_l(dense, validation, noise).value < 0
    with pytest.raises(ValueError, match="held-out"):
        delta_l(dense, discovery, noise)


BALANCED = (lambda u: np.sin(2 * np.pi * np.asarray(u)), lambda u: np.cos(2 * np.pi * np.asarray(u)))


def _span_r2(law, profiles):
    fitted = law.profiles(VALUES)
    truth = np.stack([profile(VALUES) - profile(np.array([BASELINE]))[0] for profile in profiles], axis=1)
    projection = fitted @ np.linalg.lstsq(fitted, truth, rcond=None)[0]
    return 1 - ((truth - projection) ** 2).sum() / (truth ** 2).sum()


def _heldout_rms(law, block):
    residuals = []
    for unit in block.units:
        mask = ~np.isin(unit.values, law.calibration_values)
        residuals.append(unit.effects[mask] - law.predict(law.calibrate(unit), unit.values[mask]))
    return float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))


def test_shared_profile_recovers_planted_rank_and_subspace():
    discovery, validation, noise = world(BALANCED)
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=4), folds=6)
    assert ranked[0].name in {"family2:K2", "family2:K2:gain"}
    law = ranked[0].law
    assert _span_r2(law, BALANCED) > 0.999 and len(law.calibration_values) == 2
    assert delta_l(law, validation, noise).value > delta_l(LocalInterpolation.fit(discovery, noise), validation, noise).value


def test_n1_code_charges_calibration_observations_as_zero_response():
    """Code N1: calibration observations are coded as zero response. With low noise and one dominant
    profile, the extra calibration of a rank-2 law can cost more bits than its better fit saves; the
    rank-2 law still recovers the planted subspace and predicts better. This is a property of the
    declared code, reported rather than hidden."""
    discovery, validation, noise = world(SMOOTH)
    one, two = (SharedProfile.fit(discovery, noise, rank=rank) for rank in (1, 2))
    assert _span_r2(two, SMOOTH) > 0.999
    assert _heldout_rms(two, validation) < 0.5 * _heldout_rms(one, validation)
    assert delta_l(one, validation, noise).value > delta_l(two, validation, noise).value


def test_gain_calibration_design_prices_calibration_observations():
    """Gain calibration design: choosing the calibration values by discovery data gain under the same code
    removes the rank inversion of the D-optimal design (the rank-2 law now compresses more than the
    rank-1 law), at the price of a larger residual on the uncalibrated values; both are reported."""
    discovery, validation, noise = world(SMOOTH)
    laws = {(rank, design): SharedProfile.fit(discovery, noise, rank=rank, design=design)
            for rank in (1, 2) for design in ("d-optimal", "gain")}
    gains = {key: delta_l(law, validation, noise).value for key, law in laws.items()}
    assert gains[(2, "gain")] > gains[(1, "gain")] >= gains[(1, "d-optimal")]
    assert gains[(2, "gain")] > gains[(2, "d-optimal")]
    assert laws[(2, "gain")].variant == "bspline-K2-gain"
    restored = laws[(2, "gain")].refit(discovery, noise)
    assert restored.calibration_values == laws[(2, "gain")].calibration_values


def test_the_world_own_law_is_read_not_a_physical_prior():
    """A rewritten world: its transition sits at 0.3 instead of 0.7; the fitted profile follows the world."""
    discovery, _, noise = world((tanh_profile(0.3, 0.05),))
    law = SharedProfile.fit(discovery, noise, rank=1)
    fine = np.linspace(0, 1, 401)
    slope = np.abs(np.gradient(law.profiles(fine)[:, 0], fine))
    assert abs(fine[int(np.argmax(slope))] - 0.3) < 0.03


def test_pure_noise_world_yields_no_law_relative_to_the_plugin_set():
    discovery, validation, noise = world(())
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=3), folds=6)
    best = ranked[0]
    assert best.name == "zero" or best.cv_delta_l <= 0
    card = law_card(ZeroLaw.fit(discovery, noise), SCOPE, discovery, validation, noise, CodeSpec(),
                    confidence=0.95, resamples=500, seed=1)
    assert card.status == "not_found" and "under plugin set law-core-dev" in card.statement


def test_trivial_linear_world_is_compressed_by_a_calibrated_law():
    discovery, validation, noise = world((lambda u: np.asarray(u),))
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=3), folds=6)
    assert ranked[0].name in {"family0:pl-axis", "family2:K1", "family2:K1:gain"}
    assert delta_l(ranked[0].law, validation, noise).value > 0


def test_backward_law_reports_identifiable_and_non_injective_cases():
    discovery, validation, noise = world((tanh_profile(0.7, 0.3),))
    law = SharedProfile.fit(discovery, noise, rank=1)
    fit = lambda block: law.refit(block, noise)  # noqa: E731
    threshold = conformal_threshold(fit, discovery, noise, alpha=0.1, folds=4)
    unit = validation.units[0]
    state = law.calibrate(unit)
    index = int(np.flatnonzero(np.isclose(unit.values, 0.9))[0])
    result = recover(law, state, unit.effects[index], noise, threshold=threshold, tolerance=0.05)
    assert result.identifiable and abs(result.estimate - 0.9) < 0.05

    folded = world((lambda u: (np.asarray(u) - 0.5) ** 2,))
    law = SharedProfile.fit(folded[0], folded[2], rank=1)
    unit = folded[1].units[0]
    threshold = conformal_threshold(lambda block: law.refit(block, folded[2]), folded[0], folded[2], alpha=0.1, folds=4)
    index = int(np.flatnonzero(np.isclose(unit.values, 0.9))[0])
    result = recover(law, law.calibrate(unit), unit.effects[index], folded[2], threshold=threshold, tolerance=0.05)
    assert not result.identifiable and result.reason == "non-injective" and result.estimate is None


def test_law_card_is_scoped_and_refuses_mixed_scopes():
    discovery, validation, noise = world(SMOOTH)
    law = SharedProfile.fit(discovery, noise, rank=2)
    card = law_card(law, SCOPE, discovery, validation, noise, CodeSpec(), confidence=0.95, resamples=500, seed=2)
    assert card.status == "found" and card.scope["world"] == SCOPE.world and card.reuse["cross_world"] == []
    assert card.reuse["context"]["interval"][0] > 0 and card.reuse["context"]["forward_r2"] > 0.99
    assert card.digest == law_card(law, SCOPE, discovery, validation, noise, CodeSpec(), confidence=0.95,
                                   resamples=500, seed=2).digest
    other = LawScope("c" * 64, SCOPE.program, SCOPE.observation, SCOPE.axis, SCOPE.readout, SCOPE.coordinate)
    with pytest.raises(ValueError, match="one scope"):
        law_card(law, other, discovery, validation, noise, CodeSpec(), confidence=0.95, resamples=10, seed=0)


def test_exactly_zero_world_fits_without_singular_calibration():
    """A world with no response at all (e.g. the outside of a support that holds every effect) stays fittable."""
    discovery = planted((), role=DataRole.DISCOVERY, contexts=6, sigma=0.0)
    noise = NoiseModel.from_replicates(planted((), role=DataRole.DISCOVERY, contexts=6, seed=1), CodeSpec())
    law = SharedProfile.fit(discovery, noise, rank=1)
    state = law.calibrate(discovery.units[0])
    assert np.all(np.isfinite(state))
    assert np.allclose(law.predict(state, VALUES), 0.0)


def test_indistinguishable_candidates_are_reported_not_hidden():
    """Law recovery is relative to the data: when two ranks give the same held-out-fold gain within the
    declared confidence, the selection names the other one; clear-cut selections name none."""
    from multiecho_framework.law_core import indistinguishable

    discovery, _, noise = world(SMOOTH)
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=3), folds=6)
    tied = indistinguishable(ranked, confidence=0.9, resamples=2000, seed=0)
    assert tied and all(name.startswith("family2:") for name in tied)
    assert tied == indistinguishable(ranked, confidence=0.9, resamples=2000, seed=0)
    discovery, _, noise = world(BALANCED)
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=3), folds=6)
    assert "zero" not in indistinguishable(ranked, confidence=0.9, resamples=2000, seed=0)
