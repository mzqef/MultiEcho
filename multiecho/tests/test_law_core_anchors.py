"""Per-unit baselines and the N2 code on known-answer data (SP25 R1/R3).

Contexts whose baselines differ on one axis share one profile pinned to zero at each unit's own baseline;
the N2 code needs no replicates. Tolerances are engineering-test tolerances on constructed data (README §5
rule 12): they verify the implementation, not a capability of any plugin on any world.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from test_law_core import SCOPE, SMOOTH, planted

from multiecho_framework import ChannelTest, DataRole, EvidenceStatus, WorldKind
from multiecho_framework.law_core import (
    Anchored,
    CodeSpec,
    DenseCurve,
    LocalInterpolation,
    NoiseModel,
    ResponseBlock,
    SharedProfile,
    Unit,
    ZeroLaw,
    candidate_ranks,
    data_gain,
    default_candidates,
    delta_l,
    forward_r2,
    law_channel,
    law_from_state,
    law_state,
    select,
)

GRID = np.linspace(0.0, 1.0, 21)
ANCHORS = (0.3, 0.5, 0.65)          # on the grid: each unit lacks the grid value equal to its own baseline


def anchored(profiles, *, role, contexts, anchors=ANCHORS, dimension=10, sigma=0.02, seed=0, replicates=2,
             shared=None):
    rng = np.random.default_rng([seed, list(DataRole).index(role), 7])
    units = []
    for c in range(contexts):
        anchor = shared if shared is not None else anchors[c % len(anchors)]
        values = GRID[~np.isclose(GRID, anchor)]
        directions = rng.normal(size=(len(profiles), dimension))
        clean = sum(np.outer(profile(values) - profile(np.array([anchor]))[0], direction)
                    for profile, direction in zip(profiles, directions, strict=True))
        for r in range(replicates):
            units.append(Unit(f"{role.value}:{c:02d}", r, values, clean + rng.normal(scale=sigma, size=clean.shape),
                              baseline=None if shared is not None else anchor))
    return ResponseBlock(SCOPE, role, shared, (0.0, 1.0), GRID, tuple(units))


def test_blocks_hold_one_shared_baseline_or_one_per_unit():
    unit = Unit("c", 0, GRID[1:], np.zeros((20, 2)), baseline=0.0)
    with pytest.raises(ValueError, match="shared baseline holds"):
        ResponseBlock(SCOPE, DataRole.DISCOVERY, 0.0, (0.0, 1.0), GRID, (unit,))
    with pytest.raises(ValueError, match="declares its own"):
        ResponseBlock(SCOPE, DataRole.DISCOVERY, None, (0.0, 1.0), GRID, (Unit("c", 0, GRID[1:], np.zeros((20, 2))),))
    with pytest.raises(ValueError, match="not a unit value"):
        ResponseBlock(SCOPE, DataRole.DISCOVERY, None, (0.0, 1.0), GRID,
                      (Unit("c", 0, GRID, np.zeros((21, 2)), baseline=0.5),))
    block = anchored(SMOOTH, role=DataRole.DISCOVERY, contexts=6)
    assert block.anchors == ANCHORS and block.digest != anchored(SMOOTH, role=DataRole.DISCOVERY, contexts=6,
                                                                 anchors=(0.3, 0.5, 0.7)).digest


def test_shared_profiles_with_per_unit_baselines_recover_a_planted_law():
    discovery = anchored(SMOOTH, role=DataRole.DISCOVERY, contexts=12)
    validation = anchored(SMOOTH, role=DataRole.VALIDATION, contexts=9)
    noise = NoiseModel.from_replicates(discovery, CodeSpec())
    law = SharedProfile.fit(discovery, noise, rank=2)
    assert law.baseline is None and not set(law.calibration_values) & set(ANCHORS)
    assert forward_r2(law, validation, discovery) > 0.99
    gain = delta_l(law, validation, noise)
    assert gain.value > 0 and gain.calibration_points == 2 * len(validation.units)
    state = law.calibrate(validation.units[0])
    assert isinstance(state, Anchored) and state.anchor == validation.units[0].baseline
    assert np.allclose(law.predict(state, [state.anchor]), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="anchor"):
        law.profiles(GRID)
    restored = law_from_state(*law_state(law))
    assert restored.code_digest == law.code_digest and restored.baseline is None
    assert np.allclose(restored.predict(restored.calibrate(validation.units[1]), GRID),
                       law.predict(law.calibrate(validation.units[1]), GRID))
    gain_design = SharedProfile.fit(discovery, noise, rank=2, design="gain")
    assert delta_l(gain_design, validation, noise).value > 0
    # A held-out unit whose baseline is one of the calibration values calibrates on the values it has.
    odd = anchored(SMOOTH, role=DataRole.VALIDATION, contexts=2, anchors=(law.calibration_values[0],), seed=4)
    assert law.calibration_mask(odd.units[0]).sum() == 1
    assert np.isfinite(delta_l(law, odd, noise).value)


def test_families_zero_and_one_and_selection_with_per_unit_baselines():
    discovery = anchored(SMOOTH, role=DataRole.DISCOVERY, contexts=12)
    validation = anchored(SMOOTH, role=DataRole.VALIDATION, contexts=6)
    noise = NoiseModel.from_replicates(discovery, CodeSpec())
    local = LocalInterpolation.fit(discovery, noise)
    prediction = local.predict(local.calibrate(validation.units[0]), [validation.units[0].baseline])
    assert np.allclose(prediction, 0.0)
    dense = DenseCurve.fit(discovery, noise)
    gains, coded, calibrated = data_gain(dense, validation, noise)
    assert coded == 0 and calibrated == sum(len(unit.values) for unit in validation.units)
    assert candidate_ranks(discovery) == tuple(range(1, (21 - 3) // 3 + 1))
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=3), folds=4)
    assert ranked[0].name.startswith("family2:K2")
    assert all(item.law is None or item.law.baseline is None for item in ranked)


@pytest.mark.parametrize("family", ["local", "dense", "shared"])
def test_one_common_anchor_matches_the_shared_baseline_path(family):
    kwargs = {"contexts": 8, "seed": 3}
    shared_d = anchored(SMOOTH, role=DataRole.DISCOVERY, shared=0.5, **kwargs)
    shared_v = anchored(SMOOTH, role=DataRole.VALIDATION, shared=0.5, **kwargs)
    unit_d = ResponseBlock(SCOPE, DataRole.DISCOVERY, None, (0.0, 1.0), GRID, tuple(
        Unit(u.context, u.replicate, u.values, u.effects, baseline=0.5) for u in shared_d.units))
    unit_v = ResponseBlock(SCOPE, DataRole.VALIDATION, None, (0.0, 1.0), GRID, tuple(
        Unit(u.context, u.replicate, u.values, u.effects, baseline=0.5) for u in shared_v.units))
    noise = NoiseModel.from_replicates(shared_d, CodeSpec())
    fit = {"local": LocalInterpolation.fit, "dense": DenseCurve.fit,
           "shared": lambda block, n: SharedProfile.fit(block, n, rank=2)}[family]
    a, b = fit(shared_d, noise), fit(unit_d, noise)
    if family != "dense":        # family 1 calibrates on every observed value; per unit it lists the whole grid
        assert a.calibration_values == b.calibration_values
    for left, right in zip(shared_v.units, unit_v.units, strict=True):
        assert np.allclose(a.predict(a.calibrate(left), GRID), b.predict(b.calibrate(right), GRID), atol=1e-8)


def test_n2_gain_is_the_log_ratio_of_residual_sums_per_context_and_group():
    code = CodeSpec(noise="N2-gauss-ml", sigma_source="ml:per-location")
    values = np.array([0.0, 0.25, 0.75, 1.0])
    effects = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [1.0, 1.0]])
    block = ResponseBlock(SCOPE, DataRole.DISCOVERY, 0.5, (0.0, 1.0), values, (Unit("c", 0, values, effects),))
    noise = NoiseModel.for_ml(block, code, groups=[7, 3])
    assert noise.groups.tolist() == [1, 0] and noise.code == "N2"
    held = ResponseBlock(SCOPE, DataRole.VALIDATION, 0.5, (0.0, 1.0), values, (Unit("v", 0, values, effects),))
    local = LocalInterpolation.fit(block, noise)          # calibrated at 0 and 1; predicts ±(…) at 0.25 / 0.75
    gains, coded, calibrated = data_gain(local, held, noise)
    prediction = local.predict(local.calibrate(held.units[0]), values[1:3])
    expected = 0.0
    for column in range(2):
        zero = float((effects[:, column] ** 2).sum())
        residual = float(((effects[1:3, column] - prediction[:, column]) ** 2).sum()
                         + (effects[[0, 3], column] ** 2).sum())
        expected += 4 / 2 * math.log2(max(zero, 4 * noise.floor) / max(residual, 4 * noise.floor))
    assert coded == 2 and calibrated == 2 and gains["v"] == pytest.approx(expected)
    assert data_gain(ZeroLaw.fit(block, noise), held, noise)[0]["v"] == 0.0
    silent = ResponseBlock(SCOPE, DataRole.VALIDATION, 0.5, (0.0, 1.0), values,
                           (Unit("s", 0, values, np.zeros((4, 2))),))
    assert data_gain(local, silent, noise)[0]["s"] == 0.0        # residual sums floored at the quantization
    with pytest.raises(ValueError, match="codes"):
        CodeSpec(noise="N2-gauss-ml", sigma_source="paired-replicates:discovery")


@pytest.mark.parametrize("source", ["ml:pooled", "ml:per-location"])
def test_n2_finds_a_planted_law_without_replicates_and_none_in_noise(source):
    code = CodeSpec(noise="N2-gauss-ml", sigma_source=source)
    groups = np.repeat(np.arange(5), 2)
    discovery = planted(SMOOTH, role=DataRole.DISCOVERY, contexts=12, seed=5, replicates=1)
    validation = planted(SMOOTH, role=DataRole.VALIDATION, contexts=8, seed=5, replicates=1)
    noise = NoiseModel.for_ml(discovery, code, groups=groups)
    ranked = select(discovery, noise, default_candidates(discovery, max_rank=3), folds=4)
    assert ranked[0].law.family == 2 and delta_l(ranked[0].law, validation, noise).value > 0
    null_d = planted((), role=DataRole.DISCOVERY, contexts=12, seed=6, replicates=1)
    null_v = planted((), role=DataRole.VALIDATION, contexts=8, seed=6, replicates=1)
    null_noise = NoiseModel.for_ml(null_d, code, groups=groups)
    best = select(null_d, null_noise, default_candidates(null_d, max_rank=3), folds=4)[0]
    assert best.law.family == -1 or delta_l(best.law, null_v, null_noise).value <= 0


def test_law_channel_pins_profiles_at_the_receivers_own_baseline():
    discovery = anchored(SMOOTH, role=DataRole.DISCOVERY, contexts=12, seed=8)
    noise = NoiseModel.from_replicates(discovery, CodeSpec())
    law = SharedProfile.fit(discovery, noise, rank=2)
    assert 0.6 not in law.calibration_values
    receiver = anchored(SMOOTH, role=DataRole.VALIDATION, contexts=8, seed=9, dimension=6, shared=0.6)
    record, _ = law_channel(law, "a" * 64, receiver, "b" * 64, source_kind=WorldKind.SIMULATION,
                            target_kind=WorldKind.LEARNED, minimum_improvement=0.0, resamples=300, seed=1)
    assert record.statuses[ChannelTest.CONSISTENCY] is EvidenceStatus.SUPPORTED
    # A receiver whose own baseline is a calibration value keeps K − 1 informative calibrations: the rank-K
    # law is then not identified there, and the channel reports that instead of substituting a value.
    pinned = anchored(SMOOTH, role=DataRole.VALIDATION, contexts=8, seed=9, dimension=6,
                      shared=law.calibration_values[0])
    record, _ = law_channel(law, "a" * 64, pinned, "b" * 64, source_kind=WorldKind.SIMULATION,
                            target_kind=WorldKind.LEARNED, minimum_improvement=0.0, resamples=300, seed=1)
    assert record.statuses[ChannelTest.CONSISTENCY] is not EvidenceStatus.SUPPORTED
