"""Observation masking: location-level selection, frozen calibration, per-context instances."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from multiecho_framework import (
    CONTEXT_CHANNEL,
    Artifact,
    Channel,
    DataRole,
    EnergyMask,
    Experiment,
    Grouping,
    Intervention,
    ManualMask,
    MaskedScope,
    MaskerCard,
    MaskInput,
    Modality,
    MonotonicSNRMask,
    NoMask,
    RepeatNoiseSNRMask,
    Statement,
    Sweep,
)
from multiecho_framework.conformance import check_report
from multiecho_framework.masking import _jaccard, monotonic_scores, steady_shares
from multiecho_framework.selftest import learned_declaration
from multiecho_framework.testing import PREDICTION_DIMS, SWEEP, InterpolationMethod, learned_chain

VALUES = tuple(Intervention((float(v),)) for v in np.linspace(-1, 1, 8))


def sweep(groups=6, width=3, seeds=1, noise=0.0, seed=0, steady=(0, 2), oscillating=(4,)):
    rng = np.random.default_rng(seed)
    u = np.linspace(-1, 1, len(VALUES))
    outputs = np.zeros((seeds, len(u), groups * width))
    for group in steady:
        outputs[:, :, group * width:(group + 1) * width] = np.outer(u, rng.normal(size=width))
    for group in oscillating:
        outputs[:, :, group * width:(group + 1) * width] = np.outer(np.sin(9 * u), rng.normal(size=width))
    return Sweep(VALUES, tuple(range(seeds)), outputs + noise * rng.normal(size=outputs.shape))


def test_grouping_blocks_whole_and_location_respect():
    grouping = Grouping.blocks(4, 3)
    assert (grouping.dims, grouping.groups) == (12, 4)
    np.testing.assert_array_equal(grouping.expand(np.array([1, 0, 0, 1], bool)), np.repeat([1, 0, 0, 1], 3).astype(bool))
    assert grouping.respects(np.repeat([1, 0, 1, 0], 3).astype(bool))
    assert not grouping.respects(np.eye(12, dtype=bool)[0])
    assert Grouping.whole(5).groups == 1
    np.testing.assert_allclose(grouping.sum(np.arange(12.0)), [3, 12, 21, 30])
    with pytest.raises(ValueError):
        Grouping("bad", np.array([0, 2, 2]))


def test_monotonic_mask_keeps_steady_locations_whole():
    grouping = Grouping.blocks(6, 3)
    scores = monotonic_scores(grouping, sweep().outputs)
    assert scores[0] > 1e6 and scores[2] > 1e6 and scores[4] < 2 and scores[1] == 0
    shares = steady_shares(grouping, sweep().outputs)
    np.testing.assert_allclose(shares[[0, 1, 2, 3, 5]], [1, 0, 1, 0, 0])
    masker = MonotonicSNRMask(grouping).calibrate([MaskInput(f"c{i}", sweep(seed=i)) for i in range(3)])
    parameters = masker.describe().parameters
    assert (parameters["channel_tau"], parameters["share"]) == (2.0, 0.1)
    assert parameters["discovery"] == {"contexts": 3, "coverage": 2 / 6, "empty_contexts": 0.0}
    selected = masker.fit(MaskInput("c", sweep()))
    np.testing.assert_array_equal(selected, [True, False, True, False, False, False])
    y = np.arange(1.0, 19.0)
    out = masker.apply(selected, y)
    assert grouping.respects(out != 0) and out[:3].tolist() == [1, 2, 3] and not out[3:6].any()
    with pytest.raises(ValueError, match="Calibrate"):
        MonotonicSNRMask(grouping).fit(MaskInput("c", sweep()))
    with pytest.raises(ValueError, match="share"):
        MonotonicSNRMask(grouping, share=1.0)


def test_monotonic_mask_scale_holds_for_wide_locations():
    """A wide location (e.g. 1408 channels of a token) whose variation is mostly not a steady trend.

    Pooling |mean step| / standard error over its channels stays near 1 however steady a
    part of it is, so a threshold on the pooled score keeps nothing; the steady share
    keeps the location when steady channels carry a substantial part of its variation.
    """
    rng = np.random.default_rng(3)
    u = np.linspace(-1, 1, len(VALUES))
    width = 1000
    outputs = np.zeros((1, len(u), 2 * width))
    outputs[0] = rng.normal(size=(len(u), 2 * width))
    outputs[0, :, :40] += np.outer(u, 8 * np.ones(40))             # location 0: 4% steady channels
    outputs[0, :, width:width + 2] += np.outer(u, 8 * np.ones(2))  # location 1: 0.2% steady channels
    grouping = Grouping.blocks(2, width)
    data = Sweep(VALUES, (0,), outputs)
    assert monotonic_scores(grouping, data.outputs).max() < 1
    shares = steady_shares(grouping, data.outputs)
    assert shares[0] > 0.5 and shares[1] < 0.1
    masker = MonotonicSNRMask(grouping).calibrate([MaskInput("c", data)])
    np.testing.assert_array_equal(masker.fit(MaskInput("c", data)), [True, False])


def test_threshold_calibration_does_not_reward_empty_masks():
    empty, some = np.zeros(10, bool), np.eye(10, dtype=bool)[0]
    assert _jaccard(empty, empty) == 0.0 and _jaccard(some, some) == 1.0
    random = np.random.default_rng(0).random((2, 2000)) < 0.5
    assert abs(_jaccard(random[0], random[1])) < 0.05
    noisy = [MaskInput(f"c{i}", sweep(seeds=3, noise=0.01, seed=i)) for i in range(3)]
    repeat = RepeatNoiseSNRMask(Grouping.blocks(6, 3), candidates=(2.0, 1e9)).calibrate(noisy)
    assert repeat.tau == 2.0 and repeat.fit(noisy[0]).any()


def test_energy_repeat_noise_manual_and_none():
    grouping = Grouping.blocks(6, 3)
    assert EnergyMask(grouping, 0.9).fit(MaskInput("c", sweep()))[[0, 2]].all()
    constant = Sweep(VALUES, (0,), np.full((1, 8, 18), 0.123456789, dtype=np.float32))
    assert not EnergyMask(grouping).fit(MaskInput("c", constant)).any()
    noisy = [MaskInput(f"c{i}", sweep(seeds=3, noise=0.01, seed=i)) for i in range(3)]
    repeat = RepeatNoiseSNRMask(grouping).calibrate(noisy)
    assert repeat.fit(noisy[0])[[0, 2]].all() and not repeat.fit(noisy[0])[[1, 3, 5]].any()
    manual = ManualMask(grouping, np.array([1, 0, 0, 0, 0, 1], bool), "ends")
    assert manual.describe().kind == "selection" and manual.fit(MaskInput("c")).sum() == 2
    with pytest.raises(ValueError, match="kind"):
        MaskerCard("pca:2", "projection", "sweep", {}, "a projection is not a mask")
    assert NoMask().apply(None, np.ones(3)).tolist() == [1, 1, 1]


def test_masked_scope_seals_the_rule_and_needs_an_instance_per_context():
    masker = MonotonicSNRMask(Grouping.blocks(2, 2))
    scope = MaskedScope("masked", ("y",), masker=masker, observed="y")
    other = MaskedScope("masked", ("y",), masker=EnergyMask(Grouping.blocks(2, 2)), observed="y")
    assert scope.record()["masker"]["masker_id"] == "monotonic-snr" and scope.digest != other.digest
    artifact = Artifact("w@x", (Channel("y", Modality.TENSOR, np.arange(4.0)),
                                Channel(CONTEXT_CHANNEL, Modality.TEXT, (Statement("context_id", "c0", "", "identity"),))))
    with pytest.raises(ValueError, match="No frozen mask instance"):
        scope.apply(artifact, via="v")
    scope.begin_sweep()
    assert scope.apply(artifact, via="v").channel("y").data.tolist() == [0, 1, 2, 3]
    scope.end_sweep()
    with pytest.raises(PermissionError, match="Freeze"):
        scope.assign("c0", np.array([True, False]))
    frozen = replace_tau(masker, 2.0)
    scope.freeze(frozen)
    scope.assign("c0", np.array([True, False]))
    assert scope.apply(artifact, via="v").channel("y").data.tolist() == [0, 1, 0, 0]
    assert scope.digest == MaskedScope("masked", ("y",), masker=masker, observed="y").digest
    with pytest.raises(PermissionError):
        scope.assign("c0", np.array([True, True]))


def replace_tau(masker, tau):
    """A frozen copy of the masker (the declared constants are kept; `tau` is the share threshold)."""
    steady = Sweep(VALUES, (0,), np.outer(np.linspace(-1, 1, len(VALUES)), np.ones(masker.grouping.dims))[None])
    return MonotonicSNRMask(masker.grouping, masker.channel_tau, min(tau / 10, 0.5)).calibrate([MaskInput("c", steady)])


def test_experiment_sweeps_freezes_on_discovery_and_masks_every_context():
    chain = learned_chain(masker=MonotonicSNRMask(Grouping.blocks(PREDICTION_DIMS, 1)))
    experiment = Experiment(learned_declaration(chain), chain)
    experiment.measure(DataRole.DISCOVERY)
    stages = [event["stage"] for event in experiment.stages]
    assert stages[:6] == ["declared", "swept", "masker-frozen", "masked", "measured", "checked"]
    discovery = {c.context_id for c in experiment.declaration.contexts_for(DataRole.DISCOVERY)}
    assert set(experiment.masking["instances"]) == discovery
    with pytest.raises(PermissionError):
        experiment.measure(DataRole.VALIDATION)
    experiment.freeze(InterpolationMethod().fit(experiment.responses()))
    experiment.test()
    report = experiment.report()
    assert all(result.passed for result in check_report(report))
    assert set(report["masking"]["instances"]) == {c.context_id for c in experiment.declaration.contexts}
    assert 0 < report["masking"]["coverage"]["validation"] < 1


def test_declaration_requires_a_disjoint_sweep_exactly_for_sweep_maskers():
    chain = learned_chain(masker=MonotonicSNRMask(Grouping.blocks(PREDICTION_DIMS, 1)))
    base = learned_declaration(chain)
    with pytest.raises(ValueError, match="exactly when"):
        replace(base, sweep=())
    with pytest.raises(ValueError, match="disjoint"):
        replace(base, sweep=(*SWEEP, base.reference))
    plain = learned_chain()
    with pytest.raises(ValueError, match="exactly when"):
        replace(learned_declaration(plain), sweep=SWEEP)
