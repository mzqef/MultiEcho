from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from multiecho_framework import (
    AccessMode,
    Axis,
    CombinationSpec,
    ComplexitySpec,
    DataRole,
    DistributionKind,
    ExactResetSpec,
    FactorDistribution,
    FactorKind,
    FactorSpec,
    Intervention,
    NaturalCoordinate,
    ObservationCondition,
    ObservationSpec,
    PhysicsRewrite,
    ReadinSpec,
    ReferenceChannelSpec,
    RegimeSpec,
    SamplingLevel,
    SceneAxis,
    SceneProgram,
    Space,
    Statement,
    StimulusSpec,
    SupplementaryQuery,
    TransitionFeature,
    TransitionKind,
    WorldSpec,
    dumps_program,
    loads_program,
    probe_isolation,
    probe_replay,
    scene_contexts,
    scene_declaration,
    scene_space,
    scene_statements,
    validate_readin,
)


def program() -> SceneProgram:
    return SceneProgram(
        "oscillator",
        "v1",
        (
            FactorSpec(
                "zeta",
                "1",
                FactorKind.HIDDEN_PROPERTY,
                FactorDistribution(DistributionKind.FIXED, (0.5,)),
                "Damping ratio",
                False,
            ),
            FactorSpec(
                "appearance",
                "1",
                FactorKind.NUISANCE,
                FactorDistribution(DistributionKind.CATEGORICAL, ("red", "blue")),
                "Appearance",
                True,
            ),
        ),
        (),
        (
            SceneAxis(
                "zeta",
                "zeta",
                "1",
                NaturalCoordinate.LOG,
                0.1,
                2.0,
                (0.1, 0.5, 1.0, 2.0),
                (
                    RegimeSpec("under", 0.1, 1.0, "Underdamped"),
                    RegimeSpec("over", 1.0, 2.0, "Overdamped"),
                ),
                (
                    TransitionFeature(
                        "critical",
                        TransitionKind.POINT,
                        (1.0,),
                        "zero_crossings",
                        "Critical damping",
                    ),
                ),
            ),
        ),
        ObservationCondition(
            "video",
            StimulusSpec(16, 0.08, 64, 64),
            8,
            "fixed final stimulus frame",
            "unoccluded",
            "fixed orthographic",
            "solid objects",
            "24 frames",
        ),
        (
            PhysicsRewrite(
                "shifted",
                "v1",
                "Shift critical damping",
                (Statement("critical_zeta", 1.2),),
                "Transition moves to zeta=1.2",
            ),
        ),
        (
            (DataRole.DISCOVERY, 2),
            (DataRole.VALIDATION, 1),
            (DataRole.CONFIRMATION, 1),
        ),
        "test",
        7,
        "analytic oscillator",
    )


def test_scene_program_generates_reproducible_contexts_and_confirmation_points():
    scene = program()
    first = scene.context("zeta", DataRole.DISCOVERY, 0)
    second = scene.context("zeta", DataRole.DISCOVERY, 0)
    assert first == second
    assert scene.derive_seed("appearance", "zeta", DataRole.DISCOVERY, 0) == scene.derive_seed(
        "appearance", "zeta", DataRole.DISCOVERY, 0
    )
    points = scene.confirmation_points("zeta", 4)
    assert points == scene.confirmation_points("zeta", 4)
    assert all(0.1 <= value <= 2.0 for value in points)
    natural_grid = np.log([0.1, 0.5, 1.0, 2.0])
    assert all(np.min(np.abs(natural_grid - np.log(value))) >= 0.1 * np.min(np.diff(natural_grid)) for value in points)
    expected_seed = int.from_bytes(
        hashlib.sha256(
            b"test|7|context|oscillator|zeta|discovery|0"
        ).digest()[:8],
        "big",
    )
    assert scene.derive_seed("context", "zeta", DataRole.DISCOVERY, 0) == expected_seed


def test_context_at_pairs_scene_contexts_across_axis_values():
    scene = program()
    low = scene.context_at("zeta", 0.5, DataRole.DISCOVERY, 0)
    high = scene.context_at("zeta", 1.0, DataRole.DISCOVERY, 0)
    assert low.exchangeability_group == high.exchangeability_group
    assert low.context_id != high.context_id
    assert scene_statements(low)["zeta"].value == 0.5
    assert scene_statements(high)["zeta"].value == 1.0
    assert {
        key: value for key, value in scene_statements(low).items() if key != "zeta"
    } == {
        key: value for key, value in scene_statements(high).items() if key != "zeta"
    }
    assert low.exchangeability_group == low.context_id.removesuffix(":0.5")


def test_context_identity_and_factor_sampling_are_stable_and_role_scoped():
    scene = program()
    first = scene.context("zeta", DataRole.DISCOVERY, 0)
    next_context = scene.context("zeta", DataRole.DISCOVERY, 1)
    validation = scene.context("zeta", DataRole.VALIDATION, 0)
    assert first.exchangeability_group == first.context_id
    assert len({first.payload_digest, next_context.payload_digest, validation.payload_digest}) == 3
    assert len({first.exchangeability_group, next_context.exchangeability_group,
                validation.exchangeability_group}) == 3
    assert scene_statements(first)["context_seed"].kind == "identity"

    reordered = SceneProgram(
        scene.family_id, scene.revision, tuple(reversed(scene.factors)), scene.relations,
        scene.axes, scene.observation, scene.rewrites, scene.context_counts,
        scene.seed_namespace, scene.master_seed, scene.reference,
    )
    assert scene_statements(first) == scene_statements(reordered.context("zeta", DataRole.DISCOVERY, 0))

    expanded = SceneProgram(
        scene.family_id, scene.revision,
        (*scene.factors, FactorSpec(
            "extra", "1", FactorKind.NUISANCE,
            FactorDistribution(DistributionKind.UNIFORM, (0.0, 1.0)),
            "Independent test factor", True,
        )),
        scene.relations, scene.axes, scene.observation, scene.rewrites,
        scene.context_counts, scene.seed_namespace, scene.master_seed, scene.reference,
    )
    assert all(
        scene_statements(first)[key] == scene_statements(expanded.context("zeta", DataRole.DISCOVERY, 0))[key]
        for key in scene_statements(first)
    )
    broad_budget = replace(
        scene,
        context_counts=(
            (DataRole.DISCOVERY, 24),
            (DataRole.VALIDATION, 12),
            (DataRole.CONFIRMATION, 12),
        ),
    )
    contexts = scene_contexts(broad_budget, "zeta")
    assert len(contexts) == 48
    assert len({context.payload_digest for context in contexts}) == 48
    assert len({context.exchangeability_group for context in contexts}) == 48


def test_replicate_level_sampling_and_branch_seed_are_deterministic():
    scene = program()
    repeated_appearance = FactorSpec(
        "appearance", "1", FactorKind.NUISANCE,
        FactorDistribution(DistributionKind.CATEGORICAL, ("red", "blue")),
        "Appearance", True, SamplingLevel.REPLICATE,
    )
    scene = SceneProgram(
        scene.family_id, scene.revision, (scene.factors[0], repeated_appearance),
        scene.relations, scene.axes, scene.observation, scene.rewrites,
        scene.context_counts, scene.seed_namespace, scene.master_seed, scene.reference,
    )
    context = scene.context("zeta", DataRole.DISCOVERY, 0)
    draw = scene.replicate_draw("zeta", DataRole.DISCOVERY, 0, 1)
    assert "appearance" not in scene_statements(context)
    assert draw == scene.replicate_draw("zeta", DataRole.DISCOVERY, 0, 1)
    assert draw.statements[0].key == "appearance"
    assert scene.derive_branch_seed("noise", "zeta", DataRole.DISCOVERY, 0, 1, 0.5) == \
        scene.derive_branch_seed("noise", "zeta", DataRole.DISCOVERY, 0, 1, 0.5)
    across_replicates = replace(
        repeated_appearance,
        distribution=FactorDistribution(DistributionKind.UNIFORM, (0.0, 1.0)),
    )
    scene = replace(scene, factors=(scene.factors[0], across_replicates))
    assert scene.replicate_draw("zeta", DataRole.DISCOVERY, 0, 1).statements != \
        scene.replicate_draw("zeta", DataRole.DISCOVERY, 0, 2).statements


def test_sampling_tier_is_a_prefix_of_the_declared_budget():
    scene = program()
    tiered = replace(
        scene,
        tiers=(("B", ((DataRole.DISCOVERY, 1), (DataRole.VALIDATION, 1),
                      (DataRole.CONFIRMATION, 1))),),
    )
    full = scene_contexts(tiered, "zeta", "A")
    small = scene_contexts(tiered, "zeta", "B")
    assert small == tuple(context for context in full if context.context_id in {
        "oscillator:zeta:discovery:00",
        "oscillator:zeta:validation:00",
        "oscillator:zeta:confirmation:00",
    })


def test_scene_program_round_trip_and_envelope_integrity():
    scene = program()
    text = dumps_program(scene)
    assert loads_program(text) == scene
    assert dumps_program(loads_program(text)) == text
    envelope = json.loads(text)
    envelope["payload"]["family_id"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        loads_program(json.dumps(envelope))
    record = scene.to_record()
    record["axes"][0]["upper"] = 2
    assert SceneProgram.from_record(record).digest == scene.digest
    assert scene.digest == replace(scene, axes=(replace(scene.axis("zeta"), upper=2.0),)).digest
    golden_path = Path(__file__).parent / "data" / "scene_program_golden.json"
    golden = golden_path.read_text(encoding="utf-8")
    assert dumps_program(scene) == golden
    assert loads_program(golden).digest == scene.digest


def test_serialization_round_trips_extended_program_contracts():
    scene = program()
    extended = replace(
        scene,
        extra_observations=(replace(scene.observation, condition_id="field"),),
        tiers=(("B", ((DataRole.DISCOVERY, 1), (DataRole.VALIDATION, 1),
                      (DataRole.CONFIRMATION, 1))),),
        combinations=(CombinationSpec("scaled", "zeta * 2", "1", "Scaled damping"),),
        complexity=(ComplexitySpec("zeta", "entity", False, "nested placement"),),
        evidence=("reference:fixture",),
    )
    assert loads_program(dumps_program(extended)) == extended


def test_combination_expression_is_a_finite_whitelisted_calculation():
    expression = CombinationSpec("critical", "sqrt(stiffness / mass)", "rad/s")
    assert expression.evaluate({"stiffness": 16.0, "mass": 4.0}) == 2.0
    with pytest.raises(ValueError, match="Unsupported"):
        CombinationSpec("unsafe", "__import__('os')").evaluate({})
    with pytest.raises(ValueError, match="undefined"):
        CombinationSpec("invalid", "1 / zero").evaluate({"zero": 0.0})


def test_measurement_digest_tracks_observation_not_design_only_fields():
    scene = program()
    changed_rewrite = replace(scene.rewrites[0], description="Changed injected-world note")
    with_rewrite_change = replace(scene, rewrites=(changed_rewrite,))
    assert with_rewrite_change.digest != scene.digest
    assert with_rewrite_change.measurement_digest("zeta") == scene.measurement_digest("zeta")

    axis = scene.axis("zeta")
    changed_grid = replace(axis, base_grid=(0.1, 0.4, 0.5, 1.0, 2.0))
    with_grid_change = replace(scene, axes=(changed_grid,))
    assert with_grid_change.measurement_digest("zeta") == scene.measurement_digest("zeta")

    changed_observation = replace(
        scene.observation,
        stimulus=replace(scene.observation.stimulus, interval=0.1),
    )
    with_observation_change = replace(scene, observation=changed_observation)
    assert with_observation_change.measurement_digest("zeta") != scene.measurement_digest("zeta")
    extra_axis = replace(
        scene.axis("zeta"),
        name="omega",
        factor="appearance",
        lower=0.1,
        upper=2.0,
        base_grid=(0.1, 0.5, 1.0, 2.0),
    )
    assert replace(scene, axes=(*scene.axes, extra_axis)).measurement_digest("zeta") == \
        scene.measurement_digest("zeta")
    extra_observation = replace(scene.observation, condition_id="alternate-video")
    assert replace(scene, extra_observations=(extra_observation,)).measurement_digest("zeta") == \
        scene.measurement_digest("zeta")
    assert replace(scene, context_counts=((DataRole.DISCOVERY, 3),)).measurement_digest("zeta") == \
        scene.measurement_digest("zeta")
    baseline_axis = replace(scene.axis("zeta"), baseline=0.6)
    assert replace(scene, axes=(baseline_axis,)).measurement_digest("zeta") != \
        scene.measurement_digest("zeta")
    changed_factor = replace(
        scene.factors[0],
        distribution=FactorDistribution(DistributionKind.FIXED, (0.6,)),
    )
    assert replace(scene, factors=(changed_factor, *scene.factors[1:])).measurement_digest("zeta") != \
        scene.measurement_digest("zeta")


def test_scene_declaration_builds_framework_declaration():
    from multiecho_framework.contracts import fingerprint

    scene = program()
    chain_record = {
        "chain_id": "fixture-chain",
        "worlds": [],
        "interfaces": [],
        "observer": {"scope": {}},
        "stimulus": None,
        "references": [],
    }
    world = WorldSpec(
        "fixture-chain",
        fingerprint(chain_record),
        scene_space(scene, "zeta"),
        ObservationSpec("response", (1,), "Fixture response"),
        AccessMode.COMPILED_INPUT,
    )

    class ChainFixture:
        def describe(self):
            return world

        def record(self):
            return chain_record

    declaration = scene_declaration(
        scene, "zeta", "video", ChainFixture(), values=(0.1, 1.0, 2.0)
    )
    assert declaration.reference.values == (0.5,)
    assert [item.values for item in declaration.interventions] == [(0.1,), (1.0,), (2.0,)]
    assert len(declaration.contexts) == 4


def test_periodic_axis_grid_rule_and_reference_contracts():
    periodic = SceneAxis(
        "phase",
        "phase",
        "rad",
        NaturalCoordinate.PERIODIC,
        0.0,
        2 * np.pi,
        (0.0, np.pi, 2 * np.pi),
        (),
        (),
        period=2 * np.pi,
        baseline=0.0,
    )
    assert periodic.to_natural(2 * np.pi) == 0.0
    assert periodic.from_natural(2 * np.pi - 0.1) == pytest.approx(2 * np.pi - 0.1)
    assert SceneProgram._grid_distance(periodic, np.asarray([0.0]), 2 * np.pi - 0.05) == pytest.approx(0.05)
    logarithmic = SceneAxis(
        "zeta", "zeta", "1", NaturalCoordinate.LOG, 0.1, 10.0,
        tuple(np.geomspace(0.1, 10.0, 5)), (), (), grid_rule="log-equal:bad+(1.0)"
    )
    with pytest.raises(ValueError, match="Invalid grid rule"):
        logarithmic.grid_from_rule()
    grid_rule = SceneAxis(
        "zeta", "zeta", "1", NaturalCoordinate.LOG, 0.1, 10.0,
        tuple(np.geomspace(0.1, 10.0, 5)), (), (),
        grid_rule="log-equal:5+(1.0)"
    )
    assert grid_rule.grid_rule_matches()

    condition = replace(
        program().observation,
        reference_channels=(
            ReferenceChannelSpec("residual", "reset_residual", (2,), "float64", "m", "Alignment residual"),
        ),
        exact_reset=ExactResetSpec(("x",), "shooting", 1e-6, "residual"),
        exact_reset_axis="zeta",
    )
    scene = program()
    rewritten = replace(scene, observation=condition)
    assert rewritten.observation.exact_reset is not None
    invalid_axis = replace(scene.axis("zeta"), factor="appearance")
    with pytest.raises(ValueError, match="hidden-property"):
        replace(scene, axes=(invalid_axis,), observation=condition)
    with pytest.raises(ValueError, match="unique"):
        replace(scene, extra_observations=(scene.observation,))


def test_scene_program_validates_supplementary_queries():
    scene = program()
    scene.validate_query(SupplementaryQuery(
        "q1", "transition", "1.0", "zeta", (0.8, 1.2), (0.9, 1.1), "Resolve transition",
    ))
    with pytest.raises(ValueError, match="outside"):
        scene.validate_query(SupplementaryQuery(
            "q2", "transition", "1.0", "zeta", (0.05, 0.2), (0.1,), "Outside domain",
        ))
    with pytest.raises(ValueError, match="inside"):
        scene.validate_query(SupplementaryQuery(
            "q3", "transition", "1.0", "zeta", (0.8, 1.0), (1.1,), "Value outside request",
        ))
    with pytest.raises(ValueError, match="baseline"):
        scene.validate_query(SupplementaryQuery(
            "q4", "transition", "1.0", "zeta", (0.4, 0.6), (0.5,), "Would have zero effect",
        ))
    with pytest.raises(ValueError, match="only on discovery"):
        scene.validate_query(SupplementaryQuery(
            "q5", "transition", "1.0", "zeta", (0.8, 1.2), (0.9,), "Wrong role",
            role=DataRole.VALIDATION,
        ))
    with pytest.raises(ValueError, match="program identity"):
        scene.validate_query(SupplementaryQuery(
            "q6", "transition", "1.0", "zeta", (0.8, 1.2), (0.9,), "Wrong program",
            program="other-family",
        ))
    with pytest.raises(ValueError, match="observation"):
        scene.validate_query(SupplementaryQuery(
            "q7", "transition", "1.0", "zeta", (0.8, 1.2), (0.9,), "Wrong observation",
            observation="missing-condition",
        ))


def test_discrete_confirmation_points_are_legal_and_sampling_is_bounded():
    scene = program()
    discrete = SceneAxis(
        "count",
        "appearance",
        "1",
        NaturalCoordinate.DISCRETE_COMPLEXITY,
        1,
        10,
        (1, 2, 4, 7, 10),
        (),
        (),
    )
    scene = SceneProgram(
        scene.family_id,
        scene.revision,
        scene.factors,
        scene.relations,
        (*scene.axes, discrete),
        scene.observation,
        scene.rewrites,
        scene.context_counts,
        scene.seed_namespace,
        scene.master_seed,
        scene.reference,
    )
    points = scene.confirmation_points("count", 2)
    assert points == scene.confirmation_points("count", 2)
    assert all(float(value).is_integer() and value not in {1, 2, 4, 7, 10} for value in points)
    with pytest.raises(ValueError, match="Not enough"):
        scene.confirmation_points("count", 6)


class _ReplayDouble:
    def __init__(self, stochastic: bool = False):
        self.stochastic = stochastic
        self.calls = 0
        self.spec = WorldSpec(
            "replay",
            "v1",
            Space("x", (Axis("x", "1", 0.0, 1.0),), "test"),
            ObservationSpec("y", (1,), "test"),
            AccessMode.COMPILED_INPUT,
        )

    def describe(self):
        return self.spec

    def open(self, context, seed):
        return {"value": 0.0}

    def fork(self, state):
        return dict(state)

    def apply(self, state, intervention, seed):
        self.calls += 1
        state["value"] = intervention.values[0] + (self.calls * 1e-3 if self.stochastic else 0.0)
        return state

    def observe(self, state):
        return np.array([state["value"]])

    def close(self, state):
        return None


def test_replay_probe_reports_exactness_and_difference():
    scene = program()
    context = scene.context("zeta", DataRole.DISCOVERY, 0)
    exact = probe_replay(_ReplayDouble(), context, Intervention((0.5,)), 3)
    assert exact.exact and exact.maximum_absolute_difference == 0.0
    changed = probe_replay(_ReplayDouble(stochastic=True), context, Intervention((0.5,)), 3)
    assert not changed.exact and changed.maximum_absolute_difference == pytest.approx(0.001)
    with pytest.raises(ValueError, match="seed"):
        probe_replay(_ReplayDouble(), context, Intervention((0.5,)), -1)


def test_framework_isolation_probe_reuses_root_but_forks_each_branch():
    context = program().context("zeta", DataRole.DISCOVERY, 0)
    assert probe_isolation(
        _ReplayDouble(), context, Intervention((0.5,)), Intervention((1.0,)), 4
    )

    class SharedReplayDouble(_ReplayDouble):
        def fork(self, state):
            return state

    with pytest.raises(ValueError, match="isolated fork"):
        probe_isolation(
            SharedReplayDouble(), context, Intervention((0.5,)), Intervention((1.0,)), 4
        )


def test_replay_probe_hashes_large_float_arrays_under_one_second():
    class LargeReplayDouble(_ReplayDouble):
        def observe(self, state):
            return np.zeros((16, 384, 384, 3), dtype=np.float64)

    context = program().context("zeta", DataRole.DISCOVERY, 0)
    start = perf_counter()
    result = probe_replay(LargeReplayDouble(), context, Intervention((0.5,)), 4)
    elapsed = perf_counter() - start
    assert result.exact
    assert len(result.first_digest) == 64
    assert elapsed < 1.0


def test_scene_axis_rejects_invalid_grids_and_transitions():
    with pytest.raises(ValueError, match="increasing"):
        SceneAxis("x", "x", "1", NaturalCoordinate.LINEAR, 0, 1, (0.0, 0.5, 0.5), (), ())
    with pytest.raises(ValueError, match="width"):
        TransitionFeature("bad", TransitionKind.POINT, (), "metric", "missing point")


def test_optional_branch_frame_nesting_and_prefix_contracts_keep_old_records():
    scene = program()
    golden = (Path(__file__).parent / "data" / "scene_program_golden.json").read_text(encoding="utf-8")
    record = scene.to_record()
    assert "branch" not in record
    assert all("frame" not in feature for axis in record["axes"] for feature in axis["transitions"])
    assert dumps_program(scene) == golden

    axis = scene.axis("zeta")
    located = replace(axis.transitions[0], feature_id="critical-at-frame-18", frame=18)
    extended = replace(scene, branch="oscillation", axes=(replace(axis, transitions=(*axis.transitions, located)),))
    restored = loads_program(dumps_program(extended))
    assert restored == extended and restored.branch == "oscillation"
    assert restored.axis("zeta").transitions[1].frame == 18
    assert extended.digest != scene.digest
    assert extended.measurement_digest("zeta") == scene.measurement_digest("zeta")
    with pytest.raises(ValueError, match="observation frame"):
        replace(axis.transitions[0], frame=-1)

    unnested = ComplexitySpec("zeta", "particle", True, "random packing", nested=False)
    assert not unnested.additive_test and ComplexitySpec("zeta", "beam", False, "nested").additive_test

    condition = scene.observation
    def readin(packing, frames=tuple(range(8, 16))):
        return ReadinSpec(condition.condition_id, frames, 64, 64, "bilinear", "none", packing, "unit", (1,), "test")
    validate_readin(condition, readin("prefix:repeat-first:3+tubelet:11"))
    validate_readin(condition, readin("prefix:repeat-first:2"))
    with pytest.raises(ValueError, match="dividing"):
        validate_readin(condition, readin("prefix:repeat-first:2+tubelet:4"))
    with pytest.raises(ValueError, match="positive number"):
        validate_readin(condition, readin("prefix:repeat-first:0"))
