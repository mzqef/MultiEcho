from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from multiecho_framework import (
    AccessMode,
    Axis,
    Context,
    DataRole,
    Intervention,
    ObservationSpec,
    PairRequest,
    PairRunner,
    SeedMode,
    Space,
    WorldSpec,
)


class RecordingAdapter:
    def __init__(self):
        self.spec = WorldSpec(
            "test-double", "fixed-v1", Space("input", (Axis("value", "1", -10, 10),), "test coordinates"),
            ObservationSpec("output", (1,), "test observation"), AccessMode.COMPILED_INPUT,
        )
        self.closed = []
        self.seen = []

    def describe(self):
        return self.spec

    def open(self, context, seed):
        return [0.0]

    def fork(self, state):
        return list(state)

    def apply(self, state, intervention, seed):
        self.seen.append((state[0], seed))
        state[0] += intervention.values[0]
        return state

    def observe(self, state):
        return np.asarray(state)

    def close(self, state):
        self.closed.append(state)


def request(mode=SeedMode.FIXED):
    return PairRequest(
        "pair", Context("context", "memory:test", "fixture-v1", DataRole.DISCOVERY, "group"),
        Intervention((1.0,)), Intervention((4.0,)), mode,
        (7,) if mode is SeedMode.FIXED else (7, 8),
        (9, 10) if mode is SeedMode.INDEPENDENT else (),
    )


def test_branches_fork_the_same_unmodified_context():
    adapter = RecordingAdapter()
    result = PairRunner(adapter).run(request())
    np.testing.assert_array_equal(result.paired_effects, [[3.0]])
    assert adapter.seen == [(0.0, 7), (0.0, 7)]
    assert len(adapter.closed) == 3


def test_paired_and_independent_randomness_remain_distinct():
    paired = PairRunner(RecordingAdapter()).run(request(SeedMode.PAIRED))
    assert paired.paired_effects.shape == (2, 1)
    independent = PairRunner(RecordingAdapter()).run(request(SeedMode.INDEPENDENT))
    np.testing.assert_array_equal(independent.mean_effect, [3.0])
    with pytest.raises(ValueError, match="pathwise"):
        _ = independent.paired_effects


def test_world_revision_changes_fail_closed():
    adapter = RecordingAdapter()
    runner = PairRunner(adapter)
    adapter.spec = replace(adapter.spec, revision="changed")
    with pytest.raises(RuntimeError, match="revision changed"):
        runner.run(request())


def test_domain_and_seed_contracts_reject_invalid_requests():
    with pytest.raises(ValueError, match="outside domain"):
        PairRunner(RecordingAdapter()).run(replace(request(), intervention=Intervention((11.0,))))
    with pytest.raises(ValueError, match="must not overlap"):
        replace(request(SeedMode.INDEPENDENT), independent_seeds=(7, 10))


def test_adapter_cannot_claim_isolation_with_the_same_handle():
    adapter = RecordingAdapter()
    adapter.fork = lambda state: state
    with pytest.raises(ValueError, match="isolated"):
        PairRunner(adapter).run(request())
    assert len(adapter.closed) == 1