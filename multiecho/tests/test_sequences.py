from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_execution import RecordingAdapter, request

from multiecho_framework.contracts import Intervention
from multiecho_framework.execution import SequenceRequest, SequenceRunner


def sequence(index=0):
    context = replace(request().context, context_id=f"trace-{index}", payload_digest=f"trace-{index}",
                      exchangeability_group=f"group-{index}")
    return SequenceRequest(f"trace-{index}", context, (Intervention((1.0,)), Intervention((2.0,))),
                           11, (12, 13))


def sequence_adapter():
    adapter = RecordingAdapter()
    adapter.spec = replace(adapter.spec, supports_sequence=True)
    return adapter


def test_sequence_observes_the_initial_state_and_each_executed_step():
    adapter = sequence_adapter()
    result = SequenceRunner(adapter).run(sequence())
    np.testing.assert_array_equal(result.observations, [[0.0], [1.0], [3.0]])
    assert adapter.seen == [(0.0, 12), (1.0, 13)]
    assert not result.observations.flags.writeable
    assert len(adapter.closed) == 2
    again = SequenceRunner(adapter).run(sequence(1))
    np.testing.assert_array_equal(again.observations, result.observations)


def test_nonsequential_adapters_and_incomplete_seed_schedules_are_rejected():
    with pytest.raises(ValueError, match="sequential"):
        SequenceRunner(RecordingAdapter())
    with pytest.raises(ValueError, match="one step seed"):
        replace(sequence(), step_seeds=(12,))


def test_sequence_failure_closes_distinct_handles_without_double_close():
    adapter = sequence_adapter()
    def allocate_then_fail(state, intervention, seed):
        if seed == 13:
            raise RuntimeError("step failed")
        return [state[0] + intervention.values[0]]
    adapter.apply = allocate_then_fail
    with pytest.raises(RuntimeError, match="step failed"):
        SequenceRunner(adapter).run(sequence())
    assert len(adapter.closed) == 3
    assert len({id(state) for state in adapter.closed}) == 3


def test_sequence_rejects_midflight_world_updates():
    adapter = sequence_adapter()
    original = adapter.apply
    def changed(state, intervention, seed):
        adapter.spec = replace(adapter.spec, revision="mutated")
        return original(state, intervention, seed)
    adapter.apply = changed
    with pytest.raises(RuntimeError, match="revision changed"):
        SequenceRunner(adapter).run(sequence())