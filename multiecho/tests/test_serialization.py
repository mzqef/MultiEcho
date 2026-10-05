from __future__ import annotations

import json

import numpy as np
import pytest
from test_evaluation import dataset
from test_sequences import sequence, sequence_adapter

from multiecho_framework import Artifact, Channel, Modality, Region, Scope, Statement
from multiecho_framework.execution import SequenceRunner
from multiecho_framework.serialization import (
    dumps_artifact,
    dumps_dataset,
    dumps_trace,
    loads_artifact,
    loads_dataset,
    loads_trace,
)


def test_dataset_exchange_preserves_identity_roles_draws_and_values():
    original = dataset()
    serialized = dumps_dataset(original)
    restored = loads_dataset(serialized)
    assert restored.world.digest == original.world.digest
    assert restored.roles == original.roles
    assert restored.groups == original.groups
    np.testing.assert_array_equal(restored.mean_effects, original.mean_effects)
    assert dumps_dataset(restored) == serialized
    assert not restored.results[0].baseline.flags.writeable


def test_trace_and_artifact_roundtrip_with_no_executable_payloads():
    trace = SequenceRunner(sequence_adapter()).run(sequence())
    restored_trace = loads_trace(dumps_trace(trace))
    np.testing.assert_array_equal(restored_trace.observations, trace.observations)
    assert restored_trace.request == trace.request
    raw = Artifact("engine@test", (
        Channel("video", Modality.VIDEO, np.random.default_rng(0).random((2, 4, 5, 1)), rate=10.0),
        Channel("truth", Modality.TEXT, (Statement("k", 4.0, "N/m"), Statement("intent", "probe", kind="intent"))),
    ))
    scoped = Scope("roi", ("video", "truth"), window=(1, 2), region=Region.box("box", (4, 5), (1, 3), (0, 5)),
                   clip=(0.0, 0.5)).apply(raw, via="interface-digest")
    for artifact in (raw, scoped):
        restored = loads_artifact(dumps_artifact(artifact))
        assert restored.digest == artifact.digest
    assert loads_artifact(dumps_artifact(scoped)).scope.region.mask.sum() == 10


def test_tampering_wrong_versions_and_duplicate_keys_are_rejected():
    original = dumps_dataset(dataset())
    altered = json.loads(original)
    altered["payload"]["results"][0]["baseline"] = [[100.0]]
    with pytest.raises(ValueError, match="digest mismatch"):
        loads_dataset(json.dumps(altered))
    altered["schema"] = "future-schema"
    with pytest.raises(ValueError, match="schema"):
        loads_dataset(json.dumps(altered))
    with pytest.raises(ValueError, match="Duplicate"):
        loads_dataset('{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError, match="Non-finite"):
        loads_dataset('{"payload":NaN}')
    with pytest.raises(ValueError, match="kind"):
        loads_trace(original)