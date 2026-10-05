from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_execution import RecordingAdapter, request
from test_laws import Candidate, plan
from test_sequences import sequence, sequence_adapter

from multiecho_framework.contracts import DataRole, Intervention, ObservationSpec
from multiecho_framework.evaluation import EvidenceStatus
from multiecho_framework.execution import PairResult, PairRunner, SequenceRunner
from multiecho_framework.laws import PathPair
from multiecho_framework.laws import test_path_law as check_path_law


class EndpointCandidate(Candidate):
    def endpoint(self, observation, context):
        return observation


def path_pairs():
    runner = SequenceRunner(sequence_adapter())
    pairs = []
    for index in range(4):
        template = sequence(index)
        template = replace(template, context=replace(template.context, role=DataRole.VALIDATION))
        left = runner.run(template)
        right = runner.run(replace(template, trace_id=f"reordered-{index}", interventions=tuple(reversed(template.interventions))))
        pairs.append(PathPair(f"paths-{index}", left, right, "The test operations commute", "Deterministic adapter; same initial seed"))
    return tuple(pairs)


def test_separately_executed_paths_support_only_the_declared_endpoint_claim():
    pairs = path_pairs()
    candidate = EndpointCandidate(pairs[0].left.world)
    report = check_path_law(candidate, pairs, plan())
    assert report.status is EvidenceStatus.SUPPORTED
    assert len(report.trace_ids) == 8
    assert report.groups == 4
    altered = tuple(replace(pair, right=replace(pair.right, observations=np.array([[0.0], [2.0], [7.0]]))) for pair in pairs)
    assert check_path_law(candidate, altered, plan()).status is EvidenceStatus.CONTRADICTED


def test_same_trace_or_different_origins_cannot_certify_path_consistency():
    pair = path_pairs()[0]
    with pytest.raises(ValueError, match="separately executed"):
        replace(pair, right=pair.left)
    with pytest.raises(ValueError, match="initial observations"):
        replace(pair, right=replace(pair.right, observations=np.array([[1.0], [2.0], [3.0]])))
    with pytest.raises(ValueError, match="total horizon"):
        replace(pair, right=replace(pair.right, request=replace(pair.right.request, interventions=(Intervention((1.0,), 2), Intervention((2.0,))))))


def test_deserialized_results_cannot_bypass_domains_or_integer_shapes():
    original = PairRunner(RecordingAdapter()).run(request())
    with pytest.raises(ValueError, match="outside domain"):
        PairResult(original.world, replace(original.request, intervention=Intervention((20.0,))), original.baseline, original.intervention)
    with pytest.raises(ValueError, match="positive, fixed"):
        ObservationSpec("bad", (1.5,), "fractional dimensions")
    with pytest.raises(ValueError, match="positive integer"):
        Intervention((1.0,), True)
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(request(), independent_seeds=[4])