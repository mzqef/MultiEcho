from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_evaluation import dataset
from test_sequences import sequence, sequence_adapter

from multiecho_framework.contracts import DataRole, Intervention
from multiecho_framework.evaluation import EvaluationPlan, EvaluationTarget, EvidenceStatus
from multiecho_framework.execution import SequenceRunner
from multiecho_framework.laws import (
    LawSpec,
    LawTestPlan,
    fit_response_law,
    predict_responses,
    propose_disagreement,
)
from multiecho_framework.laws import (
    test_regimes as check_regimes,
)
from multiecho_framework.laws import (
    test_residual_law as check_residual,
)
from multiecho_framework.laws import (
    test_trajectory_law as check_trajectory,
)


def traces():
    runner = SequenceRunner(sequence_adapter())
    return tuple(runner.run(replace(sequence(index), context=replace(sequence(index).context, role=DataRole.VALIDATION)))
                 for index in range(4))


def plan():
    return LawTestPlan(EvaluationPlan("law-test", EvaluationTarget.MEAN_RESPONSE, DataRole.VALIDATION,
                                     0, resamples=99), 0.1, "Declared response residual", "1")


class Candidate:
    def __init__(self, world, factor=1.0, fitted_on=()):
        self.spec = LawSpec(f"candidate-{factor}", "v1", world, "Test response rule", "Declared numerical test domain", fitted_on)
        self.factor = factor

    def describe(self):
        return self.spec

    def in_domain(self, *args):
        return True

    def predict(self, initial, actions, context):
        return initial + self.factor * np.cumsum([action.values for action in actions], axis=0)

    def residual(self, trace):
        return np.diff(trace.observations, axis=0) - self.factor * np.array([action.values for action in trace.request.interventions])


def test_frozen_rules_can_be_supported_or_contradicted_without_reference_physics():
    data = traces()
    good = Candidate(data[0].world)
    bad = Candidate(data[0].world, factor=0)
    assert check_residual(good, data, plan()).status is EvidenceStatus.SUPPORTED
    assert check_residual(bad, data, plan()).status is EvidenceStatus.CONTRADICTED
    assert all(row.status is EvidenceStatus.SUPPORTED for row in check_trajectory(good, data, plan()))
    assert all(row.status is EvidenceStatus.CONTRADICTED for row in check_trajectory(bad, data, plan()))
    assert check_residual(good, data[:1], plan()).status is EvidenceStatus.UNRESOLVED


def test_rollout_is_scored_against_executed_future_not_endpoint_algebra():
    data = traces()
    candidate = Candidate(data[0].world)
    altered = tuple(replace(trace, observations=np.array([[0.0], [1.0], [9.0]])) for trace in data)
    scores = check_trajectory(candidate, altered, plan())
    assert scores[0].status is EvidenceStatus.SUPPORTED
    assert scores[1].status is EvidenceStatus.CONTRADICTED
    assert scores[0].confidence > plan().evaluation.confidence


def test_fitting_leakage_world_drift_and_out_of_domain_are_not_silent():
    data = traces()
    candidate = Candidate(data[0].world, fitted_on=(replace(data[0].request.context, role=DataRole.DISCOVERY),))
    with pytest.raises(ValueError, match="leakage"):
        check_residual(candidate, data, plan())
    candidate = Candidate(replace(data[0].world, revision="other"))
    with pytest.raises(ValueError, match="identities"):
        check_residual(candidate, data, plan())
    candidate = Candidate(data[0].world)
    candidate.in_domain = lambda *args: False
    with pytest.raises(ValueError, match="outside"):
        check_residual(candidate, data, plan())


def test_regime_reports_preserve_failures_and_adjust_declared_family():
    data = traces()
    bad = tuple(replace(trace, observations=trace.observations * 2) for trace in data)
    report = check_regimes(Candidate(data[0].world), {"domain-a": data, "domain-b": bad}, plan())
    assert report["domain-a"].status is EvidenceStatus.SUPPORTED
    assert report["domain-b"].status is EvidenceStatus.CONTRADICTED
    assert report["domain-a"].confidence == pytest.approx(0.975)


class ResponseCandidate(Candidate):
    def predict(self, query, baseline_mean):
        return self.factor * np.subtract(query.intervention.values, query.baseline.values)


def test_active_selection_uses_candidate_predictions_not_observed_outcomes():
    data = dataset(DataRole.DISCOVERY)
    context = data.results[0].request.context
    candidates = (ResponseCandidate(data.world), ResponseCandidate(data.world, factor=2))
    choices = (Intervention((2.0,)), Intervention((5.0,)))
    proposal = propose_disagreement(candidates, data.world, context, Intervention((0.0,)), np.array([0.0]), choices)
    assert proposal.intervention == choices[1]
    assert proposal.disagreement_rms == 2.5
    assert len(proposal.candidate_digests) == 2
    with pytest.raises(ValueError, match="discovery"):
        propose_disagreement(candidates, data.world, replace(context, role=DataRole.CONFIRMATION),
                             Intervention((0.0,)), np.array([0.0]), choices)


def test_external_fitting_tracks_contexts_and_evaluates_without_targets():
    training, evaluation = dataset(DataRole.TRAIN, "train"), dataset(DataRole.VALIDATION, "test")
    class Fitter:
        def fit(self, data):
            return ResponseCandidate(data.world, fitted_on=tuple(row.request.context for row in data.results))
    candidate = fit_response_law(Fitter(), training)
    np.testing.assert_array_equal(predict_responses(candidate, evaluation), evaluation.mean_effects)
    with pytest.raises(ValueError, match="Training export"):
        fit_response_law(Fitter(), evaluation)
    with pytest.raises(ValueError, match="leakage"):
        predict_responses(candidate, training)