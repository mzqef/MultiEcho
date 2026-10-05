from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_execution import RecordingAdapter, request

from multiecho_framework.contracts import DataRole, Intervention
from multiecho_framework.datasets import PairDataset, require_disjoint
from multiecho_framework.evaluation import (
    BranchPrediction,
    EvaluationPlan,
    EvaluationTarget,
    EvidenceStatus,
    energy_score,
    evaluate_distributions,
    evaluate_predictions,
    grouped_values,
)
from multiecho_framework.execution import PairRunner


def dataset(role=DataRole.VALIDATION, prefix="test"):
    runner = PairRunner(RecordingAdapter())
    template = request()
    return PairDataset(tuple(runner.run(replace(
        template, pair_id=f"{prefix}-{index}", intervention=Intervention((float(index + 2),)),
        context=replace(template.context, context_id=f"{prefix}-{index}", payload_digest=f"{prefix}-{index}",
                        exchangeability_group=f"{prefix}-group-{index}", role=role),
    )) for index in range(4)))


def test_target_matched_evaluation_and_explicit_equivalence():
    data = dataset()
    plan = EvaluationPlan("metric", EvaluationTarget.MEAN_RESPONSE, DataRole.VALIDATION, 0.01, equivalence_margin=0.01, resamples=99)
    report = evaluate_predictions(data, data.mean_effects, {"zero": np.zeros_like(data.mean_effects)}, plan)
    assert report.comparisons[0].status is EvidenceStatus.SUPPORTED
    assert report.descriptive_r2 == 1
    equal = evaluate_predictions(data, data.mean_effects, {"same": data.mean_effects}, plan)
    assert equal.comparisons[0].status is EvidenceStatus.EQUIVALENT


def test_unequal_group_sizes_do_not_count_as_independent_replication():
    np.testing.assert_array_equal(grouped_values(np.array([0, 0, 0, 4]), ("a", "a", "a", "b")), [0, 4])
    data = dataset()
    one_group = PairDataset(tuple(replace(row, request=replace(row.request, context=replace(
        row.request.context, exchangeability_group="one"
    ))) for row in data.results))
    plan = EvaluationPlan("metric", EvaluationTarget.MEAN_RESPONSE, DataRole.VALIDATION, 0, resamples=99)
    result = evaluate_predictions(one_group, one_group.mean_effects, {"zero": np.zeros_like(one_group.mean_effects)}, plan)
    assert result.comparisons[0].interval is None
    assert result.comparisons[0].status is EvidenceStatus.UNRESOLVED


def test_leakage_and_mismatched_comparators_are_rejected():
    data = dataset()
    with pytest.raises(ValueError, match="leakage"):
        require_disjoint(data, data)
    plan = EvaluationPlan("metric", EvaluationTarget.MEAN_RESPONSE, DataRole.VALIDATION, 0, resamples=99)
    with pytest.raises(ValueError, match="Comparator target mismatch"):
        evaluate_predictions(data, data.mean_effects, {"wrong": np.zeros((4, 2))}, plan)
    with pytest.raises(ValueError, match="roles"):
        evaluate_predictions(dataset(DataRole.TRAIN), data.mean_effects, {"zero": data.mean_effects}, plan)


def test_distribution_score_detects_spread_that_mean_response_misses():
    spread = np.array([[-1.0], [1.0]])
    collapsed = np.zeros_like(spread)
    assert spread.mean() == collapsed.mean()
    assert energy_score(spread, spread) < energy_score(collapsed, spread)


def test_data_roles_cannot_share_an_exchangeability_group():
    train = dataset(DataRole.TRAIN, "train").results[0]
    validation = dataset().results[0]
    validation = replace(validation, request=replace(validation.request, context=replace(
        validation.request.context, exchangeability_group=train.request.context.exchangeability_group
    )))
    with pytest.raises(ValueError, match="straddle"):
        PairDataset((train, validation))


def test_distributional_evaluation_detects_spread_and_rejects_point_target_confusion():
    from multiecho_framework.contracts import SeedMode

    template = request(SeedMode.PAIRED)
    runner = PairRunner(RecordingAdapter())
    rows = []
    for index in range(4):
        context = replace(template.context, context_id=f"distribution-{index}", payload_digest=f"distribution-{index}",
                          role=DataRole.VALIDATION, exchangeability_group=f"distribution-{index}")
        result = runner.run(replace(template, pair_id=f"distribution-{index}", context=context))
        rows.append(replace(result, baseline=np.zeros((2, 1)), intervention=np.array([[-1.0], [1.0]])))
    data = PairDataset(tuple(rows))
    predicted = BranchPrediction(tuple(row.baseline for row in rows), tuple(row.intervention for row in rows))
    collapsed = BranchPrediction(predicted.baseline, predicted.baseline)
    plan = EvaluationPlan("spread", EvaluationTarget.BRANCH_DISTRIBUTIONS, DataRole.VALIDATION, 0.01, resamples=99)
    report = evaluate_distributions(data, predicted, {"same-mean-collapsed": collapsed}, plan)
    assert report.comparisons[0].status is EvidenceStatus.SUPPORTED
    with pytest.raises(ValueError, match="Point evaluation"):
        evaluate_predictions(data, data.mean_effects, {"zero": data.mean_effects}, plan)
    with pytest.raises(ValueError, match="fixed realization"):
        evaluate_distributions(dataset(), predicted, {"zero": collapsed}, plan)