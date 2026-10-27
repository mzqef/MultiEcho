from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_evaluation import dataset
from test_execution import RecordingAdapter, request

from multiecho_framework.contracts import DataRole, SeedMode
from multiecho_framework.datasets import PairDataset
from multiecho_framework.execution import PairRunner
from multiecho_framework.training import (
    TrainingTarget,
    branch_energy_loss,
    export_training,
    response_mse,
)


def distribution_dataset(mode=SeedMode.PAIRED):
    template = request(mode)
    template = replace(template, context=replace(template.context, role=DataRole.TRAIN))
    result = PairRunner(RecordingAdapter()).run(template)
    return PairDataset((replace(result, baseline=np.zeros((2, 1)), intervention=np.array([[-1.0], [1.0]])),))


def test_export_is_explicit_and_keeps_world_context_seed_and_domain_identity():
    training, evaluation = dataset(DataRole.TRAIN, "train"), dataset()
    batch = export_training(training, TrainingTarget.MEAN_RESPONSE, evaluation=evaluation)
    np.testing.assert_array_equal(np.stack(batch.responses), training.mean_effects)
    assert batch.world.digest == training.world.digest
    assert batch.context_refs[0] == training.results[0].request.context.payload_ref
    assert batch.baseline_seeds[0] == (7,)
    assert batch.horizons == (1, 1, 1, 1)
    assert not batch.intervention_values.flags.writeable
    assert not batch.responses[0].flags.writeable


def test_training_cannot_silently_consume_heldout_or_shared_groups():
    with pytest.raises(ValueError, match="Training export"):
        export_training(dataset(), TrainingTarget.MEAN_RESPONSE)
    with pytest.raises(ValueError, match="Training export"):
        export_training(dataset(DataRole.DISCOVERY), TrainingTarget.MEAN_RESPONSE)
    exported = export_training(dataset(DataRole.DISCOVERY), TrainingTarget.MEAN_RESPONSE, allow_discovery=True)
    assert len(exported.pair_ids) == 4
    training = dataset(DataRole.TRAIN)
    with pytest.raises(ValueError, match="leakage"):
        export_training(training, TrainingTarget.MEAN_RESPONSE, evaluation=dataset())


def test_distribution_targets_preserve_a_mean_preserving_intervention():
    data = distribution_dataset()
    mean_batch = export_training(data, TrainingTarget.MEAN_RESPONSE)
    np.testing.assert_array_equal(mean_batch.responses[0], [0.0])
    branch_batch = export_training(data, TrainingTarget.BRANCH_DISTRIBUTIONS)
    assert branch_batch.responses is None
    good = branch_energy_loss(branch_batch.baseline_samples, branch_batch.intervention_samples, branch_batch)
    collapsed = branch_energy_loss(branch_batch.baseline_samples, branch_batch.baseline_samples, branch_batch)
    assert good < collapsed


def test_independent_draws_are_not_paired_targets_and_one_draw_is_not_a_distribution():
    independent = distribution_dataset(SeedMode.INDEPENDENT)
    with pytest.raises(ValueError, match="pathwise"):
        export_training(independent, TrainingTarget.PAIRED_RESPONSE)
    exported = export_training(independent, TrainingTarget.BRANCH_DISTRIBUTIONS)
    assert exported.baseline_seeds[0] != exported.intervention_seeds[0]
    with pytest.raises(ValueError, match="multiple"):
        export_training(dataset(DataRole.TRAIN), TrainingTarget.BRANCH_DISTRIBUTIONS)


def test_reference_response_loss_uses_equal_group_weights():
    zeros = np.zeros((4, 1))
    predictions = np.array([[0.0], [0.0], [0.0], [2.0]])
    assert response_mse(zeros, predictions, zeros, group_ids=("a", "a", "a", "b")) == 2.0
    assert response_mse(zeros + 7, predictions + 7, zeros, group_ids=("a", "a", "a", "b")) == 2.0
    with pytest.raises(ValueError, match="share batch"):
        response_mse(zeros, predictions, np.zeros(4), group_ids=("a", "a", "a", "b"))