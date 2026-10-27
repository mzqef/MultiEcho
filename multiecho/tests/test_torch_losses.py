from __future__ import annotations

import os

import numpy as np
import pytest

from multiecho_framework import torch_losses
from multiecho_framework.evaluation import energy_score as numpy_energy_score

pytestmark = pytest.mark.skipif(
    os.environ.get("MULTIECHO_TEST_TORCH") != "1",
    reason="Optional backend: install the torch extra and set MULTIECHO_TEST_TORCH=1",
)


def test_response_loss_matches_reference_and_backpropagates():
    import torch

    baseline = torch.zeros((4, 1), requires_grad=True)
    intervention = torch.tensor([[0.0], [0.0], [0.0], [2.0]], requires_grad=True)
    target = torch.zeros_like(intervention)
    loss = torch_losses.response_mse(baseline, intervention, target, group_ids=("a", "a", "a", "b"))
    assert loss.item() == 2.0
    loss.backward()
    assert intervention.grad[-1].item() == 2.0
    assert baseline.grad[-1].item() == -2.0


def test_distribution_loss_keeps_spread_and_gradients():
    import torch

    target = torch.tensor([[-1.0], [1.0]])
    prediction = torch.tensor([[-0.5], [0.5]], requires_grad=True)
    score = torch_losses.energy_score(prediction, target)
    assert score.item() == pytest.approx(numpy_energy_score(np.array([[-0.5], [0.5]]), np.array([[-1.0], [1.0]])))
    score.backward()
    assert prediction.grad[0].item() > 0
    assert prediction.grad[1].item() < 0
    zero = torch.zeros_like(target)
    good = torch_losses.branch_energy_loss((zero,), (target,), (zero,), (target,), group_ids=("group",))
    collapsed = torch_losses.branch_energy_loss((zero,), (zero,), (zero,), (target,), group_ids=("group",))
    assert good.item() < collapsed.item()


def test_torch_loss_rejects_target_and_group_mismatch():
    import torch

    with pytest.raises(ValueError, match="matching shapes"):
        torch_losses.response_mse(torch.zeros((2, 1)), torch.zeros((2, 1)), torch.zeros((2, 2)), group_ids=("a", "b"))
    with pytest.raises(ValueError, match="group IDs"):
        torch_losses.response_mse(torch.zeros((2, 1)), torch.zeros((2, 1)), torch.zeros((2, 1)), group_ids=("a",))
    with pytest.raises(ValueError, match="finite"):
        torch_losses.energy_score(torch.tensor([[float("nan")]]), torch.zeros((2, 1)))