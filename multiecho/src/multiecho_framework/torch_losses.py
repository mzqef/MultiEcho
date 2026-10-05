"""Optional differentiable PyTorch losses; the core package never imports PyTorch."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _group_mean(values: torch.Tensor, group_ids: tuple[str, ...]) -> torch.Tensor:
    import torch

    if values.ndim != 1 or values.shape[0] != len(group_ids) or not group_ids:
        raise ValueError("Loss values need matching nonempty group IDs")
    if any(not isinstance(group, str) or not group for group in group_ids):
        raise ValueError("Each training item needs an exchangeability group")
    means = []
    for group in sorted(set(group_ids)):
        positions = [index for index, value in enumerate(group_ids) if value == group]
        means.append(values[positions].mean())
    return torch.stack(means).mean()


def _finite_float(*values: torch.Tensor) -> None:
    import torch

    for value in values:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("Training losses require floating-point tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Training tensors must be finite")
    if len({value.device for value in values}) != 1 or len({value.dtype for value in values}) != 1:
        raise ValueError("Loss tensors must share device and dtype")


def response_mse(
    baseline_prediction: torch.Tensor,
    intervention_prediction: torch.Tensor,
    response_target: torch.Tensor,
    *,
    group_ids: tuple[str, ...],
) -> torch.Tensor:
    _finite_float(baseline_prediction, intervention_prediction, response_target)
    if baseline_prediction.shape != intervention_prediction.shape or baseline_prediction.shape != response_target.shape:
        raise ValueError("Point predictions and response targets must have matching shapes")
    if response_target.ndim < 2 or any(size == 0 for size in response_target.shape):
        raise ValueError("Response targets need nonempty batch and observation dimensions")
    axes = tuple(range(1, response_target.ndim))
    errors = ((intervention_prediction - baseline_prediction) - response_target).square().mean(dim=axes)
    return _group_mean(errors, group_ids)


def energy_score(samples: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    import torch

    _finite_float(samples, outcomes)
    if samples.ndim < 2 or samples.ndim != outcomes.ndim or samples.shape[1:] != outcomes.shape[1:]:
        raise ValueError("Sample and outcome distributions must share observation shape")
    if any(size == 0 for size in samples.shape + outcomes.shape):
        raise ValueError("Distributional losses require nonempty draws and observations")
    draws = samples.reshape(samples.shape[0], -1)
    truth = outcomes.reshape(outcomes.shape[0], -1)
    cross = torch.cdist(draws, truth).mean()
    within = torch.cdist(draws, draws).mean()
    return cross - within / 2


def branch_energy_loss(
    baseline_predictions: tuple[torch.Tensor, ...],
    intervention_predictions: tuple[torch.Tensor, ...],
    baseline_targets: tuple[torch.Tensor, ...],
    intervention_targets: tuple[torch.Tensor, ...],
    *,
    group_ids: tuple[str, ...],
) -> torch.Tensor:
    import torch

    count = len(group_ids)
    if not count or any(len(values) != count for values in (
        baseline_predictions, intervention_predictions, baseline_targets, intervention_targets,
    )):
        raise ValueError("Both branches' predicted and target distributions must cover every group item")
    losses = [
        (energy_score(baseline, baseline_truth) + energy_score(intervention, intervention_truth)) / 2
        for baseline, intervention, baseline_truth, intervention_truth in zip(
            baseline_predictions, intervention_predictions, baseline_targets, intervention_targets, strict=True,
        )
    ]
    return _group_mean(torch.stack(losses), group_ids)