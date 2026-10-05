"""The three-reference estimator plugins (PL, linear, kernel) inside the stage-enforcing Experiment."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from examples.plugins.methods import InterpolationForward, ProjectionBackward
from examples.plugins.three_reference import (
    METHODS,
    ThreeReferenceBackward,
    ThreeReferenceForward,
    predict_effects,
    recover_positions,
    weights,
)
from multiecho_framework import DataRole, Experiment, Intervention
from multiecho_framework.analysis import views
from multiecho_framework.conformance import check_report
from multiecho_framework.selftest import control_declaration
from multiecho_framework.testing import control_chain

pytest.importorskip("scipy")
QUERIES = np.array([0.1, 0.25, 0.4, 0.6, 0.8, 0.95])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("middle", [0.5, 0.3])
def test_weights_are_affine_and_reproduce_the_references(method, middle):
    knots = np.array([0.0, middle, 1.0])
    w = weights(method, knots, middle)
    if method != "linear":
        np.testing.assert_allclose(w, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(weights(method, QUERIES, middle) @ knots, QUERIES, atol=1e-9)
    with pytest.raises(ValueError, match="interpolate"):
        weights(method, [1.2], middle)


@pytest.mark.parametrize("method", METHODS)
def test_affine_effects_are_recovered_both_ways(method):
    rng = np.random.default_rng(0)
    slope = rng.normal(size=(4, 5))
    low, high = -0.5 * slope, 0.5 * slope
    truth = (QUERIES[None, :, None] - 0.5) * slope[:, None, :]
    np.testing.assert_allclose(predict_effects(method, low, high, QUERIES), truth, atol=1e-8)
    np.testing.assert_allclose(recover_positions(method, low, high, truth), np.broadcast_to(QUERIES, (4, 6)), atol=1e-3)
    np.testing.assert_array_equal(recover_positions(method, 0 * low, 0 * high, truth), 0.5)




@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("coordinate", ["raw", "log"])
def test_plugins_run_in_an_experiment_and_pass_every_report_check(method, coordinate):
    chain = control_chain()
    declaration = replace(control_declaration(chain), tasks=("forward", "backward"), resamples=199,
                          reference_tolerance=None, reference_world="")
    experiment = Experiment(declaration, chain)
    experiment.measure(DataRole.DISCOVERY)
    responses = experiment.responses()
    forward = ThreeReferenceForward(method, coordinate).fit(responses)
    backward = ThreeReferenceBackward(method, coordinate).fit(responses)
    assert forward.spec.law_id == f"three-reference-{method}-forward"
    assert (forward.epsilon in (0.3, 1.0, 3.0)) and (method == "kernel" or forward.epsilon == 1.0)
    experiment.freeze(forward, "forward")
    experiment.freeze(backward, "backward")
    experiment.test()
    report = experiment.report()
    assert all(result.passed for result in check_report(report))
    view = views(experiment.datasets[DataRole.VALIDATION])[experiment.declaration.contexts_for(DataRole.VALIDATION)[0].context_id]
    if method != "linear":  # PL and kernel interpolate the references; linear is a least-squares fit
        for item in view.calibration:
            np.testing.assert_allclose(forward.predict(view, item.intervention), item.response, atol=1e-8)
            assert abs(backward.recover(view, item.response)[0] - item.intervention.values[0]) < 1e-2


def test_raw_pl_equals_the_interpolation_example():
    chain = control_chain()
    experiment = Experiment(replace(control_declaration(chain), resamples=199), chain)
    experiment.measure(DataRole.DISCOVERY)
    responses = experiment.responses()
    ours, theirs = ThreeReferenceForward("pl").fit(responses), InterpolationForward().fit(responses)
    ours_back, theirs_back = ThreeReferenceBackward("pl").fit(responses), ProjectionBackward().fit(responses)
    for view, queries in zip(responses.views, responses.queries, strict=True):
        for query in queries:
            np.testing.assert_allclose(ours.predict(view, query.intervention), theirs.predict(view, query.intervention),
                                       atol=1e-10)
            assert abs(ours_back.recover(view, query.response)[0] - theirs_back.recover(view, query.response)[0]) < 1e-2
    with pytest.raises(ValueError, match="interpolate"):
        ours.predict(responses.views[0], Intervention((9.0,), 4))
