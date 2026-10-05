"""Keyframe protocol: the stimulus gate, reference routes and learned-world contracts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from multiecho_framework import (
    Chain,
    LearnedContract,
    ObservationSpec,
    ReferenceRoute,
    Scope,
    ScopeInterface,
    StimulusSpec,
    WorldCard,
    WorldKind,
)
from multiecho_framework.conformance import check_keyframe, check_stimulus_identity
from multiecho_framework.selftest import QUERIES, control_declaration
from multiecho_framework.testing import (
    STIMULUS_FRAMES,
    ExtrapolatorDouble,
    PersistenceModelDouble,
    control_chain,
    learned_chain,
)

CONTEXT = control_declaration(control_chain()).contexts[0]


def test_stimulus_spec_and_learned_contract_validate():
    spec = StimulusSpec(8, 0.25, 6, 12)
    assert spec.keyframe == 7 and spec.observation_time == 0.25
    for bad in (dict(frames=0), dict(interval=0.0), dict(observe_after=0)):
        with pytest.raises(ValueError):
            replace(spec, **bad)
    with pytest.raises(ValueError, match="autoregressively"):
        LearnedContract(8, 4, 1, False, "as delivered", "float64")
    with pytest.raises(ValueError, match="adapted"):
        LearnedContract(8, 1, 1, False, "", "float64")
    with pytest.raises(ValueError, match="only learned worlds"):
        WorldCard("m", "r", WorldKind.LEARNED, accepts=("video",), emits=("y",))
    with pytest.raises(ValueError, match="only learned worlds"):
        WorldCard("s", "r", WorldKind.SIMULATION, emits=("y",), contract=LearnedContract(8, 1, 1, False, "a", "p"))


def test_learned_world_receives_only_the_stimulus_and_the_analyst_gets_the_reference():
    chain = learned_chain()
    root = chain.open(CONTEXT, 0)
    state = chain.apply(chain.fork(root), QUERIES[0], 0)
    delivered = chain.delivered(state, "extrapolator-double")
    assert [c.name for c in delivered.channels] == ["video"] and len(delivered.channel("video").data) == STIMULUS_FRAMES
    joined = chain.trace(state)[-2]
    assert {"prediction", "future"} <= {c.name for c in joined.channels}
    assert chain.record()["stimulus"]["frames"] == STIMULUS_FRAMES and chain.record()["references"]
    assert all(result.passed for result in check_keyframe(chain, CONTEXT, QUERIES[0]))
    chain.close(state)
    chain.close(root)


def test_chain_refuses_learned_worlds_without_stimulus_or_with_reference_access():
    chain = learned_chain()
    scene, track, model = chain.worlds
    analyst = ScopeInterface("analyst", model.describe().world_id, "analyst", Scope("prediction", ("prediction",)))
    spec = ObservationSpec("prediction", (72,), "prediction")
    with pytest.raises(ValueError, match="StimulusSpec"):
        Chain("c", chain.worlds, chain.interfaces, analyst, spec)
    with pytest.raises(ValueError, match="whole 7-frame"):
        Chain("c", chain.worlds, chain.interfaces, analyst, spec, stimulus=replace(chain.stimulus, frames=7))

    class Peeking(ExtrapolatorDouble):
        world_id = "extrapolator-double"
        accepts = ("video", "future")

    peeking = (scene, track, Peeking())
    with pytest.raises(ValueError, match="accepts reference channels"):
        Chain("c", peeking, chain.interfaces, analyst, spec, stimulus=chain.stimulus,
              references=(ReferenceRoute("track-double", Scope("reference", ("future",))),))


def test_gate_refuses_future_frames_at_run_time():
    chain = learned_chain(input_scope=Scope("model-input", ("video", "future")))
    with pytest.raises(ValueError, match="does not accept|Keyframe protocol violated"):
        chain.apply(chain.fork(chain.open(CONTEXT, 0)), QUERIES[0], 0)


def test_subjects_share_the_protocol_stimulus():
    pair = (learned_chain(), learned_chain(PersistenceModelDouble()))
    assert all(result.passed for result in check_stimulus_identity(pair, CONTEXT, QUERIES[0]))
    clipped = learned_chain(PersistenceModelDouble(), input_scope=Scope("model-input", ("video",), clip=(0.0, 0.5)))
    assert not check_stimulus_identity((pair[0], clipped), CONTEXT, QUERIES[0])[0].passed
    state = pair[0].apply(pair[0].fork(pair[0].open(CONTEXT, 0)), QUERIES[0], 0)
    assert np.isfinite(pair[0].observe(state)).all()
