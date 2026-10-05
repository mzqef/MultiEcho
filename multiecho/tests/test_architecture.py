from __future__ import annotations

import numpy as np
import pytest

from multiecho_framework import (
    CONTEXT_CHANNEL,
    Artifact,
    Axis,
    Chain,
    Channel,
    CommunicationRequest,
    CommunicationRunner,
    ComposedInterface,
    DataRole,
    Intervention,
    Modality,
    ObservationSpec,
    ParameterBridge,
    ParameterMap,
    PhysicalScene,
    Region,
    Scope,
    ScopeInterface,
    SeedMode,
    Space,
    Statement,
    scene_context,
)
from multiecho_framework.testing import OscillatorDouble, control_chain, control_contexts

HANDLE = Space("stiffness", (Axis("stiffness", "N/m", 1.0, 9.0),), "Spring stiffness")


def video_artifact() -> Artifact:
    frames = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5, 1) / 60
    return Artifact("engine@test", (Channel("video", Modality.VIDEO, frames, rate=10.0),
                                    Channel("truth", Modality.TEXT, (Statement("k", 4.0, "N/m"),
                                                                     Statement("intent", "probe", kind="intent")))))


def test_channels_enforce_modality_contracts():
    with pytest.raises(ValueError, match="video channel"):
        Channel("video", Modality.VIDEO, np.zeros((4, 5)))
    with pytest.raises(ValueError, match="statements"):
        Channel("text", Modality.TEXT, ("free text",))
    with pytest.raises(ValueError, match="unique"):
        Channel("text", Modality.TEXT, (Statement("k", 1.0), Statement("k", 2.0)))
    with pytest.raises(ValueError, match="finite"):
        Statement("k", float("nan"))
    assert not Channel("latent", Modality.TENSOR, np.ones(3)).data.flags.writeable


def test_scope_masks_windows_clips_and_keeps_the_scope_with_the_data():
    raw = video_artifact()
    region = Region.box("upper", (4, 5), (0, 2), (0, 5))
    scope = Scope("roi", ("video", "truth"), window=(1, 3), region=region, keys=("k",), clip=(0.0, 0.5))
    scoped = scope.apply(raw, via="interface")
    video = scoped.channel("video").data
    assert video.shape == (2, 4, 5, 1) and video.max() <= 0.5
    assert np.all(video[:, 2:] == 0)
    assert list(scoped.channel("truth").statements) == ["k"]
    assert scoped.parent == raw.digest and scoped.scope is scope
    assert scope.vector(scoped).shape == (2 * 2 * 5 + 1,)
    with pytest.raises(ValueError, match="this exact scope"):
        Scope("other", ("video",)).vector(scoped)
    with pytest.raises(ValueError, match="no channel"):
        Scope("missing", ("audio",)).apply(raw, via="interface")


def test_parameter_bridge_maps_units_and_adds_constants():
    scene = Artifact("scene@test", (Channel("scene", Modality.TEXT, (
        Statement("stiffness", 4.0, "N/m"), Statement("position", 0.5, "m"))),))
    bridge = ParameterBridge("bridge", "scene", "engine", (ParameterMap("stiffness", "k"),
                                                           ParameterMap("position", "x0_cm", scale=100.0, unit="cm")),
                             (Statement("dt", 0.25, "s"),))
    message = bridge.transmit(scene)
    statements = message.statements()
    assert statements["x0_cm"].value == 50.0 and statements["x0_cm"].unit == "cm"
    assert statements["dt"].value == 0.25 and message.via == bridge.describe().digest
    with pytest.raises(ValueError, match="lacks numeric statement"):
        bridge.transmit(Artifact("scene@test", (Channel("scene", Modality.TEXT, (Statement("k", 1.0),)),)))


def test_composed_interface_records_itself_and_the_original_parent():
    raw = video_artifact()
    first = ScopeInterface("mask", "engine", "reader", Scope("roi", ("video",), window=(0, 2)))
    second = ScopeInterface("clip", "reader", "receiver", Scope("clip", ("video",), clip=(0.0, 0.1)))
    composed = ComposedInterface("a-to-b", (first, second))
    message = composed.transmit(raw)
    assert message.parent == raw.digest and message.via == composed.describe().digest
    assert composed.describe().source == "engine" and composed.describe().target == "receiver"


def test_physical_scene_verifies_payload_applies_interventions_and_receives():
    scene = PhysicalScene("scene", "v1", HANDLE)
    context = scene_context("c", (Statement("position", 0.5, "m"),), DataRole.DISCOVERY)
    state = scene.apply(scene.fork(scene.open(context, 0)), Intervention((3.0,), 2), 0)
    statements = scene.emit(state).statements()
    assert statements["stiffness"].value == 3.0 and statements["horizon"].value == 2.0
    received = scene.receive(state, Artifact("reader@test", (Channel("scene", Modality.TEXT,
                                                                       (Statement("stiffness", 5.5, "N/m"),)),)), 0)
    assert scene.emit(received).statements()["stiffness"].value == 5.5
    with pytest.raises(ValueError, match="digest"):
        scene.open(scene_context("c", (Statement("position", 0.5, "m"),), DataRole.DISCOVERY).__class__(
            "c", context.payload_ref, "0" * 64, DataRole.DISCOVERY, "c"), 0)


def test_chain_validates_its_topology_and_seals_every_member():
    chain = control_chain()
    record = chain.record()
    assert [world["kind"] for world in record["worlds"]] == ["physical", "simulation"]
    assert record["observer"]["scope"]["channels"] == ["state"]
    scene, engine = chain.worlds
    with pytest.raises(ValueError, match="does not join"):
        Chain("broken", (scene, engine), (ScopeInterface("x", "somewhere", "else", Scope("s", ("scene",))),),
              chain.observer, chain.observation)
    with pytest.raises(ValueError, match="intervention space"):
        Chain("no-handle", (scene, engine), chain.interfaces, chain.observer, chain.observation, handle=1)
    retrained = control_chain(OscillatorDouble(noise=2e-4))
    assert retrained.describe().revision != chain.describe().revision


def test_chain_branches_are_isolated_and_traced():
    chain = control_chain(OscillatorDouble(noise=0.0))
    context = control_contexts({DataRole.DISCOVERY: 1})[0]
    root = chain.open(context, 0)
    first = chain.apply(chain.fork(root), Intervention((3.0,), 4), 0)
    second = chain.apply(chain.fork(root), Intervention((6.0,), 4), 0)
    again = chain.apply(chain.fork(root), Intervention((3.0,), 4), 0)
    np.testing.assert_array_equal(chain.observe(first), chain.observe(again))
    assert not np.array_equal(chain.observe(first), chain.observe(second))
    trace = chain.trace(first)
    assert [artifact.source.split("@")[0] for artifact in trace] == [
        "physical-scene", "physical-scene", "analytic-oscillator-double", "analytic-oscillator-double",
        "analytic-oscillator-double"]
    assert trace[-1].scope is chain.observer.scope and trace[1].via == chain.interfaces[0].describe().digest
    assert trace[-2].channel(CONTEXT_CHANNEL).statements["context_id"].value == context.context_id
    assert chain.emitted(first, "analytic-oscillator-double").channel("truth").statements["position"].numeric
    for state in (first, second, again, root):
        chain.close(state)


def test_chain_refuses_channels_a_world_does_not_accept():
    chain = control_chain()
    chain.interfaces = (ScopeInterface("leaky", "physical-scene", "analytic-oscillator-double",
                                       Scope("scene", ("scene",))),)
    context = control_contexts({DataRole.DISCOVERY: 1})[0]
    with pytest.raises(ValueError, match="does not accept"):
        chain.apply(chain.fork(chain.open(context, 0)), Intervention((3.0,), 4), 0)


def test_communication_runner_isolates_no_correct_and_wrong_messages():
    scene = PhysicalScene("receiver", "v1", HANDLE)
    observer = ScopeInterface("read", "receiver", "analyst", Scope("k", ("scene",), keys=("stiffness",)))
    context = scene_context("r", (Statement("stiffness", 4.0, "N/m"),), DataRole.VALIDATION)

    def message(value: float) -> Artifact:
        return Artifact("reader@test", (Channel("scene", Modality.TEXT, (Statement("stiffness", value, "N/m"),)),),
                        parent="p", via="reader-interface")

    result = CommunicationRunner(scene, observer).run(CommunicationRequest(
        "trial", context, message(6.0), message(2.0), "different stiffness", SeedMode.FIXED, (0,)))
    assert (result.without_message[0, 0], result.with_message[0, 0], result.with_wrong_message[0, 0]) == (4.0, 6.0, 2.0)
    with pytest.raises(ValueError, match="same declared interface"):
        CommunicationRequest("t", context, message(6.0), Artifact("x@y", message(2.0).channels, via="other"),
                             "r", SeedMode.FIXED, (0,))
    with pytest.raises(ValueError, match="cannot be the correct message"):
        CommunicationRequest("t", context, message(6.0), message(6.0), "r", SeedMode.FIXED, (0,))


def test_chain_observation_contract_is_enforced():
    chain = control_chain()
    wrong = Chain("wrong-shape", chain.worlds, chain.interfaces, chain.observer,
                  ObservationSpec("phase-state", (3,), "declared too wide"))
    context = control_contexts({DataRole.DISCOVERY: 1})[0]
    with pytest.raises(ValueError):
        wrong.observe(wrong.apply(wrong.fork(wrong.open(context, 0)), Intervention((3.0,), 4), 0))
