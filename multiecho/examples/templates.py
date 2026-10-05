"""Copyable plugin templates: an engine, a learned model and a translator, joined into a chain.

Replace the marked parts with your simulator, model or translator, then probe:

    multiecho-conformance world     examples.templates:probe_engine
    multiecho-conformance world     examples.templates:probe_model
    multiecho-conformance interface examples.templates:probe_translator
    multiecho-conformance adapter   examples.templates:probe_chain
    multiecho-conformance keyframe  examples.templates:probe_chain

As shipped, each template is trivially simple and passes every probe. The engine
emits the stimulus (ending at the keyframe) and the true future; the future reaches
the analyst only through a reference route, never the learned world.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiecho_framework import (
    Artifact,
    Axis,
    Chain,
    Channel,
    Context,
    DataRole,
    InterfaceSpec,
    Intervention,
    LearnedContract,
    Modality,
    ObservationSpec,
    ParameterBridge,
    ParameterMap,
    PhysicalScene,
    ReferenceRoute,
    Scope,
    ScopeInterface,
    Space,
    Statement,
    StimulusSpec,
    WorldCard,
    WorldKind,
    scene_context,
)
from multiecho_framework.contracts import fingerprint


@dataclass
class Handle:
    """Everything one branch owns. It must not alias another branch's mutable state."""

    received: dict[str, np.ndarray | tuple]
    seed: int = 0


class EngineTemplate:
    """A simulation world: receives parameters, simulates and renders."""

    # (1) Identity: change the revision with any change to engine, parameters or renderer.
    card = WorldCard("my-engine", fingerprint({"engine": "my-engine", "version": 1}), WorldKind.SIMULATION,
                     accepts=("parameters",), emits=("video", "future", "truth"))

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> Handle:
        return Handle({})

    def fork(self, state: Handle) -> Handle:
        return Handle(dict(state.received), state.seed)

    def receive(self, state: Handle, message: Artifact | None, seed: int) -> Handle:
        if message is not None:
            state.received = {key: item.value for key, item in message.statements().items()}
        state.seed = seed
        return state

    def apply(self, state: Handle, intervention: Intervention, seed: int) -> Handle:
        raise ValueError("Engines receive parameters through a bridge")

    def emit(self, state: Handle) -> Artifact:
        # (2) Simulate and render here. Use only np.random.default_rng(state.seed) for randomness.
        # The stimulus ends at the keyframe; frames after it are the true future (reference only).
        level = float(state.received.get("level", 0.0))
        frames = level * np.linspace(0.5, 1.0, 6)[:, None, None, None] * np.ones((6, 8, 8, 1))
        return Artifact(self.card.source, (Channel("video", Modality.VIDEO, frames[:4], rate=10.0),
                                           Channel("future", Modality.VIDEO, frames[4:], rate=10.0),
                                           Channel("truth", Modality.TEXT, (Statement("level", level, "1"),))))

    def close(self, state: Handle) -> None:
        return None


class ModelTemplate:
    """A learned world: adapts the protocol stimulus to its own input and emits its native output."""

    # (3) Contract: read the whole stimulus; declare the prediction horizon (0: an encoder),
    # how the stimulus is adapted, and the numerical precision. Outputs stay native.
    card = WorldCard("my-model", fingerprint({"checkpoint": "sha256-of-weights"}), WorldKind.LEARNED,
                     accepts=("video",), emits=("latent",),
                     contract=LearnedContract(history=4, horizon=0, step=1, autoregressive=False,
                                              adaptation="frames used as delivered", precision="float64"))

    def describe(self) -> WorldCard:
        return self.card

    def open(self, context: Context, seed: int) -> Handle:
        return Handle({})

    def fork(self, state: Handle) -> Handle:
        return Handle(dict(state.received), state.seed)

    def receive(self, state: Handle, message: Artifact | None, seed: int) -> Handle:
        if message is not None:
            state.received = {"video": message.channel("video").data}
        state.seed = seed
        return state

    def apply(self, state: Handle, intervention: Intervention, seed: int) -> Handle:
        # (4) Optional: internal-state interventions. Declare a handle Space in the card to use them.
        raise ValueError("This model declares no internal-state handle")

    def emit(self, state: Handle) -> Artifact:
        # (5) Run the frozen model here; emit its native output (latents, tokens or frames).
        latent = state.received["video"].reshape(len(state.received["video"]), -1).mean(axis=1, keepdims=True)
        return Artifact(self.card.source, (Channel("latent", Modality.TENSOR, latent),))

    def close(self, state: Handle) -> None:
        return None


@dataclass(frozen=True)
class TranslatorTemplate:
    """An interface that needs a model: e.g. real2sim, or reading parameters from video."""

    interface_id: str = "my-translator"
    source: str = "my-engine"
    target: str = "physical-scene"

    def describe(self) -> InterfaceSpec:
        return InterfaceSpec(self.interface_id, fingerprint({"translator": "v1"}), self.source, self.target,
                             "model-translator", "Reads a level statement from mean frame brightness")

    def transmit(self, artifact: Artifact) -> Artifact:
        # (6) Map the artifact to the receiver's channels. Never modify the input artifact.
        level = float(artifact.channel("video").data.mean())
        return Artifact(artifact.source, (Channel("scene", Modality.TEXT, (Statement("level", level, "1"),)),),
                        parent=artifact.digest, via=self.describe().digest)


HANDLE = Space("level", (Axis("level", "1", 0.0, 1.0),), "Brightness level of the scene")


def template_chain() -> Chain:
    scene, engine, model = PhysicalScene("physical-scene", "v1", HANDLE), EngineTemplate(), ModelTemplate()
    bridge = ParameterBridge("scene-to-engine", "physical-scene", "my-engine", (ParameterMap("level", "level"),))
    to_model = ScopeInterface("engine-to-model", "my-engine", "my-model", Scope("frames", ("video",)))
    analyst = ScopeInterface("analyst", "my-model", "analyst", Scope("latent", ("latent",)))
    return Chain("template-chain", (scene, engine, model), (bridge, to_model), analyst,
                 ObservationSpec("latent", (4,), "Mean brightness per frame"),
                 stimulus=StimulusSpec(frames=4, interval=0.1, height=8, width=8),
                 references=(ReferenceRoute("my-engine", Scope("reference", ("future",))),))


def _context() -> Context:
    return scene_context("probe", (Statement("level", 0.2, "1"),), DataRole.DISCOVERY)


def probe_engine():
    message = ParameterBridge("b", "s", "e", (ParameterMap("level", "level"),)).transmit(
        Artifact("scene@probe", (Channel("scene", Modality.TEXT, (Statement("level", 0.5, "1"),)),)))
    return EngineTemplate(), _context(), message


def probe_model():
    return ModelTemplate(), _context(), EngineTemplate().emit(Handle({"level": 0.5}))


def probe_translator():
    return TranslatorTemplate(), EngineTemplate().emit(Handle({"level": 0.5}))


def probe_chain():
    return template_chain(), _context(), Intervention((0.7,))
