"""Known-answer self-test: every conformance check must pass a compliant fixture and fail
its targeted violation. A check that cannot fail is not a check."""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .artifacts import Artifact, Channel, Modality, Scope
from .conformance import (
    REPORT_CHECKS,
    check_adapter,
    check_interface,
    check_keyframe,
    check_report,
    check_stimulus_identity,
    check_world,
)
from .contracts import DataRole, Intervention
from .experiment import Experiment, seal
from .interfaces import ScopeInterface
from .masking import Grouping, MonotonicSNRMask
from .protocol import Declaration, Outcome
from .study import StudyLock
from .testing import (
    PREDICTION_DIMS,
    STIFFNESS,
    SWEEP,
    ExtrapolatorDouble,
    InterpolationMethod,
    NullDouble,
    OscillatorDouble,
    OscillatorLaw,
    OscillatorMethod,
    PersistenceModelDouble,
    ScrambledDouble,
    TrackDouble,
    control_chain,
    control_contexts,
    learned_chain,
    release_condition,
)
from .worlds import PhysicalScene

REFERENCE = Intervention((4.0,), 4)
QUERIES = tuple(Intervention((value,), 4) for value in (3.0, 5.0, 6.0))
CALIBRATION = (Intervention((2.0,), 4), Intervention((8.0,), 4))


def control_declaration(chain, experiment_id: str = "oscillator-stiffness") -> Declaration:
    contexts = control_contexts({DataRole.DISCOVERY: 12, DataRole.VALIDATION: 8, DataRole.CONFIRMATION: 8})
    return Declaration(
        experiment_id, "How does the end-of-horizon phase state respond to spring stiffness?",
        chain.describe(), chain.record(), "stiffness set in the physical scene and bridged to the engine",
        "end of the four-step horizon", REFERENCE, QUERIES, contexts, "one independently drawn initial condition",
        calibration=CALIBRATION, conditions=(release_condition(contexts),),
        reference_world="analytic unit-mass oscillator", reference_tolerance=0.05,
    )


def run_control(simulator: OscillatorDouble | None = None, *, reference: bool = True) -> dict:
    chain = control_chain(simulator)
    experiment = Experiment(control_declaration(chain), chain)
    experiment.measure(DataRole.DISCOVERY)
    law = OscillatorMethod().fit(experiment.responses())
    experiment.freeze(law)
    experiment.test()
    if reference:
        experiment.compare_reference(OscillatorLaw(law.describe(), 0.3))
    return experiment.report()


def learned_declaration(chain, study: bool = True) -> Declaration:
    """The learned control's declaration; with `study`, it belongs to a study locked at its own values."""
    contexts = control_contexts({DataRole.DISCOVERY: 10, DataRole.VALIDATION: 6, DataRole.CONFIRMATION: 6}, seed=1)
    declaration = Declaration(
        "track-extrapolation", "How does a learned next-frame prediction respond to spring stiffness?",
        chain.describe(), chain.record(), "stiffness set in the physical scene and bridged to the renderer",
        "the frame after the keyframe", REFERENCE, QUERIES, contexts, "one independently drawn initial condition",
        calibration=CALIBRATION, stimulus=chain.stimulus, sweep=SWEEP if chain.observer.scope.record().get("masker")
        and chain.observer.scope.rule["needs"] == "sweep" else ())
    if not study:
        return declaration
    return replace(declaration, study=StudyLock.from_declaration("track-study", declaration))


def run_learned_control(masker=None) -> dict:
    masker = masker or MonotonicSNRMask(Grouping.blocks(PREDICTION_DIMS, 1))
    chain = learned_chain(masker=masker)
    experiment = Experiment(learned_declaration(chain), chain)
    experiment.measure(DataRole.DISCOVERY)
    experiment.freeze(InterpolationMethod().fit(experiment.responses()))
    experiment.test()
    return experiment.report()


def _mutations() -> dict[str, Callable[[dict], dict]]:
    def resealed(change: Callable[[dict], None]) -> Callable[[dict], dict]:
        def apply(report: dict) -> dict:
            corrupted = copy.deepcopy(report)
            change(corrupted)
            return seal(corrupted)
        return apply

    def unsealed(report: dict) -> dict:
        corrupted = copy.deepcopy(report)
        corrupted["analyses"]["forward"]["outcome"]["meaning"] += " (edited)"
        return corrupted

    def reorder(report: dict) -> None:
        stages = report["stages"]
        frozen = next(i for i, event in enumerate(stages) if event["stage"] == "frozen")
        stages.insert(len(stages) - 1, stages.pop(frozen))
        for index, event in enumerate(stages):
            event["index"] = index

    def leak(report: dict) -> None:
        contexts = report["declaration"]["contexts"]
        held = next(c for c in contexts if c["role"] == "validation")
        held["exchangeability_group"] = next(c for c in contexts if c["role"] == "discovery")["exchangeability_group"]

    def forward(change: Callable[[dict], None]) -> Callable[[dict], None]:
        return lambda report: change(report["analyses"]["forward"])

    def widen(analysis: dict) -> None:
        analysis["claims"][0]["regime"] = {"release": ["at-rest", "moving", "unmeasured"]}

    def rescope(report: dict) -> None:
        report["declaration"]["chain"]["observer"]["scope"]["channels"] = ["state", "truth"]

    def rechain(report: dict) -> None:
        report["declaration"]["chain"]["worlds"][1]["revision"] = "retrained"

    return {
        "C01": resealed(lambda r: r.__setitem__("protocol_version", "9.9")),
        "C02": unsealed,
        "C03": resealed(lambda r: r["declaration"].__setitem__("handle", "")),
        "C04": resealed(reorder),
        "C05": resealed(lambda r: r["stages"][-1].__setitem__("world", "0" * 64)),
        "C06": resealed(forward(lambda a: a["candidate"]["fitted_on"].append("validation-00"))),
        "C07": resealed(leak),
        "C08": resealed(forward(lambda a: a["evaluations"]["validation"].__setitem__("controls", {}))),
        "C09": resealed(lambda r: r["response"].__setitem__("responsive", {"validation": False})),
        "C10": resealed(forward(lambda a: a["evaluations"]["validation"].__setitem__("descriptive_r2", None))),
        "C11": resealed(forward(lambda a: a["evaluations"]["validation"]["controls"]["zero-response"]
                                .__setitem__("status", "contradicted"))),
        "C12": resealed(forward(lambda a: a["outcome"].__setitem__("value", "no_law"))),
        "C13": resealed(forward(widen)),
        "C14": resealed(lambda r: r["reference"].__setitem__("status", "physical")),
        "C15": resealed(rescope),
        "C16": resealed(rechain),
    }


def _learned_mutations() -> dict[str, Callable[[dict], dict]]:
    def resealed(change: Callable[[dict], None]) -> Callable[[dict], dict]:
        def apply(report: dict) -> dict:
            corrupted = copy.deepcopy(report)
            change(corrupted)
            return seal(corrupted)
        return apply

    def drop_instance(report: dict) -> None:
        report["masking"]["instances"].pop(next(iter(report["masking"]["instances"])))

    def overlap_sweep(report: dict) -> None:
        report["declaration"]["sweep"][0]["values"] = list(report["declaration"]["reference"]["values"])

    def short_history(report: dict) -> None:
        learned = next(w for w in report["declaration"]["chain"]["worlds"] if w["kind"] == "learned")
        learned["contract"]["history"] -= 1

    def relock(report: dict) -> None:
        report["declaration"]["study"]["locked"]["stimulus"]["frames"] += 8

    def drop_keyframes(report: dict) -> None:
        report["keyframes"]["contexts"].pop(next(iter(report["keyframes"]["contexts"])))

    return {
        "C17": resealed(drop_instance),
        "C18": resealed(overlap_sweep),
        "C19": resealed(short_history),
        "C20": resealed(lambda r: r["response"]["determinism"].__setitem__("identical", False)),
        "C21": resealed(relock),
        "C22": resealed(drop_keyframes),
    }


class _CountingTrack(TrackDouble):
    """Breaks hard control: every emission shifts the frames by a counter shared across branches."""

    def emit(self, state):
        self._count = getattr(self, "_count", 0) + 1
        artifact = super().emit(state)
        video = artifact.channel("video")
        shifted = Channel("video", Modality.VIDEO, video.data + 1e-3 * self._count, video.rate, video.unit, video.semantics)
        return Artifact(artifact.source, (shifted, artifact.channel("future")))


class _FutureIntoVideo:
    """Breaks the keyframe protocol: appends the true future to the stimulus."""

    def __init__(self, inner: ScopeInterface):
        self.inner = inner

    def describe(self):
        return self.inner.describe()

    def transmit(self, artifact):
        video, future = artifact.channel("video"), artifact.channel("future")
        joined = Artifact(artifact.source, (Channel("video", Modality.VIDEO, np.concatenate([video.data, future.data]),
                                                    video.rate, video.unit, video.semantics),))
        return self.inner.transmit(joined)


class _HintInterface:
    """Breaks future invariance without breaking the gate: smuggles the future in a tensor channel."""

    def __init__(self, inner: ScopeInterface):
        self.inner = inner

    def describe(self):
        return self.inner.describe()

    def transmit(self, artifact):
        message = self.inner.transmit(artifact)
        hint = Channel("hint", Modality.TENSOR, artifact.channel("future").data.reshape(-1))
        return Artifact(message.source, (*message.channels, hint), message.parent, message.via, message.scope)


class _HintedModel(ExtrapolatorDouble):
    world_id = "hinted-model-double"
    accepts = ("video", "hint")

    def receive(self, state, message, seed):
        state = super().receive(state, message, seed)
        if message is not None:
            state.extra = message.channel("hint").data
        return state

    def predict(self, state):
        return super().predict(state) + 0.5 * state.extra[:PREDICTION_DIMS]


class _LeakyDouble(OscillatorDouble):
    """Forks alias one state, and each run shifts the shared state."""

    def fork(self, state):
        return state

    def receive(self, state, message, seed):
        state = super().receive(state, message, seed)
        state.parameters["x0"] = state.parameters.get("x0", 0.0) + getattr(self, "_shift", 0.0)
        self._shift = getattr(self, "_shift", 0.0) + 0.1
        return state


class _GlobalRandomDouble(OscillatorDouble):
    def emit(self, state):
        artifact = super().emit(state)
        noisy = replace(artifact.channels[0], data=artifact.channels[0].data + np.random.normal(0, 1e-3, 2))
        return Artifact(artifact.source, (noisy, *artifact.channels[1:]))


class _DriftingDouble(OscillatorDouble):
    def describe(self):
        self._calls = getattr(self, "_calls", 0) + 1
        return replace(self.card, revision=f"{self.card.revision}-{self._calls}")


def _unverified(context):
    import json

    from .artifacts import Statement

    return {item["key"]: Statement(**item) for item in json.loads(context.payload_ref.removeprefix("inline:"))}, ()


class _RandomInterface:
    def __init__(self, inner):
        self.inner = inner

    def describe(self):
        return self.inner.describe()

    def transmit(self, artifact):
        message = self.inner.transmit(artifact)
        return replace(message, parent=f"{message.parent}#{np.random.randint(1 << 30)}")


class _MutatingInterface(_RandomInterface):
    def transmit(self, artifact):
        object.__setattr__(artifact, "via", "tampered")
        return self.inner.transmit(artifact)


def run_self_test(verbose: bool = True) -> bool:
    rows: list[tuple[str, str, bool]] = []
    expected = {"known-law": (None, Outcome.STABLE_LAW), "null-response": (NullDouble(), Outcome.UNIDENTIFIED),
                "scrambled-response": (ScrambledDouble(), Outcome.UNIDENTIFIED)}
    reports = {}
    for name, (simulator, outcome) in expected.items():
        reports[name] = run_control(simulator, reference=simulator is None)
        rows.append((f"outcome/{name}", f"returns {outcome.value}",
                     reports[name]["analyses"]["forward"]["outcome"]["value"] == outcome.value))
        failed = [result.check_id for result in check_report(reports[name]) if not result.passed]
        rows.append((f"report/{name}", "passes every report check", not failed))
    rows.append(("response/null-response", "reports a zero response RMS (described, not adjudicated)",
                 all(value == 0 for value in reports["null-response"]["response"]["response_rms"].values())))
    compliant = reports["known-law"]
    for check_id, mutate in _mutations().items():
        results = {result.check_id: result.passed for result in check_report(mutate(compliant))}
        title = next(title for cid, title, _ in REPORT_CHECKS if cid == check_id)
        rows.append((check_id, f"detects violation: {title}", not results[check_id]))
    learned = run_learned_control()
    failed = [result.check_id for result in check_report(learned) if not result.passed]
    rows.append(("report/learned-masked", "passes every report check (learned world, masked scope)", not failed))
    for check_id, mutate in _learned_mutations().items():
        results = {result.check_id: result.passed for result in check_report(mutate(learned))}
        title = next(title for cid, title, _ in REPORT_CHECKS if cid == check_id)
        rows.append((check_id, f"detects violation: {title}", not results[check_id]))

    chain = control_chain()
    context, action = control_declaration(chain).contexts[0], QUERIES[0]
    rows.append(("adapter/control-chain", "passes every adapter check",
                 all(result.passed for result in check_adapter(chain, context, action))))
    unverified = control_chain()
    unverified.worlds = (PhysicalScene("physical-scene", "parameters-only-v1", STIFFNESS, loader=_unverified),
                         *unverified.worlds[1:])
    for check_id, broken in (("A01", control_chain(_DriftingDouble())), ("A02", control_chain(_LeakyDouble())),
                             ("A04", control_chain(_GlobalRandomDouble())), ("A05", unverified)):
        results = {result.check_id: result.passed for result in check_adapter(broken, context, action)}
        rows.append((check_id, f"detects a broken chain ({check_id})", not results[check_id]))

    track = learned_chain()
    rows.append(("adapter/learned-chain", "passes every adapter check",
                 all(result.passed for result in check_adapter(track, context, action))))
    counting = learned_chain()
    counting.worlds = (counting.worlds[0], _CountingTrack(), counting.worlds[2])
    results = {result.check_id: result.passed for result in check_adapter(counting, context, action)}
    rows.append(("A09", "detects a branch that depends on other branches", not results["A09"]))
    rows.append(("keyframe/learned-chain", "passes K01 and K02",
                 all(result.passed for result in check_keyframe(track, context, action))))
    leaky = learned_chain(input_interface=_FutureIntoVideo(track.interfaces[1]))
    results = {result.check_id: result.passed for result in check_keyframe(leaky, context, action)}
    rows.append(("K01", "detects future frames delivered to a learned world", not results["K01"]))
    hinted_model = _HintedModel()
    hinted = learned_chain(hinted_model, input_interface=_HintInterface(ScopeInterface(
        "track-to-model", "track-double", hinted_model.describe().world_id, Scope("model-input", ("video",)))))
    results = {result.check_id: result.passed for result in check_keyframe(hinted, context, action)}
    rows.append(("K02", "detects a learned output that depends on the true future", not results["K02"]))
    pair = (track, learned_chain(PersistenceModelDouble()))
    rows.append(("stimulus/two-subjects", "passes K04",
                 all(result.passed for result in check_stimulus_identity(pair, context, action))))
    clipped = learned_chain(PersistenceModelDouble(), input_scope=Scope("model-input", ("video",), clip=(0.0, 0.5)))
    results = {result.check_id: result.passed for result in check_stimulus_identity((track, clipped), context, action)}
    rows.append(("K04", "detects a subject-specific stimulus", not results["K04"]))

    scene, bridge = chain.worlds[0], chain.interfaces[0]
    root = scene.open(context, 0)
    message = bridge.transmit(scene.emit(scene.apply(scene.fork(root), action, 0)))
    rows.append(("world/oscillator-double", "passes every world check",
                 all(result.passed for result in check_world(OscillatorDouble(), context, message))))
    for check_id, broken in (("W01", _DriftingDouble()), ("W02", _LeakyDouble()), ("W04", _GlobalRandomDouble())):
        results = {result.check_id: result.passed for result in check_world(broken, context, message)}
        rows.append((check_id, f"detects {type(broken).__name__.strip('_')}", not results[check_id]))
    artifact = scene.emit(scene.apply(scene.fork(root), action, 0))
    rows.append(("interface/parameter-bridge", "passes every interface check",
                 all(result.passed for result in check_interface(bridge, artifact))))
    for check_id, broken in (("I02", _RandomInterface(bridge)), ("I04", _MutatingInterface(bridge))):
        results = {result.check_id: result.passed for result in check_interface(broken, artifact)}
        rows.append((check_id, f"detects {type(broken).__name__.strip('_')}", not results[check_id]))

    if verbose:
        for check_id, title, passed in rows:
            print(f"{'PASS' if passed else 'FAIL'} {check_id:<28} {title}")
        print(f"{sum(passed for *_, passed in rows)}/{len(rows)} self-test expectations met")
    return all(passed for *_, passed in rows)


__all__ = ["control_declaration", "learned_declaration", "run_control", "run_learned_control", "run_self_test"]
