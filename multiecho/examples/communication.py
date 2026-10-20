"""Communication: world A runs, an A-B interface masks, clips and translates, world B receives.

Run:  python examples/communication.py

    B1 world A: a simulation (physical scene -> engine) runs a spring with unknown stiffness
    B2 interface A->B: mask the video to a region, clip it, then a model-based translator
       reads stiffness from the track and emits a text statement
    B3 world B: a physical-world layer receives the statement into its scene description

Three tests stay separate: the interface's own conformance, translation agreement
on aligned scenes, and utility for B against no message and a matched wrong message.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.plugins.simulation import HEIGHT, WIDTH, AnalyticOscillatorEngine  # noqa: E402
from examples.plugins.translators import StiffnessReadout  # noqa: E402
from multiecho_framework import (  # noqa: E402
    Alignment,
    Axis,
    Chain,
    CommunicationRequest,
    CommunicationRunner,
    ComposedInterface,
    DataRole,
    EvaluationPlan,
    EvaluationTarget,
    Intervention,
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
    evaluate_communication,
    evaluate_translation,
    scene_context,
)
from multiecho_framework.conformance import check_interface  # noqa: E402

HANDLE = Space("spring-stiffness", (Axis("stiffness", "N/m", 1.0, 9.0),), "Stiffness of the spring")


def world_a() -> Chain:
    scene, engine = PhysicalScene("spring-scene", "parameters-only-v1", HANDLE), AnalyticOscillatorEngine()
    bridge = ParameterBridge("physical-to-sim", "spring-scene", engine.describe().world_id,
                             (ParameterMap("stiffness", "k"), ParameterMap("position", "x0"),
                              ParameterMap("velocity", "v0"), ParameterMap("horizon", "steps")),
                             (Statement("damping", 0.3, "1/s"), Statement("dt", 0.25, "s")))
    observer = ScopeInterface("a-output", engine.describe().world_id, "b", Scope("video", ("video",)))
    return Chain("world-a", (scene, engine), (bridge,), observer,
                 ObservationSpec("video", (16 * HEIGHT * WIDTH,), "Rendered frames"))


def main() -> int:
    sender, receiver = world_a(), PhysicalScene("lab-notebook", "parameters-only-v1", HANDLE)
    engine = sender.worlds[1].describe().world_id
    mask = ScopeInterface("mask-and-clip", engine, "stiffness-readout",
                          Scope("track-region", ("video",), region=Region.box("track", (HEIGHT, WIDTH), (4, 12), (0, WIDTH)),
                                clip=(0.0, 1.0)))
    interface = ComposedInterface("a-to-b", (mask, StiffnessReadout("stiffness-readout", "stiffness-readout", "lab-notebook")))
    observe_b = ScopeInterface("b-analyst", "lab-notebook", "analyst", Scope("stiffness", ("scene",), keys=("stiffness",)))

    rng = np.random.default_rng(3)
    alignments, trials, targets = [], [], []
    for index in range(8):
        truth = float(rng.uniform(2.5, 7.5))
        wrong_truth = 10.0 - truth
        context = scene_context(f"scene-{index:02d}", (
            Statement("position", float(rng.uniform(0.5, 1.0)), "m", "initial-condition"),
            Statement("velocity", 0.0, "m/s", "initial-condition"),
            Statement("stiffness", 4.0, "N/m", "prior"),
        ), DataRole.VALIDATION)

        def run_a(stiffness: float, context=context):
            state = sender.apply(sender.fork(sender.open(context, 0)), Intervention((stiffness,), 8), 0)
            return interface.transmit(sender.emitted(state, engine))

        message, wrong = run_a(truth), run_a(wrong_truth)
        own = receiver.emit(receiver.apply(receiver.fork(receiver.open(context, 0)), Intervention((truth,), 8), 0))
        alignments.append(Alignment(f"align-{index}", message, own, context.exchangeability_group,
                                    "the same scene, read by the translator and described by B"))
        trials.append(CommunicationRunner(receiver, observe_b).run(CommunicationRequest(
            f"trial-{index}", context, message, wrong, "the same interface applied to a different stiffness",
            SeedMode.FIXED, (0,))))
        targets.append([truth])

    probe = sender.apply(sender.fork(sender.open(context, 0)), Intervention((5.0,), 8), 0)
    interface_checks = check_interface(interface, sender.emitted(probe, engine))
    prior = np.full((len(alignments), 1), 4.0)
    translation = evaluate_translation(interface, tuple(alignments), observe_b.scope, {"prior": prior},
                                       EvaluationPlan("translation", EvaluationTarget.MESSAGE_AGREEMENT, DataRole.VALIDATION, 0.0))
    utility = evaluate_communication(tuple(trials), np.array(targets),
                                     EvaluationPlan("utility", EvaluationTarget.RECEIVER_OBSERVATION, DataRole.VALIDATION, 0.0),
                                     target_semantics="the scene's true stiffness")

    print(f"1. Interface    {interface.describe().description}: "
          f"{sum(r.passed for r in interface_checks)}/{len(interface_checks)} interface checks passed")
    print(f"2. Translation  stiffness RMS error {np.sqrt(translation.group_weighted_mse):.3f} N/m; "
          + ", ".join(f"vs {c.comparator}: {c.status.value}" for c in translation.comparisons))
    print(f"3. Utility      B's stiffness RMS error {np.sqrt(utility.group_weighted_mse):.3f} N/m; "
          + ", ".join(f"vs {c.comparator}: {c.status.value}" for c in utility.comparisons))
    supported = all(r.passed for r in interface_checks) and all(
        c.status.value == "supported" for c in (*translation.comparisons, *utility.comparisons))
    print("All three tests supported." if supported else "Not every test is supported.")
    return 0 if supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
