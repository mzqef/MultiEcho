"""MultiEcho quickstart: one sim2learned experiment, built as a communication chain.

Run:  python examples/quickstart.py [--report report.json]

    A1 physical scene ─A2 parameter bridge→ A4 engine ─→ A4 appearance sampler
       ─A5 video interface→ A6 learned model ─A7 analyst scope→ A8 forward/backward analysis

A3 declares the counterfactual branches of every context. Engines, the sampler, the
model and the analysis methods are example plugins in examples/plugins.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.plugins.learned import FUTURE, LATENT, TinyVideoModel  # noqa: E402
from examples.plugins.methods import InterpolationForward, ProjectionBackward  # noqa: E402
from examples.plugins.simulation import (  # noqa: E402
    FRAMES,
    HEIGHT,
    WIDTH,
    AnalyticOscillatorEngine,
    AppearanceSampler,
    EulerOscillatorEngine,
)
from multiecho_framework import (  # noqa: E402
    Axis,
    Chain,
    ConditionDimension,
    DataRole,
    Declaration,
    Experiment,
    Intervention,
    ObservationSpec,
    ParameterBridge,
    ParameterMap,
    PhysicalScene,
    Scope,
    ScopeInterface,
    SeedMode,
    Space,
    Statement,
    StimulusSpec,
    dumps_report,
    scene_context,
    scene_statements,
)
from multiecho_framework.conformance import check_report  # noqa: E402

REFERENCE = Intervention((4.0,), steps=4)
CALIBRATION = (Intervention((3.0,), steps=4), Intervention((5.0,), steps=4))
QUERIES = tuple(Intervention((k,), steps=4) for k in (3.5, 4.5))
#: The stimulus is fixed by the protocol: 16 frames over the 4 x 0.25 s horizon, keyframe last.
STIMULUS = StimulusSpec(frames=FRAMES, interval=1.0 / (FRAMES - 1), height=HEIGHT, width=WIDTH)


def physical_world() -> PhysicalScene:
    """A1: the physical world layer. Here a scene is described by parameters only."""
    handle = Space("spring-stiffness", (Axis("stiffness", "N/m", 1.0, 9.0),), "Stiffness of the spring holding the mass")
    return PhysicalScene("spring-scene", "parameters-only-v1", handle, description="A unit mass on a spring, released or pushed")


def contexts() -> tuple:
    import numpy as np

    rng, output = np.random.default_rng(0), []
    for role, count in ((DataRole.DISCOVERY, 12), (DataRole.VALIDATION, 8), (DataRole.CONFIRMATION, 8)):
        for index in range(count):
            moving = index % 2 == 1
            output.append(scene_context(f"{role.value}-{index:02d}", (
                Statement("position", round(float(rng.uniform(0.4, 1.0) * rng.choice([-1, 1])), 6), "m", "initial-condition"),
                Statement("velocity", round(float(rng.uniform(-1, 1)), 6) if moving else 0.0, "m/s", "initial-condition"),
                Statement("intent", "How does a learned world respond to a stiffer spring?", "", "intent"),
                Statement("stiffness_range", "1-9", "N/m", "range"),
            ), role))
    return tuple(output)


def chain(engine) -> Chain:
    scene, sampler, model = physical_world(), AppearanceSampler(gain=0.02, offset=0.005), TinyVideoModel()
    card = engine.describe()
    bridge = ParameterBridge(  # A2: this translator is only parameters
        "physical-to-sim", "spring-scene", card.world_id,
        (ParameterMap("stiffness", "k"), ParameterMap("position", "x0"), ParameterMap("velocity", "v0"),
         ParameterMap("horizon", "steps")),
        (Statement("damping", 0.3, "1/s"), Statement("dt", 0.25, "s")),
    )
    to_sampler = ScopeInterface("sim-to-sampler", card.world_id, "appearance-sampler", Scope("rendered", ("video", "truth")))
    to_model = ScopeInterface("sim-to-model", "appearance-sampler", "tiny-video-model",  # A5: video only
                              Scope("model-input", ("video",), clip=(0.0, 1.5)))
    analyst = ScopeInterface("analyst", "tiny-video-model", "analyst",  # A7: predicted future latents only
                             Scope("future-latents", ("latent",), window=(FRAMES, FRAMES + FUTURE)))
    return Chain(f"spring-to-{card.world_id}-to-tiny-model", (scene, engine, sampler, model),
                 (bridge, to_sampler, to_model), analyst,
                 ObservationSpec("future-latents", (FUTURE * LATENT,), "Predicted future latents of the tiny model"),
                 stimulus=STIMULUS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=Path("quickstart_report.json"))
    arguments = parser.parse_args()
    start = time.perf_counter()
    subject = chain(AnalyticOscillatorEngine())
    declared = contexts()
    release = ConditionDimension("release", tuple(
        (c.context_id, "at-rest" if scene_statements(c)["velocity"].value == 0 else "moving") for c in declared))
    declaration = Declaration(  # A3: branches of every context, fixed before measuring
        "quickstart-spring", "How does a learned world respond to a stiffer spring?",
        subject.describe(), subject.record(), "stiffness set in the physical scene",
        "four predicted future latent frames", REFERENCE, QUERIES, declared,
        "one independently drawn initial condition", calibration=CALIBRATION, tasks=("forward", "backward"),
        coupling=SeedMode.CONTEXT, draws=3, conditions=(release,), stimulus=STIMULUS,
    )

    experiment = Experiment(declaration, subject)
    discovery = experiment.measure(DataRole.DISCOVERY)  # A4-A7 on every discovery branch
    state = subject.fork(subject.open(declared[0], 0))
    trace = subject.trace(subject.apply(state, QUERIES[0], 0))
    print("One branch through the chain (raw artifacts, messages, and the analyst's scoped view):")
    for artifact in trace:
        channels = ", ".join(f"{c.name}:{c.modality.value}{list(getattr(c.data, 'shape', [len(c.data)]))}"
                             for c in artifact.channels)
        print(f"  {artifact.source.split('@')[0]:<34} {channels}" + (f"   scope={artifact.scope.scope_id}" if artifact.scope else ""))

    responses = experiment.responses()
    experiment.freeze(InterpolationForward().fit(responses), "forward")    # A8
    experiment.freeze(ProjectionBackward().fit(responses), "backward")
    experiment.test()
    report = experiment.report()
    arguments.report.write_text(dumps_report(report), encoding="utf-8")

    response = report["response"]
    print(f"\nMeasured {len(discovery.results)} discovery pairs x {declaration.draws} context draws "
          f"(every branch of a context shares each draw); determinism "
          f"{'identical' if response['determinism']['identical'] else 'VIOLATED'}; held-out response RMS "
          + ", ".join(f"{role} {value:.3f}" for role, value in response["response_rms"].items()))
    for task, analysis in report["analyses"].items():
        stages = "; ".join(f"{role}: " + ", ".join(f"{name} {row['status']}" for name, row in evaluation["controls"].items())
                           for role, evaluation in analysis["evaluations"].items())
        print(f"  {task:<8} {analysis['outcome']['value']:<22} {stages}")
    for analysis in report["analyses"].values():
        print(f"  claim: {analysis['claims'][0]['statement']}")

    finals = []
    for member in (subject, chain(EulerOscillatorEngine())):
        state = member.apply(member.fork(member.open(declared[0], 0)), QUERIES[0], 0)
        engine = member.worlds[1].describe().world_id
        finals.append(member.emitted(state, engine).statements()["final_position"].value)
    print(f"\nOne physical scene, two engines (1:N): final position {finals[0]:.4f} m (analytic) vs {finals[1]:.4f} m (Euler)")

    results = check_report(report)
    passed = sum(result.passed for result in results)
    print(f"\nConformance: {passed}/{len(results)} report checks passed")
    print(f"Sealed report: {arguments.report} (seal {report['seal'][:16]}...)")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
