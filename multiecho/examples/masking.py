"""Observation masking: one learned world, observed under three maskers.

Run:  python examples/masking.py

    A1 physical scene ─A2 bridge→ A4 engine ─A5 stimulus→ A6 next-frame predictor
       ─A7 masked analyst scope→ A8 forward/backward analysis

Masking is a plugin of the analyst scope and selects whole locations (here: pixels),
never single channels. Data-driven maskers see only a counterfactual sweep at
declared values disjoint from the evaluation; their rule is frozen on discovery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.plugins.learned import TinyFramePredictor  # noqa: E402
from examples.plugins.methods import InterpolationForward, ProjectionBackward  # noqa: E402
from examples.plugins.simulation import HEIGHT, WIDTH, AnalyticOscillatorEngine  # noqa: E402
from examples.quickstart import (  # noqa: E402
    CALIBRATION,
    QUERIES,
    REFERENCE,
    STIMULUS,
    contexts,
    physical_world,
)
from multiecho_framework import (  # noqa: E402
    Chain,
    DataRole,
    Declaration,
    Experiment,
    Grouping,
    Intervention,
    ManualMask,
    MaskedScope,
    MonotonicSNRMask,
    NoMask,
    ObservationSpec,
    ParameterBridge,
    ParameterMap,
    Scope,
    ScopeInterface,
    Statement,
)
from multiecho_framework.conformance import check_report  # noqa: E402

PIXELS = Grouping.blocks(HEIGHT * WIDTH, 1, "pixels")
SWEEP = tuple(Intervention((k,), steps=4) for k in (3.1, 3.3, 3.7, 3.9, 4.1, 4.3, 4.7, 4.9))


def masked_chain(masker) -> Chain:
    scene, engine, model = physical_world(), AnalyticOscillatorEngine(), TinyFramePredictor(HEIGHT, WIDTH)
    bridge = ParameterBridge(
        "physical-to-sim", "spring-scene", engine.describe().world_id,
        (ParameterMap("stiffness", "k"), ParameterMap("position", "x0"), ParameterMap("velocity", "v0"),
         ParameterMap("horizon", "steps")),
        (Statement("damping", 0.3, "1/s"), Statement("dt", 0.25, "s")))
    to_model = ScopeInterface("sim-to-model", engine.describe().world_id, model.describe().world_id,
                              Scope("model-input", ("video",)))
    scope = MaskedScope(f"frame:{masker.describe().masker_id}", ("frame",), masker=masker, observed="frame")
    analyst = ScopeInterface("analyst", model.describe().world_id, "analyst", scope)
    return Chain(f"spring-to-frame-predictor:{masker.describe().masker_id}", (scene, engine, model), (bridge, to_model),
                 analyst, ObservationSpec("frame", (masker.output_dims(PIXELS.dims),), "Masked predicted next frame"),
                 stimulus=STIMULUS)


def run(masker) -> dict:
    subject = masked_chain(masker)
    needs_sweep = masker.describe().needs == "sweep"
    declaration = Declaration(
        f"masking-{masker.describe().masker_id}", "Where in a predicted frame does the response to stiffness live?",
        subject.describe(), subject.record(), "stiffness set in the physical scene", "the frame after the keyframe",
        REFERENCE, QUERIES, contexts(), "one independently drawn initial condition", calibration=CALIBRATION,
        tasks=("forward", "backward"), stimulus=STIMULUS, sweep=SWEEP if needs_sweep else ())
    experiment = Experiment(declaration, subject)
    experiment.measure(DataRole.DISCOVERY)
    responses = experiment.responses()
    experiment.freeze(InterpolationForward().fit(responses), "forward")
    experiment.freeze(ProjectionBackward().fit(responses), "backward")
    experiment.test()
    return experiment.report()


def main() -> int:
    left = np.zeros(PIXELS.groups, dtype=bool)
    left.reshape(HEIGHT, WIDTH)[:, : WIDTH // 2] = True
    maskers = (NoMask(), ManualMask(PIXELS, left, "left-half", "the left half of the frame"), MonotonicSNRMask(PIXELS))
    passed = True
    for masker in maskers:
        report = run(masker)
        checks = check_report(report)
        passed &= all(result.passed for result in checks)
        coverage = (report["masking"] or {}).get("coverage", {})
        line = "; ".join(f"{task} {analysis['outcome']['value']} (R2 " + ", ".join(
            f"{e['descriptive_r2']:.2f}" if e["descriptive_r2"] is not None else "—" for e in analysis["evaluations"].values())
            + ")" for task, analysis in report["analyses"].items())
        print(f"{masker.describe().masker_id:<16} coverage {coverage.get('validation', 1.0):.3f}  {line}  "
              f"[{sum(r.passed for r in checks)}/{len(checks)} report checks]")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
