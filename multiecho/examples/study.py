"""Studies: experiments that may be compared, and one that may not.

Run:  python examples/study.py [--workspace DIR]

A pilot experiment (16-frame stimulus, unmasked) fixes a study lock. A second
experiment of the same study changes only the analyst's masker: it is admitted and is
comparable with the pilot. A third experiment delivers an 8-frame stimulus to a model
that reads 8 frames: the lock refuses its declaration before anything runs, and a
field-by-field comparison of the two reports names the stimulus as the difference.

Every experiment also reports keyframe alignment: the spring's stiffness changes where
the mass is at the keyframe, so the keyframes of a context's branches differ and are
flagged above the declared warning threshold.

With --workspace, the reports, the lock and keyframe galleries are written for the
`multiecho-ui` viewer.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

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
    KeyframeAlignment,
    MaskedScope,
    MonotonicSNRMask,
    NoMask,
    ObservationSpec,
    ParameterBridge,
    ParameterMap,
    Scope,
    ScopeInterface,
    Statement,
    StimulusSpec,
    StudyLock,
    compare_reports,
    dumps_report,
    dumps_study,
    save_gallery,
)
from multiecho_framework.conformance import check_report  # noqa: E402
from multiecho_framework.contracts import fingerprint  # noqa: E402

PIXELS = Grouping.blocks(HEIGHT * WIDTH, 1, "pixels")
SWEEP = tuple(Intervention((k,), steps=4) for k in (3.1, 3.3, 3.7, 3.9, 4.1, 4.3, 4.7, 4.9))
KEYFRAMES = KeyframeAlignment("report", warn=0.05)


class ShortHistoryPredictor(TinyFramePredictor):
    """The same predictor, declared to read an 8-frame history."""

    def __init__(self, frames: int):
        super().__init__(HEIGHT, WIDTH)
        self.card = replace(self.card, world_id=f"tiny-frame-predictor-{frames}f",
                            revision=fingerprint({"base": self.card.revision, "history": frames}),
                            contract=replace(self.card.contract, history=frames))


def study_chain(masker, frames: int = STIMULUS.frames) -> Chain:
    scene, engine = physical_world(), AnalyticOscillatorEngine()
    model = TinyFramePredictor(HEIGHT, WIDTH) if frames == STIMULUS.frames else ShortHistoryPredictor(frames)
    stimulus = STIMULUS if frames == STIMULUS.frames else StimulusSpec(frames, STIMULUS.interval, HEIGHT, WIDTH)
    bridge = ParameterBridge(
        "physical-to-sim", "spring-scene", engine.describe().world_id,
        (ParameterMap("stiffness", "k"), ParameterMap("position", "x0"), ParameterMap("velocity", "v0"),
         ParameterMap("horizon", "steps")),
        (Statement("damping", 0.3, "1/s"), Statement("dt", 0.25, "s")))
    window = None if frames == STIMULUS.frames else (STIMULUS.frames - frames, STIMULUS.frames)
    to_model = ScopeInterface("sim-to-model", engine.describe().world_id, model.describe().world_id,
                              Scope("model-input", ("video",), window=window))
    card = masker.describe()
    scope = (Scope("frame", ("frame",)) if card.kind == "identity"
             else MaskedScope(f"frame:{card.masker_id}", ("frame",), masker=masker, observed="frame"))
    analyst = ScopeInterface("analyst", model.describe().world_id, "analyst", scope)
    return Chain(f"spring-{frames}f-{card.masker_id}", (scene, engine, model), (bridge, to_model), analyst,
                 ObservationSpec("frame", (PIXELS.dims,), "Predicted next frame"), stimulus=stimulus)


def declaration(chain: Chain, experiment_id: str, study: StudyLock | None = None) -> Declaration:
    needs_sweep = chain.record()["observer"]["scope"].get("masker", {}).get("needs") == "sweep"
    return Declaration(
        experiment_id, "Where in a predicted frame does the response to stiffness live?", chain.describe(),
        chain.record(), "stiffness set in the physical scene", "the frame after the keyframe", REFERENCE, QUERIES,
        contexts(), "one independently drawn initial condition", calibration=CALIBRATION,
        tasks=("forward", "backward"), stimulus=chain.stimulus, sweep=SWEEP if needs_sweep else (),
        keyframes=KEYFRAMES, study=study)


def run(chain: Chain, declared: Declaration) -> tuple[dict, Experiment]:
    frames = chain.stimulus.frames
    experiment = Experiment(declared, chain, keep_frames=(0, frames // 2, frames - 2, frames - 1))
    experiment.measure(DataRole.DISCOVERY)
    responses = experiment.responses()
    experiment.freeze(InterpolationForward().fit(responses), "forward")
    experiment.freeze(ProjectionBackward().fit(responses), "backward")
    experiment.test()
    return experiment.report(), experiment


def summary(name: str, report: dict) -> str:
    checks = check_report(report)
    keyframes = report["keyframes"]
    outcome = ", ".join(f"{task} {a['outcome']['value']}" for task, a in report["analyses"].items())
    return (f"{name:<22} {outcome}; keyframes {keyframes['verdict']} (max rms "
            f"{max(s['max_rms'] for s in keyframes['summary'].values()):.3f}, {len(keyframes['warnings'])} flagged) "
            f"[{sum(r.passed for r in checks)}/{len(checks)} report checks]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, help="write reports, the lock and galleries for multiecho-ui")
    arguments = parser.parse_args()

    pilot_chain = study_chain(NoMask())
    pilot = declaration(pilot_chain, "spring-pilot")
    lock = StudyLock.from_declaration("spring-stiffness", pilot, description="16-frame spring stimulus, v2 contract")
    pilot = replace(pilot, study=lock)
    runs = {"pilot (none)": run(pilot_chain, pilot)}

    masked_chain = study_chain(MonotonicSNRMask(PIXELS))
    runs["masked (monotonic)"] = run(masked_chain, declaration(masked_chain, "spring-masked", lock))

    short_chain = study_chain(NoMask(), frames=8)
    try:
        declaration(short_chain, "spring-8-frames", lock)
        refused = "admitted (unexpected)"
    except ValueError as error:
        refused = f"refused: {str(error)[:110]}..."
    print(f"8-frame variant in study {lock.study_id!r}: {refused}")
    runs["8 frames (no study)"] = run(short_chain, declaration(short_chain, "spring-8-frames"))

    passed = True
    for name, (report, _) in runs.items():
        print(summary(name, report))
        passed &= all(result.passed for result in check_report(report))
    reports = {name: report for name, (report, _) in runs.items()}
    for other in ("masked (monotonic)", "8 frames (no study)"):
        result = compare_reports(reports["pilot (none)"], reports[other])
        verdict = "comparable" if result["comparable"] else f"NOT comparable (differs in {', '.join(result['different'])})"
        print(f"pilot vs {other}: {verdict}")
    passed &= compare_reports(reports["pilot (none)"], reports["masked (monotonic)"])["comparable"]
    passed &= not compare_reports(reports["pilot (none)"], reports["8 frames (no study)"])["comparable"]

    if arguments.workspace:
        root = arguments.workspace
        for folder in ("reports", "studies", "keyframes"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "studies" / f"{lock.study_id}.lock.json").write_text(dumps_study(lock), encoding="utf-8")
        for report, experiment in runs.values():
            stem = report["declaration"]["experiment_id"]
            (root / "reports" / f"{stem}.report.json").write_text(dumps_report(report), encoding="utf-8")
            save_gallery(root / "keyframes" / f"{stem}.npz", experiment.keyframe_gallery())
        (root / "workspace.json").write_text(json.dumps({"title": "multiecho examples: spring study"}, indent=2),
                                             encoding="utf-8")
        print(f"workspace written to {root}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
