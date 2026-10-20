"""Cross-world law discovery and law channels on analytic worlds (worked example for 0.4 experimental).

Run:  python examples/cross_world_laws.py

The same pipeline as the pilot-0 milestone (docs/pilot-0.md), on worlds small enough to run in seconds:

    gated campaign (ledger + job graph) → response blocks per world → law selection by CV-ΔL with an
    indistinguishability report → law cards on held-out contexts → law channels between worlds

Worlds, all answering "how does the response change along the damping-ratio axis ζ?":
    declared    the analytic damped oscillator; response = positions at 24 sample times
    rewritten   the same oscillator with rewritten physics (effective damping 2ζ: critical point at ζ = 0.5)
    observer    a fixed random nonlinear read-out of the declared world into 32 channels
                (stands in for a learned world: same physics, a different coordinate system)

Contexts differ in initial amplitude; two appearance replicates per context add observation noise.
Every methodological level (folds, ranks, confidence, resamples) is declared in LEVELS below.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiecho_framework import DataRole, WorldKind  # noqa: E402
from multiecho_framework.campaign import GateRule, JobGraph, Ledger, Node, run_worker  # noqa: E402
from multiecho_framework.law_core import (  # noqa: E402
    CodeSpec,
    LawScope,
    NoiseModel,
    ResponseBlock,
    Unit,
    default_candidates,
    indistinguishable,
    law_card,
    law_channel,
    select,
)

GRID = np.geomspace(0.05, 5.0, 21)
BASELINE = float(GRID[6])                       # ζ0 ≈ 0.15, a grid value
TIMES = np.linspace(0.0, 1.84, 24)
CONTEXTS = {DataRole.DISCOVERY: 16, DataRole.VALIDATION: 8}
REPLICATES, NOISE = 2, 0.01
LEVELS = {"max_rank": 3, "folds": 4, "confidence": 0.95, "resamples": 1000, "seed": 20261020}
WORLDS = {"declared": WorldKind.SIMULATION, "rewritten": WorldKind.SIMULATION, "observer": WorldKind.LEARNED}
MIXING = np.random.default_rng(3).normal(size=(24, 32)) / np.sqrt(24)


def oscillator(zeta: float, amplitude: float, omega: float = 2 * np.pi) -> np.ndarray:
    """x(t) of x'' + 2ζωx' + ω²x = 0 with x(0) = amplitude, x'(0) = 0."""
    if np.isclose(zeta, 1.0):
        return amplitude * (1 + omega * TIMES) * np.exp(-omega * TIMES)
    if zeta < 1:
        damped = omega * np.sqrt(1 - zeta ** 2)
        return amplitude * np.exp(-zeta * omega * TIMES) * (
            np.cos(damped * TIMES) + zeta * omega / damped * np.sin(damped * TIMES))
    root = omega * np.sqrt(zeta ** 2 - 1)
    fast, slow = -zeta * omega - root, -zeta * omega + root
    return amplitude * (slow * np.exp(fast * TIMES) - fast * np.exp(slow * TIMES)) / (slow - fast)


def observe(world: str, zeta: float, amplitude: float) -> np.ndarray:
    if world == "rewritten":
        return oscillator(2 * zeta, amplitude)
    state = oscillator(zeta, amplitude)
    return np.tanh(state @ MIXING) if world == "observer" else state


def measure(world: str, role: DataRole, context: int) -> list[Unit]:
    """One context job: baseline and every grid value, per replicate; effects are branch − baseline."""
    rng = np.random.default_rng([list(WORLDS).index(world), list(DataRole).index(role), context])
    amplitude = rng.uniform(0.5, 1.5)
    units = []
    for replicate in range(REPLICATES):
        noise = np.random.default_rng([list(WORLDS).index(world), list(DataRole).index(role), context, replicate])
        baseline = observe(world, BASELINE, amplitude) + noise.normal(scale=NOISE, size=observe(world, 1, 1).shape)
        values = [v for v in GRID if v != BASELINE]
        effects = [observe(world, v, amplitude) + noise.normal(scale=NOISE, size=baseline.shape) - baseline
                   for v in values]
        units.append(Unit(f"{role.value}-{context:02d}", replicate, np.log(values), np.stack(effects)))
    return units


def main() -> int:
    store: dict[str, list[Unit]] = {}
    jobs = {f"{world}:{role.value}:{c:02d}": (world, role, c)
            for world in WORLDS for role, count in CONTEXTS.items() for c in range(count)}

    class Job:
        def __init__(self, job_id):
            self.job_id, self.priority = job_id, 0

    nodes = [Node("lock", "lock", gate=True)] + [
        Node(f"measure:{world}", "measure", ("lock",), tuple(j for j, (w, _, _) in jobs.items() if w == world))
        for world in WORLDS]
    graph = JobGraph(nodes, rules=(GateRule("lock", "measure"),))
    with tempfile.TemporaryDirectory() as directory, Ledger(Path(directory) / "ledger.sqlite") as ledger:
        ledger.add(graph, [Job(job_id) for job_id in jobs])
        ledger.satisfy("lock", "grid, contexts and levels declared above before any measurement")

        def run(job_id):
            store[job_id] = measure(*jobs[job_id])
            return len(store[job_id])

        report = run_worker(ledger, graph, run, worker="example", lease_s=60, max_attempts=1)
        print(f"campaign: {len(report.done)} jobs done, {report.produced} units, ledger {ledger.counts()}")

    blocks, noises, laws = {}, {}, {}
    print(f"\n{'world':10s} {'selected law':28s} {'ΔL bits':>10s} {'held-out R²':>11s}  indistinguishable")
    for world in WORLDS:
        scope = LawScope(world, "damped-oscillator/zeta", "24-times", "zeta", "output", "log",
                         plugin_set="example:law-core")
        for role in CONTEXTS:
            units = tuple(unit for job_id, (w, r, _) in jobs.items() if w == world and r == role
                          for unit in store[job_id])
            blocks[world, role] = ResponseBlock(scope, role, float(np.log(BASELINE)),
                                                (float(np.log(GRID[0])), float(np.log(GRID[-1]))),
                                                np.log(GRID), units)
        discovery, validation = blocks[world, DataRole.DISCOVERY], blocks[world, DataRole.VALIDATION]
        noises[world] = NoiseModel.from_replicates(discovery, CodeSpec())
        ranked = select(discovery, noises[world], default_candidates(discovery, max_rank=LEVELS["max_rank"]),
                        folds=LEVELS["folds"])
        laws[world] = ranked
        card = law_card(ranked[0].law, scope, discovery, validation, noises[world], CodeSpec(),
                        confidence=LEVELS["confidence"], resamples=LEVELS["resamples"], seed=LEVELS["seed"],
                        cv_delta_l=ranked[0].cv_delta_l)
        tied = indistinguishable(ranked, confidence=LEVELS["confidence"], resamples=LEVELS["resamples"],
                                 seed=LEVELS["seed"])
        context = card.reuse["context"]
        print(f"{world:10s} {card.status + ': ' + card.law_id.split('@')[0]:28s} {context['delta_l']:10.4g} "
              f"{context['forward_r2']:11.3f}  {', '.join(tied) or '–'}")

    source = next(item.law for item in laws["declared"] if item.law is not None and item.law.family == 2)
    print(f"\nlaw channels from 'declared' (message: {source.variant}, calibrated at "
          f"{np.round(np.exp(source.calibration_values), 3).tolist()}):")
    for target in ("rewritten", "observer"):
        record, _ = law_channel(source, "declared", blocks[target, DataRole.VALIDATION], target,
                                source_kind=WORLDS["declared"], target_kind=WORLDS[target],
                                minimum_improvement=0.0, resamples=LEVELS["resamples"], seed=LEVELS["seed"])
        statuses = {test.value: status.value for test, status in record.statuses.items()}
        print(f"  declared -> {target:9s} outcome={record.outcome.value:8s} {statuses}")
    print("\nΔL is the compression gain over the zero-response code; held-out R² compares with the discovery mean")
    print("effect at each value, so the two can disagree. Readings are relative to this plugin set; 'not_found'")
    print("never means that a world has no law.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
