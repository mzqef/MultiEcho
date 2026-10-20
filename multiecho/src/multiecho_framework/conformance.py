"""Protocol conformance: static checks on sealed reports, dynamic probes on plugins.

Report checks (C01-C22) read only the JSON report, so they apply to any
implementation. Probes check measured worlds (A01-A09), chain member worlds such as
engines and models (W01-W06), interfaces such as translators (I01-I04), and the
keyframe protocol of chains with learned worlds (K01, K02, K04). Study commands check
reports against a study lock and tell whether two experiments are comparable.
Requirement identifiers in PROTOCOL.md cite these check identifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .artifacts import Artifact, Channel, Modality
from .chain import Chain, stimulus_violations
from .contracts import Context, Intervention, SeedMode, fingerprint
from .evaluation import EvidenceStatus
from .execution import WorldAdapter
from .interfaces import Interface
from .keyframes import audit as audit_keyframes
from .protocol import HELD_OUT_ROLES, PROTOCOL_VERSION, TASKS, Outcome, adjudicate, regime_verdict
from .study import DEFAULT_LOCKED, StudyLock, compare_reports, dumps_study, loads_study
from .worlds import World, WorldKind

REPORT_SCHEMA = "multiecho-report/0.3"
CLAIM_FIELDS = ("claim_id", "task", "outcome", "statement", "subject", "interface", "handle", "intervention",
                "contexts", "regime", "instrument", "validity_domain", "stages")
DECLARATION_FIELDS = ("experiment_id", "question", "world", "chain", "handle", "observation_time", "reference",
                      "interventions", "contexts", "independence_unit", "tasks", "coupling", "draws", "repeats")


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    title: str
    passed: bool
    detail: str = ""


class _Fail(Exception):
    pass


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise _Fail(detail)


def _roles(report: dict) -> dict[str, list[dict]]:
    roles: dict[str, list[dict]] = {}
    for context in report["declaration"]["contexts"]:
        roles.setdefault(context["role"], []).append(context)
    return roles


def _analyses(report: dict) -> dict[str, dict]:
    analyses = report["analyses"]
    _require(isinstance(analyses, dict) and set(analyses) == set(report["declaration"]["tasks"]),
             "analyses must cover exactly the declared tasks")
    return analyses


def _c01(report: dict) -> None:
    _require(report.get("schema") == REPORT_SCHEMA, f"schema must be {REPORT_SCHEMA}")
    _require(report.get("protocol_version") == PROTOCOL_VERSION, f"protocol_version must be {PROTOCOL_VERSION}")
    implementation = report.get("implementation", {})
    _require(bool(implementation.get("name")) and bool(implementation.get("version")), "implementation identity missing")


def _c02(report: dict) -> None:
    _require(report.get("seal") == fingerprint({k: v for k, v in report.items() if k != "seal"}),
             "seal does not match the canonical report body")


def _c03(report: dict) -> None:
    declaration = report["declaration"]
    missing = [name for name in DECLARATION_FIELDS if declaration.get(name) in (None, "", [], {})]
    _require(not missing, f"declaration fields missing: {missing}")
    _require(set(declaration["tasks"]) <= set(TASKS), "unknown analysis task")
    for context in declaration["contexts"]:
        for name in ("context_id", "payload_digest", "role", "exchangeability_group"):
            _require(bool(context.get(name)), f"context.{name} missing")
    body = {key: value for key, value in declaration.items() if key != "digest"}
    _require(declaration.get("digest") == fingerprint(body), "declaration digest does not match its content")


def _c04(report: dict) -> None:
    stages = report["stages"]
    _require([event["index"] for event in stages] == list(range(len(stages))), "stage indices are not sequential")
    _require(stages and stages[0]["stage"] == "declared", "the first stage must be the declaration")
    _require(stages[0]["digest"] == report["declaration"]["digest"], "declared stage does not seal the declaration")
    kinds = [(event["stage"], event["role"], event.get("task")) for event in stages]
    _require(("measured", "discovery", None) in kinds, "discovery was never measured")
    discovery = kinds.index(("measured", "discovery", None))
    analyses = _analyses(report)
    frozen = []
    for task, analysis in analyses.items():
        positions = [i for i, kind in enumerate(kinds) if kind == ("frozen", None, task)]
        _require(len(positions) == 1, f"exactly one frozen {task} law is required")
        _require(positions[0] > discovery, f"{task} law frozen before discovery was measured")
        _require(stages[positions[0]]["digest"] == analysis["candidate"]["digest"], f"frozen {task} stage mismatch")
        frozen.append(positions[0])
    last_freeze = max(frozen)
    declared = [role for role in (role.value for role in HELD_OUT_ROLES) if role in _roles(report)]
    for role in declared:
        _require(("measured", role, None) in kinds, f"held-out stage {role} was not measured")
        for task in analyses:
            _require(("tested", role, task) in kinds, f"held-out stage {role} was not tested for {task}")
    for index, (stage, role, _) in enumerate(kinds):
        if role in declared and stage in ("measured", "tested", "swept", "masked"):
            _require(index > last_freeze, f"{role} {stage} before every declared law was frozen")
    checked = ("checked", "discovery", None)
    _require(checked in kinds and discovery < kinds.index(checked) < min(frozen),
             "determinism must be checked after discovery is measured and before any law is frozen")


def _c05(report: dict) -> None:
    world = fingerprint(report["declaration"]["world"])
    _require(all(event["world"] == world for event in report["stages"]), "world identity drifted between stages")


def _c06(report: dict) -> None:
    discovery = {context["context_id"] for context in _roles(report).get("discovery", [])}
    for task, analysis in _analyses(report).items():
        fitted = set(analysis["candidate"]["fitted_on"])
        _require(bool(fitted), f"{task} law does not record its fitting contexts")
        _require(fitted <= discovery, f"{task} law fitted on non-discovery contexts: {sorted(fitted - discovery)}")


def _c07(report: dict) -> None:
    roles = _roles(report)
    for name in ("context_id", "payload_digest", "exchangeability_group"):
        seen = {context[name] for context in roles.get("discovery", [])}
        for role in (role.value for role in HELD_OUT_ROLES):
            overlap = seen & {context[name] for context in roles.get(role, [])}
            _require(not overlap, f"discovery/{role} leakage in {name}: {sorted(overlap)[:3]}")


def _c08(report: dict) -> None:
    for task, analysis in _analyses(report).items():
        documented = analysis.get("controls", {})
        _require(bool(documented) and all(documented.values()), f"{task} controls must be named and described")
        for role, evaluation in analysis["evaluations"].items():
            _require(bool(evaluation["controls"]), f"{task}/{role} has no control comparison")
            _require(set(evaluation["controls"]) <= set(documented), f"{task}/{role} uses undocumented controls")


GATE_FIELDS = ("resolved", "responsive", "repeat_floor_rms", "factor", "zero_tolerance")


def _c09(report: dict) -> None:
    response = report["response"]
    _require(bool(response.get("method")), "response method missing")
    _require("resolution" not in report, "a resolution section implies a gate; the protocol makes no triviality judgment")
    gates = [name for name in GATE_FIELDS if name in response]
    _require(not gates, f"gate fields in the response statistics: {gates}")
    tested = {role for analysis in _analyses(report).values() for role in analysis["evaluations"]}
    _require(set(response["response_rms"]) == tested, "response statistics not reported for every tested stage")
    for role, value in response["response_rms"].items():
        _require(isinstance(value, (int, float)) and value >= 0, f"{role} response RMS must be a nonnegative number")


def _nulls(value, path: str) -> list[str]:
    if value is None:
        return [path]
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _nulls(child, f"{path}.{key}")]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in _nulls(child, f"{path}[{index}]")]
    return []


def _c10(report: dict) -> None:
    accounted = set(report.get("undefined", {}))
    evidence = {"response": {k: v for k, v in report["response"].items() if k != "determinism"},
                "analyses": {task: {"evaluations": analysis["evaluations"], "regimes": analysis["regimes"]}
                             for task, analysis in report["analyses"].items()}}
    if report.get("masking") is not None:
        evidence["masking"] = {"coverage": report["masking"]["coverage"]}
    if report.get("reference") is not None:
        evidence["reference"] = report["reference"]
    missing = [path for section, value in evidence.items() for path in _nulls(value, section) if path not in accounted]
    _require(not missing, f"undefined values not accounted for: {missing[:5]}")
    _require(all(report["undefined"].values()), "every undefined value needs a stated reason")


def _c11(report: dict) -> None:
    declaration = report["declaration"]
    margin, improvement = declaration["equivalence_margin"], declaration["minimum_improvement"]
    for task, analysis in _analyses(report).items():
        for role, evaluation in analysis["evaluations"].items():
            for name, row in evaluation["controls"].items():
                interval, expected = row["interval"], EvidenceStatus.UNRESOLVED.value
                if interval is not None:
                    lower, upper = interval
                    if margin is not None and -margin < lower <= upper < margin:
                        expected = EvidenceStatus.EQUIVALENT.value
                    elif lower > improvement:
                        expected = EvidenceStatus.SUPPORTED.value
                    elif upper < -improvement:
                        expected = EvidenceStatus.CONTRADICTED.value
                _require(row["status"] == expected, f"{task}/{role}/{name} status {row['status']} should be {expected}")


def _c12(report: dict) -> None:
    for task, analysis in _analyses(report).items():
        outcome = analysis["outcome"]["value"]
        _require(outcome in {item.value for item in Outcome}, f"unknown {task} outcome {outcome!r} (\"no law\" is not an outcome)")
        stage_status = {role: {name: row["status"] for name, row in evaluation["controls"].items()}
                        for role, evaluation in analysis["evaluations"].items()}
        regime_status = {}
        for name, values in analysis["regimes"].items():
            regime_status[name] = {}
            for value, row in values.items():
                combined = regime_verdict([status for controls in row["stages"].values() for status in controls.values()])
                _require(row["status"] == combined, f"{task} regime {name}={value} status inconsistent")
                regime_status[name][value] = combined
        expected, holds = adjudicate(stage_status, regime_status)
        _require(outcome == expected.value, f"{task} outcome {outcome} does not follow from the evidence ({expected.value})")
        _require(analysis["outcome"]["holds_in"] == {k: list(v) for k, v in holds.items()}, f"{task} holds_in inconsistent")


def _c13(report: dict) -> None:
    for task, analysis in _analyses(report).items():
        claims, holds = analysis.get("claims", []), analysis["outcome"]["holds_in"]
        _require(bool(claims), f"{task} needs at least one scoped claim")
        for claim in claims:
            missing = [name for name in CLAIM_FIELDS if claim.get(name) in (None, "", [])]
            _require(not missing, f"claim {claim.get('claim_id')} missing scope fields {missing}")
            _require(claim["task"] == task and claim["outcome"] == analysis["outcome"]["value"],
                     "claim task or outcome differs from its analysis")
            if claim["outcome"] in (Outcome.STABLE_LAW.value, Outcome.REGIME_SPECIFIC.value):
                for name, values in claim["regime"].items():
                    _require(set(values) <= set(holds.get(name, [])), f"claim extends beyond supported regime {name}")


def _c14(report: dict) -> None:
    reference = report.get("reference")
    if reference is None:
        return
    _require(reference.get("status") in {"agrees", "deviates", "not_compared"}, "unknown reference status")
    _require(bool(report["declaration"].get("reference_world")), "reference compared without a declared reference world")
    stages = [event["stage"] for event in report["stages"]]
    _require("referenced" in stages and stages.index("referenced") > max(
        index for index, stage in enumerate(stages) if stage == "frozen"), "reference comparison preceded freezing")


def _c15(report: dict) -> None:
    observer = report["declaration"]["chain"].get("observer", {})
    scope = observer.get("scope", {})
    _require(bool(scope.get("scope_id")) and bool(scope.get("channels")), "the analyst scope must name its channels")
    _require(observer.get("revision") == fingerprint(scope), "the analyst interface revision must seal its scope")
    for task, analysis in _analyses(report).items():
        for claim in analysis["claims"]:
            _require(claim["interface"].get("scope") == scope, f"{task} claim does not carry the analyst scope")


def _c16(report: dict) -> None:
    declaration = report["declaration"]
    chain, world = declaration["chain"], declaration["world"]
    _require(chain.get("chain_id") == world["world_id"], "chain identity differs from the measured world")
    _require(fingerprint(chain) == world["revision"], "measured world revision does not seal the chain")
    worlds, interfaces = chain["worlds"], chain["interfaces"]
    _require(len(worlds) >= 1 and len(interfaces) == len(worlds) - 1, "a chain joins consecutive worlds by interfaces")
    for index, spec in enumerate(interfaces):
        _require((spec["source"], spec["target"]) == (worlds[index]["world_id"], worlds[index + 1]["world_id"]),
                 f"interface {spec['interface_id']} does not join consecutive worlds")
    for member in (*worlds, *interfaces, chain["observer"]):
        body = {key: value for key, value in member.items() if key not in ("digest", "scope")}
        _require(member.get("digest") == fingerprint(body), "a chain member digest does not match its record")


def _declared_masker(report: dict) -> dict | None:
    return report["declaration"]["chain"].get("observer", {}).get("scope", {}).get("masker")


def _c17(report: dict) -> None:
    masker, masking = _declared_masker(report), report.get("masking")
    kinds = [(event["stage"], event["role"]) for event in report["stages"]]
    if masker is None:
        _require(masking is None, "a masking section without a masked analyst scope")
        _require(not any(stage in ("swept", "masker-frozen", "masked") for stage, _ in kinds),
                 "masking stages without a masked analyst scope")
        return
    _require(masking is not None and masking["rule"] == masker, "the masking section does not seal the declared rule")
    _require(masking["frozen"] is not None and masking["frozen"]["masker_id"] == masker["masker_id"],
             "the frozen masker differs from the declared rule")
    frozen = [i for i, (stage, _) in enumerate(kinds) if stage == "masker-frozen"]
    _require(len(frozen) == 1, "the masker must be frozen exactly once")
    _require(report["stages"][frozen[0]]["digest"] == fingerprint(masking["frozen"]), "frozen masker stage mismatch")
    swept = bool(report["declaration"]["sweep"])
    for role in _roles(report):
        _require(("masked", role) in kinds and ("measured", role) in kinds, f"{role} was never masked and measured")
        _require(kinds.index(("masked", role)) < kinds.index(("measured", role)),
                 f"{role} measured before its mask instances were assigned")
        if swept:
            _require(("swept", role) in kinds and kinds.index(("swept", role)) < kinds.index(("masked", role)),
                     f"{role} instances assigned without its sweep")
    _require(frozen[0] < kinds.index(("masked", "discovery")), "discovery instances assigned before the masker froze")
    if swept:
        _require(kinds.index(("swept", "discovery")) < frozen[0], "the masker froze before the discovery sweep")
    contexts = {context["context_id"] for context in report["declaration"]["contexts"]}
    _require(set(masking["instances"]) == contexts, "every declared context needs exactly one mask instance")


def _seed(digest: str, *parts: object) -> int:
    key = json.dumps([digest, *parts], sort_keys=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def _c18(report: dict) -> None:
    declaration = report["declaration"]
    sweep = [(tuple(action["values"]), action["steps"]) for action in declaration["sweep"]]
    if not sweep:
        return
    evaluation = [(tuple(action["values"]), action["steps"]) for action in
                  (declaration["reference"], *declaration["calibration"], *declaration["interventions"])]
    _require(not set(sweep) & set(evaluation), "sweep values overlap evaluation values")
    digest, ids = declaration["digest"], [context["context_id"] for context in declaration["contexts"]]
    draws = {_seed(digest, c, "draw", d) for c in ids for d in range(declaration["draws"])}
    swept = {_seed(digest, c, "sweep", d) for c in ids for d in range(declaration["sweep_draws"])}
    _require(not draws & swept, "sweep seeds overlap evaluation seeds")


def _c19(report: dict) -> None:
    declaration = report["declaration"]
    chain, stimulus = declaration["chain"], declaration["stimulus"]
    learned = [world for world in chain["worlds"] if world["kind"] == WorldKind.LEARNED.value]
    _require(chain.get("stimulus") == stimulus, "the declared stimulus differs from the chain's")
    if not learned:
        return
    _require(stimulus is not None, "a chain with learned worlds must declare the protocol stimulus")
    _require(stimulus["observe_after"] >= 1, "the observation must lie after the keyframe")
    references = {name for route in chain.get("references", []) for name in route["scope"]["channels"]}
    for world in learned:
        contract = world.get("contract") or {}
        _require(contract.get("history") == stimulus["frames"], f"{world['world_id']} does not read the whole stimulus")
        _require(contract.get("horizon", 0) <= 1 or contract.get("autoregressive") is True,
                 f"{world['world_id']} rolls out beyond one step without declaring autoregression")
        _require(bool(contract.get("adaptation")) and bool(contract.get("precision")),
                 f"{world['world_id']} does not declare its stimulus adaptation and precision")
        _require(stimulus["channel"] in world["accepts"], f"{world['world_id']} does not accept the stimulus channel")
        _require(not references & set(world["accepts"]), f"{world['world_id']} accepts reference channels")


def _c20(report: dict) -> None:
    coupling = report["declaration"]["coupling"]
    _require(coupling in (SeedMode.FIXED.value, SeedMode.CONTEXT.value),
             f"coupling {coupling!r} lets branches of one context differ beyond the intervention")
    determinism = report["response"].get("determinism") or {}
    _require(determinism.get("identical") is True, "repeating a reference branch with its seed did not reproduce it")


def _c21(report: dict) -> None:
    declaration = report["declaration"]
    study = declaration.get("study")
    if study is None:
        return
    lock = StudyLock(**study)
    differences = lock.differences(declaration)
    _require(not differences, "declaration differs from its study lock in " + ", ".join(d["field"] for d in differences))


def _c22(report: dict) -> None:
    declaration, section = report["declaration"], report.get("keyframes")
    if declaration.get("stimulus") is None:
        _require(section is None, "a keyframe section without a declared stimulus")
        return
    contexts = [context["context_id"] for context in declaration["contexts"]]
    branches = len(declaration["calibration"]) + len(declaration["interventions"])
    problems = audit_keyframes(declaration["keyframes"], section, contexts, branches)
    _require(not problems, "; ".join(problems))


REPORT_CHECKS: tuple[tuple[str, str, Callable[[dict], None]], ...] = (
    ("C01", "Schema and protocol version are declared", _c01),
    ("C02", "Report seal matches canonical content", _c02),
    ("C03", "Declaration is complete and sealed", _c03),
    ("C04", "Stage order: declare, discovery, freeze every task, then held-out", _c04),
    ("C05", "World identity is stable across all stages", _c05),
    ("C06", "Every law is fitted on discovery contexts only", _c06),
    ("C07", "Held-out contexts do not leak discovery identities", _c07),
    ("C08", "Every evaluation names equal-information controls", _c08),
    ("C09", "Response statistics are reported for every stage; no triviality gate", _c09),
    ("C10", "Undefined values are counted with reasons, never dropped", _c10),
    ("C11", "Evidence statuses follow the declared decision margins", _c11),
    ("C12", "Each outcome follows from the evidence by the adjudication rule", _c12),
    ("C13", "Claims are scoped and do not exceed supported regimes", _c13),
    ("C14", "Reference comparison is a second layer after freezing", _c14),
    ("C15", "The analyst observation scope is explicit and carried by claims", _c15),
    ("C16", "The measured world seals its chain of worlds and interfaces", _c16),
    ("C17", "Masking: rule declared, frozen on discovery sweeps, one instance per context before measuring", _c17),
    ("C18", "Sweep values and seeds are disjoint from the evaluation's", _c18),
    ("C19", "Learned worlds read the whole declared stimulus and are observed after the keyframe", _c19),
    ("C20", "Hard control: branches of a context share every draw and repeat bit-identically", _c20),
    ("C21", "A declared study lock holds: every locked field equals the lock", _c21),
    ("C22", "Keyframe alignment is measured as declared and satisfies its policy", _c22),
)


def check_report(report: dict) -> list[CheckResult]:
    results = []
    for check_id, title, check in REPORT_CHECKS:
        try:
            check(report)
            results.append(CheckResult(check_id, title, True))
        except _Fail as error:
            results.append(CheckResult(check_id, title, False, str(error)))
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration) as error:
            results.append(CheckResult(check_id, title, False, f"malformed report: {error!r}"))
    return results


def _run(checks: tuple[tuple[str, str, Callable[[], None]], ...]) -> list[CheckResult]:
    results = []
    for check_id, title, check in checks:
        try:
            check()
            results.append(CheckResult(check_id, title, True))
        except _Fail as error:
            results.append(CheckResult(check_id, title, False, str(error)))
        except Exception as error:  # noqa: BLE001 - a crashing plugin fails its probe
            results.append(CheckResult(check_id, title, False, f"plugin raised {error!r}"))
    return results


def check_adapter(adapter: WorldAdapter, context: Context, action: Intervention, seed: int = 0) -> list[CheckResult]:
    """Probe a measured world, typically a Chain (A01-A08)."""
    spec = adapter.describe()

    def branch(root: object, branch_seed: int) -> np.ndarray:
        state = adapter.fork(root)
        output = adapter.apply(state, action, branch_seed)
        try:
            return np.array(adapter.observe(output), dtype=float, copy=True)
        finally:
            if output is not state:
                adapter.close(output)
            adapter.close(state)

    def fresh(branch_seed: int = seed) -> np.ndarray:
        root = adapter.open(context, seed)
        try:
            return branch(root, branch_seed)
        finally:
            adapter.close(root)

    def a02() -> None:
        root = adapter.open(context, seed)
        try:
            first, second = branch(root, seed), branch(root, seed)
        finally:
            adapter.close(root)
        _require(np.array_equal(first, second), "applying on a fork changed the shared root state")

    def a04() -> None:
        state = np.random.get_state()
        try:
            np.random.seed(1)
            first = fresh()
            np.random.seed(2)
            second = fresh()
        finally:
            np.random.set_state(state)
        _require(np.array_equal(first, second), "observations depend on global random state")

    def a05() -> None:
        tampered = replace(context, payload_digest="0" * 64 if context.payload_digest != "0" * 64 else "1" * 64)
        try:
            handle = adapter.open(tampered, seed)
        except (ValueError, PermissionError):
            return
        adapter.close(handle)
        raise _Fail("open() accepted a context whose payload digest does not match")

    def a06() -> None:
        value = fresh()
        _require(value.shape == spec.observation.shape and np.isfinite(value).all(),
                 "observation violates the declared shape or finite-value contract")

    def a07() -> None:
        root = adapter.open(context, seed)
        adapter.close(root)
        adapter.close(root)

    def a08() -> None:
        if spec.stochastic:
            values = [fresh(seed + offset) for offset in range(1, 6)]
            _require(any(not np.array_equal(values[0], value) for value in values[1:]),
                     "a stochastic world ignored the supplied branch seed")

    def a09() -> None:
        lower = Intervention(tuple(axis.lower for axis in spec.intervention_space.axes), action.steps)
        upper = Intervention(tuple(axis.upper for axis in spec.intervention_space.axes), action.steps)
        root = adapter.open(context, seed)
        try:
            first = branch(root, seed)
            for other in (lower, upper):
                state = adapter.fork(root)
                output = adapter.apply(state, other, seed)
                adapter.observe(output)
                if output is not state:
                    adapter.close(output)
                adapter.close(state)
            again = branch(root, seed)
        finally:
            adapter.close(root)
        _require(np.array_equal(first, again) and np.array_equal(first, fresh()),
                 "a branch depends on which other branches of its context ran before it")

    return _run((
        ("A01", "Declared world identity is stable",
         lambda: _require(adapter.describe().digest == spec.digest, "describe() is not stable")),

        ("A02", "Forks isolate branches from the shared root", a02),
        ("A03", "Seeded execution is reproducible",
         lambda: _require(np.array_equal(fresh(), fresh()), "the same context, action and seed differed")),
        ("A04", "Execution is independent of global random state", a04),
        ("A05", "Context payloads are verified against their digest", a05),
        ("A06", "Observations honor the declared contract", a06),
        ("A07", "Handles close idempotently", a07),
        ("A08", "Stochastic worlds use the supplied seed", a08),
        ("A09", "Hard control: a branch does not depend on other branches of its context", a09),
    ))


def check_world(world: World, context: Context, message: Artifact | None, seed: int = 0) -> list[CheckResult]:
    """Probe a chain member such as an engine, sampler or model plugin (W01-W06)."""
    card = world.describe()

    def run(root: object, run_seed: int) -> Artifact:
        state = world.fork(root)
        handles = [state]
        try:
            state = world.receive(state, message, run_seed)
            handles.append(state)
            _require(state is not root, "receive() returned the shared root")
            return world.emit(state)
        finally:
            for handle in {id(h): h for h in handles}.values():
                world.close(handle)

    def fresh(run_seed: int = seed) -> Artifact:
        root = world.open(context, seed)
        try:
            return run(root, run_seed)
        finally:
            world.close(root)

    def w02() -> None:
        root = world.open(context, seed)
        try:
            _require(run(root, seed).digest == run(root, seed).digest, "a branch changed the shared root")
        finally:
            world.close(root)

    def w04() -> None:
        state = np.random.get_state()
        try:
            np.random.seed(1)
            first = fresh()
            np.random.seed(2)
            second = fresh()
        finally:
            np.random.set_state(state)
        _require(first.digest == second.digest, "emitted artifacts depend on global random state")

    def w05() -> None:
        artifact = fresh()
        _require(tuple(channel.name for channel in artifact.channels) == card.emits,
                 f"emitted channels differ from the declared {card.emits}")
        _require(artifact.source == card.source, "artifact source does not name this world revision")

    def w06() -> None:
        root = world.open(context, seed)
        world.close(root)
        world.close(root)

    return _run((
        ("W01", "Declared world identity is stable",
         lambda: _require(world.describe().digest == card.digest, "describe() is not stable")),
        ("W02", "Forks isolate branches from the shared root", w02),
        ("W03", "Seeded execution is reproducible",
         lambda: _require(fresh().digest == fresh().digest, "the same input and seed gave different artifacts")),
        ("W04", "Execution is independent of global random state", w04),
        ("W05", "Emitted channels and source match the declaration", w05),
        ("W06", "Handles close idempotently", w06),
    ))


def check_interface(interface: Interface, artifact: Artifact) -> list[CheckResult]:
    """Probe an interface such as a translator or bridge (I01-I04)."""
    spec = interface.describe()
    original = artifact.digest

    def i03() -> None:
        message = interface.transmit(artifact)
        _require(message.parent == artifact.digest, "message does not record its parent artifact")
        _require(bool(message.via), "message does not record the interface that produced it")

    def i04() -> None:
        interface.transmit(artifact)
        _require(artifact.digest == original, "transmission modified its input artifact")

    return _run((
        ("I01", "Declared interface identity is stable",
         lambda: _require(interface.describe().digest == spec.digest, "describe() is not stable")),
        ("I02", "Transmission is deterministic",
         lambda: _require(interface.transmit(artifact).digest == interface.transmit(artifact).digest,
                          "the same artifact produced different messages")),
        ("I03", "Messages record provenance: parent artifact and interface", i03),
        ("I04", "Transmission leaves its input unchanged", i04),
    ))


class _ScrambledSource:
    """A chain member whose emitted reference channels are replaced by fixed noise (probe K02 only)."""

    def __init__(self, inner: World, channels: frozenset[str]):
        self.inner, self.channels = inner, channels

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def emit(self, state: object) -> Artifact:
        artifact = self.inner.emit(state)
        rng = np.random.default_rng(12345)
        channels = []
        for channel in artifact.channels:
            if channel.name in self.channels and channel.modality is not Modality.TEXT:
                noise = rng.uniform(0, 1, np.shape(channel.data))
                channel = Channel(channel.name, channel.modality, noise, channel.rate, channel.unit, channel.semantics)
            channels.append(channel)
        return Artifact(artifact.source, tuple(channels), artifact.parent, artifact.via, artifact.scope)


def _rebuilt(chain: Chain, worlds: tuple) -> Chain:
    return Chain(chain.chain_id, worlds, chain.interfaces, chain.observer, chain.observation, handle=chain.handle,
                 access=chain.access, stimulus=chain.stimulus, references=chain.references)


def _learned(chain: Chain) -> list[str]:
    return [world.describe().world_id for world in chain.worlds if world.describe().kind is WorldKind.LEARNED]


def check_keyframe(chain: Chain, context: Context, action: Intervention, seed: int = 0) -> list[CheckResult]:
    """Probe the keyframe protocol of a chain with learned worlds (K01, K02)."""
    learned = _learned(chain)

    def run(subject: Chain):
        root = subject.open(context, seed)
        state = subject.apply(subject.fork(root), action, seed)
        try:
            return ({w: subject.delivered(state, w) for w in learned},
                    {w: subject.emitted(state, w).digest for w in learned})
        finally:
            subject.close(state)
            subject.close(root)

    def k01() -> None:
        _require(bool(learned), "the chain has no learned world")
        delivered, _ = run(chain)
        for world_id, message in delivered.items():
            problems = stimulus_violations(message, chain.stimulus, chain.reference_channels)
            _require(not problems, f"{world_id}: {'; '.join(problems)}")

    def k02() -> None:
        _require(bool(learned), "the chain has no learned world")
        _, emitted = run(chain)
        for route in chain.references:
            worlds = tuple(_ScrambledSource(world, frozenset(route.scope.channels))
                           if world.describe().world_id == route.source else world for world in chain.worlds)
            _, scrambled = run(_rebuilt(chain, worlds))
            _require(scrambled == emitted, f"learned outputs changed when the reference data of {route.source} changed")

    return _run((
        ("K01", "Learned worlds receive exactly the stimulus ending at the keyframe, and no reference channel", k01),
        ("K02", "Learned outputs are invariant to the reference data (true future), black box", k02),
    ))


def check_stimulus_identity(chains: tuple[Chain, ...], context: Context, action: Intervention,
                            seed: int = 0) -> list[CheckResult]:
    """K04: every subject of a comparison receives the bit-identical protocol stimulus."""

    def k04() -> None:
        stimuli = []
        for chain in chains:
            learned = _learned(chain)
            _require(bool(learned), f"chain {chain.chain_id} has no learned world")
            root = chain.open(context, seed)
            state = chain.apply(chain.fork(root), action, seed)
            try:
                stimuli += [chain.delivered(state, w).channel(chain.stimulus.channel).data for w in learned]
            finally:
                chain.close(state)
                chain.close(root)
        _require(all(np.array_equal(stimuli[0], other) for other in stimuli[1:]),
                 "subjects received different stimuli; the stimulus belongs to the protocol, not the subject")

    return _run((("K04", "Every subject receives the bit-identical protocol stimulus", k04),))


def _print(
results: list[CheckResult]) -> bool:
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.check_id} {result.title}"
              + (f": {result.detail}" if result.detail else ""))
    print(f"{sum(r.passed for r in results)}/{len(results)} checks passed")
    return all(result.passed for result in results)


def _load_report(path: Path) -> dict:
    from .experiment import loads_report

    return loads_report(Path(path).read_text(encoding="utf-8"))


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 70 else text[:67] + "..."


def _print_comparison(result: dict, a: str, b: str) -> bool:
    print(f"A = {a}\nB = {b}")
    for row in result["fields"]:
        detail = "" if row["same"] else f"   A {_short(row['a'])}  |  B {_short(row['b'])}"
        print(f"{'same' if row['same'] else 'DIFF'} {row['field']}{detail}")
    for key in ("subjects", "masker", "observed_channels"):
        va, vb = result["varying"]["a"][key], result["varying"]["b"][key]
        print(f"vary {key}: A {_short(va)}  |  B {_short(vb)}")
    study = result["study"]
    if study["a"] or study["b"]:
        print(f"study: A {study['a']}  |  B {study['b']}  (same lock: {study['same_lock']})")
    print("COMPARABLE" if result["comparable"] else f"NOT COMPARABLE: differs in {', '.join(result['different'])}")
    return result["comparable"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multiecho-conformance", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("report", help="check a sealed report JSON file from any implementation").add_argument("path", type=Path)
    for name, returns in (("adapter", "(chain, context, intervention)"), ("world", "(world, context, message)"),
                          ("interface", "(interface, artifact)"), ("keyframe", "(chain, context, intervention)"),
                          ("stimulus", "(chains, context, intervention); probe K04 across subjects")):
        commands.add_parser(name, help=f"probe a factory 'module:function' returning {returns}").add_argument("factory")
    lock = commands.add_parser("lock", help="create a study lock from a (pilot) report's declaration")
    lock.add_argument("report", type=Path)
    lock.add_argument("--study-id", required=True)
    lock.add_argument("--fields", default=",".join(DEFAULT_LOCKED), help="comma-separated lockable fields")
    lock.add_argument("--description", default="")
    lock.add_argument("--output", "-o", type=Path, help="write the lock here (default: print)")
    study = commands.add_parser("study", help="check reports against a study lock")
    study.add_argument("lock", type=Path)
    study.add_argument("reports", type=Path, nargs="+")
    compare = commands.add_parser("compare", help="tell whether two reports are comparable")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--lock", type=Path, help="compare the fields of this lock (default: the default fields)")
    commands.add_parser("self-test", help="verify every check on known-answer fixtures")
    arguments = parser.parse_args(argv)
    if arguments.command == "report":
        return 0 if _print(check_report(json.loads(arguments.path.read_text(encoding="utf-8")))) else 1
    if arguments.command == "self-test":
        from .selftest import run_self_test

        return 0 if run_self_test() else 1
    if arguments.command == "lock":
        fields = tuple(name.strip() for name in arguments.fields.split(",") if name.strip())
        text = dumps_study(StudyLock.from_declaration(arguments.study_id, _load_report(arguments.report), fields,
                                                      arguments.description))
        if arguments.output:
            arguments.output.write_text(text, encoding="utf-8")
            print(f"wrote {arguments.output}")
        else:
            print(text, end="")
        return 0
    if arguments.command == "study":
        lock_ = loads_study(arguments.lock.read_text(encoding="utf-8"))
        passed = True
        for path in arguments.reports:
            report = _load_report(path)
            differences = lock_.differences(report)
            claimed = (report["declaration"].get("study") or {}).get("study_id")
            ok = not differences
            passed &= ok
            print(f"{'PASS' if ok else 'FAIL'} {path} (declares study {claimed!r})")
            for row in differences:
                print(f"     {row['field']}: locked {_short(row['locked'])}, found {_short(row['found'])}")
        print(f"{'all reports belong to' if passed else 'not every report belongs to'} study {lock_.study_id!r}")
        return 0 if passed else 1
    if arguments.command == "compare":
        fields = loads_study(arguments.lock.read_text(encoding="utf-8")).fields if arguments.lock else DEFAULT_LOCKED
        result = compare_reports(_load_report(arguments.a), _load_report(arguments.b), fields)
        return 0 if _print_comparison(result, str(arguments.a), str(arguments.b)) else 1
    module, _, name = arguments.factory.partition(":")
    target = getattr(importlib.import_module(module), name)()
    probe = {"adapter": check_adapter, "world": check_world, "interface": check_interface,
             "keyframe": check_keyframe, "stimulus": check_stimulus_identity}[arguments.command]
    return 0 if _print(probe(*target)) else 1


if __name__ == "__main__":
    sys.exit(main())
