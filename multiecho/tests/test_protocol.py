from __future__ import annotations

import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

import multiecho_framework as framework
from multiecho_framework import (
    DataRole,
    Experiment,
    Intervention,
    Outcome,
    adjudicate,
    dumps_report,
    loads_report,
)
from multiecho_framework.selftest import control_declaration, run_control
from multiecho_framework.testing import (
    NullDouble,
    OscillatorMethod,
    ScrambledDouble,
    control_chain,
    control_contexts,
    release_condition,
    transition,
)

ROOT = Path(__file__).resolve().parents[1]


def declaration(subject=None, **changes):
    subject = subject or control_chain()
    return replace(control_declaration(subject), resamples=199, **changes)


def test_declaration_requires_discovery_held_out_chain_and_known_tasks():
    chain = control_chain()
    with pytest.raises(ValueError, match="held-out"):
        declaration(chain, contexts=control_contexts({DataRole.DISCOVERY: 4}), conditions=())
    with pytest.raises(ValueError, match="chain record"):
        declaration(chain, chain={"chain_id": "other"})
    with pytest.raises(ValueError, match="seal its chain"):
        declaration(chain, chain={**chain.record(), "handle": 1})
    with pytest.raises(ValueError, match="analysis tasks"):
        declaration(chain, tasks=("forward", "sideways"))


def test_declaration_rejects_leakage_and_overlapping_branches():
    base = declaration()
    discovery, held = base.contexts_for(DataRole.DISCOVERY)[0], base.contexts_for(DataRole.VALIDATION)[0]
    leaked = replace(held, exchangeability_group=discovery.exchangeability_group)
    contexts = tuple(leaked if context is held else context for context in base.contexts)
    with pytest.raises(ValueError, match="share a exchangeability_group"):
        declaration(contexts=contexts, conditions=())
    with pytest.raises(ValueError, match="differ from the reference"):
        declaration(calibration=(base.reference,))
    with pytest.raises(ValueError, match="unique"):
        declaration(calibration=(base.interventions[0],))
    with pytest.raises(ValueError, match="reference horizon"):
        declaration(interventions=(Intervention((3.0,), 3),))


def test_session_enforces_stage_order_and_one_freeze_per_task():
    chain = control_chain()
    experiment = Experiment(declaration(chain, tasks=("forward", "backward")), chain)
    with pytest.raises(PermissionError, match="every declared task is frozen"):
        experiment.measure(DataRole.VALIDATION)
    with pytest.raises(PermissionError, match="before fitting"):
        experiment.responses()
    experiment.measure(DataRole.DISCOVERY)
    law = OscillatorMethod().fit(experiment.responses())
    experiment.freeze(law, "forward")
    with pytest.raises(PermissionError, match="already frozen"):
        experiment.freeze(law, "forward")
    with pytest.raises(PermissionError, match="every declared task is frozen"):
        experiment.measure(DataRole.VALIDATION)
    with pytest.raises(PermissionError, match="Freeze every declared task"):
        experiment.test()


def test_session_rejects_laws_fitted_outside_discovery():
    chain = control_chain()
    experiment = Experiment(declaration(chain), chain)
    experiment.measure(DataRole.DISCOVERY)
    contexts = control_contexts({DataRole.DISCOVERY: 6, DataRole.VALIDATION: 4}, seed=9)
    other = Experiment(declaration(chain, contexts=contexts, conditions=(release_condition(contexts),)), chain)
    other.measure(DataRole.DISCOVERY)
    with pytest.raises(PermissionError, match="discovery only"):
        experiment.freeze(OscillatorMethod().fit(other.responses()))


def test_methods_see_calibration_but_never_held_out_queries():
    chain = control_chain()
    experiment = Experiment(declaration(chain), chain)
    experiment.measure(DataRole.DISCOVERY)
    data = experiment.responses()
    assert all(view.context.role is DataRole.DISCOVERY for view in data.views)
    assert all(len(view.calibration) == 2 for view in data.views)
    assert all(len(rows) == 3 for rows in data.queries)


def test_known_law_is_recovered_and_reported_as_stable():
    report = run_control()
    forward = report["analyses"]["forward"]
    assert forward["outcome"]["value"] == Outcome.STABLE_LAW.value
    assert forward["outcome"]["holds_in"] == {"release": ["at-rest", "moving"]}
    damping = float(re.search(r"damping (\d+\.\d+)", forward["candidate"]["hypothesis"]).group(1))
    assert abs(damping - 0.3) < 0.005
    assert forward["claims"][0]["interface"]["scope"]["channels"] == ["state"]
    assert report["reference"]["status"] == "agrees"


def test_negative_control_is_not_described():
    report = run_control(ScrambledDouble(), reference=False)
    assert report["analyses"]["forward"]["outcome"]["value"] == Outcome.UNIDENTIFIED.value


def test_report_seal_detects_tampering():
    report = run_control(reference=False)
    assert loads_report(dumps_report(report)) == report
    with pytest.raises(ValueError, match="seal"):
        loads_report(dumps_report(report).replace('"stable_law"', '"regime_specific"', 1))


@pytest.mark.parametrize(("stages", "regimes", "expected", "holds"), [
    ({"validation": {"zero": "supported"}}, {}, Outcome.STABLE_LAW, {}),
    ({"validation": {"zero": "supported"}},
     {"view": {"a": "supported", "b": "contradicted"}}, Outcome.REGIME_SPECIFIC, {"view": ("a",)}),
    ({"validation": {"zero": "unresolved"}},
     {"view": {"a": "supported", "b": "unresolved"}}, Outcome.REGIME_SPECIFIC, {"view": ("a",)}),
    ({"validation": {"zero": "contradicted"}},
     {"view": {"a": "unresolved", "b": "contradicted"}}, Outcome.UNIDENTIFIED, {}),
    ({"validation": {"zero": "supported"}, "confirmation": {"zero": "equivalent"}}, {}, Outcome.UNIDENTIFIED, {}),
])
def test_adjudication_table(stages, regimes, expected, holds):
    outcome, where = adjudicate(stages, regimes)
    assert outcome is expected
    assert {name: values for name, values in where.items() if values} == {k: v for k, v in holds.items() if v}


def test_there_is_no_absence_outcome():
    assert "no_law" not in {outcome.value for outcome in Outcome}


def test_analytic_double_matches_numerical_integration():
    stiffness, damping, dt, state = 5.0, 0.3, 1e-4, [0.7, -0.2]
    for _ in range(int(1.0 / dt)):
        acceleration = -stiffness * state[0] - damping * state[1]
        state = [state[0] + dt * state[1], state[1] + dt * acceleration]
    exact = transition(stiffness, damping, 1.0) @ [0.7, -0.2]
    assert abs(exact[0] - state[0]) < 1e-3 and abs(exact[1] - state[1]) < 1e-3


def test_version_is_single_sourced_and_protocol_version_matches_document():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "multiecho_framework._version.__version__"
    header = re.search(r"\*\*Protocol Version (\d+\.\d+)\*\*", (ROOT / "PROTOCOL.md").read_text(encoding="utf-8"))
    assert header and header.group(1) == framework.__protocol_version__ == framework.PROTOCOL_VERSION
    assert f'version: "{framework.__version__}"' in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert set(framework.EXPERIMENTAL) <= set(framework.__all__)


def test_declaration_requires_hard_control_and_the_chain_stimulus():
    from multiecho_framework import SeedMode
    from multiecho_framework.selftest import learned_declaration
    from multiecho_framework.testing import learned_chain

    with pytest.raises(ValueError, match="Hard control"):
        declaration(coupling=SeedMode.PAIRED, draws=2)
    with pytest.raises(ValueError, match="several"):
        declaration(coupling=SeedMode.CONTEXT, draws=1)
    assert declaration(coupling=SeedMode.CONTEXT, draws=2).coupling is SeedMode.CONTEXT
    chain = learned_chain()
    base = learned_declaration(chain)
    with pytest.raises(ValueError, match="stimulus"):
        replace(base, stimulus=None)
    with pytest.raises(ValueError, match="stimulus"):
        replace(base, stimulus=replace(chain.stimulus, interval=0.5))


def test_no_triviality_judgment_is_made():
    assert {outcome.value for outcome in Outcome} == {"stable_law", "regime_specific", "unidentified_in_scope"}
    report = run_control(NullDouble(), reference=False)
    assert report["analyses"]["forward"]["outcome"]["value"] == Outcome.UNIDENTIFIED.value
    assert all(value == 0 for value in report["response"]["response_rms"].values())
    assert not {"resolved", "responsive", "zero_tolerance"} & set(report["response"]) and "resolution" not in report
