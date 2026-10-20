from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from multiecho_framework.conformance import REPORT_CHECKS, check_adapter, check_report, main
from multiecho_framework.selftest import (
    QUERIES,
    _learned_mutations,
    _mutations,
    control_declaration,
    run_control,
    run_learned_control,
    run_self_test,
)
from multiecho_framework.testing import control_chain

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compliant():
    return run_control()


@pytest.fixture(scope="module")
def learned():
    return run_learned_control()


def test_compliant_report_passes_every_check(compliant, learned):
    assert all(result.passed for result in check_report(compliant))
    assert all(result.passed for result in check_report(learned))


@pytest.mark.parametrize("check_id", [check_id for check_id, _, _ in REPORT_CHECKS])
def test_each_check_detects_its_targeted_violation(compliant, learned, check_id):
    mutations, report = (_mutations(), compliant) if check_id in _mutations() else (_learned_mutations(), learned)
    results = {result.check_id: result for result in check_report(mutations[check_id](report))}
    assert not results[check_id].passed, results[check_id]


def test_malformed_reports_fail_rather_than_crash():
    results = {result.check_id: result.passed for result in check_report({})}
    assert not all(results.values()) and not results["C01"] and not results["C12"]


def test_every_protocol_requirement_is_traced():
    protocol = (ROOT / "PROTOCOL.md").read_text(encoding="utf-8")
    body, appendix = protocol.split("## Appendix A", 1)
    requirements = set(re.findall(r"\*\*(R-\d+\.\d+)\*\*", body))
    assert requirements and requirements <= set(re.findall(r"R-\d+\.\d+", appendix)), \
        sorted(requirements - set(re.findall(r"R-\d+\.\d+", appendix)))
    checks = {check_id for check_id, _, _ in REPORT_CHECKS} | {f"A0{i}" for i in range(1, 10)} \
        | {f"W0{i}" for i in range(1, 7)} | {f"I0{i}" for i in range(1, 5)} | {"K01", "K02", "K04"}
    traced = set(re.findall(r"\b([CAWIK]\d\d)\b", appendix))
    assert checks <= traced, sorted(checks - traced)


def test_adapter_probes_pass_on_a_control_chain():
    chain = control_chain()
    context = control_declaration(chain).contexts[0]
    assert all(result.passed for result in check_adapter(chain, context, QUERIES[0]))


def test_self_test_meets_every_expectation():
    assert run_self_test(verbose=False)


def test_cli_checks_report_files(tmp_path, compliant, capsys):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(compliant), encoding="utf-8")
    assert main(["report", str(path)]) == 0
    assert "22/22 checks passed" in capsys.readouterr().out
    path.write_text(json.dumps({**compliant, "protocol_version": "9.9"}), encoding="utf-8")
    assert main(["report", str(path)]) == 1


@pytest.mark.parametrize(("command", "factory", "expected"), [
    ("world", "examples.templates:probe_engine", "6/6"),
    ("world", "examples.templates:probe_model", "6/6"),
    ("interface", "examples.templates:probe_translator", "4/4"),
    ("adapter", "examples.templates:probe_chain", "9/9"),
    ("keyframe", "examples.templates:probe_chain", "2/2"),
])
def test_cli_probes_the_plugin_templates(command, factory, expected):
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    completed = subprocess.run([sys.executable, "-m", "multiecho_framework.conformance", command, factory],
                               cwd=ROOT, capture_output=True, text=True, env=environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"{expected} checks passed" in completed.stdout
