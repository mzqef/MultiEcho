from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *arguments: str, cwd: Path) -> subprocess.CompletedProcess:
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    return subprocess.run([sys.executable, str(ROOT / "examples" / script), *arguments],
                          cwd=cwd, capture_output=True, text=True, env=environment, timeout=600)


def test_quickstart_runs_every_layer_and_reports_both_analyses(tmp_path):
    completed = run("quickstart.py", "--report", str(tmp_path / "report.json"), cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    for world in ("spring-scene", "analytic-oscillator-engine", "appearance-sampler", "tiny-video-model"):
        assert world in output
    assert "scope=future-latents" in output
    assert "forward  stable_law" in output and "backward stable_law" in output
    assert "two engines (1:N)" in output
    assert "22/22 report checks passed" in output
    assert "determinism identical" in output
    assert (tmp_path / "report.json").is_file()


def test_masking_example_reports_three_maskers(tmp_path):
    completed = run("masking.py", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for masker in ("none", "manual:left-half", "monotonic-snr"):
        assert masker in completed.stdout
    assert completed.stdout.count("[22/22 report checks]") == 3


def test_study_example_admits_a_masker_and_refuses_an_eight_frame_stimulus(tmp_path):
    completed = run("study.py", "--workspace", str(tmp_path / "ws"), cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "8-frame variant in study 'spring-stiffness': refused" in output
    assert "pilot vs masked (monotonic): comparable" in output
    assert "pilot vs 8 frames (no study): NOT comparable (differs in stimulus)" in output
    assert output.count("[22/22 report checks]") == 3
    assert len(list((tmp_path / "ws" / "reports").glob("*.report.json"))) == 3
    assert len(list((tmp_path / "ws" / "keyframes").glob("*.npz"))) == 3


def test_communication_example_supports_all_three_tests(tmp_path):
    completed = run("communication.py", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "4/4 interface checks passed" in completed.stdout
    assert "All three tests supported." in completed.stdout


def test_cross_world_laws_example_reads_laws_and_channels(tmp_path):
    completed = run("cross_world_laws.py", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "72 jobs done" in completed.stdout
    for world in ("declared", "rewritten", "observer"):
        assert world in completed.stdout
    assert completed.stdout.count("outcome=") == 2
