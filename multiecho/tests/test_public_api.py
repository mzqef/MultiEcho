from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import multiecho_framework as framework


def test_public_symbols_are_importable_and_optional_torch_is_not_loaded():
    for name in framework.__all__:
        assert getattr(framework, name) is not None
    source = str(Path(framework.__file__).resolve().parents[1])
    subprocess.run([
        sys.executable, "-I", "-c",
        f"import sys; sys.path.insert(0, {source!r}); import multiecho_framework; "
        "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules); "
        "assert not any(name == 'multiecho' or name.startswith('multiecho.') for name in sys.modules)",
    ], check=True, capture_output=True, text=True)


def test_runtime_has_no_research_scene_or_campaign_imports():
    package = Path(framework.__file__).parent
    allowed = set(sys.stdlib_module_names) | {"numpy", "torch"}
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] in allowed, path


def test_core_ships_no_estimator_engine_or_model():
    assert not hasattr(framework, "RidgeResponseFitter")
    kinds = {kind.value for kind in framework.WorldKind}
    assert kinds == {"physical", "simulation", "learned"}
    concrete_worlds = [name for name in framework.__all__
                       if isinstance(getattr(framework, name), type) and hasattr(getattr(framework, name), "emit")
                       and not getattr(getattr(framework, name), "_is_protocol", False)]
    assert concrete_worlds == ["PhysicalScene"]
    assert framework.EvaluationTarget.MESSAGE_AGREEMENT.value == "message_agreement"