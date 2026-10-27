"""The example plugins as a workbench catalog: what `multiecho-ui` may build experiments from.

A plugin module lists its factories in `MULTIECHO_PLUGINS`, a sequence of `(kind, factory)` pairs with kind
`world`, `interface`, `masker`, `forward` or `backward`. Keyword parameters with defaults become form fields;
`source`, `target` and `interface_id` of an interface, and `grouping` of a masker, are filled by the workbench.
List the module in the workspace's `workspace.json` (`"plugins": ["examples.plugins.catalog"]`).
"""
from __future__ import annotations

from examples.plugins.learned import TinyFramePredictor, TinyVideoModel
from examples.plugins.methods import InterpolationForward, ProjectionBackward
from examples.plugins.simulation import (
    AnalyticOscillatorEngine,
    AppearanceSampler,
    EulerOscillatorEngine,
)
from examples.plugins.three_reference import ThreeReferenceBackward, ThreeReferenceForward
from examples.plugins.translators import StiffnessReadout

MULTIECHO_PLUGINS = (
    ("world", AnalyticOscillatorEngine),
    ("world", EulerOscillatorEngine),
    ("world", AppearanceSampler),
    ("world", TinyVideoModel),
    ("world", TinyFramePredictor),
    ("interface", StiffnessReadout),
    ("forward", InterpolationForward),
    ("backward", ProjectionBackward),
    ("forward", ThreeReferenceForward),
    ("backward", ThreeReferenceBackward),
)
