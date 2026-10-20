from __future__ import annotations

from dataclasses import replace

import pytest

from multiecho_framework import ReadinSpec, WorldRevision, revision_lineage

READIN = ReadinSpec("video", tuple(range(8, 16)), 64, 64, "bilinear", "none", "tubelet:2", "unit", (1,), "test")


def revision(**changes) -> WorldRevision:
    base = WorldRevision("toy:predictor", "learned", "a" * 64, '{"layers": 2}', READIN, "fp32", "pretrained",
                         (("steps", "1"),), "none", "b" * 64)
    return replace(base, **changes)


def test_any_world_kind_has_a_revision_and_any_change_is_another_world():
    physical = revision(world_id="real:recording", kind="physical", weights_sha256="none", readin=None,
                        init="none")
    simulated = revision(world_id="reference:pendulum", kind="simulation", weights_sha256="none", readin=None,
                         init="none")
    learned = revision()
    assert len({physical.digest, simulated.digest, learned.digest}) == 3
    assert revision(precision="bf16").digest != learned.digest
    assert revision(architecture='{ "layers" : 2 }').digest == learned.digest
    with pytest.raises(ValueError, match="kind"):
        revision(kind="dream")
    with pytest.raises(ValueError, match="read-in"):
        revision(readin=None)


def test_lineage_follows_parents_to_the_root_and_rejects_gaps():
    root = revision()
    tuned = revision(init="trained:counterfactual@step=100", parent=root.digest)
    edited = revision(init="trained:edit@step=0", parent=tuned.digest)
    assert revision_lineage(edited, (root, tuned)) == (edited, tuned, root)
    with pytest.raises(ValueError, match="Unknown parent"):
        revision_lineage(edited, (root,))
