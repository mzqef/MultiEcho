"""The law channel on known-answer worlds: a shared-profile law reaches a world with the same law only."""
from __future__ import annotations

import numpy as np
from test_law_core import BALANCED, world

from multiecho_framework import ChannelOutcome, ChannelTest, DataRole, EvidenceStatus, WorldKind
from multiecho_framework.law_core import SharedProfile, law_channel

LEVELS = {"minimum_improvement": 0.0, "resamples": 500, "seed": 7}


def test_shared_profile_law_transfers_to_a_world_with_the_same_law_only():
    discovery, _, noise = world(BALANCED, seed=1)
    law = SharedProfile.fit(discovery, noise, rank=2)
    _, same, _ = world(BALANCED, seed=2, dimension=7)
    record, _ = law_channel(law, "a" * 64, same, "b" * 64, source_kind=WorldKind.SIMULATION,
                            target_kind=WorldKind.LEARNED, **LEVELS)
    assert record.outcome is ChannelOutcome.PARTIAL
    assert record.statuses[ChannelTest.CONSISTENCY] is EvidenceStatus.SUPPORTED
    assert set(record.not_run) == {ChannelTest.UTILITY, ChannelTest.COMMUTATION, ChannelTest.ROUND_TRIP}
    other = (lambda u: np.exp(-((np.asarray(u) - 0.15) / 0.05) ** 2), lambda u: np.asarray(u) ** 3)
    _, different, _ = world(other, seed=3)
    record, _ = law_channel(law, "a" * 64, different, "c" * 64, source_kind=WorldKind.SIMULATION,
                            target_kind=WorldKind.SIMULATION, **LEVELS)
    assert record.statuses[ChannelTest.CONSISTENCY] is not EvidenceStatus.SUPPORTED
    assert record.outcome is ChannelOutcome.FAILED and record.fitted_on == (DataRole.DISCOVERY,)
