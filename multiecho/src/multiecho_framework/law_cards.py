"""Law cards: a law as a scoped reading of one world (experimental; docs/protocol-0.4-draft.md §20).

A law is stated for one world revision, scene program, input family (observation condition),
axis and readout, under a plugin set. The card records its form, parameters, the reuse
dimensions tested on held-out data, its compression gain and its evidence; cross-world reuse
cites channel-record digests. Fitting laws is plugin work; this module only fixes the record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .contracts import fingerprint

REUSE_DIMENSIONS = ("context", "time", "space", "intervention_scale", "between_laws", "composition", "cross_world")



@dataclass(frozen=True)
class LawScope:
    """What a law is about: one world revision, scene program, input family, axis and readout.

    A law is a reading of this scope under a plugin set; it never generalizes across scopes by
    itself, and readings from different observation conditions are never merged into one law.
    """

    world: str
    program: str
    observation: str
    axis: str
    readout: str
    coordinate: str
    plugin_set: str = ""
    mask_reading: str = ""
    reduce_reading: str = ""

    def __post_init__(self) -> None:
        if not all((self.world, self.program, self.observation, self.axis, self.readout)):
            raise ValueError("A law scope names world, program, observation, axis and readout")
        if self.coordinate not in {"linear", "log", "periodic", "discrete_complexity"}:
            raise ValueError("Unknown natural coordinate")


@dataclass(frozen=True)
class LawCard:
    law_id: str
    family: int
    form: str
    scope: dict
    status: str
    statement: str
    parameters: dict
    reuse: dict
    failed_regimes: tuple = ()
    delta_l: dict = field(default_factory=dict)
    coordinates: tuple[str, ...] = ()
    readout_dependence: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"found", "not_found"}:
            raise ValueError("A law card is a found or not-found reading")
        if set(self.reuse) - set(REUSE_DIMENSIONS):
            raise ValueError(f"Reuse dimensions are {REUSE_DIMENSIONS}")
        if self.reuse.get("cross_world") and any(not item for item in self.reuse["cross_world"]):
            raise ValueError("Cross-world reuse cites channel-record digests")

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

