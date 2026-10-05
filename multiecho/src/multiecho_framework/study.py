"""Study locks: what must be identical for experiments to be comparable.

One experiment seals its own declaration, so two experiments that each pass every
check may still be incomparable: an 8-frame stimulus in one and a 16-frame stimulus in
the other, different contexts, different intervention values. A study lock names the
protocol content shared by every experiment of a study and freezes its values:

    stimulus, handle, intervention space, observation time, reference, calibration and
    queried interventions, contexts, independence unit, coupling and draws, keyframe
    alignment policy                                            (DEFAULT_LOCKED)

Everything else may vary between the experiments of a study: the subject and every
other world of the chain, the interfaces, the analyst's masker (and so its sweep) and
the analysis methods. Other fields (`LOCKABLE`, e.g. the sweep or the tasks) can be
locked explicitly. A lock is created once (typically from a pilot declaration), saved as JSON,
and passed to every later declaration as `Declaration(study=lock)`. The declaration
then refuses to exist if any locked field differs, and the report carries the lock so
that check C21 verifies it from the JSON alone.

`compare_declarations` and `compare_reports` tell, for any two experiments, which
locked fields agree and what varies, with or without a lock.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .contracts import fingerprint

STUDY_SCHEMA = "multiecho-study/0.1"

#: Fields a lock may freeze, with what they are. Paths are into a declaration record.
LOCKABLE = {
    "stimulus": "the protocol stimulus: frames, interval, resolution, channel, observation offset",
    "handle": "the intervention handle (what is intervened on)",
    "intervention_space": "the intervention coordinates, units and bounds",
    "observation_time": "when the observation is taken",
    "reference": "the reference intervention",
    "calibration": "the labeled calibration interventions",
    "interventions": "the queried interventions",
    "contexts": "the declared contexts: identities, roles, payload digests, exchangeability groups",
    "independence_unit": "the declared unit of independence",
    "coupling": "the random coupling of branches",
    "draws": "the number of context draws",
    "sweep": "the masker sweep values",
    "sweep_draws": "the number of sweep draws",
    "keyframes": "the keyframe alignment policy",
    "tasks": "the analysis tasks",
    "conditions": "the declared condition dimensions",
    "question": "the research question text",
}
DEFAULT_LOCKED = ("stimulus", "handle", "intervention_space", "observation_time", "reference", "calibration",
                  "interventions", "contexts", "independence_unit", "coupling", "draws", "keyframes")
_CONTEXT_FIELDS = ("context_id", "role", "payload_digest", "exchangeability_group")


def _record(declaration) -> dict:
    """A declaration record from a `Declaration`, a declaration record, or a report."""
    if hasattr(declaration, "as_record"):
        return declaration.as_record()
    if isinstance(declaration, dict) and "declaration" in declaration and "seal" in declaration:
        return declaration["declaration"]
    if isinstance(declaration, dict):
        return declaration
    raise TypeError("Expected a Declaration, a declaration record or a report")


def locked_view(declaration, fields: tuple[str, ...] = DEFAULT_LOCKED) -> dict:
    """The values of `fields` in a declaration, in canonical plain form."""
    record, out = _record(declaration), {}
    for name in fields:
        if name not in LOCKABLE:
            raise ValueError(f"{name!r} is not a lockable field; choose from {sorted(LOCKABLE)}")
        if name == "intervention_space":
            out[name] = record["world"]["intervention_space"]
        elif name == "contexts":
            out[name] = [{key: context[key] for key in _CONTEXT_FIELDS} for context in record["contexts"]]
        else:
            out[name] = record.get(name)
    return json.loads(json.dumps(out, sort_keys=True))


@dataclass(frozen=True)
class StudyLock:
    """Frozen protocol content shared by every experiment of a study."""

    study_id: str
    locked: dict
    description: str = ""
    schema: str = STUDY_SCHEMA

    def __post_init__(self) -> None:
        if not self.study_id or not isinstance(self.locked, dict) or not self.locked:
            raise ValueError("A study lock needs an identity and at least one locked field")
        unknown = set(self.locked) - set(LOCKABLE)
        if unknown:
            raise ValueError(f"Unknown locked fields {sorted(unknown)}; choose from {sorted(LOCKABLE)}")
        if self.schema != STUDY_SCHEMA:
            raise ValueError(f"Unsupported study schema {self.schema!r}")
        object.__setattr__(self, "locked", json.loads(json.dumps(self.locked, sort_keys=True)))

    @classmethod
    def from_declaration(cls, study_id: str, declaration, fields: tuple[str, ...] = DEFAULT_LOCKED,
                         description: str = "") -> StudyLock:
        """Lock `fields` at the values of a (pilot) declaration, declaration record or report."""
        return cls(study_id, locked_view(declaration, tuple(fields)), description)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted(self.locked))

    def differences(self, declaration) -> list[dict]:
        """Locked fields whose values differ in `declaration` (empty if it belongs to the study)."""
        found = locked_view(declaration, self.fields)
        return [{"field": name, "locked": self.locked[name], "found": found[name]}
                for name in self.fields if found[name] != self.locked[name]]

    def require(self, declaration) -> None:
        differences = self.differences(declaration)
        if differences:
            raise ValueError(f"Declaration differs from study {self.study_id!r} in locked fields: "
                             + "; ".join(_describe(row) for row in differences))

    def record(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return fingerprint(self.record())


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 120 else text[:117] + "..."


def _describe(row: dict) -> str:
    return f"{row['field']}: locked {_short(row['locked'])}, found {_short(row['found'])}"


def dumps_study(lock: StudyLock) -> str:
    return json.dumps({**lock.record(), "digest": lock.digest}, sort_keys=True, indent=2) + "\n"


def loads_study(text: str) -> StudyLock:
    data = json.loads(text)
    digest = data.pop("digest", None)
    lock = StudyLock(**data)
    if digest is not None and digest != lock.digest:
        raise ValueError("Study lock digest does not match its content")
    return lock


def _varying(record: dict) -> dict:
    chain = record.get("chain", {})
    worlds = chain.get("worlds", [])
    scope = chain.get("observer", {}).get("scope", {})
    return {
        "subjects": [f"{w['world_id']}@{w['revision'][:12]}" for w in worlds if w.get("kind") == "learned"],
        "worlds": [f"{w['world_id']}@{w['revision'][:12]}" for w in worlds],
        "interfaces": [i.get("interface_id") for i in chain.get("interfaces", [])],
        "masker": (scope.get("masker") or {}).get("masker_id", "none"),
        "observed_channels": scope.get("channels"),
        "observation": record.get("world", {}).get("observation"),
    }


def compare_declarations(a, b, fields: tuple[str, ...] = DEFAULT_LOCKED) -> dict:
    """Which of `fields` agree between two declarations, and what varies between them.

    `comparable` is true when every compared field agrees. Each side's study lock (if
    any) is reported; two experiments of the same lock are comparable by construction.
    """
    ra, rb = _record(a), _record(b)
    va, vb = locked_view(ra, tuple(fields)), locked_view(rb, tuple(fields))
    rows = [{"field": name, "same": va[name] == vb[name], "a": va[name], "b": vb[name]} for name in fields]
    studies = {side: (record.get("study") or {}).get("study_id") for side, record in (("a", ra), ("b", rb))}
    digests = {side: StudyLock(**record["study"]).digest if record.get("study") else None
               for side, record in (("a", ra), ("b", rb))}
    return {"comparable": all(row["same"] for row in rows), "fields": rows,
            "different": [row["field"] for row in rows if not row["same"]],
            "varying": {"a": _varying(ra), "b": _varying(rb)},
            "study": {"a": studies["a"], "b": studies["b"],
                      "same_lock": digests["a"] is not None and digests["a"] == digests["b"]}}


def compare_reports(a: dict, b: dict, fields: tuple[str, ...] = DEFAULT_LOCKED) -> dict:
    return compare_declarations(a["declaration"], b["declaration"], fields)


__all__ = ["DEFAULT_LOCKED", "LOCKABLE", "STUDY_SCHEMA", "StudyLock", "compare_declarations", "compare_reports",
           "dumps_study", "loads_study", "locked_view"]
