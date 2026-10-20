"""World revisions and their lineage (experimental).

A world version is fixed by its revision: weights, architecture, read-in, precision,
initialization, inference settings and code. Any change makes another world.
Construction (training, fine-tuning, editing, pruning) produces a new revision whose `parent`
names the revision it was built from, so a lineage can be followed back to its root.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .contracts import fingerprint
from .scene_program import ReadinSpec
from .worlds import WorldKind

_HASH = re.compile(r"[0-9a-f]{64}")
_INIT = re.compile(
    r"(pretrained|none|constructed(?::seed=\d+)?|"
    r"(?:untrained|permuted)(?:\[[\w.,:-]+\])?:seed=\d+|"
    r"selftrained:seed=\d+|trained:[\w.-]+@step=\d+)"
)
WORLD_KINDS = frozenset(kind.value for kind in WorldKind)


def require_digest(value: str, name: str, *, none: bool = False) -> None:
    if (none and value == "none") or (isinstance(value, str) and _HASH.fullmatch(value)):
        return
    raise ValueError(f"{name} must be a SHA-256 digest" + (" or none" if none else ""))


@dataclass(frozen=True)
class WorldRevision:
    world_id: str
    kind: str
    weights_sha256: str
    architecture: str
    readin: ReadinSpec | None
    precision: str
    init: str
    inference: tuple[tuple[str, str], ...]
    parent: str
    code_revision: str

    def __post_init__(self):
        if not self.world_id or self.kind not in WORLD_KINDS or not self.precision:
            raise ValueError("Declare world identity, a physical/simulation/learned kind and precision")
        weights = self.weights_sha256.removeprefix("set:")
        require_digest(weights, "weights_sha256", none=True)
        require_digest(self.code_revision, "code_revision")
        require_digest(self.parent, "parent", none=True)
        if not _INIT.fullmatch(self.init):
            raise ValueError(f"Invalid initialization: {self.init}")
        architecture = json.loads(self.architecture)
        if not isinstance(architecture, dict):
            raise ValueError("Architecture must be a JSON object")
        object.__setattr__(self, "architecture", json.dumps(
            architecture, sort_keys=True, separators=(",", ":"), allow_nan=False))
        if self.readin is not None and not isinstance(self.readin, ReadinSpec):
            raise TypeError("readin must be ReadinSpec or None")
        if self.kind == "learned" and self.readin is None:
            raise ValueError("Learned worlds require a declared read-in")
        if (not isinstance(self.inference, tuple)
                or any(not isinstance(pair, tuple) or len(pair) != 2
                       or not all(isinstance(v, str) for v in pair) or not pair[0]
                       for pair in self.inference)
                or len({p[0] for p in self.inference}) != len(self.inference)):
            raise ValueError("Inference settings are unique immutable string pairs")
        object.__setattr__(self, "inference", tuple(sorted(self.inference)))

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


def revision_lineage(head: WorldRevision, known: Iterable[WorldRevision]) -> tuple[WorldRevision, ...]:
    """Follow `parent` links from `head` to a root (parent "none"); every ancestor must be known."""
    if not isinstance(head, WorldRevision):
        raise TypeError("Lineage starts from a WorldRevision")
    by_digest = {revision.digest: revision for revision in known}
    chain, seen = [head], {head.digest}
    while chain[-1].parent != "none":
        parent = by_digest.get(chain[-1].parent)
        if parent is None:
            raise ValueError(f"Unknown parent revision {chain[-1].parent[:16]}")
        if parent.digest in seen:
            raise ValueError("Revision lineage contains a cycle")
        chain.append(parent)
        seen.add(parent.digest)
    return tuple(chain)
