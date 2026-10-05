"""Keyframe alignment: how close the keyframes of a context's branches must be, and what was measured.

Every branch of a context ends its stimulus at the keyframe (R-2.6). Whether the
keyframe *content* should agree across the branches depends on the question. A
keyframe-anchored design makes every branch pass through the same keyframe state, so
the branches differ only in their history; other designs let the intervention change
the keyframe as well. The declaration states a `KeyframeAlignment` policy:

    free        nothing is measured or required
    report      differences are measured and reported; `warn` flags large ones (default)
    tolerance   every branch's difference to the reference branch is at most `tolerance`
    identical   every branch's keyframe equals the reference branch's, bit for bit

The difference of a branch is the root mean square of its keyframe minus the keyframe
of its context's reference branch under the same draw seed, in stimulus units, taken as
the largest value over draws. `Experiment` measures it on the stimulus the chain
delivered to the learned world and seals the result in the report's `keyframes`
section; check C22 recomputes the verdict from the numbers.

A `KeyframeGallery` keeps the frames themselves (keyframe and optionally earlier
stimulus frames) for inspection, for example in a viewer; it is data, not evidence.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KEYFRAME_MODES = ("free", "report", "tolerance", "identical")
GALLERY_SCHEMA = "multiecho-keyframes/0.1"
METRIC = ("root mean square of the keyframe minus the context's reference-branch keyframe under the same draw "
          "seed, in stimulus units; the largest value over draws")


def _threshold(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"The keyframe {name} is a finite nonnegative number")


@dataclass(frozen=True)
class KeyframeAlignment:
    """The declared keyframe alignment policy of an experiment (see the module docstring)."""

    mode: str = "report"
    warn: float | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in KEYFRAME_MODES:
            raise ValueError(f"The keyframe alignment mode is one of {KEYFRAME_MODES}")
        _threshold(self.warn, "warning threshold")
        _threshold(self.tolerance, "tolerance")
        if (self.mode == "tolerance") != (self.tolerance is not None):
            raise ValueError("Declare a tolerance exactly when the keyframe alignment mode is 'tolerance'")
        if self.mode == "free" and self.warn is not None:
            raise ValueError("A free keyframe policy measures nothing, so it cannot warn")

    @property
    def requires_capture(self) -> bool:
        return self.mode in ("tolerance", "identical")

    def record(self) -> dict:
        return asdict(self)


def keyframe_difference(frame: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(rms, max_abs) of a keyframe against its reference keyframe."""
    a, b = np.asarray(frame, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Keyframes of one context differ in shape: {a.shape} vs {b.shape}")
    difference = a - b
    return float(np.sqrt(np.mean(difference ** 2))), float(np.max(np.abs(difference)))


def _label(context_id: str, values: list) -> str:
    return f"{context_id}@{','.join(f'{v:g}' for v in values)}"


def verdict(policy: dict, contexts: dict) -> tuple[str, list[str], list[str]]:
    """(verdict, labels above `warn`, labels above `tolerance`) recomputed from a section's numbers."""
    rows = [(cid, branch) for cid, entry in contexts.items() for branch in entry["branches"]]
    warn, tolerance = policy.get("warn"), policy.get("tolerance")
    warned = sorted(_label(cid, b["values"]) for cid, b in rows if warn is not None and b["rms"] > warn)
    exceeded = sorted(_label(cid, b["values"]) for cid, b in rows if tolerance is not None and b["rms"] > tolerance)
    mode = policy["mode"]
    if mode == "free":
        return "free", [], []
    if mode == "identical":
        return ("identical" if all(b["max_abs"] == 0 for _, b in rows) else "differs"), warned, exceeded
    if mode == "tolerance":
        return ("within_tolerance" if not exceeded else "exceeds_tolerance"), warned, exceeded
    return "reported", warned, exceeded


def summarize(policy: KeyframeAlignment, reference: tuple[float, ...], entries: list[dict],
              roles: dict[str, str]) -> dict:
    """The report section from captured keyframes.

    `entries` hold `context_id`, `values` (intervention coordinates), `seed` and `frame`
    (the keyframe) of every branch run; `roles` maps context ids to their data role.
    """
    references = {(e["context_id"], e["seed"]): e["frame"] for e in entries if tuple(e["values"]) == tuple(reference)}
    contexts: dict[str, dict] = {}
    for entry in entries:
        values = tuple(entry["values"])
        if values == tuple(reference):
            continue
        base = references.get((entry["context_id"], entry["seed"]))
        if base is None:
            raise ValueError(f"No reference keyframe for context {entry['context_id']} under seed {entry['seed']}")
        rms, largest = keyframe_difference(entry["frame"], base)
        row = contexts.setdefault(entry["context_id"], {"role": roles[entry["context_id"]], "branches": {}})
        previous = row["branches"].setdefault(values, {"values": list(values), "rms": rms, "max_abs": largest})
        previous["rms"], previous["max_abs"] = max(previous["rms"], rms), max(previous["max_abs"], largest)
    contexts = {cid: {"role": row["role"], "branches": [row["branches"][v] for v in sorted(row["branches"])]}
                for cid, row in sorted(contexts.items())}
    section = {"policy": policy.record(), "status": "captured", "metric": METRIC, "contexts": contexts}
    return _finish(section)


def not_captured(policy: KeyframeAlignment, reason: str) -> dict:
    return _finish({"policy": policy.record(), "status": "not_captured" if policy.mode != "free" else "not_requested",
                    "reason": reason, "metric": METRIC, "contexts": {}})


def _finish(section: dict) -> dict:
    result, warned, exceeded = verdict(section["policy"], section["contexts"])
    summary = {}
    for role in sorted({entry["role"] for entry in section["contexts"].values()}):
        rows = [b for entry in section["contexts"].values() if entry["role"] == role for b in entry["branches"]]
        summary[role] = {"branches": len(rows), "identical": sum(b["max_abs"] == 0 for b in rows),
                         "max_rms": max(b["rms"] for b in rows), "mean_rms": float(np.mean([b["rms"] for b in rows]))}
    if section["status"] != "captured":
        result = "free" if section["policy"]["mode"] == "free" else "not_captured"
    return {**section, "summary": summary, "warnings": warned, "exceeding_tolerance": exceeded, "verdict": result}


def audit(policy: dict, section: dict | None, contexts: list[str], branches: int) -> list[str]:
    """Problems of a report's keyframe section (empty if it is consistent and satisfies its policy)."""
    if section is None:
        return ["the keyframe section is missing"]
    problems = []
    if section.get("policy") != policy:
        problems.append("the keyframe section's policy differs from the declared policy")
    status = section.get("status")
    if policy["mode"] == "free":
        return problems + ([] if status in ("not_requested", "captured") else [f"unexpected status {status!r}"])
    if status != "captured":
        if KeyframeAlignment(**policy).requires_capture:
            problems.append(f"the policy {policy['mode']!r} needs measured keyframes, but they were {status}")
        return problems
    measured = section.get("contexts", {})
    if sorted(measured) != sorted(contexts):
        problems.append("keyframes are not reported for exactly the declared contexts")
    for cid, entry in measured.items():
        if len(entry.get("branches", [])) != branches:
            problems.append(f"context {cid} reports {len(entry.get('branches', []))} branches, declared {branches}")
    result, warned, exceeded = verdict(policy, measured)
    if (section.get("verdict"), section.get("warnings"), section.get("exceeding_tolerance")) != (result, warned, exceeded):
        problems.append("the keyframe verdict or its flagged branches do not follow from the reported differences")
    if result == "differs":
        problems.append("keyframes differ across branches, but the policy declares them identical")
    if result == "exceeds_tolerance":
        problems.append(f"{len(exceeded)} branch keyframes exceed the declared tolerance {policy['tolerance']}")
    return problems


@dataclass
class KeyframeGallery:
    """Frames of every captured branch for inspection: frames[n, f] is frame `frame_labels[f]` of branch n."""

    frames: np.ndarray
    frame_labels: tuple[str, ...]
    context_ids: tuple[str, ...]
    roles: tuple[str, ...]
    values: np.ndarray
    seeds: tuple[int, ...]
    reference: tuple[bool, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        count = len(self.context_ids)
        if frames.ndim != 5 or frames.shape[0] != count or frames.shape[1] != len(self.frame_labels):
            raise ValueError("Gallery frames are (branches, frames, height, width, colours)")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != count:
            raise ValueError("Gallery values are (branches, coordinates)")
        if not (len(self.roles) == len(self.seeds) == len(self.reference) == count):
            raise ValueError("Every gallery branch has a role, a seed and a reference flag")
        self.frames, self.values = frames, values

    def differences(self, frame: int = -1) -> np.ndarray:
        """(branches, 2): rms and max_abs of each branch's frame against its context's reference (same seed)."""
        index = {(c, s): i for i, (c, s, r) in enumerate(zip(self.context_ids, self.seeds, self.reference, strict=True))
                 if r}
        out = np.full((len(self.context_ids), 2), np.nan)
        for i, (c, s) in enumerate(zip(self.context_ids, self.seeds, strict=True)):
            j = index.get((c, s))
            if j is not None:
                out[i] = keyframe_difference(self.frames[i, frame], self.frames[j, frame])
        return out


def save_gallery(path: str | Path, gallery: KeyframeGallery) -> Path:
    """Write a gallery as one compressed `.npz` (no pickle)."""
    path = Path(path)
    meta = {"schema": GALLERY_SCHEMA, **gallery.meta}
    with open(path, "wb") as handle:
        np.savez_compressed(handle, frames=gallery.frames, frame_labels=np.array(gallery.frame_labels),
                            context_ids=np.array(gallery.context_ids), roles=np.array(gallery.roles),
                            values=gallery.values, seeds=np.array(gallery.seeds, dtype=np.int64),
                            reference=np.array(gallery.reference, dtype=bool), meta=np.array(json.dumps(meta)))
    return path


def load_gallery(path: str | Path) -> KeyframeGallery:
    with np.load(io.BytesIO(Path(path).read_bytes()), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != GALLERY_SCHEMA:
            raise ValueError(f"Not a keyframe gallery ({GALLERY_SCHEMA})")
        return KeyframeGallery(data["frames"], tuple(str(x) for x in data["frame_labels"]),
                               tuple(str(x) for x in data["context_ids"]), tuple(str(x) for x in data["roles"]),
                               data["values"], tuple(int(x) for x in data["seeds"]),
                               tuple(bool(x) for x in data["reference"]), meta)


__all__ = ["GALLERY_SCHEMA", "KEYFRAME_MODES", "KeyframeAlignment", "KeyframeGallery", "audit", "keyframe_difference",
           "load_gallery", "not_captured", "save_gallery", "summarize", "verdict"]
