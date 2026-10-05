"""Versioned data-only JSON exchange; no pickle, code loading, or implicit I/O."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .artifacts import Artifact, Channel, Modality, Region, Scope, Statement
from .contracts import (
    AccessMode,
    Axis,
    Context,
    DataRole,
    Intervention,
    ObservationSpec,
    PairRequest,
    SeedMode,
    Space,
    WorldSpec,
    fingerprint,
)
from .datasets import PairDataset
from .execution import PairResult, SequenceRequest, TraceResult

SCHEMA = "multiecho-framework-artifact/0.1"


def _world(data: dict) -> WorldSpec:
    intervention = data["intervention_space"]
    observation = data["observation"]
    return WorldSpec(
        data["world_id"], data["revision"],
        Space(intervention["space_id"], tuple(Axis(**axis) for axis in intervention["axes"]), intervention["semantics"]),
        ObservationSpec(observation["space_id"], tuple(observation["shape"]), observation["semantics"], observation["unit"]),
        AccessMode(data["access"]), data["stochastic"], data["supports_sequence"], data["supports_messages"],
    )


def _context(data: dict) -> Context:
    return Context(data["context_id"], data["payload_ref"], data["payload_digest"],
                   DataRole(data["role"]), data["exchangeability_group"])


def _intervention(data: dict) -> Intervention:
    return Intervention(tuple(data["values"]), data["steps"])


def _request(data: dict) -> PairRequest:
    return PairRequest(data["pair_id"], _context(data["context"]), _intervention(data["baseline"]),
                       _intervention(data["intervention"]), SeedMode(data["mode"]),
                       tuple(data["seeds"]), tuple(data["independent_seeds"]))


def _pack(kind: str, payload: dict) -> str:
    body = {"schema": SCHEMA, "kind": kind, "payload": payload}
    envelope = {**body, "sha256": fingerprint(body)}
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _unpack(text: str, kind: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique_keys(pairs: list[tuple[str, object]]) -> dict:
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Duplicate JSON key: {key}")
            output[key] = value
        return output

    envelope = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(envelope, dict) or set(envelope) != {"schema", "kind", "payload", "sha256"}:
        raise ValueError("Invalid framework artifact envelope")
    if envelope["schema"] != SCHEMA or envelope["kind"] != kind:
        raise ValueError("Unsupported framework schema or artifact kind")
    body = {key: envelope[key] for key in ("schema", "kind", "payload")}
    if fingerprint(body) != envelope["sha256"]:
        raise ValueError("Framework artifact digest mismatch")
    if not isinstance(envelope["payload"], dict):
        raise ValueError("Artifact payload must be an object")
    return envelope["payload"]


def dumps_dataset(dataset: PairDataset) -> str:
    return _pack("paired-dataset", {
        "world": asdict(dataset.world),
        "results": [{"request": asdict(row.request), "baseline": row.baseline.tolist(),
                     "intervention": row.intervention.tolist()} for row in dataset.results],
    })


def loads_dataset(text: str) -> PairDataset:
    payload = _unpack(text, "paired-dataset")
    world = _world(payload["world"])
    return PairDataset(tuple(PairResult(world, _request(row["request"]), np.asarray(row["baseline"]),
                                       np.asarray(row["intervention"])) for row in payload["results"]))


def dumps_trace(trace: TraceResult) -> str:
    return _pack("executed-trace", {"world": asdict(trace.world), "request": asdict(trace.request),
                                     "observations": trace.observations.tolist()})


def loads_trace(text: str) -> TraceResult:
    payload = _unpack(text, "executed-trace")
    request = payload["request"]
    return TraceResult(_world(payload["world"]), SequenceRequest(
        request["trace_id"], _context(request["context"]),
        tuple(_intervention(action) for action in request["interventions"]),
        request["context_seed"], tuple(request["step_seeds"]),
    ), np.asarray(payload["observations"]))


def _scope(scope: Scope | None) -> dict | None:
    if scope is None:
        return None
    record = scope.record()
    if scope.region is not None:
        record["region"] = {"label": scope.region.label, "mask": scope.region.mask.tolist()}
    return record


def dumps_artifact(artifact: Artifact) -> str:
    """Data-only message exchange between worlds; large media belong in an artifact store."""
    channels = [{"name": channel.name, "modality": channel.modality.value, "rate": channel.rate,
                 "unit": channel.unit, "semantics": channel.semantics,
                 "data": ([asdict(item) for item in channel.data] if channel.modality is Modality.TEXT
                          else channel.data.tolist())} for channel in artifact.channels]
    return _pack("artifact", {"source": artifact.source, "parent": artifact.parent, "via": artifact.via,
                              "scope": _scope(artifact.scope), "channels": channels, "digest": artifact.digest})


def loads_artifact(text: str) -> Artifact:
    payload = _unpack(text, "artifact")
    scope = payload["scope"]
    if scope is not None:
        region = scope["region"]
        scope = Scope(scope["scope_id"], tuple(scope["channels"]),
                      tuple(scope["window"]) if scope["window"] else None,
                      Region(region["label"], np.asarray(region["mask"], dtype=bool)) if region else None,
                      tuple(scope["coordinates"]) if scope["coordinates"] else None,
                      tuple(scope["keys"]) if scope["keys"] else None,
                      tuple(scope["clip"]) if scope["clip"] else None)
    channels = tuple(Channel(
        row["name"], Modality(row["modality"]),
        tuple(Statement(**item) for item in row["data"]) if row["modality"] == Modality.TEXT.value
        else np.asarray(row["data"], dtype=float), row["rate"], row["unit"], row["semantics"],
    ) for row in payload["channels"])
    artifact = Artifact(payload["source"], channels, payload["parent"], payload["via"], scope)
    if artifact.digest != payload["digest"]:
        raise ValueError("Artifact content does not reproduce its recorded digest")
    return artifact