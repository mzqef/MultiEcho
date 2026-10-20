"""Job graph with gates.

Nodes either carry measurement jobs or are gates (a lock is sealed, a probe passed, a capability reviewed).
A gate is satisfied only by an explicit record in the ledger; a job node is complete when all its jobs are
done. Ordering rules ("every node of kind X must have a node of kind Y upstream") are declared by the
study, not built in, so the same graph serves any world, scene and study.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    requires: tuple[str, ...] = ()
    jobs: tuple[str, ...] = ()
    gate: bool = False

    def __post_init__(self):
        if not self.node_id or not self.kind:
            raise ValueError("A node has an identity and a kind")
        if self.gate and self.jobs:
            raise ValueError("A gate carries no measurement jobs")
        if len(set(self.requires)) != len(self.requires) or self.node_id in self.requires:
            raise ValueError("Requirements are unique and exclude the node itself")


@dataclass(frozen=True)
class GateRule:
    """Every node of kind `after` must have a node of kind `before` among its ancestors."""

    before: str
    after: str


class JobGraph:
    def __init__(self, nodes, rules=()):
        self.nodes = {node.node_id: node for node in nodes}
        if len(self.nodes) != len(tuple(nodes)):
            raise ValueError("Node identities are unique")
        self.rules = tuple(rules)
        owners: dict[str, str] = {}
        for node in self.nodes.values():
            for job_id in node.jobs:
                if job_id in owners:
                    raise ValueError(f"Job {job_id[:16]} belongs to {owners[job_id]} and {node.node_id}")
                owners[job_id] = node.node_id
        self.owner = owners
        self.order = self._topological()
        self.validate()

    def _topological(self) -> tuple[str, ...]:
        missing = {req for node in self.nodes.values() for req in node.requires if req not in self.nodes}
        if missing:
            raise ValueError(f"Unknown requirements: {sorted(missing)}")
        remaining = {node_id: set(node.requires) for node_id, node in self.nodes.items()}
        order = []
        while remaining:
            ready = sorted(node_id for node_id, requires in remaining.items() if not requires)
            if not ready:
                raise ValueError(f"The job graph has a cycle among {sorted(remaining)}")
            for node_id in ready:
                order.append(node_id)
                del remaining[node_id]
            for requires in remaining.values():
                requires.difference_update(ready)
        return tuple(order)

    def ancestors(self, node_id: str) -> set[str]:
        seen, stack = set(), list(self.nodes[node_id].requires)
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self.nodes[current].requires)
        return seen

    def validate(self) -> None:
        errors = []
        for rule in self.rules:
            for node in self.nodes.values():
                if node.kind == rule.after and not any(
                        self.nodes[ancestor].kind == rule.before for ancestor in self.ancestors(node.node_id)):
                    errors.append(f"{node.node_id} ({rule.after}) has no upstream {rule.before}")
        if errors:
            raise ValueError("; ".join(errors))

    def ready(self, complete: set[str]) -> tuple[str, ...]:
        """Nodes not yet complete whose requirements are all complete, in topological order."""
        return tuple(node_id for node_id in self.order
                     if node_id not in complete and set(self.nodes[node_id].requires) <= complete)

    def complete(self, done_jobs: set[str], satisfied_gates: set[str]) -> set[str]:
        result = set()
        for node_id in self.order:
            node = self.nodes[node_id]
            if set(node.requires) <= result and (
                    node_id in satisfied_gates if node.gate else set(node.jobs) <= done_jobs):
                result.add(node_id)
        return result
