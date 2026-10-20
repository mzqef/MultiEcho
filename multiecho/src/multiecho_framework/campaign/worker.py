"""Generic worker loop: claim a ready job from the ledger, run it, record the outcome.

`run(job_id)` executes one job (a measurement job, a plugin reading, anything idempotent) and returns a
count of produced records or None. Exceptions of the `permanent` types fail the job at once; any other
exception is retried up to `max_attempts`. Idempotent `run` makes a killed worker harmless: its job is
redone after the lease expires.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class WorkerReport:
    worker: str
    done: list[str] = field(default_factory=list)
    retried: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    produced: int = 0


def run_worker(ledger, graph, run: Callable[[str], int | None], *, worker: str, lease_s: float, max_attempts: int,
               permanent: tuple[type[BaseException], ...] = (), max_jobs: int | None = None,
               kinds=None) -> WorkerReport:
    """Run until no ready job is claimable (or `max_jobs` jobs were attempted); `kinds` restricts the worker
    to nodes of those kinds, so several specialised workers can share one ledger."""
    report = WorkerReport(worker)
    attempted = 0
    while max_jobs is None or attempted < max_jobs:
        complete = graph.complete(ledger.done(), ledger.satisfied())
        ready = [node for node in graph.ready(complete) if not graph.nodes[node].gate
                 and (kinds is None or graph.nodes[node].kind in kinds)]
        job_id = ledger.claim(worker, ready, lease_s=lease_s)
        if job_id is None:
            break
        attempted += 1
        try:
            produced = run(job_id)
        except Exception as error:   # classified below; the ledger keeps the message
            state = ledger.fail(job_id, worker, f"{type(error).__name__}: {error}",
                                permanent=isinstance(error, permanent), max_attempts=max_attempts)
            (report.failed if state == "failed" else report.retried).append(job_id)
            continue
        ledger.complete(job_id, worker)
        report.done.append(job_id)
        report.produced += int(produced or 0)
    return report
