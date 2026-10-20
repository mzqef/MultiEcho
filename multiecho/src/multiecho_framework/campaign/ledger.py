"""Claim ledger: SQLite in WAL mode; job state only, never measurements.

A claim is an atomic conditional update inside `BEGIN IMMEDIATE`, so concurrent workers never hold the same
job. A claim carries a lease; a worker that stops heartbeating loses its jobs back to `pending` when the
lease expires. Gates are satisfied by explicit records with a reason.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

STATES = ("pending", "claimed", "done", "failed")


class Ledger:
    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.connection = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, node TEXT NOT NULL, priority INTEGER NOT NULL,
                state TEXT NOT NULL, worker TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS gates(node TEXT PRIMARY KEY, reason TEXT NOT NULL, satisfied REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events(at REAL NOT NULL, job_id TEXT, event TEXT NOT NULL, detail TEXT);
        """)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _event(self, job_id, event, detail=None):
        self.connection.execute("INSERT INTO events VALUES(?,?,?,?)",
                                (self.clock(), job_id, event, json.dumps(detail) if detail is not None else None))

    def add(self, graph, jobs) -> int:
        """Register the graph's jobs; existing rows keep their state (re-adding is idempotent)."""
        priorities = {job.job_id: job.priority for job in jobs}
        if set(priorities) != set(graph.owner):
            raise ValueError("Ledger jobs and graph jobs differ")
        now = self.clock()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            before = self.connection.total_changes
            self.connection.executemany(
                "INSERT OR IGNORE INTO jobs(job_id,node,priority,state,updated) VALUES(?,?,?,'pending',?)",
                [(job_id, graph.owner[job_id], priorities[job_id], now) for job_id in sorted(priorities)])
            added = self.connection.total_changes - before
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return added

    def satisfy(self, node: str, reason: str) -> None:
        if not reason:
            raise ValueError("A gate is satisfied with a recorded reason")
        self.connection.execute("INSERT OR IGNORE INTO gates VALUES(?,?,?)", (node, reason, self.clock()))

    def satisfied(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT node FROM gates")}

    def done(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT job_id FROM jobs WHERE state='done'")}

    def counts(self) -> dict[str, int]:
        rows = dict(self.connection.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall())
        return {state: rows.get(state, 0) for state in STATES}

    def claim(self, worker: str, nodes, *, lease_s: float) -> str | None:
        """Atomically claim the most urgent pending (or lease-expired) job of the given ready nodes."""
        nodes = tuple(nodes)
        if not nodes:
            return None
        now = self.clock()
        marks = ",".join("?" * len(nodes))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                f"SELECT job_id, state, worker FROM jobs WHERE node IN ({marks}) AND (state='pending' OR "
                f"(state='claimed' AND lease_until < ?)) ORDER BY priority, job_id LIMIT 1", (*nodes, now)).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            job_id, state, previous = row
            self.connection.execute(
                "UPDATE jobs SET state='claimed', worker=?, lease_until=?, attempts=attempts+1, updated=? "
                "WHERE job_id=?", (worker, now + lease_s, now, job_id))
            if state == "claimed":
                self._event(job_id, "reclaimed", {"from": previous, "to": worker})
            self._event(job_id, "claimed", {"worker": worker})
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return job_id

    def release(self, worker: str, reason: str) -> int:
        """Operator action for a worker known to be dead: its claims return to pending before lease expiry."""
        if not reason:
            raise ValueError("Releasing a worker's claims needs a recorded reason")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            jobs = [row[0] for row in self.connection.execute(
                "SELECT job_id FROM jobs WHERE state='claimed' AND worker=?", (worker,))]
            self.connection.execute("UPDATE jobs SET state='pending', worker=NULL, lease_until=NULL, updated=? "
                                    "WHERE state='claimed' AND worker=?", (self.clock(), worker))
            for job_id in jobs:
                self._event(job_id, "released", {"worker": worker, "reason": reason})
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return len(jobs)

    def requeue_failed(self, reason: str, *, error_prefix: str = "") -> int:
        """Operator action after fixing the cause of failures: failed jobs (optionally only those whose error
        starts with `error_prefix`) return to pending with a fresh attempt count; the event is recorded."""
        if not reason:
            raise ValueError("Requeueing failed jobs needs a recorded reason")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            jobs = [row[0] for row in self.connection.execute(
                "SELECT job_id FROM jobs WHERE state='failed' AND substr(COALESCE(error,''),1,?)=?",
                (len(error_prefix), error_prefix))]
            for job_id in jobs:
                self.connection.execute("UPDATE jobs SET state='pending', worker=NULL, attempts=0, updated=? "
                                        "WHERE job_id=?", (self.clock(), job_id))
                self._event(job_id, "requeued", {"reason": reason})
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return len(jobs)

    def _owned(self, job_id, worker, update, params):
        cursor = self.connection.execute(
            f"UPDATE jobs SET {update}, updated=? WHERE job_id=? AND worker=? AND state='claimed'",
            (*params, self.clock(), job_id, worker))
        if cursor.rowcount != 1:
            raise RuntimeError(f"Job {job_id[:16]} is not claimed by {worker} (lease expired or reclaimed)")

    def heartbeat(self, job_id: str, worker: str, *, lease_s: float) -> None:
        self._owned(job_id, worker, "lease_until=?", (self.clock() + lease_s,))

    def complete(self, job_id: str, worker: str) -> None:
        self._owned(job_id, worker, "state='done', lease_until=NULL, error=NULL", ())
        self._event(job_id, "done", {"worker": worker})

    def fail(self, job_id: str, worker: str, error: str, *, permanent: bool, max_attempts: int) -> str:
        attempts = self.connection.execute("SELECT attempts FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
        state = "failed" if permanent or attempts >= max_attempts else "pending"
        self._owned(job_id, worker, "state=?, lease_until=NULL, error=?", (state, error))
        self._event(job_id, "failed" if state == "failed" else "retry", {"worker": worker, "error": error})
        return state
