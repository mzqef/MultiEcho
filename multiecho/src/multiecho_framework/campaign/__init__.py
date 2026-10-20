"""Experimental (0.4): campaign orchestration that any study can reuse.

A gated job graph (studies declare the gates and ordering rules), a SQLite claim ledger with leases and
recorded operator actions, and a generic worker loop. Nothing here knows about worlds, scenes or plugins;
jobs are identified by string ids and executed by a caller-supplied `run(job_id)`.
"""
from .graph import GateRule, JobGraph, Node
from .ledger import Ledger
from .worker import WorkerReport, run_worker

__all__ = ["GateRule", "JobGraph", "Ledger", "Node", "WorkerReport", "run_worker"]
