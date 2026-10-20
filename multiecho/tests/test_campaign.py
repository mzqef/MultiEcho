"""Campaign backbone: gated job graph, claim ledger with leases and operator actions, generic worker."""
from __future__ import annotations

import threading

import pytest

from multiecho_framework.campaign import GateRule, JobGraph, Ledger, Node, run_worker


class Job:
    def __init__(self, job_id, priority=0):
        self.job_id, self.priority = job_id, priority


def test_graph_rejects_cycles_unknown_requirements_and_rule_violations():
    with pytest.raises(ValueError, match="cycle"):
        JobGraph((Node("a", "x", ("b",)), Node("b", "x", ("a",))))
    with pytest.raises(ValueError, match="Unknown"):
        JobGraph((Node("a", "x", ("missing",)),))
    with pytest.raises(ValueError, match="no upstream lock"):
        JobGraph((Node("lock", "lock", gate=True), Node("m", "measure")), rules=(GateRule("lock", "measure"),))
    graph = JobGraph((Node("c", "x", ("a", "b")), Node("b", "x"), Node("a", "x")))
    assert graph.order == ("a", "b", "c") and graph.ready(set()) == ("a", "b")


def test_concurrent_claims_never_duplicate(tmp_path):
    ids = tuple(f"{i:064x}" for i in range(400))
    path = tmp_path / "ledger.sqlite"
    with Ledger(path) as ledger:
        assert ledger.add(JobGraph((Node("n", "measure", (), ids),)), [Job(i) for i in ids]) == 400
    claimed = []

    def claim(worker):
        with Ledger(path) as own:
            while (job_id := own.claim(worker, ("n",), lease_s=3600)) is not None:
                claimed.append(job_id)

    threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(claimed) == len(set(claimed)) == 400


def test_leases_operator_actions_gates_and_retries(tmp_path):
    clock = [0.0]
    ids = tuple(f"{i:064x}" for i in range(4))
    graph = JobGraph((Node("lock", "lock", gate=True), Node("n", "measure", ("lock",), ids)),
                     rules=(GateRule("lock", "measure"),))
    with Ledger(tmp_path / "ledger.sqlite", clock=lambda: clock[0]) as ledger:
        ledger.add(graph, [Job(i) for i in ids])
        runs = []
        assert not run_worker(ledger, graph, runs.append, worker="w", lease_s=10, max_attempts=2).done
        ledger.satisfy("lock", "lock verified")
        assert ledger.claim("dead", ("n",), lease_s=10) == ids[0]
        clock[0] = 11.0
        assert ledger.claim("alive", ("n",), lease_s=10) == ids[0]
        with pytest.raises(RuntimeError, match="not claimed by dead"):
            ledger.complete(ids[0], "dead")
        assert ledger.release("alive", "operator stopped the worker") == 1
        flaky = {ids[1]: 1}

        def run(job_id):
            if job_id == ids[2]:
                raise ValueError("contract violation")
            if flaky.get(job_id):
                flaky[job_id] -= 1
                raise RuntimeError("transient")
            return 1

        report = run_worker(ledger, graph, run, worker="w", lease_s=10, max_attempts=2, permanent=(ValueError,))
        assert report.failed == [ids[2]] and report.retried == [ids[1]] and report.produced == 3
        assert ledger.requeue_failed("cause fixed", error_prefix="ValueError") == 1
        report = run_worker(ledger, graph, lambda job_id: 1, worker="w", lease_s=10, max_attempts=2)
        assert report.done == [ids[2]] and ledger.counts()["done"] == 4
