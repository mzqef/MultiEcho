"""Study locks (R-6.4, C21) and keyframe alignment (R-6.5, C22)."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from multiecho_framework import (
    DataRole,
    Experiment,
    Intervention,
    KeyframeAlignment,
    StudyLock,
    compare_declarations,
    compare_reports,
    dumps_report,
    dumps_study,
    load_gallery,
    loads_study,
    save_gallery,
)
from multiecho_framework.conformance import check_report, main
from multiecho_framework.experiment import seal
from multiecho_framework.selftest import learned_declaration
from multiecho_framework.testing import InterpolationMethod, PersistenceModelDouble, learned_chain


def run(chain, declaration, keep_frames=()):
    experiment = Experiment(declaration, chain, keep_frames=keep_frames)
    experiment.measure(DataRole.DISCOVERY)
    experiment.freeze(InterpolationMethod().fit(experiment.responses()))
    experiment.test()
    return experiment, experiment.report()


def results(report):
    return {result.check_id: result for result in check_report(report)}


def test_a_lock_admits_other_subjects_and_refuses_other_protocol_content():
    base = learned_declaration(learned_chain(), study=False)
    lock = StudyLock.from_declaration("spring-study", base, description="pilot")
    assert lock.differences(base) == [] and "stimulus" in lock.fields and "contexts" in lock.fields
    other_chain = learned_chain(PersistenceModelDouble())
    other = replace(learned_declaration(other_chain, study=False), study=lock)
    assert other.study.study_id == "spring-study"
    with pytest.raises(ValueError, match="interventions"):
        replace(other, interventions=(Intervention((5.5,), 4),))
    with pytest.raises(ValueError, match="contexts"):
        replace(other, contexts=other.contexts[1:])
    assert loads_study(dumps_study(lock)) == lock
    tampered = json.loads(dumps_study(lock))
    tampered["locked"]["stimulus"]["frames"] = 16
    with pytest.raises(ValueError, match="digest"):
        loads_study(json.dumps(tampered))
    with pytest.raises(ValueError, match="lockable"):
        StudyLock.from_declaration("bad", base, ("subject",))


def test_compare_tells_comparable_from_incomparable_experiments():
    base = learned_declaration(learned_chain(), study=False)
    other = learned_declaration(learned_chain(PersistenceModelDouble()), study=False)
    same = compare_declarations(base, other)
    assert same["comparable"] and same["different"] == []
    assert same["varying"]["a"]["subjects"] != same["varying"]["b"]["subjects"]
    longer = copy.deepcopy(base.as_record())
    longer["stimulus"]["frames"] = 16
    result = compare_declarations(base, longer)
    assert not result["comparable"] and result["different"] == ["stimulus"]


def test_reports_carry_and_verify_their_study(tmp_path):
    chain = learned_chain()
    declaration = learned_declaration(chain)
    _, report = run(chain, declaration)
    assert report["declaration"]["study"]["study_id"] == "track-study" and results(report)["C21"].passed
    other_chain = learned_chain(PersistenceModelDouble())
    _, other = run(other_chain, replace(learned_declaration(other_chain, study=False), study=declaration.study))
    assert compare_reports(report, other)["study"]["same_lock"]
    a, b, lock = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "lock.json"
    a.write_text(dumps_report(report))
    b.write_text(dumps_report(other))
    assert main(["lock", str(a), "--study-id", "cli-study", "-o", str(lock)]) == 0
    assert main(["study", str(lock), str(a), str(b)]) == 0
    assert main(["compare", str(a), str(b)]) == 0
    edited = copy.deepcopy(other)
    edited["declaration"]["stimulus"]["frames"] = 16
    edited["declaration"]["chain"]["stimulus"]["frames"] = 16
    c = tmp_path / "c.json"
    c.write_text(dumps_report(seal(edited)))
    assert main(["study", str(lock), str(c)]) == 1
    assert main(["compare", str(a), str(c)]) == 1


def test_keyframes_are_reported_by_default():
    chain = learned_chain()
    experiment, report = run(chain, learned_declaration(chain, study=False))
    section = report["keyframes"]
    assert section["status"] == "captured" and section["verdict"] == "reported" and results(report)["C22"].passed
    contexts = experiment.declaration.contexts
    assert sorted(section["contexts"]) == sorted(c.context_id for c in contexts)
    branches = len(experiment.declaration.calibration) + len(experiment.declaration.interventions)
    assert all(len(entry["branches"]) == branches for entry in section["contexts"].values())
    assert section["summary"]["validation"]["max_rms"] > 0        # stiffness changes the spring's keyframe


@pytest.mark.parametrize(("policy", "passes", "verdict"), [
    (KeyframeAlignment("identical"), False, "differs"),
    (KeyframeAlignment("tolerance", tolerance=10.0), True, "within_tolerance"),
    (KeyframeAlignment("tolerance", tolerance=1e-6), False, "exceeds_tolerance"),
    (KeyframeAlignment("report", warn=1e-6), True, "reported"),
    (KeyframeAlignment("free"), True, "free"),
])
def test_keyframe_policies_are_enforced_by_c22(policy, passes, verdict):
    chain = learned_chain()
    _, report = run(chain, replace(learned_declaration(chain, study=False), keyframes=policy))
    assert report["keyframes"]["verdict"] == verdict
    assert results(report)["C22"].passed is passes, results(report)["C22"].detail
    if policy.warn is not None:
        assert report["keyframes"]["warnings"]


def test_c22_recomputes_the_verdict_from_the_numbers():
    chain = learned_chain()
    _, report = run(chain, replace(learned_declaration(chain, study=False),
                                   keyframes=KeyframeAlignment("tolerance", tolerance=10.0)))
    corrupted = copy.deepcopy(report)
    entry = next(iter(corrupted["keyframes"]["contexts"].values()))
    entry["branches"][0]["rms"] = 99.0
    assert not results(seal(corrupted))["C22"].passed


def test_keyframe_policy_validation():
    with pytest.raises(ValueError, match="tolerance"):
        KeyframeAlignment("tolerance")
    with pytest.raises(ValueError, match="tolerance"):
        KeyframeAlignment("report", tolerance=0.1)
    with pytest.raises(ValueError, match="mode"):
        KeyframeAlignment("aligned")
    with pytest.raises(ValueError, match="free"):
        KeyframeAlignment("free", warn=0.1)


def test_keyframe_gallery_round_trip(tmp_path):
    chain = learned_chain()
    experiment, _ = run(chain, learned_declaration(chain, study=False), keep_frames=(0, -1))
    gallery = experiment.keyframe_gallery()
    assert gallery.frames.shape[1] == 2 and gallery.frame_labels[-1].endswith("(keyframe)")
    differences = gallery.differences()
    reference = np.array(gallery.reference)
    np.testing.assert_array_equal(differences[reference], 0.0)
    assert np.nanmax(differences[~reference, 0]) > 0
    loaded = load_gallery(save_gallery(tmp_path / "g.npz", gallery))
    np.testing.assert_array_equal(loaded.frames, gallery.frames)
    assert loaded.meta["experiment_id"] == "track-extrapolation" and loaded.context_ids == gallery.context_ids
    with pytest.raises(ValueError, match="keep_frames"):
        run(chain, learned_declaration(chain, study=False))[0].keyframe_gallery()
