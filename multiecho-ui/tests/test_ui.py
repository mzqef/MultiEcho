"""multiecho-ui: workspace model, PNG encoding and every HTTP endpoint on a real workspace."""
from __future__ import annotations

import copy
import json
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from multiecho_framework import (
    DataRole,
    Experiment,
    KeyframeAlignment,
    StudyLock,
    dumps_report,
    dumps_study,
    save_gallery,
)
from multiecho_framework.experiment import seal
from multiecho_framework.selftest import learned_declaration
from multiecho_framework.testing import InterpolationMethod, PersistenceModelDouble, learned_chain

from multiecho_ui.png import encode_png, heatmap, to_uint8
from multiecho_ui.server import SCRIPTS, serve
from multiecho_ui.workspace import NotFound, Workspace


def _run(chain, declaration):
    experiment = Experiment(declaration, chain, keep_frames=(0, -1))
    experiment.measure(DataRole.DISCOVERY)
    experiment.freeze(InterpolationMethod().fit(experiment.responses()))
    experiment.test()
    return experiment.report(), experiment.keyframe_gallery()


FRAMEWORK = Path(__file__).resolve().parents[2] / "multiecho"   # the framework checkout, for examples.plugins
DESIGNS = Path(__file__).resolve().parents[1] / "examples" / "workspace" / "experiments"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    for folder in ("reports", "studies", "experiments", "keyframes"):
        (root / folder).mkdir()
    chain = learned_chain()
    base = replace(learned_declaration(chain, study=False), keyframes=KeyframeAlignment("report", warn=0.01))
    lock = StudyLock.from_declaration("ui-study", base)
    report, gallery = _run(chain, replace(base, study=lock))
    other_chain = learned_chain(PersistenceModelDouble())
    other_base = replace(learned_declaration(other_chain, study=False), keyframes=KeyframeAlignment("report", warn=0.01))
    other, _ = _run(other_chain, replace(other_base, study=lock))
    edited = copy.deepcopy(other)
    edited["declaration"]["stimulus"]["frames"] = 16
    (root / "reports" / "a.json").write_text(dumps_report(report))
    (root / "reports" / "b.json").write_text(dumps_report(other))
    (root / "reports" / "c.json").write_text(dumps_report(seal(edited)))
    (root / "studies" / "ui-study.json").write_text(dumps_study(lock))
    save_gallery(root / "keyframes" / "a.npz", gallery)
    for name in ("spring-latents", "spring-frame-masked"):
        (root / "experiments" / f"{name}.json").write_text((DESIGNS / f"{name}.json").read_text())
    (root / "experiments" / "broken.json").write_text("{")
    (root / "workspace.json").write_text(json.dumps({"title": "test workspace", "plugins": ["examples.plugins.catalog"],
                                                     "paths": [str(FRAMEWORK)]}))
    return root


@pytest.fixture(scope="module")
def server(workspace):
    httpd = serve(str(workspace), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never route localhost through a proxy


def get(base, path):
    with _DIRECT.open(base + path, timeout=30) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def get_json(base, path):
    return json.loads(get(base, path)[2])


def test_png_encoding_round_trips_through_zlib():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    data = encode_png(image)
    assert data.startswith(b"\x89PNG\r\n\x1a\n") and b"IHDR" in data and data.endswith(b"IEND\xaeB`\x82")
    start = data.index(b"IDAT") + 4
    length = int.from_bytes(data[start - 8:start - 4], "big")
    raw = zlib.decompress(data[start:start + length])
    assert raw == b"\x00" + image[0].tobytes() + b"\x00" + image[1].tobytes()
    assert to_uint8(np.array([[0.0, 0.5, 1.2]])).tolist() == [[0, 128, 255]]
    colours = heatmap(np.array([[0.0, 1.0, 2.0]]), 1.0)
    assert colours[0, 0].tolist() == [0, 0, 4] and colours[0, 1].tolist() == colours[0, 2].tolist()
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2), dtype=np.float32))


def post(base, path, body, headers=None):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    with _DIRECT.open(request, timeout=60) as response:
        return json.loads(response.read())


def error_of(call) -> tuple[int, str]:
    with pytest.raises(urllib.error.HTTPError) as caught:
        call()
    return caught.value.code, json.loads(caught.value.read())["error"]


def finish(base, run_id, timeout=120.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        record = get_json(base, f"/api/run?id={run_id}")
        if not record["busy"] or time.monotonic() > deadline:
            return record
        time.sleep(0.1)


def test_workspace_model(workspace):
    view = Workspace(workspace)
    assert view.info() == {"title": "test workspace", "root": str(workspace.resolve()), "plugins": ["examples.plugins.catalog"],
                           "experiments": 3, "runs": 0, "reports": 3, "studies": 1, "keyframes": 1}
    summaries = {row["name"]: row for row in view.reports()}
    assert summaries["a.json"]["checks"] == {"passed": 22, "total": 22, "failed": []}
    assert summaries["a.json"]["study"] == "ui-study" and summaries["a.json"]["keyframes"]["warnings"] > 0
    assert "C21" in summaries["c.json"]["checks"]["failed"]
    assert view.compare("a.json", "b.json")["comparable"]
    assert view.compare("a.json", "c.json")["different"] == ["stimulus"]
    members = {m["report"]: m for m in view.studies()[0]["members"]}
    assert members["a.json"]["belongs"] and members["b.json"]["belongs"] and members["c.json"]["different"] == ["stimulus"]
    index = view.gallery("a.npz")
    context = index["contexts"][0]
    assert [b["values"] for b in context["branches"]] == sorted(b["values"] for b in context["branches"])
    reference = next(b for b in context["branches"] if b["reference"])
    assert reference["rms"] == [0.0, 0.0]
    with pytest.raises(NotFound):
        view.report("../a.json")
    designs = {row["name"]: row for row in view.designs()}
    assert designs["spring-latents"]["worlds"][0] == "spring-scene" and designs["broken"]["error"]
    with pytest.raises(ValueError):
        view.save_design("../escape", view.design("spring-latents"))
    with pytest.raises(ValueError):
        view.save_design("plain", {"schema": "something else"})


def test_catalog_lists_framework_maskers_and_workspace_plugins(workspace):
    from multiecho_ui.catalog import Catalog, CatalogError

    catalog = Catalog(("examples.plugins.catalog", "no.such.module"), (FRAMEWORK,))
    entries = {entry["id"]: entry for entry in catalog.records()["plugins"]}
    assert "no.such.module" in catalog.errors
    sampler = entries["examples.plugins.simulation:AppearanceSampler"]
    assert sampler["kind"] == "world" and [(p["name"], p["type"], p["default"]) for p in sampler["parameters"]] == [
        ("gain", "float", 0.05), ("offset", "float", 0.02)]
    readout = {p["name"]: p["type"] for p in entries["examples.plugins.translators:StiffnessReadout"]["parameters"]}
    assert readout == {"interface_id": "auto", "source": "auto", "target": "auto", "damping_prior": "float"}
    manual = {p["name"]: p["type"] for p in entries["multiecho_framework.masking:ManualMask"]["parameters"]}
    assert manual["grouping"] == "auto" and manual["selected"] == "selection"
    assert catalog.build("examples.plugins.simulation:AppearanceSampler", "world", {"gain": "0.5"}).describe().world_id
    for plugin_id, kind, params in (("examples.plugins.simulation:AppearanceSampler", "world", {"unknown": 1}),
                                    ("examples.plugins.simulation:AppearanceSampler", "forward", {}),
                                    ("examples.plugins.learned:TinyVideoModel", "world", {"seed": 1.5}),
                                    ("builtins:eval", "world", {})):
        with pytest.raises(CatalogError):
            catalog.build(plugin_id, kind, params)


def test_design_builds_traces_and_names_the_failing_part(workspace):
    from multiecho_ui.catalog import Catalog
    from multiecho_ui.design import blank, validate

    catalog = Catalog(("examples.plugins.catalog",), (FRAMEWORK,))
    document = json.loads((DESIGNS / "spring-latents.json").read_text())
    result = validate(document, catalog)
    assert result["ok"] and result["chain"]["observation"]["dims"] == 32
    assert [row["label"] for row in result["trace"]][-1] == "the analyst sees (scope future-latents)"
    assert result["declaration"]["contexts"] == {"discovery": 12, "validation": 8, "confirmation": 8}
    assert result["declaration"]["conditions"] == {"release": ["at-rest", "moving"]}
    masked = validate(json.loads((DESIGNS / "spring-frame-masked.json").read_text()), catalog)
    assert masked["chain"]["analyst"]["masker"]["locations"] == 512 and masked["declaration"]["sweep"] == 8
    assert validate(blank("demo"), catalog)["ok"]
    broken = copy.deepcopy(document)
    broken["chain"]["interfaces"].pop()
    assert validate(broken, catalog)["error"]["where"] == "chain.interfaces"
    broken = copy.deepcopy(document)
    broken["chain"]["worlds"][1]["plugin"] = "os:system"
    assert validate(broken, catalog)["error"]["where"] == "chain.worlds[1]"
    broken = copy.deepcopy(document)
    broken["design"]["contexts"]["factors"][2]["when"]["key"] = "later"
    assert validate(broken, catalog)["error"]["where"].startswith("design.contexts.factors[2]")
    unswept = json.loads((DESIGNS / "spring-frame-masked.json").read_text())
    unswept["design"]["sweep"]["values"] = []
    error = validate(unswept, catalog)["error"]
    assert error["where"] == "design" and "sweep" in error["message"]


def test_every_endpoint(server):
    status, kind, body = get(server, "/")
    assert status == 200 and kind.startswith("text/html") and b"MultiEcho workbench" in body
    for script in SCRIPTS:
        assert f'src="/{script}"'.encode() in body
        status, kind, _ = get(server, f"/{script}")
        assert status == 200 and kind.startswith("text/javascript")
    assert get(server, "/style.css")[0] == 200
    info = get_json(server, "/api/workspace")
    assert info["reports"] == 3 and info["experiments"] == 3 and "stimulus" in info["lockable"]
    catalog = get_json(server, "/api/catalog")
    assert any(entry["kind"] == "forward" for entry in catalog["plugins"]) and not catalog["errors"]
    assert {row["name"] for row in get_json(server, "/api/designs")} == {"broken", "spring-latents", "spring-frame-masked"}
    assert get_json(server, "/api/design?name=spring-latents")["name"] == "spring-latents"
    reports = get_json(server, "/api/reports")
    assert [r["name"] for r in reports] == ["a.json", "b.json", "c.json"]
    detail = get_json(server, "/api/report?name=a.json")
    assert len(detail["checks"]) == 22 and detail["keyframes"]["verdict"] == "reported"
    assert detail["locked"]["stimulus"]["frames"] == 8 and detail["analyses"]["forward"]["outcome"]["value"]
    compare = get_json(server, "/api/compare?a=a.json&b=c.json&study=ui-study.json")
    assert not compare["comparable"] and compare["different"] == ["stimulus"]
    assert get_json(server, "/api/studies")[0]["study_id"] == "ui-study"
    galleries = get_json(server, "/api/galleries")
    assert galleries[0]["frame_labels"][-1].endswith("(keyframe)")
    gallery = get_json(server, "/api/gallery?name=a.npz")
    branch = next(b for b in gallery["contexts"][0]["branches"] if not b["reference"])
    for kind in ("frame", "diff"):
        status, content, png = get(server, f"/api/image?gallery=a.npz&branch={branch['index']}&frame=1&kind={kind}")
        assert status == 200 and content == "image/png" and png.startswith(b"\x89PNG")
    for path, code in (("/api/report?name=../../etc/passwd", 404), ("/api/report", 400), ("/nope", 404),
                       ("/api/image?gallery=a.npz&branch=9999&frame=0", 404), ("/api/design?name=../workspace", 400),
                       ("/api/run?id=missing", 400)):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server, path)
        assert error.value.code == code


def test_actions_only_from_this_hosts_pages(server):
    host = server.split("//", 1)[1]
    code, message = error_of(lambda: post(server, "/api/design/validate", {"document": {}}, {"Origin": "http://evil.example"}))
    assert code == 403 and "cross-origin" in message
    request = urllib.request.Request(server + "/api/design/delete", data=b'{"name": "spring-latents"}', method="POST",
                                     headers={"Content-Type": "text/plain"})
    code, message = error_of(lambda: _DIRECT.open(request, timeout=30))
    assert code == 403 and "JSON" in message
    request = urllib.request.Request(server + "/api/workspace", headers={"Host": "rebound.example"})
    assert error_of(lambda: _DIRECT.open(request, timeout=30))[0] == 403
    assert post(server, "/api/design/validate", {"document": {"schema": "x"}}, {"Origin": f"http://{host}"})["ok"] is False
    assert error_of(lambda: post(server, "/api/design/save", {"name": "../escape", "document": {}}))[0] == 400
    assert error_of(lambda: post(server, "/api/nope", {}))[0] == 404


def test_a_run_follows_the_protocol_order_and_seals_a_checked_report(server, workspace):
    run_id = post(server, "/api/run/start", {"design": "spring-latents"})["id"]
    record = get_json(server, f"/api/run?id={run_id}")
    assert record["stage"] == "declared" and record["allowed"] == {"measure": True, "freeze": False, "test": False, "seal": False}
    assert error_of(lambda: post(server, "/api/run/act", {"id": run_id, "action": "test"}))[0] == 409
    post(server, "/api/run/act", {"id": run_id, "action": "measure"})
    record = finish(server, run_id)
    assert record["stage"] == "measured" and record["discovery"]["pairs"] == 48 and record["discovery"]["determinism"]["identical"]
    assert len(record["discovery"]["contexts"]) == 12 and len(record["discovery"]["explained"]) == 2
    forward = {"plugin": "examples.plugins.methods:InterpolationForward", "params": {}}
    post(server, "/api/run/act", {"id": run_id, "action": "freeze", "task": "forward", "method": forward})
    record = finish(server, run_id)
    assert record["laws"]["forward"]["law_id"] == "in-context-interpolation" and record["allowed"]["test"] is False
    code, message = error_of(lambda: post(server, "/api/run/act", {"id": run_id, "action": "freeze", "task": "forward", "method": forward}))
    assert code == 409 and "refitted" in message
    assert error_of(lambda: post(server, "/api/run/act", {"id": run_id, "action": "freeze", "task": "backward",
                                                          "method": forward}))[0] == 400
    post(server, "/api/run/act", {"id": run_id, "action": "all"})
    record = finish(server, run_id)
    assert record["stage"] == "sealed" and not record["error"]
    assert record["report"]["checks"] == {"passed": 22, "total": 22, "failed": []}
    assert set(record["report"]["outcomes"]) == {"forward", "backward"}
    rows = record["tests"]["diagnostics"]["validation"]
    assert len(rows) == 16 and all("recovered" in row["backward"] and "error_rms" in row["forward"] for row in rows)
    assert (workspace / "reports" / f"{run_id}.json").is_file() and (workspace / "keyframes" / f"{run_id}.npz").is_file()
    assert (workspace / "runs" / f"{run_id}.json").is_file()
    assert any(row["id"] == run_id and row["report"] == f"{run_id}.json" for row in get_json(server, "/api/runs"))
    assert get_json(server, f"/api/report?name={run_id}.json")["summary"]["checks"]["failed"] == []
    assert error_of(lambda: post(server, "/api/run/act", {"id": run_id, "action": "seal"}))[0] == 409
    made = post(server, "/api/study/create", {"report": f"{run_id}.json", "study_id": "spring-study", "description": "springs"})
    assert made["name"] == "spring-study.json" and (workspace / "studies" / "spring-study.json").is_file()
    assert error_of(lambda: post(server, "/api/study/create", {"report": f"{run_id}.json", "study_id": "spring-study"}))[0] == 400
    created = post(server, "/api/design/new", {"name": "copy-of-spring", "from": "spring-latents"})["name"]
    assert get_json(server, f"/api/design?name={created}")["design"]["experiment_id"] == "copy-of-spring"
    post(server, "/api/design/delete", {"name": created})
    assert created not in {row["name"] for row in get_json(server, "/api/designs")}


def test_a_masked_run_freezes_its_masker_on_discovery(workspace):
    from multiecho_ui.catalog import Catalog
    from multiecho_ui.runs import Runs

    runs = Runs(workspace, Catalog(("examples.plugins.catalog",), (FRAMEWORK,)))
    run_id = runs.start("spring-frame-masked", json.loads((DESIGNS / "spring-frame-masked.json").read_text()))
    runs.act(run_id, "all")
    runs.wait(120)
    record = runs.get(run_id)
    assert record["stage"] == "sealed" and record["discovery"]["masking"]["frozen"]["masker_id"] == "monotonic-snr"
    assert 0 < record["discovery"]["masking"]["coverage"]["discovery"] < 1
    assert record["report"]["checks"]["failed"] == []
