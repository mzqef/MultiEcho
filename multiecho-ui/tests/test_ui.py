"""multiecho-ui: workspace model, PNG encoding and every HTTP endpoint on a real workspace."""
from __future__ import annotations

import copy
import json
import threading
import urllib.error
import urllib.request
import zlib
from dataclasses import replace

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
from multiecho_ui.server import serve
from multiecho_ui.workspace import NotFound, Workspace


def _run(chain, declaration):
    experiment = Experiment(declaration, chain, keep_frames=(0, -1))
    experiment.measure(DataRole.DISCOVERY)
    experiment.freeze(InterpolationMethod().fit(experiment.responses()))
    experiment.test()
    return experiment.report(), experiment.keyframe_gallery()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    for folder in ("reports", "studies", "keyframes"):
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
    (root / "workspace.json").write_text(json.dumps({"title": "test workspace"}))
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


def test_workspace_model(workspace):
    view = Workspace(workspace)
    assert view.info() == {"title": "test workspace", "root": str(workspace.resolve()), "reports": 3, "studies": 1,
                           "keyframes": 1}
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


def test_every_endpoint(server):
    status, kind, body = get(server, "/")
    assert status == 200 and kind.startswith("text/html") and b"multiecho" in body
    for path in ("/app.js", "/style.css"):
        assert get(server, path)[0] == 200
    assert get_json(server, "/api/workspace")["reports"] == 3
    reports = get_json(server, "/api/reports")
    assert [r["name"] for r in reports] == ["a.json", "b.json", "c.json"]
    detail = get_json(server, "/api/report?name=a.json")
    assert len(detail["checks"]) == 22 and detail["keyframes"]["verdict"] == "reported"
    assert detail["locked"]["stimulus"]["frames"] == 8
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
                       ("/api/image?gallery=a.npz&branch=9999&frame=0", 404)):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server, path)
        assert error.value.code == code
