"""multiecho-ui: a local workbench for running MultiEcho experiments.

    multiecho-ui --workspace DIR [--host 127.0.0.1] [--port 8765]

Build an experiment from the workspace's plugin catalog (worlds, interfaces, maskers, analysis methods), declare
it, run it stage by stage in the order the protocol enforces (measure discovery, freeze each task's law, test
held-out contexts, seal the report), and inspect sealed reports, comparisons, study locks and keyframe galleries.

Reading is GET; every action is a JSON POST accepted only from this host's own pages (Host and Origin checks).
Plugins run in this process and come only from the modules listed in the workspace's `workspace.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .catalog import Catalog, CatalogError
from .design import DesignError, blank, validate
from .runs import RunError, Runs
from .workspace import NotFound, Workspace

SCRIPTS = ("app.js", "experiments.js", "runs.js", "plugins.js", "records.js")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8"),
          **{f"/{name}": (name, "text/javascript; charset=utf-8") for name in SCRIPTS}}
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_BODY = 4 * 1024 * 1024


def _clean(value):
    """JSON-safe: non-finite floats become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


class Forbidden(Exception):
    pass


class Bench:
    """The workspace, its plugin catalog and its runs."""

    def __init__(self, root: str):
        self.workspace = Workspace(root)
        settings = self.workspace.settings()
        self.catalog = Catalog(settings["plugins"], settings["paths"])
        self.runs = Runs(self.workspace.root, self.catalog, self.workspace.lock)

    def info(self) -> dict:
        from multiecho_framework.study import DEFAULT_LOCKED, LOCKABLE

        return {**self.workspace.info(), "catalog_errors": self.catalog.errors, "lockable": LOCKABLE,
                "default_locked": list(DEFAULT_LOCKED)}

    # ----------------------------------------------------------------- actions
    def post(self, path: str, body: dict) -> dict:
        workspace = self.workspace
        if path == "/api/design/validate":
            return validate(body["document"], self.catalog, workspace.lock)
        if path == "/api/design/save":
            return {"name": workspace.save_design(body["name"], body["document"])}
        if path == "/api/design/new":
            name = workspace._name(body["name"])
            if name in {row["name"] for row in workspace.designs()}:
                raise ValueError(f"A design named {name!r} already exists")
            source = body.get("from")
            document = workspace.design(source) if source else blank(name)
            if source:
                document["design"] = {**(document.get("design") or {}), "experiment_id": name}
            return {"name": workspace.save_design(name, document)}
        if path == "/api/design/delete":
            workspace.delete_design(body["name"])
            return {"deleted": body["name"]}
        if path == "/api/run/start":
            name = body["design"]
            return {"id": self.runs.start(name, workspace.design(name))}
        if path == "/api/run/act":
            self.runs.act(body["id"], body["action"], body)
            return {"id": body["id"], "action": body["action"]}
        if path == "/api/run/discard":
            self.runs.discard(body["id"])
            return {"discarded": body["id"]}
        if path == "/api/study/create":
            fields = tuple(body.get("fields") or ())
            name = workspace.create_lock(body["report"], body["study_id"], fields or _default_fields(),
                                         body.get("description") or "")
            return {"name": name}
        raise NotFound(path)


def _default_fields() -> tuple[str, ...]:
    from multiecho_framework.study import DEFAULT_LOCKED

    return DEFAULT_LOCKED


def make_handler(bench: Bench, allowed_hosts: set[str]):
    workspace = bench.workspace

    class Handler(BaseHTTPRequestHandler):
        server_version = "multiecho-ui"

        def log_message(self, format, *args):  # noqa: A002
            if getattr(self.server, "verbose", False):
                super().log_message(format, *args)

        def _send(self, status: int, body: bytes, kind: str, cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(_clean(payload), allow_nan=False).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def _host(self) -> str:
            host = (self.headers.get("Host") or "").strip()
            if host.startswith("["):
                return host[1:].split("]", 1)[0]
            return host.rsplit(":", 1)[0] if host.count(":") == 1 else host

        def _check_origin(self) -> None:
            """Accept actions only from this server's own pages: a known Host, a matching Origin, a JSON body."""
            host = self._host()
            if host not in allowed_hosts:
                raise Forbidden(f"host {host!r} is not served here")
            origin = self.headers.get("Origin")
            if origin is not None and urlparse(origin).netloc != self.headers.get("Host"):
                raise Forbidden("cross-origin request")
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                raise Forbidden("actions take a JSON body")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            try:
                if self._host() not in allowed_hosts:
                    raise Forbidden(f"host {self._host()!r} is not served here")
                if url.path in STATIC:
                    name, kind = STATIC[url.path]
                    self._send(HTTPStatus.OK, resources.files("multiecho_ui.static").joinpath(name).read_bytes(), kind)
                elif url.path == "/api/workspace":
                    self._json(bench.info())
                elif url.path == "/api/catalog":
                    self._json(bench.catalog.records())
                elif url.path == "/api/designs":
                    self._json(workspace.designs())
                elif url.path == "/api/design":
                    self._json(workspace.design(query["name"]))
                elif url.path == "/api/runs":
                    self._json(bench.runs.records())
                elif url.path == "/api/run":
                    self._json(bench.runs.get(query["id"]))
                elif url.path == "/api/reports":
                    self._json(workspace.reports())
                elif url.path == "/api/report":
                    self._json(workspace.report(query["name"]))
                elif url.path == "/api/compare":
                    self._json(workspace.compare(query["a"], query["b"], query.get("study") or None))
                elif url.path == "/api/studies":
                    self._json(workspace.studies())
                elif url.path == "/api/galleries":
                    self._json(workspace.galleries())
                elif url.path == "/api/gallery":
                    self._json(workspace.gallery(query["name"]))
                elif url.path == "/api/image":
                    vmax = float(query["vmax"]) if query.get("vmax") else None
                    body = workspace.image(query["gallery"], int(query["branch"]), int(query.get("frame", -1)),
                                           query.get("kind", "frame"), vmax)
                    self._send(HTTPStatus.OK, body, "image/png")
                else:
                    self._json({"error": f"unknown path {url.path}"}, HTTPStatus.NOT_FOUND)
            except Forbidden as error:
                self._json({"error": f"forbidden: {error}"}, HTTPStatus.FORBIDDEN)
            except (NotFound, FileNotFoundError) as error:
                self._json({"error": f"not found: {error}"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as error:
                self._json({"error": f"bad request: {error}"}, HTTPStatus.BAD_REQUEST)

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            try:
                self._check_origin()
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    raise ValueError("the body is empty or too large")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("the body is a JSON object")
                self._json(bench.post(url.path, body))
            except Forbidden as error:
                self._json({"error": f"forbidden: {error}"}, HTTPStatus.FORBIDDEN)
            except (NotFound, FileNotFoundError) as error:
                self._json({"error": f"not found: {error}"}, HTTPStatus.NOT_FOUND)
            except RunError as error:
                self._json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (DesignError, CatalogError, KeyError, ValueError, TypeError) as error:
                self._json({"error": f"{error}" if not isinstance(error, KeyError) else f"missing field {error}"},
                           HTTPStatus.BAD_REQUEST)

    return Handler


def serve(workspace: str, host: str = "127.0.0.1", port: int = 8765, verbose: bool = False) -> ThreadingHTTPServer:
    """A server bound to (host, port); call `serve_forever()` on it (port 0 picks a free port)."""
    bench = Bench(workspace)
    server = ThreadingHTTPServer((host, port), make_handler(bench, LOOPBACK | {host}))
    server.verbose = verbose
    server.daemon_threads = True
    server.bench = bench
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multiecho-ui", description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True,
                        help="folder with workspace.json, experiments/, runs/, reports/, studies/ and keyframes/")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: this machine only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    server = serve(arguments.workspace, arguments.host, arguments.port, arguments.verbose)
    host, port = server.server_address[:2]
    errors = server.bench.catalog.errors
    print(f"multiecho-ui: workbench for {arguments.workspace} at http://{host}:{port}/ (Ctrl+C to stop)", flush=True)
    for module, error in errors.items():
        print(f"  plugin module {module} failed to load: {error}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
