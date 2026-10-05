"""multiecho-ui: a local, read-only web viewer for MultiEcho workspaces.

    multiecho-ui --workspace DIR [--host 127.0.0.1] [--port 8765]

Pages: reports with their conformance checks; field-by-field comparison of two
reports; study locks with the reports that belong to them; keyframe galleries laid out
along the counterfactual axis, with differences to each context's reference branch.
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

from .workspace import NotFound, Workspace

STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}


def _clean(value):
    """JSON-safe: non-finite floats become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def make_handler(workspace: Workspace):
    class Handler(BaseHTTPRequestHandler):
        server_version = "multiecho-ui"

        def log_message(self, format, *args):  # noqa: A002
            if getattr(self.server, "verbose", False):
                super().log_message(format, *args)

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(_clean(payload), allow_nan=False).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            try:
                if url.path in STATIC:
                    name, kind = STATIC[url.path]
                    self._send(HTTPStatus.OK, resources.files("multiecho_ui.static").joinpath(name).read_bytes(), kind)
                elif url.path == "/api/workspace":
                    self._json(workspace.info())
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
            except (NotFound, FileNotFoundError) as error:
                self._json({"error": f"not found: {error}"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as error:
                self._json({"error": f"bad request: {error}"}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(workspace: str, host: str = "127.0.0.1", port: int = 8765, verbose: bool = False) -> ThreadingHTTPServer:
    """A server bound to (host, port); call `serve_forever()` on it (port 0 picks a free port)."""
    server = ThreadingHTTPServer((host, port), make_handler(Workspace(workspace)))
    server.verbose = verbose
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="multiecho-ui", description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, help="folder with reports/, studies/ and keyframes/")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    server = serve(arguments.workspace, arguments.host, arguments.port, arguments.verbose)
    host, port = server.server_address[:2]
    print(f"multiecho-ui serving {arguments.workspace} at http://{host}:{port}/ (read-only; Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
