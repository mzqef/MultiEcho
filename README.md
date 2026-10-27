# MultiEcho

A counterfactual framework and measurement protocol for experiments on learned worlds
(physical scenes, simulators and world models), built on communication between worlds.

| Directory | Package | Contents |
|---|---|---|
| [`multiecho/`](multiecho/) | `multiecho-framework` | Protocol (`PROTOCOL.md`), framework, conformance checks, examples and tests |
| [`multiecho-ui/`](multiecho-ui/) | `multiecho-ui` | Local workbench for building, running and inspecting experiments |

Start with [`multiecho/README.md`](multiecho/README.md). Tags: `v0.3.0` (protocol 0.3, the
reviewed version), `v0.4.0a1` and `v0.4.0a2` (experimental pre-releases; protocol 0.3 plus
draft 0.4 additions). Python 3.11+ and NumPy are the only core requirements.

```console
python -m pip install -e "multiecho[dev]" -e multiecho-ui[dev]
python multiecho/examples/quickstart.py
multiecho-ui --workspace multiecho-ui/examples/workspace     # http://127.0.0.1:8765/
```
