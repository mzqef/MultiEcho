# Contributing to the Framework

This directory is a self-contained community library. Keep concrete worlds, model
weights, scene assets, engines, experimental thresholds, campaign scripts and result
tables in external integrations or the parent research tree, not in its runtime package.

## Development

```console
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m multiecho_framework.conformance self-test
python -m build
```

For optional losses, install `.[torch]` and set `MULTIECHO_TEST_TORCH=1` before
running the tests. Core installation and imports must work without PyTorch, GPU
libraries, networking, model downloads, or the parent `multiecho` package.

## Contribution Boundaries

1. Add worlds, interfaces and analysis methods as plugins in their own packages,
   implementing the protocols documented in [the plugin guide](docs/plugins.md). No
   plugin discovery executes code from a received message or artifact path.
2. For core changes, provide a small numerical conformance test, not a physical
   benchmark. Cover correct execution and a discriminating invalid case.
3. Preserve the distinctions between tasks, kinds, data roles, random coupling,
   exchangeability units, scopes, unsupported evidence and equivalence. Do not
   silently turn a numerical score into a physical-understanding verdict.
4. Test fork isolation and cleanup for chain changes; scope, shape and identity
   failures for artifacts and interfaces; no training or calibration overlap and no
   held-out visibility for analysis changes; and versioned roundtrip/tampering for
   exchange changes.
5. Do not change published JSON semantics in place. Introduce a version and an
   explicit migration path. Content hashes detect changes, not author authenticity.
6. Update the [architecture](docs/architecture.md) and public API docs with actual
   support and limitations. Do not claim universal translators, native physical
   mechanisms, or improved model training from unit tests.

## Extension Priorities

- example and third-party plugins: engines, samplers, world models, translators and
  analysis methods, each shipping its own conformance probe results;
- recorded-media physical loaders (video, audio) and their scope defaults;
- distributional laws and stochastic path tests (a candidate extension for a later
  protocol version);
- nonlinear and learned translators with explicit calibration and independent
  validation;
- safe array-store and transport integrations for large artifacts;
- conformance reports from independent implementations of this protocol.

## Changing the Protocol

`PROTOCOL.md` is versioned independently of this package. Propose protocol changes as
an issue that states the requirement text, the conformance check that would detect
its violation, and whether any existing report's outcome would change (which requires
a major version). Every new MUST needs either a check with a known-answer fixture in
`selftest.py` or an explicit entry in the "not mechanically checkable" list.

Discuss dependencies and compatibility before adding a backend to the core. An
external integration is preferable when it has its own heavyweight runtime or
scientific assumptions. The project retains its existing MIT license; no release
or remote publication is performed by the test/build workflow.