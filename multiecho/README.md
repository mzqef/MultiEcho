# MultiEcho

**A counterfactual framework for experiments on learned worlds, built on communication
between worlds.**

Physical scenes, simulators and world models are all **worlds**: they receive, evolve
and emit. **Interfaces** carry what one world emits to another: masking, clipping,
parameter bridges and translators. An **experiment** is a chain of worlds that ends at
an analyst, who observes under an explicit scope and fits laws. Communication and
experiment share one architecture, one protocol and one set of checks.

```
communication:  world A ──interface──▶ world B
experiment:     physical ──bridge──▶ simulation ──interface──▶ learned ──analyst scope──▶ analysis
```

## What Is in the Package

| Layer | What it does | Provided here | Plugin (yours or examples/) |
|---|---|---|---|
| **A1** physical world | Valid data describing a real scene: video, audio, intent, units, parameters, ranges | `PhysicalScene`, `scene_context`, loaders | recorded-media loaders |
| **A2** physical → simulation | A bridge or translator from the scene to engine inputs | `ParameterBridge` (parameters only) | real2sim translators |
| **A3** experiment setting | Counterfactual branches of every context, fixed in advance | `Declaration` | — |
| **A4** effective simulating | Simulate and render branches, e.g. engine then fast visual sampler | — | engines, samplers |
| **A5** learned-world interface | Deliver the protocol stimulus, ending at the keyframe; nothing after it | `StimulusSpec`, keyframe gate, `ScopeInterface` | — |
| **A6** learned-world running | The model adapts the stimulus itself and evolves each branch on an isolated fork | `Chain`, `LearnedContract` | world models |
| **A7** analyst interface | Observe under an explicit scope; mask whole locations; receive the true future by reference route | `Scope`, `MaskedScope`, `ReferenceRoute` | maskers |
| **A8** analysis | Forward and backward fitting, judged by fixed statistics | `Experiment`, outcomes, sealed reports | analysis methods |
| **B1–B3** communication | World A → interface → world B, with interface, translation and utility tests | `ComposedInterface`, `evaluate_translation`, `CommunicationRunner` | translators |

The framework core ships **no engine, model, translator or estimator**. Those are plugins
implementing its protocols. The experimental `multiecho_framework.law_core` subpackage
(0.4.0a1, imported only on request) is one *reference* law plugin implementation, not a
protocol requirement. Examples live in [`examples/plugins`](examples/plugins),
including the paper's three-reference estimators (PL, linear, kernel) in
[`examples/plugins/three_reference.py`](examples/plugins/three_reference.py), and
copyable templates live in [`examples/templates.py`](examples/templates.py).

## Ten Minutes

```console
python -m pip install -e ".[dev]"
python examples/quickstart.py
python examples/masking.py
python examples/communication.py
```

The quickstart runs one sim2learned experiment through all eight layers:
- a parameter-only spring scene is bridged to an engine;
- the engine renders video, and a sampler draws appearance variants;
- a tiny frozen video model reads the declared 16-frame stimulus and predicts future
  latents;
- the analyst observes those latents under a scope and fits forward and backward laws.

```text
One branch through the chain (raw artifacts, messages, and the analyst's scoped view):
  spring-scene                       scene:text[6]
  spring-scene                       parameters:text[6]
  analytic-oscillator-engine         video:video[16, 16, 32, 1], truth:text[3]
  ...
  tiny-video-model                   latent:tensor[4, 8]   scope=future-latents
  forward  stable_law             validation: ... supported; confirmation: ... supported
  backward stable_law             validation: ... supported; confirmation: ... supported
One physical scene, two engines (1:N): final position -0.1433 m (analytic) vs -0.1440 m (Euler)
Conformance: 22/22 report checks passed
```

The masking example observes a next-frame predictor under three maskers: the full
response, a fixed region, and the locations that change steadily along a sweep at
declared values disjoint from the evaluation.

The communication example runs a simulation as world A. Its interface masks and clips
the video, then a model-based translator reads the spring stiffness. World B is a
physical-world description that receives the reading. Interface conformance,
translation agreement and utility against no and wrong messages are tested separately.

## One Experiment

```python
chain = Chain("my-chain", (scene, engine, sampler, model), (bridge, to_sampler, to_model),
              analyst_scope, observation, stimulus=StimulusSpec(16, 0.08, 256, 256),
              references=(ReferenceRoute("my-engine", Scope("reference", ("future",))),))
declaration = Declaration(..., world=chain.describe(), chain=chain.record(), stimulus=chain.stimulus,
                          calibration=..., interventions=..., tasks=("forward", "backward"),
                          coupling=SeedMode.CONTEXT, draws=3)
experiment = Experiment(declaration, chain)
experiment.measure(DataRole.DISCOVERY)
experiment.freeze(my_forward_method.fit(experiment.responses()), "forward")
experiment.freeze(my_backward_method.fit(experiment.responses()), "backward")
experiment.test()                  # validation and confirmation, against fixed controls
report = experiment.report()       # sealed, self-contained, checkable by anyone
```

The chain and the session enforce:
- a learned world receives exactly the declared stimulus, ending at the keyframe, and
  never the true future;
- every branch of a context shares each draw: only the intervention differs;
- a masker is frozen on discovery sweeps and selects whole locations;
- held-out contexts cannot be measured before every declared task is frozen;
- a frozen law cannot be refitted;
- the chain's identity, which seals every world, interface and the analyst scope,
  cannot change.

Each task ends in one of three outcomes:
- `stable_law`
- `regime_specific` (the claim is restricted to the conditions where the law held)
- `unidentified_in_scope`

"No law" is not an outcome. Triviality is not judged by the protocol either: the
report states the response statistics, and whether a response is trivial is analysed
case by case. Physics, another engine or another model can serve as a second-layer
reference, which never changes an outcome.

## Check Anything

```console
multiecho-conformance report  path/to/report.json          # any implementation's report
multiecho-conformance world   examples.templates:probe_engine
multiecho-conformance world   examples.templates:probe_model
multiecho-conformance interface examples.templates:probe_translator
multiecho-conformance adapter examples.templates:probe_chain
multiecho-conformance keyframe examples.templates:probe_chain
multiecho-conformance self-test
```

There are 22 report checks, 9 chain probes, 6 world probes, 4 interface probes and 3
keyframe probes.
The self-test proves that every check passes compliant fixtures and fails its targeted
violation.

### Studies: keeping experiments comparable

A study lock freezes the protocol content that experiments must share to be compared
(stimulus, interventions, contexts, observation time, keyframe policy, ...); subjects,
maskers and analysis methods vary. Pass it as `Declaration(study=lock)`: a
declaration that differs in any locked field is refused before anything runs, and the
report carries the lock (check C21).

```console
multiecho-conformance lock pilot.report.json --study-id my-study -o my-study.lock.json
multiecho-conformance study my-study.lock.json run-b.report.json run-c.report.json
multiecho-conformance compare run-b.report.json run-c.report.json
```

### Keyframe alignment

`Declaration(keyframes=KeyframeAlignment(mode, warn, tolerance))` states how close the
keyframes of a context's branches must be: `free`, `report` (default: measured and
reported, flagged above `warn`), `tolerance`, or `identical`. The experiment measures
the keyframe the chain delivered to the learned world for every branch and seals the
differences in the report (check C22). `Experiment(..., keep_frames=(0, -1))` also
keeps frames for a `KeyframeGallery`, for example for the `multiecho-ui` workbench.

Run `python examples/study.py` for both, including a refused 8-frame variant.

### Scene programs (experimental)

`SceneProgram` records a scene family's sampled factors and relations, continuous,
logarithmic, periodic or discrete axes, reference regimes and transitions, observation
conditions, rewritten-physics controls, sampling tiers, context budgets, and reference
identity. Context factors are sampled independently from a SHA-256-derived context
seed; replicate-level factors and appearance/noise/physics seeds are separately
derived for each draw. The serialized program has a canonical integrity-checked JSON
envelope, a full declaration digest, and a measurement digest that excludes
design-only changes such as rewrites, grids and budgets.

Confirmation positions are reproducibly sampled in each axis's natural coordinate
and excluded from the base grid (including circular distance on periodic axes).
Plugins may submit versioned `SupplementaryQuery` records, which are checked against
the declared axis, role, baseline and requested interval. `scene_declaration` adapts
a program to the framework's `Declaration`; `probe_replay` compares observations
from isolated roots using byte-based SHA-256 digests.

These contracts do not include scene solvers, renderers, or model adapters. Those
remain experiment-specific plugins; declaring a scene program alone does not establish
that its reference dynamics or visual observability have been validated. Capability
testing evaluates the plugin; the observation condition independently defines the
measurement and does not define its own success criterion.

### World-to-world channels (experimental)

A channel is a frozen translator from world A to world B for one kind of content
(`ContentKind`: scene, object, law, mechanism). Any two `World` implementations can
be joined: physical, simulated or learned, in either direction. A channel is judged
by four tests, all on validation or confirmation data and all against named controls:
consistency (`evaluate_translation`), utility (`evaluate_communication`), the
commuting square (`evaluate_commutation`: evolve in A then translate, against
translate then evolve in B), and the round trip (`evaluate_round_trip`: what
A → B → A loses, reported per scoped coordinate).

`ChannelRecord` collects the evidence for one translator and derives its outcome:
`established` only when all four tests ran and each was supported, `failed` when none
was, `partial` otherwise, with the unsupported and missing tests named. What counts as
support is the `minimum_improvement` (and optional equivalence margin) that the
study declares in each `EvaluationPlan`; the framework ships no pass level. A failed or
partial channel is relative to the translator revision and the declared plans, not
evidence that no channel between the two worlds exists.

### World revisions (experimental)

`WorldRevision` fixes a world version — physical, simulated or learned — by weights,
architecture, read-in, precision, initialization, inference settings, code and parent; any
change is another world. `revision_lineage` follows `parent` links of constructed revisions
(training, fine-tuning, editing) back to their root. Draft 0.4 protocol clauses for revisions,
measurement, plugins, observation conditions, law cards and channels are collected in
[`docs/protocol-0.4-draft.md`](docs/protocol-0.4-draft.md); `PROTOCOL.md` 0.3 is unchanged.

### Law core and campaigns (experimental, 0.4.0a1)

`multiecho_framework.law_core` reads laws in any world's measured effects along a scene axis:
response blocks, a declared two-part code and its compression gain ΔL, law families 0–2
(three-reference interpolation, dense curve, shared profiles with two calibration designs) and the
zero-response law, selection by cross-validated ΔL with a report of the candidates the data cannot
tell apart, backward recovery with a conformal set, law cards, and the law channel (a frozen
shared-profile law of world A, calibrated in world B and judged against no-law and wrong-law
controls). `multiecho_framework.campaign` is a gated job graph, a SQLite claim ledger with leases
and recorded operator actions, and a generic worker loop. Both are NumPy/stdlib only; every level
(folds, ranks, confidence, resamples, error rates) is the caller's declaration, and a `not_found`
law or a failed channel is relative to the plugin set, never a statement that a world has no law.

```
python examples/cross_world_laws.py     # declared, rewritten-physics and observer worlds; laws and channels
```

[`docs/pilot-0.md`](docs/pilot-0.md) reports the first end-to-end run of this pipeline on a
damped-oscillator scene with simulated worlds and V-JEPA 2-AC (a pipeline milestone, not a study
result).

## Documents

- [`PROTOCOL.md`](PROTOCOL.md): the normative protocol (version 0.3, draft), with
  requirement traceability and a layer map.
- [`docs/architecture.md`](docs/architecture.md): worlds, artifacts, interfaces,
  chains and analysis.
- [`docs/plugins.md`](docs/plugins.md): how to write engines, models, translators and
  analysis methods.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): how to change the protocol or the framework.

Python 3.11+ and NumPy are the only core requirements. Training export and
differentiable losses are `EXPERIMENTAL` and outside the protocol.
