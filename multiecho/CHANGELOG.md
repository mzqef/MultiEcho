# Changelog

Protocol and implementation are versioned independently. Reports state the protocol
version (`protocol_version`) and the implementation version (`implementation.version`).

## Protocol 0.4 (draft additions) · multiecho-framework 0.4.0a1 (pre-release)

Experimental preview. `PROTOCOL.md` 0.3 is unchanged; draft clauses are in
`docs/protocol-0.4-draft.md`. Nothing in the 0.3 API changed behaviour, and
`import multiecho_framework` loads no new module.

### Added (experimental)
- `channels`: world-to-world channel records for scene, object, law and mechanism content;
  `evaluate_commutation` and `evaluate_round_trip`; outcomes derived from evidence.
- `revision`: `WorldRevision` (physical, simulation or learned) and `revision_lineage`.
- `law_cards`: `LawScope`, `LawCard`, reuse dimensions including `cross_world`.
- `SceneProgram.branch`, `TransitionFeature.frame`, non-nested `ComplexitySpec`.
- `law_core` subpackage (NumPy only): response blocks, code N1 and ΔL with a cluster-bootstrap
  interval, law families 0–2 and the zero-response law, D-optimal and gain calibration designs,
  CV-ΔL selection with an indistinguishability report, backward recovery with a conformal set,
  law cards, frozen-state round trip, and the shared-profile law channel.
- `campaign` subpackage (stdlib only): gated job graph with study-declared ordering rules,
  SQLite claim ledger (atomic claims, leases, `release`, `requeue_failed`), generic worker.
- Example `examples/cross_world_laws.py`; document `docs/pilot-0.md`.

## Protocol 0.3 (draft) · multiecho-framework 0.3.0

Every change below closes a gap that let an invalid experiment pass all checks in 0.2.

### Added
- Keyframe protocol. `StimulusSpec` declares the stimulus (frames, interval,
  resolution, channel; the last frame is the keyframe; the observation lies after it)
  as a property of the protocol. A chain with learned worlds refuses, before a learned
  world receives, any video beyond the stimulus and any reference channel (R-2.6).
- Learned contracts. `LearnedContract` on every learned `WorldCard`: the whole stimulus
  is read; the default observation is the first prediction step; longer rollouts are
  autoregressive and declared; stimulus adaptation and numerical precision are
  declared (R-2.5, R-2.7).
- Reference routes. `ReferenceRoute` delivers the true future or engine-derived data
  straight to the analyst; the analyst input also carries the context identity (R-4.4).
- Hard control. `SeedMode.CONTEXT`: every branch of a context shares each draw;
  experiments may declare only `fixed_realization` or `context_draws`. A determinism
  check repeats reference branches (R-5.6, R-8.3).
- Observation masking. `Grouping`, `Sweep`, `MaskInput`, `MaskerCard`, the `Masker`
  protocol, `MaskedScope`, and the maskers `NoMask`, `ManualMask`, `MonotonicSNRMask`,
  `RepeatNoiseSNRMask` and `EnergyMask`. A masker only selects, by location: a
  location (patch token, pixel) is kept or dropped whole; projections are not masks.
  Dimensionality reduction is a reserved stage of the analysis method (identity in 0.3). `Experiment` runs the sweep at
  declared values disjoint from the evaluation's, freezes the masker on discovery and
  assigns one instance per context (R-9.6 to R-9.8).
- Checks C17 (masking stage discipline), C18 (sweep disjointness), C19 (stimulus and
  learned contracts), C20 (hard control); probes A09 (branch independence within a
  context), K01 (keyframe gate), K02 (future invariance), K04 (stimulus identity
  across subjects); `multiecho-conformance keyframe`.
- Example `examples/masking.py`; `TinyFramePredictor` plugin.
- Study locks (R-6.4). `StudyLock` freezes the protocol content experiments of a study
  share (by default stimulus, handle, intervention space, observation time, reference,
  calibration and queried interventions, contexts, independence unit, coupling, draws,
  keyframe policy); `Declaration(study=lock)` refuses a declaration that differs and
  carries the lock into the report (check C21). `compare_declarations` /
  `compare_reports` tell field by field whether two experiments are comparable.
  Commands `multiecho-conformance lock | study | compare`, and `stimulus` (K04).
- Keyframe alignment (R-6.5). `KeyframeAlignment` (`free`, `report` with an optional
  warning threshold, `tolerance`, `identical`) in the declaration; `Experiment`
  measures the keyframe the chain delivered to the learned world for every branch
  against its context's reference branch and seals a `keyframes` report section
  (check C22). `keep_frames` collects a `KeyframeGallery` (`save_gallery`,
  `load_gallery`) for inspection.
- Example `examples/study.py` (a comparable masker variant; a refused 8-frame variant;
  `--workspace` output for the `multiecho-ui` viewer).
- Estimator plugins `examples/plugins/three_reference.py`: `ThreeReferenceForward` and
  `ThreeReferenceBackward` with methods `pl`, `linear` and `kernel` (Gaussian RBF with an
  affine tail, width chosen on discovery) on a raw or log coordinate.

### Changed
- No triviality judgment (R-8.1): the resolution gate and the `trivial` outcome are
  removed; outcomes are `stable_law`, `regime_specific` and `unidentified_in_scope`.
  The report's descriptive `response` section (response RMS, draw spread, determinism)
  replaces `resolution`; triviality is analysed case by case. `adjudicate` takes the
  control and regime statuses only.
- `Declaration`: `resolution_factor` removed; `stimulus`, `sweep` and `sweep_draws`
  added; `repeats` counts determinism repeats.
- Draw seeds are per context, shared by all its branches.
- The analyst input of a chain is the last world's emission joined with reference
  routes and the context identity; `Chain.trace` includes it.
- Report schema `multiecho-report/0.3`.
- `MonotonicSNRMask` scores a location by the share of its sweep variation carried by
  steady dimensions (`steady_shares`: per-dimension |mean step| / standard error above
  `channel_tau` = 2, location kept above `share` = 0.1), both declared and frozen on
  discovery. Pooling the ratio over a wide location (1408 channels of a V-JEPA token)
  kept it near 1 whenever most of the variation was not a steady trend, so the former
  thresholds {2, 3, 5} selected nothing. `monotonic_scores` stays as a diagnostic.
- Threshold calibration (`RepeatNoiseSNRMask`) uses chance-corrected split-half
  agreement: two empty selections no longer agree perfectly, and agreement no longer
  grows with selection size by chance; exact ties go to the smaller threshold.

## Protocol 0.2 (draft) · multiecho-framework 0.2.0

The framework is rebuilt on communication between worlds. An experiment is a chain of
worlds ending at an analyst.

### Added
- Worlds: `World`, `WorldCard`, `WorldKind` (physical, simulation, learned), and
  `PhysicalScene` for the physical world layer (parameters or recorded media), which
  can also receive.
- Artifacts: `Artifact`, `Channel`, `Modality` (video, audio, text, tensor), and
  `Statement` for implicit quantities carried as text with units.
- Observation scopes: `Scope` and `Region` (channels, time window, spatial region,
  coordinates, statement keys, clipping). Raw data and the scope travel together.
- Interfaces: `Interface`, `ScopeInterface`, `ParameterBridge`, `ComposedInterface`;
  model-based translators are plugins.
- `Chain`: counterfactual branches run through every world on isolated forks. The
  chain seals every member and is itself a measured world.
- Analysis: forward and backward tasks, `ContextView`, `ResponseSet`, method
  protocols, fixed controls per task, labeled calibration branches, and one outcome per
  task.
- Communication tests on artifacts: `evaluate_translation`, `CommunicationRunner`,
  `evaluate_communication`.
- Conformance: 16 report checks, 8 chain probes, 6 world probes, 4 interface probes,
  and a known-answer self-test.
- Artifact exchange: `dumps_artifact` / `loads_artifact`.
- Examples: an eight-layer quickstart, world A → interface → world B communication,
  example plugins, and copyable plugin templates.

### Removed
- The built-in ridge estimator. The framework ships no estimator; methods are plugins.
- The vector-only bridge and message API (`AffineBridge`, `ResponseMessage`,
  `Translator`, `MessageAdapter`), replaced by artifacts and interfaces.
- The 1.0.0rc1 protocol draft, superseded before any release.

## 0.1.0

- Initial alpha interfaces: contracts, paired and sequence execution, evaluation,
  response, trajectory, residual and path laws, serialization and training export.
