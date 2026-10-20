# The MultiEcho Protocol

**Protocol Version 0.3** · Status: draft · License: MIT

This document defines how worlds communicate, how an experiment is built from that
communication, what counts as one counterfactual measurement, and how measurements
become scoped claims that anyone can check. It is independent of any engine, model,
translator, estimator or programming language. `multiecho-framework` is a reference
implementation. Any implementation whose reports pass the report checks of Appendix A
conforms.

The key words **MUST**, **MUST NOT**, **SHOULD** and **MAY** are to be interpreted as
in RFC 2119. Requirements carry identifiers (`R-x.y`). Appendix A maps each one to the
check that detects its violation, or states why it cannot be checked mechanically.

---

## 1. Everything Is Communication

A **world** receives, evolves and emits. An **interface** carries what one world emits
to another world. A **chain** is a sequence of worlds joined by interfaces.

```
communication:  world A ──interface A→B──▶ world B
experiment:     physical ──bridge──▶ simulation ──interface──▶ learned ──analyst scope──▶ analysis
```

An experiment is the special case of communication whose last receiver is an analyst.
The same objects, rules and checks serve both.

- **R-1.1** Every experiment MUST be expressed as a chain of declared worlds and
  interfaces that ends in an explicit analyst scope.
- **R-1.2** Every claim MUST name the chain, the handle, the observation and the
  observation time. A response measured at one coordinate is not evidence at
  another.

## 2. Worlds

| Kind | What it holds or does | Examples |
|---|---|---|
| `physical` | The valid data describing a real-world scene: recorded video and audio, and text statements such as intent, units, parameters and ranges | A laboratory recording; a parameter-only scene description |
| `simulation` | Turns received parameters into simulated, rendered artifacts | A physics engine; a fast visual sampler |
| `learned` | Evolves received inputs with frozen weights and emits native outputs | A world model, video generator or encoder |

Every world declares a card: identity, revision, kind, accepted channels, emitted
channels, optional intervention handle, and stochasticity. A learned world also
declares a **learned contract**: the stimulus frames it reads, its prediction horizon
(0 for a world that only represents its input) and step, whether steps beyond the
first are rolled out from its own predictions, how it adapts the protocol stimulus to
its native input, and the numerical precision of inference.

- **R-2.1** A revision MUST change whenever weights, engine, parameters, preprocessing,
  sampler, rendering or native horizon changes.
- **R-2.2** A world MUST emit exactly its declared channels, with its own revision as
  the source, and MUST reject channels it does not accept.
- **R-2.3** A physical world MAY receive. A physical scene is a world like any other,
  and messages can update its description.
- **R-2.4** Engines, samplers, models, translators and analysis methods are plugins.
  A conforming framework MUST NOT require any particular one. An effective simulator is
  a chain of two plugin worlds: an engine that generates scenes, then a sampler that
  draws many variants quickly. Each variant is indexed by its seed.
- **R-2.5** A learned world MUST declare a learned contract. It MUST read the whole
  declared stimulus. Its default observation is the first prediction step after the
  keyframe; steps beyond the first MUST be rolled out from its own predictions and
  declared as such. How it adapts the stimulus (resizing, tubelets, frame encoding,
  constant actions) belongs to the subject, and so do the losses of that adaptation.
- **R-2.6** A learned world MUST NOT receive anything after the keyframe: every video
  channel it receives holds exactly the stimulus frames, and no reference channel
  (§4) reaches it. Delivering future frames and masking them inside the model does not
  satisfy this requirement. Its outputs MUST NOT depend on the true future.
- **R-2.7** A learned world emits its native outputs. Transforms of those outputs on
  the analyst's side MUST be declared: selecting dimensions is a masker (§9);
  normalizing, projecting or reducing dimensions belongs to the analysis method.

## 3. Artifacts and Scopes

An **artifact** is what a world emits and what an interface carries. It consists of
named channels, each of one modality:

| Modality | Shape | Carries |
|---|---|---|
| `video` | frames × height × width × colour | images over time |
| `audio` | samples × channels | sound over time |
| `text` | statements (key, value, unit, kind) | implicit quantities no other modality can carry: intent, laws, physical parameters, ranges |
| `tensor` | any fixed shape | native numeric outputs such as latents or tokens |

A **scope** states what an observer may see. It selects channels, a time window, a
spatial region, tensor coordinates and text statement keys, and it may clip values.

- **R-3.1** Physical quantities MUST be carried as text statements with units. Text
  MUST NOT duplicate what another channel of the same artifact already carries.
- **R-3.2** A scoped artifact MUST carry its scope, its parent artifact's digest and
  the digest of the interface that produced it. Raw data and scope are reported
  together.
- **R-3.3** An observation vector MUST contain only in-scope values. Values masked out
  by a region do not enter the vector.

The **stimulus** is what every learned world receives: a declared number of frames
at a declared interval and resolution on a declared channel. Its last frame is the
**keyframe**. The declared observation lies a declared number of frames after it.

- **R-3.4** The stimulus is a property of the protocol, never of a subject. It MUST be
  declared before measurement, and every subject of a comparison MUST receive the
  bit-identical stimulus. A subject that needs another resolution, frame rate or
  history length adapts the stimulus itself (R-2.5); the stimulus is not changed for it.

## 4. Interfaces

An interface is the declared protocol between two worlds. The reference
implementation provides three kinds:

| Kind | Declares |
|---|---|
| `scope` | masking, clipping and selection under an explicit scope |
| `parameter-bridge` | a translator that is only parameters: renaming, unit scaling and fixed constants |
| `composed` | several interfaces applied in sequence |

Translators that need a model are plugins that implement the same protocol. Examples
are real2sim, reading parameters from video, or learned mappings.

A **reference route** delivers channels of one upstream world's emission (for
example the true future, or observation supports derived by an engine) directly to
the analyst, joined to the last world's emission together with the context identity.

- **R-4.1** An interface MUST declare its identity, revision, source world, target
  world, kind and description. Its revision MUST change when its behaviour changes.
- **R-4.2** Transmission MUST be deterministic and MUST NOT modify its input.
- **R-4.3** Every message MUST record its parent artifact and the interface that
  produced it.
- **R-4.4** Data that must not reach a learned world MUST travel by a declared
  reference route. A learned world MUST NOT accept a reference channel.

## 5. One Measurement

For a context $c$, a reference $u_0$ and an intervention $u$ applied at the handle
world, one measurement runs the chain twice, once per branch:

1. **open** every world with the context, verifying the context payload against its
   digest;
2. **fork** every world into an isolated branch;
3. along the chain: each world **receives** the incoming message, the handle world
   **applies** the intervention, the world **emits** an artifact, and the interface
   **transmits** it onward;
4. the analyst scope **observes** the final artifact;
5. **close** every handle.

The response is $\Delta Y_{W,o}(c;u,u_0)=Y_{W,o}(c,u)-Y_{W,o}(c,u_0)$. The
analyst's scoped vector is $Y$.

- **R-5.1** Branches MUST be isolated. A branch MUST NOT change the shared root or
  another branch.
- **R-5.2** Execution MUST be reproducible from the declared seeds and MUST NOT depend
  on global random state.
- **R-5.3** A world that opens context payloads MUST verify them against their digest.
- **R-5.4** The chain's identity MUST seal every member world, every interface and the
  analyst scope, and MUST remain constant within an experiment.
- **R-5.5** Observations MUST have the declared shape and finite values.
- **R-5.6** Hard control. Within a context, branches MUST differ only by the
  intervention. Every branch of a context shares each of the context's draw seeds;
  appearance, background, lighting and every other stochastic draw are context
  variables. A branch MUST NOT depend on which other branches of its context ran
  before it, and repeating a branch with its seed MUST reproduce it bit for bit.

## 6. Declaration

Before any measurement, an experiment MUST seal a declaration containing: identity and
question; the measured chain and its record; the handle and observation time; the
stimulus; the reference, labeled calibration branches and queried branches (all with
one horizon); contexts with payload digest, role and exchangeability group; the
independence unit; the coupling (`fixed_realization` or `context_draws`) and draws;
the repeats of the determinism check; the masker sweep values and
sweep draws, when the analyst scope's masker needs a sweep; condition dimensions; the
declared analysis tasks; the decision margins; and an optional reference world with
its tolerance; the keyframe alignment policy; and, when the experiment belongs to a
study, the study lock.

- **R-6.1** The declaration MUST be sealed before measurement, and the report MUST
  carry the same digest.
- **R-6.2** Nuisance and condition families that vary MUST be declared as condition
  dimensions. Examples are viewpoint, appearance, engine choice, initial-condition
  family and prompt.
- **R-6.3** The independence unit MUST describe actual independence. Sampled variants
  of one scene belong to one group.
- **R-6.4** Experiments compared with one another MUST agree on the protocol content
  the comparison relies on. A **study lock** names that content (by default: stimulus,
  handle, intervention space, observation time, reference, calibration and queried
  interventions, contexts, independence unit, coupling and draws, keyframe policy; the
  sweep, tasks, conditions and question can be added) and freezes its values. A
  declaration that names a study lock MUST equal it in every locked field; it carries
  the lock, so the report shows it. Subjects, interfaces, maskers (with their sweeps)
  and analysis methods are what a study varies; they are not locked by default.
- **R-6.5** The declaration MUST state a keyframe alignment policy for the branches of
  a context: `free` (not measured), `report` (measured and reported, optionally with a
  warning threshold), `tolerance` (every branch's keyframe within a declared distance
  of its context's reference-branch keyframe) or `identical` (bit-identical). Except
  under `free`, the keyframe the chain delivered to the learned world MUST be measured
  for every branch and reported; `tolerance` and `identical` MUST hold.

## 7. Stages

Contexts have one of three roles: discovery, validation or confirmation.

```
declare ─▶ [sweep discovery ─▶ freeze masker ─▶ assign instances] ─▶ measure discovery ─▶ check determinism
        ─▶ fit and freeze each declared task ─▶ [sweep, assign] and test validation ─▶ … confirmation ─▶ report
```

- **R-7.1** Fitting, selection and preprocessing MUST use discovery contexts only.
- **R-7.2** Exactly one law per declared task MUST be frozen, with its fitting contexts
  recorded, before any held-out context is measured. A frozen law MUST NOT be refitted.
- **R-7.3** Held-out contexts MUST NOT share a context identity, payload digest or
  exchangeability group with discovery contexts.
- **R-7.4** Every declared task MUST be tested on every declared held-out stage, and
  every result reported.
- **R-7.5** Sweeps and mask instances of held-out contexts are measurements of those
  contexts: they MUST follow the freezing of every declared law.

## 8. Response and Determinism

The protocol makes no triviality judgment: there is no resolution gate and no
`trivial` outcome. Whether a response is trivial depends on the question asked, and
is analysed case by case from the reported statistics.

- **R-8.1** Outcomes MUST NOT be conditioned on a noise floor, a tolerance or any other
  pre-judgment of triviality. A triviality claim, when one is made, is an analysis of
  the specific problem stated with its scope; it is not an outcome of this protocol.
- **R-8.2** The report MUST state, per held-out stage, the response RMS and, with
  several context draws, the RMS standard error of each pair's effect over the draws.
  These statistics are descriptive.
- **R-8.3** The determinism check MUST repeat the reference branch of every discovery
  context with its own seed, and its result MUST be reported. A chain that does not
  reproduce its branches violates hard control (R-5.6).

## 9. Analysis

Two tasks are defined. Methods are plugins; the protocol fixes only what a method may
see and the statistics that judge it.

| Task | A frozen law answers | Target |
|---|---|---|
| `forward` | the response to intervention $u$ in context $c$ | mean paired response |
| `backward` | the intervention that produced a response in context $c$ | intervention difference from $u_0$ |

In every stage a law sees a **context view**: the context, the reference
observation, and the labeled calibration branches of that same context. Outcomes of
queried branches are visible only in discovery, for fitting.

- **R-9.1** Held-out query outcomes MUST NOT be visible to a frozen law.
- **R-9.2** Every evaluation MUST compare the frozen law with named controls that use
  no more information. Forward controls are the zero response and the discovery mean
  response at the same intervention. Backward controls are the reference value and the
  discovery mean queried value.
- **R-9.3** Statuses MUST follow the declared margins. `supported` means the interval
  of the law's error advantage lies above the minimum improvement, and `contradicted`
  means it lies below the negative of that improvement. `equivalent` means the whole
  interval lies inside the equivalence margin, and it takes precedence. Otherwise the
  status is `unresolved`, which is not evidence of absence.
- **R-9.4** Uncertainty MUST resample the independence unit and MUST be corrected
  across the named controls.
- **R-9.5** A law is identified by identifier, revision and hypothesis. The protocol
  does not prescribe the method.

### Observation Masking

A **masker** is an analyst-side plugin of the observation scope. It may select
locations of the native output, or project it. Data-driven maskers see a
counterfactual **sweep**: the same contexts measured at declared intervention values
disjoint from the evaluation's. A masker is independent of the subject and of the
analysis methods; the reference implementation provides `none`, `manual`,
`monotonic-snr`, `repeat-noise-snr` and `energy`.

- **R-9.6** The masker rule MUST be declared and sealed with the analyst scope. Sweep
  values and seeds MUST be disjoint from the evaluation's. Rule parameters MUST be
  frozen on discovery sweeps only. Every context MUST receive exactly one mask
  instance, from its own sweep, before it is measured; all branches of a context
  share it.
- **R-9.7** Selection is by location. A declared grouping states which observation
  dimensions form one location (for example the channels of one patch token, or the
  colour values of one pixel). A selection keeps or drops whole locations and MUST NOT
  select individual channels. A vector without location structure is one location.
- **R-9.8** A masker selects; it MUST NOT project, rescale or combine dimensions.
  Dimensionality reduction is not masking. It is reserved as a stage of the analysis
  method, applied after masking and evaluated in the masked observation space; in
  protocol 0.3 it is the identity.

## 10. Outcomes

Each declared task ends in exactly one outcome:

| Outcome | Meaning |
|---|---|
| `stable_law` | The frozen law beats every control in every held-out stage and condition |
| `regime_specific` | It holds only for some declared condition values; the claim is restricted to them |
| `unidentified_in_scope` | The frozen law does not beat the controls; nothing is identified in this scope |

- **R-10.1** "No law" is not an outcome. A failure to describe a response is a
  statement about the instrument and scope, and MUST be reported as
  `unidentified_in_scope`. This also covers a response that is absent or negligible;
  whether it is trivial is analysed case by case (R-8.1).
- **R-10.2** Each outcome MUST follow from the sealed evidence by the rule of
  Appendix B.

## 11. Undefined Values

- **R-11.1** An undefined statistic MUST be reported as undefined, with a reason, in
  the report's `undefined` ledger. It MUST NOT be dropped, imputed or silently replaced.

## 12. Reference Worlds

- **R-12.1** A comparison with a reference world MAY be reported after freezing. It
  MUST be declared in advance with a tolerance. Examples of reference worlds are
  physics, another engine or another model.
- **R-12.2** The reference comparison MUST NOT change any outcome. A world can have a
  stable law that deviates from physics; that is a finding.

## 13. Claims

- **R-13.1** Each task MUST state its outcome as a scoped claim naming: task, subject
  (chain and its worlds), interface (observation, time and analyst scope), handle,
  intervention (space, reference, calibration, queries, horizon), contexts (counts and
  independence unit), regime, instrument (law, controls, response), validity domain
  and tested stages.
- **R-13.2** A positive claim MUST NOT extend beyond the condition values on which the
  law was individually supported.

## 14. Communication Tests

Communication between two worlds consists of three tests. A report MUST NOT
substitute one for another.

1. **Interface.** The interface conforms (§4; checks I01–I04).
2. **Translation.** On semantically aligned queries, messages agree with the
   receiver's own artifacts under the receiver's scope, better than named controls.
3. **Utility.** The receiver acts on isolated forks of one state with no message, the
   correct message, and a matched wrong message. The wrong message uses the same
   interface, channels and budget. The correct message must beat both controls on a
   declared receiver target.

- **R-14.1** Agreement is not utility, and conformance is not agreement. Each MUST be
  tested and reported separately.

## 15. Report

A report is a canonical JSON object sealed by the SHA-256 of its canonical encoding
(sorted keys, no whitespace, no non-finite numbers). It contains the following fields:
- `schema`, `protocol_version`, `implementation`;
- `declaration` (with `chain` and `digest`);
- `stages`: ordered `declared`, `swept`, `masker-frozen`, `masked`, `measured`,
  `checked`, `frozen`, `tested` and `referenced` events, each with role, task, content
  digest and chain digest;
- `response`: response RMS, draw spread and the determinism check (descriptive);
- `masking`: the declared rule, the frozen masker, instance digests per context,
  coverage per stage and the sweep;
- `keyframes`: the keyframe policy, each branch's keyframe difference to its context's
  reference branch, flagged branches and the verdict (absent without a stimulus);
- `analyses`, per task: law, controls, evaluations, regimes, outcome and claims;
- `reference`, `undefined`, `artifacts` and `seal`.

- **R-15.1** A report MUST be self-contained. Every status and outcome MUST be
  re-derivable from the report alone.
- **R-15.2** The protocol is versioned independently of implementations. Reports MUST
  state the protocol version they conform to. Once a version reaches 1.0, a minor
  version MUST NOT change the outcome of an existing report, and schemas MUST NOT
  change semantics in place.

---

## Appendix A. Requirement Traceability

Run the checks with `multiecho-conformance` (or `python -m multiecho_framework.conformance`):
- `C` checks read a report's JSON.
- `A` checks probe a measured chain.
- `W` checks probe a plugin world.
- `I` checks probe an interface.
- `K` checks probe the keyframe protocol of a chain with learned worlds.

| Requirement | Check | Detects |
|---|---|---|
| R-15.2 | C01 | missing schema, protocol version or implementation identity |
| R-15.1 | C02 | report altered after sealing |
| R-6.1, R-1.2 | C03 | incomplete or unsealed declaration |
| R-7.2, R-7.4 | C04 | held-out measured before every task is frozen; missing or reordered stages |
| R-5.4 | C05 | chain identity drift between stages |
| R-7.1 | C06 | a law fitted on non-discovery contexts |
| R-7.3 | C07 | discovery/held-out leakage of identity, payload or group |
| R-9.2 | C08 | evaluations without named, documented controls |
| R-8.1, R-8.2 | C09 | missing response statistics; a resolution section or gate fields |
| R-8.3, R-5.6 | C20, A09 | couplings that let branches differ; branches that depend on other branches or do not repeat |
| R-11.1, R-8.2 | C10 | undefined values without a ledger entry and reason |
| R-9.3, R-15.1 | C11 | statuses inconsistent with the declared margins |
| R-10.1, R-10.2, R-12.2, R-15.1 | C12 | outcomes not derivable from evidence (re-derived without reference fields); "no law" |
| R-13.1, R-13.2 | C13 | unscoped claims; claims beyond supported regimes |
| R-12.1 | C14 | undeclared or pre-freeze reference comparison |
| R-3.2, R-1.1 | C15 | an analyst scope that is not explicit, sealed and carried by claims |
| R-1.1, R-5.4, R-2.1, R-4.1 | C16 | a chain record that does not seal its worlds, interfaces and topology |
| R-5.4, R-2.1 | A01, W01 | unstable identity |
| R-5.1 | A02, W02 | forks aliasing the shared root |
| R-5.2 | A03, A04, A08, W03, W04 | irreproducible seeds; global randomness; ignored seeds |
| R-5.3 | A05 | unverified context payloads |
| R-5.5, R-3.3 | A06 | observation contract violations |
| §5 step 5 | A07, W06 | handles that cannot be closed idempotently |
| R-2.2 | W05 | emitted channels or source differing from the card |
| R-4.1 | I01 | unstable interface identity |
| R-4.2 | I02, I04 | nondeterministic transmission; modified inputs |
| R-4.3, R-3.2 | I03 | messages without parent or interface provenance |
| R-9.6, R-7.5 | C17 | masker undeclared, frozen off discovery, or contexts measured without an instance |
| R-9.6 | C18 | sweep values or seeds that overlap the evaluation |
| R-2.5, R-3.4, R-4.4 | C19 | learned worlds not reading the whole declared stimulus, accepting reference channels, or rolling out undeclared |
| R-2.6, R-4.4 | K01 | stimulus beyond the keyframe or reference channels delivered to a learned world |
| R-2.6 | K02 | learned outputs that change with the true future |
| R-3.4 | K04 | subjects receiving different stimuli |
| R-6.4 | C21 | a declaration that differs from its declared study lock |
| R-6.5 | C22 | keyframes not measured as declared; a verdict that does not follow from the reported differences; a tolerance exceeded; identical keyframes declared but not delivered |

Studies are checked with `multiecho-conformance lock` (create a lock from a pilot
report), `study` (check reports against a lock) and `compare` (tell whether two
reports are comparable, field by field).

Enforced by the reference implementation, not re-derivable from a report:
- R-2.2: the chain refuses a message channel that the next world does not accept.
- R-3.3: `Scope.vector` reads only in-scope values, and only from artifacts carrying
  that exact scope.
- R-9.1: laws receive `ContextView` objects without query outcomes.
- R-9.4: equal-group bootstrap with Bonferroni correction across controls.
- R-9.5: the law is identified in the report, and C04 binds its digest to the freeze.
- R-14.1: separate `evaluate_translation` and `evaluate_communication`, with
  `CommunicationRequest` matching the wrong message's interface, channels and budget.
- R-2.6: the chain refuses, before a learned world receives, any video channel that
  does not hold exactly the stimulus frames and any reference channel.
- R-9.7: selection maskers return one flag per location and expand it to every
  dimension of that location.
- R-9.8: masker cards accept only the kinds `identity` and `selection`, and a
  selection instance is one flag per location.

Not mechanically checkable, and therefore the responsibility of the declaring party:
- R-2.3 and R-2.4: plugin choices.
- R-3.1: that text carries only quantities other channels cannot.
- R-6.2: that every varying nuisance was declared.
- R-6.3: that independence units are truly independent.
- R-6.4: that the locked fields are the ones a comparison actually relies on.
- R-7.1: that no selection happened outside the recorded fit.
- R-14.1: the semantic adequacy of an alignment.
- R-2.7: that a learned world's emitted channels are its native outputs.
- R-2.6: that no side channel outside the chain carries the future (K02 tests the
  chain's data paths).

The self-test (`multiecho-conformance self-test`) verifies that:
- every check passes compliant fixtures and fails its targeted violation;
- the known-law, null and scrambled control chains return `stable_law`,
  `unidentified_in_scope` and `unidentified_in_scope`, and the null chain reports a
  zero response RMS (described, not adjudicated);
- a learned, masked control chain passes every report and keyframe check, and broken
  variants (future appended to the stimulus, the future smuggled in another channel,
  a subject-specific stimulus, a branch depending on other branches) are detected.

## Appendix B. Adjudication Rule

Inputs per task:
- per held-out stage: each control's status;
- per condition value: a combined status, which is `contradicted` if any stage or
  control is contradicted, `supported` if all are supported, and `unresolved`
  otherwise.

```
if every stage has every control supported:
    if some condition value is contradicted:          regime_specific (holds where supported)
    else:                                             stable_law
elif some condition value is supported:               regime_specific (holds where supported)
else:                                                 unidentified_in_scope
```

## Appendix C. Layers

| Layer | Protocol | Reference implementation |
|---|---|---|
| A1 physical world layer | §2 (`physical`), §3 text statements | `PhysicalScene`, `scene_context`, custom loaders for recorded media |
| A2 physical → simulation bridge | §4 | `ParameterBridge`; model-based real2sim translators as plugins |
| A3 experiment setting | §6 | `Declaration`: reference, calibration, queries, contexts, conditions |
| A4 effective simulating | §2 (`simulation`), §3 | engine and sampler plugins emitting video, audio and text |
| A5 learned-world interface | §3, §4 | `StimulusSpec`, the chain's keyframe gate, `ScopeInterface` selecting what the model accepts |
| A6 learned-world running | §5 | model plugins; `Chain` runs each branch on isolated forks |
| A7 analyst interface | §3 scopes, §9 masking | `ScopeInterface` + `Scope` or `MaskedScope` with a masker plugin; `ReferenceRoute` |
| A8 analysis | §9, §10 | `Experiment` with forward and backward method plugins |
| B1 world A running | §2, §5 | any world or chain |
| B2 A→B interface | §4, §14 | `ScopeInterface`, `ParameterBridge`, `ComposedInterface`, translator plugins |
| B3 world B running | §14 | `CommunicationRunner`: no, correct and wrong messages |

Correspondence with the MultiEcho paper: $Y_{W,o}(c,u)=O_{W,o}(M_W(A_W(\kappa_h(c,u);c)))$
is a chain in which $\kappa_h$ is the bridge, engine and renderer, $A_W$ the
learned-world interface, $M_W$ the learned world and $O_{W,o}$ the analyst scope.
