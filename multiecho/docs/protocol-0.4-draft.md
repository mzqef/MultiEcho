# MultiEcho Protocol 0.4 — Draft Additions

Status: **draft**. Protocol 0.3 (`PROTOCOL.md`) remains normative for every 0.3 report and is not
changed by this document. The clauses below are candidates for 0.4. Each one names what already
enforces it in code, or states that it is not yet mechanically checked. Symbols referred to here
are experimental (`multiecho_framework.EXPERIMENTAL`) until 0.4 is released.

The protocol has one aim: the same measurement, law and channel contracts apply to **any** world
(physical, simulated, learned), so that experiments, law discovery and communication between
worlds can be compared across worlds without ranking them. The protocol fixes semantics and
checks. It ships no pass levels: what counts as support is declared by each study.

## 2. Worlds (additions)

- **R-2.1 (revised)** A world revision fixes weights, architecture, read-in (history length,
  frame rate, resolution, tubelet or per-frame packing), numerical precision, initialization
  (pretrained, untrained, permuted, constructed, self-trained, trained), inference settings,
  code and parent. Any change is another world. The kind is `physical`, `simulation` or `learned`.
  *Enforced:* `WorldRevision` validation and digest.
- **R-2.11 (new) Lineage.** A constructed revision (training, fine-tuning, editing, pruning)
  names its parent revision. A lineage is followed to a root whose parent is `none`; unknown
  parents and cycles are errors. *Enforced:* `revision_lineage`.

## 16. Measurement and replay (new)

- **R-16.1** A measurement is identified by world revision, scene program, observation
  condition, axis, context, role, value, baseline, replicate, readout, form and origin.
- **R-16.2** A measurement depends only on the scene program, the observation condition and the
  world revision, and is replayable bit for bit from its declaration. Caches only accelerate:
  a reading missing from a cache is re-measured, never reported as "no data". Plugins never
  change measurements.
- **R-16.3** An effect is a branch observation minus the baseline observation of the same
  context and replicate.
- **R-16.4** Each world reports three noise floors: repeated identical input, a minimal input
  perturbation, and the paired effect noise under resampled appearance. Noise floors describe;
  they are not gates.
- **R-16.5** Every engine and subject first passes a replay probe (one branch from two
  independent roots). Non-identical replay is added to the noise floor and reported.

*Enforced in the reference measurement store (not shipped); framework contracts pending.*

## 17. Plugins, readings and capability cards (new)

- **R-17.1** A plugin card declares identity, kind, semantic version, code digest, parameters,
  fitted roles (train, discovery or calibration only), assumptions, inputs and outputs.
- **R-17.2** Every reading names plugin, version, code digest, plugin-set version and input digest.
- **R-17.3** A reading is `found`, `not_found` or `error`. `not_found` is stated as "not found under
  plugin set vN", never as absence in the world. A `found` reading established on held-out data
  stays valid when plugins change.
- **R-17.4** Changing any member of a plugin set increments its version; old readings are kept.
- **R-17.5** A plugin version used for a confirmatory conclusion has a capability card on a
  declared known-answer suite that meets the levels declared **by the study's lock**. The
  protocol declares no levels.
- **R-17.6** A reading that is equivalent (within a declared margin) when the subject is replaced
  by its untrained or permuted revision describes that world only; it is not used to compare
  worlds or to support a mechanism.
- **R-17.7** Quantities that depend on a plugin's partition of a response (for example "inherited
  from the input" versus "inferred by the world") are not reported as properties of the world.
- **R-17.8** Known answers come from construction; no plugin computes them or sees them.

*Enforced in the reference plugin system (not shipped); framework contracts pending.*

## 18. Observation conditions (new)

- **R-18.1** Every reading, law and conclusion names its observation condition (input family):
  history length, frame rate and time step, keyframe position, visibility, viewpoint, field
  visualization. The observation condition is the domain of a law.
- **R-18.2** Readings from different observation conditions are not merged into one law.
  Whether a law holds across conditions is stated as a separate reuse dimension.
  *Enforced:* `LawScope.observation` in the lab law core; a law card carries one scope.

## 19. Base grids, queries and confirmation (new)

- **R-19.1** Base grids depend only on the scene program and the reference world.
- **R-19.2** Supplementary queries are proposed from discovery measurements only and record
  plugin, version, rationale, interval and values; the measured points join the common data.
  *Enforced:* `SupplementaryQuery` validation.
- **R-19.3** Confirmatory conclusions use only base-grid points and sealed confirmation positions
  on held-out contexts.
- **R-19.4** Confirmation positions are generated and sealed before any learned-world output is seen.
  *Enforced:* `SceneProgram.confirmation_points` is a function of the declaration only.

## 20. Law cards (new)

- **R-20.1** A law card records: scope (world revision, scene program, observation condition,
  axis, readout, plugin set), family and form, parameters (shared, per context, per time),
  tested reuse dimensions with held-out errors (context, time, space, intervention and scale,
  between laws, composition, cross-world), failed regimes, ΔL with its code, combination
  coordinates, readout dependence and evidence digests.
- **R-20.2** ΔL = L(D | zero response) − [L(law) + L(D | law)], with D every non-baseline response
  of held-out contexts, calibration observations coded as zero response, residuals under the
  declared noise code, and parameter cost (k/2)·log₂ n. The code is declared. ΔL compares
  candidate laws of one world on one data set only.
- **R-20.3** A cross-world reuse entry cites the digest of a channel record whose content is the law.
- **R-20.4** No world rankings and no total scores.

*Partly enforced:* the lab law core (`LawCard`) rejects mixed scopes and undeclared reuse
dimensions; sealed analysis reports and atlas checks (G01–G05) are pending.

## 21. Channels (new)

- **R-21.1** A channel is a frozen translator from world A to world B for one kind of content
  (scene, object, law, mechanism). A and B are any two distinct world revisions.
- **R-21.2** Translators are fitted on train, discovery or calibration data only. The four tests
  (consistency, utility, commuting square, round trip) run on validation or confirmation data,
  each against at least one named control.
- **R-21.3** The commuting square compares "evolve (or intervene) in A, then translate" with
  "translate, then evolve in B". The round trip reports, per scoped coordinate of A, what
  A → B → A loses.
- **R-21.4** The channel outcome is derived from the evidence: established only when all four
  tests ran and each was supported under its declared plan; failed when none was supported;
  partial otherwise, naming unsupported and unrun tests.
- **R-21.5** "Failed" and "partial" are relative to the translator revision and the declared
  plans; they are not statements that no channel exists between the two worlds.

*Enforced:* `evaluate_commutation`, `evaluate_round_trip`, `ChannelEvidence` and `ChannelRecord`
(roles, provenance, derived outcome, claim boundary).

## 22. Reference law core (informative)

`multiecho_framework.law_core` implements the law card semantics of §20 for one family of law
plugins. It is informative: a conforming study may use other law plugins, provided their cards
carry the fields of §20 and their negative readings are scoped to the plugin set.

- **Code.** ΔL = L(D | zero response) − [L(law) + L(D | law)] over all non-baseline held-out
  observations; calibration observations are coded as zero response (a law that needs fewer
  calibrations compresses more); residuals use a Gaussian code with per-dimension paired-effect noise
  frozen on discovery; the law costs (k/2)·log2 n plus structural bits.
- **Selection.** Candidates are ranked by cross-validated ΔL over context folds on discovery; the
  choice costs log2(number of candidates) bits. Candidates whose paired CV-ΔL difference to the
  selected law reaches zero at the declared confidence are reported as indistinguishable: the
  selected form is then not identified by the data.
- **Known limitation.** Folds are over contexts, not values. A profile law can behave differently
  between grid values than on them; sealed evaluation positions between grid values (§19) are what
  reveal this. Value-held-out selection is planned.

## 23. Campaigns (informative)

`multiecho_framework.campaign` orchestrates measurement or analysis jobs of any study: a job
graph whose gates (a sealed lock, a reviewed capability, …) are satisfied only by recorded ledger
entries, ordering rules declared by the study, atomic claims with leases, and recorded operator
actions (`release` for a stopped worker, `requeue_failed` after the cause of a failure was fixed).
Workers run an idempotent `run(job_id)`, so a killed worker is harmless.

