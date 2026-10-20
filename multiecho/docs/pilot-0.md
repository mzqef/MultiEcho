# pilot-0: the stage-one pipeline end to end (pipeline milestone)

This is the first run of the full pipeline on real measurements: a sealed study lock → job
expansion → a gated campaign over several worlds → law readings and support masks → law channels →
a sealed atlas. It is a **pipeline milestone, not a study result**: the scene program is a development
program, the law and mask plugins are development versions without capability cards, confirmation
contexts were neither expanded nor read, and two of the four feasibility criteria were not evaluated.
The numbers below illustrate what the pipeline produces; they are not claims about the worlds.

## Design

- **Scene and axes.** A 2-D damped oscillator (16 stimulus frames, 8 future frames, 384 × 384). Two
  axes, damping ratio ζ (27 base-grid values on a log scale, 0.05–5) and natural frequency ω0
  (20 values), with all values taken from the sealed lock; validation also contains four sealed
  random positions per axis that fall between grid values.
- **Contexts.** 24 discovery and 12 validation contexts per axis, two appearance replicates each.
- **Worlds.** A *declared* simulation world (two read-outs: the rendered first future frame,
  block-averaged to 64 × 64 × 3, and the engine state x, v at 24 frame times); one *rewritten-physics*
  simulation world per axis (ζ: damping doubled, so critical damping moves from ζ = 1 to ζ = 0.5;
  ω0: frequency × 1.3); the V-JEPA 2-AC predictor (first predicted tubelet, 256 tokens × 1408;
  held-out effects stored as a 128-dimensional Gaussian random projection per token).
- **Campaign.** 576 context jobs (≈ 21,000 measurement keys) on one claim ledger; CPU workers for
  the simulation worlds, one GPU worker for the learned world, and CPU processes rendering ahead of it.

## Feasibility checks

| check | result |
|---|---|
| jobs complete | 576 / 576, no failure left |
| idempotent re-run | 0 keys missing |
| stored-replay sample (≥ 1 % and ≥ 20 keys per world; 213 keys) | all bit-identical |
| noise floors; transition visibility | not evaluated in this milestone |

## Law readings (validation contexts; ΔL in bits)

| axis | world | read-out | reading | ΔL | held-out R² |
|---|---|---|---|---|---|
| ζ | declared | state | shared profiles, rank 3 | 3.65e5 | 0.85 |
| ζ | rewritten | state | shared profiles, rank 3 | 2.11e5 | 0.82 |
| ζ | declared | future frame | shared profiles, rank 3 | 9.5e4 | 0.31 |
| ζ | V-JEPA 2-AC | predicted tubelet (sketch) | no law with ΔL > 0 under this plugin set | −8.1e7 | −2.69 |
| ω0 | declared | state | shared profiles, rank 3 | 5.18e5 | 0.44 |
| ω0 | rewritten | state | shared profiles, rank 3 | 6.61e5 | 0.29 |
| ω0 | V-JEPA 2-AC | predicted tubelet (sketch) | local three-reference law | 6.5e6 | 0.13 |

Support masks (within-context unit pairs, BH at a declared q) select 2.5–3.8 % of the pixel
locations of the future frame, and 23 of the 24 state time steps: the one never selected is t = 0,
where the initial state is fixed by construction — a known answer.

**What the negative ζ reading on V-JEPA 2-AC is.** On the base-grid values the held-out ΔL is
+1.2e7 bits, consistent with discovery; the loss comes entirely from the four sealed positions between
grid values (−9.3e7 bits). The selected spline profiles have a knot at every grid value and a weak
penalty, and oscillate between knots; selection by context folds only sees grid values. The sealed
off-grid positions exist to catch exactly this. Value-held-out selection is the planned remedy
(`docs/protocol-0.4-draft.md` §22).

## Law channels (declared → other worlds, consistency test only)

| axis | target | outcome |
|---|---|---|
| ζ | rewritten physics | partial (consistency supported) |
| ζ | V-JEPA 2-AC | failed (a profile-reversed control predicted better) |
| ω0 | rewritten physics | partial (consistency supported) |
| ω0 | V-JEPA 2-AC | partial (consistency supported) |

Consistency alone cannot tell "the law is preserved" from "the law changed but can still be
calibrated"; reading how a rewritten world's law differs needs the commuting-square test.

## Problems found and fixed on the way

- A cross-process stimulus-cache lock covered rendering, so parallel workers ran one at a time.
- Stored-replay verification did not require the single-threaded BLAS environment that random-projection
  sketches are measured in; a multi-threaded verification reported a change (max 1e-3) that was not in
  the data. Verification now refuses to run outside the measurement environment, and lifting the
  resulting block is a recorded operator action.
- The study lock did not yet provide the allowed values per role that the executor asks for; added.
- Editing the working tree during a formal run made the clean-tree guard refuse 297 jobs; they were
  requeued through a recorded ledger action after the edits were committed, and completed.
