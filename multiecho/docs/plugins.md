# Writing Plugins

Engines, samplers, models, translators and analysis methods are plugins. Start from
[`examples/templates.py`](../examples/templates.py), then probe your plugin before
running experiments:

```console
multiecho-conformance world     your_package.module:probe   # returns (world, context, message)
multiecho-conformance interface your_package.module:probe   # returns (interface, artifact)
multiecho-conformance adapter   your_package.module:probe   # returns (chain, context, intervention)
multiecho-conformance keyframe  your_package.module:probe   # same; for chains with learned worlds
```

## Worlds: Engines, Samplers and Models

- **Card.** Put everything that changes behaviour into the revision: checkpoint
  digest, engine version, internal parameters, renderer, preprocessing and native
  horizon. Declare `accepts` and `emits` exactly. Declare a `handle` only if the world
  itself takes interventions, for example internal-state edits on a model.
- **`fork`.** Return a state that shares nothing mutable with its origin. Different
  Python objects are not enough if they alias one buffer, cache or remote session.
- **`receive`.** Read only declared channels. A message may be `None` (no message),
  and the chain rejects channels you do not accept.
- **Randomness.** Use only the seed passed to `receive` or `apply`, via
  `np.random.default_rng(seed)`. Never use global random state, and never keep state
  across branches (a counter, a cache keyed on call order). A fast visual sampler draws
  one variant per seed; every branch of a context receives the same seed.
- **`emit`.** Return an `Artifact` whose source is `card.source` and whose channels
  are exactly the declared ones. Put physical quantities in a text channel as
  statements with units.
- **`close`.** Release one handle. It must be idempotent and must never release shared
  weights.

A world that opens context payloads must verify them against their digest. Only
physical-world loaders normally do this.

## Learned Worlds

- **Contract.** Declare a `LearnedContract`: `history` equals the declared stimulus
  length; `horizon` is 1 for a next-step predictor (0 for an encoder); longer rollouts
  use only your own predictions and set `autoregressive=True`.
- **Adapt the stimulus yourself.** The stimulus (frames, interval, resolution) is the
  protocol's. If your model needs another input form (tubelets, per-frame encoding,
  resizing, constant actions), convert it inside the plugin and describe the conversion
  in `adaptation`. Never ask for a different stimulus.
- **Nothing after the keyframe.** You receive exactly the stimulus frames. Do not
  request future frames to mask them internally; the chain refuses them, and K02
  verifies that your outputs do not change with the true future.
- **Native outputs.** Emit the model's own outputs. Selections chosen by the analyst
  are maskers; normalizations or projections belong to the analysis method. Neither is
  part of the model. Record the
  numerical precision of inference in `precision` and in the revision.

## Maskers

A masker implements `describe`, `calibrate`, `fit`, `apply` and `output_dims`. Select
whole locations of a `Grouping` (`GroupSelection` does the expansion); never select
single channels, and never project or rescale: a masker only selects. A masker that needs a
sweep sees only `MaskInput`: the context's sweep outputs and the reference-route data
of that sweep, never evaluation responses.

## Interfaces: Translators

- **Identity.** Declare the identity, a revision that changes with the mapping, the
  source and target world identities, the kind and a description.
- **Transmission.** `transmit` must be deterministic and must not modify its input.
  Return a message whose `parent` is the input digest and whose `via` is your
  interface digest.
- **Model-based translators.** A translator that needs a model (real2sim, reading
  parameters from video) is an interface like any other. Its revision must seal the
  model.

## Analysis Methods

Methods fit on the discovery `ResponseSet` and return frozen laws:

```python
class MyForward:
    def fit(self, data: ResponseSet) -> ForwardLaw: ...    # law.predict(view, intervention)

class MyBackward:
    def fit(self, data: ResponseSet) -> BackwardLaw: ...   # law.recover(view, response)
```

`examples/plugins/three_reference.py` provides the paper's three-reference estimators
as ready plugins: `ThreeReferenceForward(method, coordinate)` and
`ThreeReferenceBackward(method, coordinate)` with `method` in `pl`, `linear`, `kernel`
and `coordinate` in `raw`, `log`. They need two calibration branches bracketing the
reference.

A law's `LawSpec` must record every discovery context it was fitted on. At test time
it sees only a `ContextView`: the reference observation and the labeled calibration
responses of that context. The framework judges it against fixed controls, and no
method can change those statistics.

## Recorded Physical Data

Give `PhysicalScene` a `loader(context) -> (statements, media_channels)` and declare
the media channel names:
- The loader dereferences the context payload, verifies its digest, and returns
  `video` and `audio` channels plus text statements.
- The framework itself never dereferences paths, URLs or credentials.
