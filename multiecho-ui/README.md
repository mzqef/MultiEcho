# multiecho-ui

A local workbench for running MultiEcho experiments. Build an experiment from the plugins
in your workspace, declare it, run it in the order the protocol enforces, and inspect the
sealed reports, comparisons, study locks and keyframes it produces. It adds no dependency
to the framework: the server is the Python standard library plus NumPy, and the page is
plain HTML and JavaScript.

```console
python -m pip install -e ../multiecho -e .
multiecho-ui --workspace examples/workspace        # http://127.0.0.1:8765/
```

The example workspace builds on the framework's example plugins (a spring scene, two
engines, an appearance sampler, two tiny learned models and generic analysis methods) and
holds two designs: `spring-latents` (the quickstart experiment) and `spring-frame-masked`
(a next-frame predictor observed through a masker frozen on discovery). Open one, press
**Check design**, then **Run**.

## How an experiment runs

1. **Build.** Compose a chain of worlds from the plugin catalog: a physical scene (its
   handle axes are what an intervention sets), engines, samplers and learned models,
   joined by interfaces (parameter bridges, scopes, translator plugins). The analyst
   observes the last world through a scope, optionally masked by a masker. Then declare
   the experiment: question, reference / calibration / query values, contexts drawn from
   declared factors, tasks, draws, statistics, sweep, keyframe policy and study lock.
   **Check design** builds the chain and the declaration without measuring anything and
   traces one discovery branch through every world and interface.
2. **Measure discovery.** Every discovery context is measured at the reference and at
   each declared value; determinism is checked; a data-driven masker is calibrated and
   frozen. The run page plots the effect size along the axis per context and the effects
   in their two main directions.
3. **Freeze laws.** Choose one analysis method per task (forward: predict the response;
   backward: recover the intervention), fit it on discovery responses only, and freeze
   it. A frozen law cannot be refitted.
4. **Test held-out contexts.** Validation and confirmation contexts are measured only now;
   each frozen law is judged against fixed controls. The run page shows the evaluation of
   every stage and condition, recovered against true values, and predicted against
   observed responses.
5. **Seal.** The report is sealed, written to `reports/`, and checked by the 22
   conformance checks; kept frames are written to `keyframes/`.

The framework enforces this order and the workbench reports what it refuses. **Run every
remaining stage** runs the rest with the methods planned in the design.

## Workspace

```
<workspace>/workspace.json      {"title": ..., "plugins": [modules], "paths": [import paths]}
<workspace>/experiments/*.json  experiment designs            (multiecho-ui.design/0.1)
<workspace>/runs/*.json         run records                   (multiecho-ui.run/0.1)
<workspace>/reports/*.json      sealed reports                (dumps_report)
<workspace>/studies/*.json      study locks                   (dumps_study / multiecho-conformance lock)
<workspace>/keyframes/*.npz     keyframe galleries            (save_gallery)
```

`paths` are relative to the workspace and added to the import path. Reports, locks and
galleries written by scripts (for example `python ../multiecho/examples/study.py
--workspace DIR`) appear alongside those of the workbench's runs.

## Plugins

A plugin module lists its factories in `MULTIECHO_PLUGINS`, a sequence of
`(kind, factory)` pairs:

```python
MULTIECHO_PLUGINS = (
    ("world", MyEngine),            # engines, samplers, learned models
    ("interface", MyTranslator),    # translators between two worlds
    ("masker", MyMasker),
    ("forward", MyForwardMethod),   # fit(ResponseSet) -> ForwardLaw
    ("backward", MyBackwardMethod), # fit(ResponseSet) -> BackwardLaw
)
```

A factory is a class or a function. Keyword parameters with a default of type `bool`,
`int`, `float` or `str`, or a tuple of numbers, become form fields; other parameters keep
their default. The workbench fills `source`, `target` and `interface_id` of an interface
from the chain, and `grouping` of a masker from the analyst scope (the observation split
into locations of a declared number of values). The framework's maskers are always
available. See [`../multiecho/examples/plugins/catalog.py`](../multiecho/examples/plugins/catalog.py)
and [`../multiecho/docs/plugins.md`](../multiecho/docs/plugins.md). Restart the workbench
after changing plugin code.

## Designs

A design (`multiecho-ui.design/0.1`) has three parts:

- `chain`: `worlds` in order (`{"type": "physical", "id", "revision", "description",
  "handle": {"id", "semantics", "axes": [{"name", "unit", "lower", "upper"}]}}` or
  `{"type": "plugin", "plugin": "module:name", "params": {...}}`), one interface between
  each pair of consecutive worlds (`bridge` with `maps` and `constants`, `scope` with a
  `scope`, or `plugin`), `handle_world` (the world taking the interventions), `analyst`
  (`scope` and an optional `masker` with `width`, the values per location), `observation`
  and `stimulus`. A scope declares `channels` and optionally `window`, `clip`, `keys`,
  `coordinates` and a box `region`.
- `design`: `experiment_id`, `question`, `handle`, `observation_time`,
  `independence_unit`, `reference`, `calibration`, `queries` (one number per handle axis),
  `steps`, `contexts` (`seed`, `counts` per role, `factors` with a distribution — `fixed`,
  `uniform`, `signed-uniform`, `log-uniform`, `choice`, `cycle` — and an optional `when`
  gate on an earlier factor, and `conditions`), `tasks`, `draws`, `repeats`,
  `statistics`, `sweep`, `keyframes`, `keep_frames` and `study` (a lock in `studies/`).
- `methods`: the planned method per task.

Designs are edited in forms or as JSON. Every build error names the part of the design it
belongs to; the framework's own validation messages are shown unchanged.

## Results

- **Reports**: every sealed report with its chain, masker, stimulus, study, outcomes,
  conformance checks (re-run when opened; a report whose seal does not match its content
  is marked) and keyframe verdict; **Compare** tells whether two reports are comparable
  under the default locked fields or a study lock's.
- **Studies**: create a study lock from a report's declaration and see which reports
  belong to each lock.
- **Keyframes**: the frames a learned world received, per context and branch, with
  differences from the reference branch.

## Safety

The server binds to 127.0.0.1 by default. Reading is `GET`; every action is a JSON
`POST` accepted only when the `Host` is this server and any `Origin` matches it. Plugins
run in the server process and come only from the modules listed in `workspace.json`;
designs refer to plugins by catalog identity. Files are read and written only in the
workspace folders above, by name.
