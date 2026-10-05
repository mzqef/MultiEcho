# multiecho-ui

A local, read-only web viewer for MultiEcho workspaces. It adds no dependency to the
framework: the server is the Python standard library plus NumPy, and the page is plain
HTML and JavaScript.

```console
python -m pip install -e ../multiecho -e .
python ../multiecho/examples/study.py --workspace demo-workspace
multiecho-ui --workspace demo-workspace          # http://127.0.0.1:8765/
```

## Workspace

```
<workspace>/workspace.json      {"title": "..."}            (optional)
<workspace>/reports/*.json      sealed reports              (dumps_report)
<workspace>/studies/*.json      study locks                 (dumps_study / multiecho-conformance lock)
<workspace>/keyframes/*.npz     keyframe galleries          (save_gallery; Experiment(keep_frames=...))
```

## Pages

- **Reports**: every report with its study, subject, masker, stimulus, outcomes and
  keyframe verdict. All 22 conformance checks are re-run when a report is loaded; a
  report whose seal does not match its content is marked.
- **Compare**: two reports field by field. Locked protocol fields (stimulus,
  interventions, contexts, observation time, keyframe policy, ...) must agree for the
  experiments to be comparable; subjects, maskers and methods are listed as what
  varies. Optionally compare the fields of a chosen study lock.
- **Studies**: each study lock, its locked values, and which reports belong to it
  (and in which fields the others differ).
- **Keyframes**: for each context, the branches' stimulus frames ordered along the
  counterfactual axis, shown as frames or as the absolute difference to the context's
  reference branch (same draw, one colour scale per frame). A chart plots each
  branch's difference against the intervention value for all contexts, with the
  policy's warning and tolerance lines. Clicking a branch shows reference, branch and
  difference side by side.

The server binds to 127.0.0.1 by default, reads only the four locations above, and
never writes.
