"""A MultiEcho workspace: experiment designs, runs, sealed reports, study locks and keyframe galleries.

Layout (every folder optional):

    <workspace>/workspace.json       {"title": ..., "plugins": [modules], "paths": [import paths, relative]}
    <workspace>/experiments/*.json   experiment designs (multiecho-ui.design/0.1)
    <workspace>/runs/*.json          run records (multiecho-ui.run/0.1)
    <workspace>/reports/*.json       sealed reports (multiecho-report)
    <workspace>/studies/*.json       study locks (multiecho-study)
    <workspace>/keyframes/*.npz      keyframe galleries (multiecho-keyframes)

Files are addressed by name only. The workbench writes designs, runs, reports produced by runs, galleries and
study locks; nothing outside these folders is read or written.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
from multiecho_framework import StudyLock, compare_reports, dumps_study, load_gallery, loads_study
from multiecho_framework.conformance import check_report
from multiecho_framework.experiment import loads_report
from multiecho_framework.keyframes import keyframe_difference
from multiecho_framework.study import DEFAULT_LOCKED, locked_view

from .design import SCHEMA as DESIGN_SCHEMA
from .png import encode_png, enlarge, heatmap, to_uint8

FOLDERS = {"experiments": "*.json", "runs": "*.json", "reports": "*.json", "studies": "*.json", "keyframes": "*.npz"}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


class NotFound(KeyError):
    pass


class Workspace:
    def __init__(self, root: str | Path, gallery_cache: int = 2):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Workspace {self.root} is not a directory")
        self._lock = threading.Lock()
        self._reports: dict[str, tuple[float, dict]] = {}
        self._galleries: dict[str, tuple[float, object, dict]] = {}
        self._gallery_cache = gallery_cache

    # ---------------------------------------------------------------- files
    def _files(self, folder: str) -> dict[str, Path]:
        directory = self.root / folder
        if not directory.is_dir():
            return {}
        return {path.name: path for path in sorted(directory.glob(FOLDERS[folder])) if path.is_file()}

    def _path(self, folder: str, name: str) -> Path:
        files = self._files(folder)
        if name not in files:
            raise NotFound(f"{folder}/{name}")
        return files[name]

    def settings(self) -> dict:
        """`workspace.json`: title, plugin modules and import paths (resolved against the workspace)."""
        meta = self.root / "workspace.json"
        settings = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        plugins = settings.get("plugins") or []
        paths = settings.get("paths") or []
        if not isinstance(plugins, list) or not all(isinstance(item, str) for item in plugins):
            raise ValueError("workspace.json: 'plugins' is a list of module names")
        if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
            raise ValueError("workspace.json: 'paths' is a list of directories")
        return {"title": settings.get("title") or self.root.name, "plugins": tuple(plugins),
                "paths": tuple((self.root / item).resolve() for item in paths)}

    def info(self) -> dict:
        settings = self.settings()
        return {"title": settings["title"], "root": str(self.root), "plugins": list(settings["plugins"]),
                **{folder: len(self._files(folder)) for folder in FOLDERS}}

    # ------------------------------------------------------------- designs
    @staticmethod
    def _name(name: str) -> str:
        stem = name.removesuffix(".json")
        if not NAME.fullmatch(stem):
            raise ValueError(f"A name uses letters, digits, '.', '_' and '-', not {name!r}")
        return stem

    def designs(self) -> list[dict]:
        out = []
        for name, path in self._files("experiments").items():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                out.append({"name": name.removesuffix(".json"), "error": str(error)})
                continue
            chain = document.get("chain") or {}
            design = document.get("design") or {}
            out.append({"name": name.removesuffix(".json"), "description": document.get("description", ""),
                        "experiment_id": design.get("experiment_id"), "question": design.get("question"),
                        "worlds": [world.get("id") or (world.get("plugin") or "").split(":")[-1]
                                   for world in chain.get("worlds") or []],
                        "masker": ((chain.get("analyst") or {}).get("masker") or {}).get("plugin"),
                        "tasks": design.get("tasks") or [], "modified": path.stat().st_mtime,
                        "error": None if document.get("schema") == DESIGN_SCHEMA else f"not a {DESIGN_SCHEMA} design"})
        return out

    def design(self, name: str) -> dict:
        return json.loads(self._path("experiments", f"{self._name(name)}.json").read_text(encoding="utf-8"))

    def save_design(self, name: str, document: dict) -> str:
        stem = self._name(name)
        if not isinstance(document, dict) or document.get("schema") != DESIGN_SCHEMA:
            raise ValueError(f"A design is a {DESIGN_SCHEMA} document")
        folder = self.root / "experiments"
        folder.mkdir(exist_ok=True)
        document = {**document, "name": stem}
        (folder / f"{stem}.json").write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return stem

    def delete_design(self, name: str) -> None:
        self._path("experiments", f"{self._name(name)}.json").unlink()

    def create_lock(self, report: str, study_id: str, fields: tuple[str, ...] = DEFAULT_LOCKED,
                    description: str = "") -> str:
        """A study lock from a report's declaration, written to `studies/<study_id>.json`."""
        stem = self._name(study_id)
        target = self.root / "studies" / f"{stem}.json"
        if target.exists():
            raise ValueError(f"A study lock {stem!r} already exists")
        lock = StudyLock.from_declaration(stem, self._report(report)["report"], tuple(fields), description)
        target.parent.mkdir(exist_ok=True)
        target.write_text(dumps_study(lock), encoding="utf-8")
        return target.name

    def lock(self, name: str) -> StudyLock:
        return self.study(name)["lock"]

    # -------------------------------------------------------------- reports
    def _report(self, name: str) -> dict:
        path = self._path("reports", name)
        stamp = path.stat().st_mtime
        with self._lock:
            cached = self._reports.get(name)
            if cached and cached[0] == stamp:
                return cached[1]
        text = path.read_text(encoding="utf-8")
        try:
            report, sealed = loads_report(text), True
        except ValueError:
            report, sealed = json.loads(text), False
        checks = [{"id": r.check_id, "title": r.title, "passed": r.passed, "detail": r.detail}
                  for r in check_report(report)]
        entry = {"report": report, "sealed": sealed, "checks": checks}
        with self._lock:
            self._reports[name] = (stamp, entry)
        return entry

    def reports(self) -> list[dict]:
        return [self._summary(name, self._report(name)) for name in self._files("reports")]

    @staticmethod
    def _summary(name: str, entry: dict) -> dict:
        report, checks = entry["report"], entry["checks"]
        declaration = report.get("declaration", {})
        chain = declaration.get("chain", {})
        scope = chain.get("observer", {}).get("scope", {})
        keyframes = report.get("keyframes") or {}
        stimulus = declaration.get("stimulus")
        return {
            "name": name, "experiment_id": declaration.get("experiment_id"), "sealed": entry["sealed"],
            "question": declaration.get("question"), "handle": declaration.get("handle"),
            "study": (declaration.get("study") or {}).get("study_id"),
            "subjects": [w["world_id"] for w in chain.get("worlds", []) if w.get("kind") == "learned"],
            "worlds": [w["world_id"] for w in chain.get("worlds", [])],
            "masker": (scope.get("masker") or {}).get("masker_id", "none"),
            "stimulus": stimulus,
            "checks": {"passed": sum(c["passed"] for c in checks), "total": len(checks),
                       "failed": [c["id"] for c in checks if not c["passed"]]},
            "outcomes": {task: analysis["outcome"]["value"] for task, analysis in report.get("analyses", {}).items()},
            "keyframes": {"verdict": keyframes.get("verdict"), "mode": (keyframes.get("policy") or {}).get("mode"),
                          "warnings": len(keyframes.get("warnings") or []),
                          "max_rms": max((s["max_rms"] for s in (keyframes.get("summary") or {}).values()),
                                         default=None)} if keyframes else None,
            "digest": declaration.get("digest"),
        }

    def report(self, name: str) -> dict:
        entry = self._report(name)
        report = entry["report"]
        r2 = {task: {role: evaluation.get("descriptive_r2") for role, evaluation in analysis["evaluations"].items()}
              for task, analysis in report.get("analyses", {}).items()}
        declaration = report.get("declaration", {})
        roles: dict[str, int] = {}
        for context in declaration.get("contexts", []):
            roles[context.get("role")] = roles.get(context.get("role"), 0) + 1
        analyses = {task: {"outcome": analysis.get("outcome"), "evaluations": analysis.get("evaluations"),
                           "law": (analysis.get("candidate") or {}).get("law_id"),
                           "hypothesis": (analysis.get("candidate") or {}).get("hypothesis"),
                           "claim": (analysis.get("claims") or [{}])[0].get("statement")}
                    for task, analysis in report.get("analyses", {}).items()}
        return {"summary": self._summary(name, entry), "checks": entry["checks"], "r2": r2, "analyses": analyses,
                "declaration": {key: declaration.get(key) for key in ("experiment_id", "question", "handle",
                                                                       "observation_time", "independence_unit",
                                                                       "tasks", "draws", "coupling")} | {"contexts": roles},
                "locked": locked_view(report["declaration"], DEFAULT_LOCKED),
                "keyframes": report.get("keyframes"), "masking": _masking(report.get("masking")),
                "response": report.get("response")}

    def compare(self, a: str, b: str, study: str | None = None) -> dict:
        fields = self.study(study)["lock"].fields if study else DEFAULT_LOCKED
        result = compare_reports(self._report(a)["report"], self._report(b)["report"], fields)
        return {**result, "names": {"a": a, "b": b}, "lock": study}

    # -------------------------------------------------------------- studies
    def study(self, name: str) -> dict:
        lock = loads_study(self._path("studies", name).read_text(encoding="utf-8"))
        return {"lock": lock}

    def studies(self) -> list[dict]:
        out = []
        for name in self._files("studies"):
            try:
                lock = self.study(name)["lock"]
            except ValueError as error:
                out.append({"name": name, "error": str(error)})
                continue
            members = []
            for report in self._files("reports"):
                declaration = self._report(report)["report"]["declaration"]
                differences = lock.differences(declaration)
                members.append({"report": report, "belongs": not differences,
                                "declares": (declaration.get("study") or {}).get("study_id"),
                                "different": [row["field"] for row in differences]})
            out.append({"name": name, "study_id": lock.study_id, "description": lock.description,
                        "digest": lock.digest, "fields": list(lock.fields), "locked": lock.locked,
                        "members": members})
        return out

    # ------------------------------------------------------------ galleries
    def _gallery(self, name: str):
        path = self._path("keyframes", name)
        stamp = path.stat().st_mtime
        with self._lock:
            cached = self._galleries.get(name)
            if cached and cached[0] == stamp:
                return cached[1], cached[2]
        gallery = load_gallery(path)
        index = _gallery_index(gallery)
        with self._lock:
            self._galleries[name] = (stamp, gallery, index)
            while len(self._galleries) > self._gallery_cache:
                self._galleries.pop(next(iter(self._galleries)))
        return gallery, index

    def galleries(self) -> list[dict]:
        """Listing from each gallery's small members only; frames are not decompressed here."""
        out = []
        for name, path in self._files("keyframes").items():
            try:
                with np.load(path, allow_pickle=False) as data:
                    meta = json.loads(str(data["meta"]))
                    labels = [str(x) for x in data["frame_labels"]]
                    ids = [str(x) for x in data["context_ids"]]
            except (ValueError, KeyError, OSError) as error:
                out.append({"name": name, "error": str(error)})
                continue
            out.append({"name": name, "meta": meta, "frame_labels": labels, "contexts": len(set(ids)),
                        "branches": len(ids)})
        return out

    def gallery(self, name: str) -> dict:
        return {"name": name, **self._gallery(name)[1]}

    def image(self, name: str, branch: int, frame: int, kind: str = "frame", vmax: float | None = None) -> bytes:
        gallery, index = self._gallery(name)
        if not 0 <= branch < len(gallery.context_ids) or not -len(gallery.frame_labels) <= frame < len(gallery.frame_labels):
            raise NotFound(f"branch {branch} frame {frame}")
        image = gallery.frames[branch, frame]
        if kind == "frame":
            picture = to_uint8(image)
        elif kind == "diff":
            reference = index["reference_of"][branch]
            if reference is None:
                raise NotFound(f"no reference branch for branch {branch}")
            difference = np.abs(image.astype(np.float64) - gallery.frames[reference, frame].astype(np.float64))
            difference = difference.max(axis=-1) if difference.ndim == 3 else difference
            if vmax is None:
                vmax = index["frame_max_abs"][frame]
            picture = heatmap(difference, vmax)
        else:
            raise NotFound(f"image kind {kind!r}")
        return encode_png(enlarge(picture))


def _masking(masking: dict | None) -> dict | None:
    if not masking:
        return None
    return {"rule": masking.get("rule"), "frozen": masking.get("frozen"), "coverage": masking.get("coverage")}


def _gallery_index(gallery) -> dict:
    """Per-context branches sorted along the counterfactual axis, with per-frame differences to the reference."""
    count, frames = len(gallery.context_ids), len(gallery.frame_labels)
    references = {(c, s): i for i, (c, s, r) in enumerate(zip(gallery.context_ids, gallery.seeds, gallery.reference,
                                                              strict=True)) if r}
    reference_of = [references.get((c, s)) for c, s in zip(gallery.context_ids, gallery.seeds, strict=True)]
    differences = np.full((count, frames, 2), np.nan)
    for i, j in enumerate(reference_of):
        if j is not None:
            for f in range(frames):
                differences[i, f] = keyframe_difference(gallery.frames[i, f], gallery.frames[j, f])
    contexts: dict[str, dict] = {}
    for i, (cid, role) in enumerate(zip(gallery.context_ids, gallery.roles, strict=True)):
        entry = contexts.setdefault(cid, {"context_id": cid, "role": role, "branches": []})
        entry["branches"].append({
            "index": i, "values": gallery.values[i].tolist(), "seed": gallery.seeds[i],
            "reference": gallery.reference[i],
            "rms": [None if np.isnan(v) else float(v) for v in differences[i, :, 0]],
            "max_abs": [None if np.isnan(v) else float(v) for v in differences[i, :, 1]]})
    for entry in contexts.values():
        entry["branches"].sort(key=lambda b: (b["values"], b["seed"]))
        rows = [b for b in entry["branches"] if not b["reference"]]
        entry["max_rms"] = [max((b["rms"][f] or 0.0 for b in rows), default=0.0) for f in range(frames)]
    frame_max_abs = [float(np.nanmax(differences[:, f, 1])) if np.isfinite(differences[:, f, 1]).any() else 0.0
                     for f in range(frames)]
    return {"meta": gallery.meta, "frame_labels": list(gallery.frame_labels),
            "shape": list(gallery.frames.shape[2:]), "branches": count,
            "contexts": sorted(contexts.values(), key=lambda e: (e["role"], e["context_id"])),
            "reference_of": reference_of, "frame_max_abs": frame_max_abs}
