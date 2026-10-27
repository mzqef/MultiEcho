"""The plugin catalog: the worlds, interfaces, maskers and analysis methods a workspace may build experiments from.

Plugin modules are listed in `workspace.json` (`"plugins": [...]`, importable from `"paths": [...]`, relative to
the workspace). Each module defines `MULTIECHO_PLUGINS`, a sequence of `(kind, factory)` pairs:

    MULTIECHO_PLUGINS = (("world", MyEngine), ("forward", MyForwardMethod))

Kinds: `world`, `interface`, `masker`, `forward`, `backward`. A factory is a class or a function. Its keyword
parameters with a default of type bool, int, float or str, or a tuple of numbers, are editable; other
parameters keep their default. The workbench fills `source`, `target` and `interface_id` of an interface (the
adjacent worlds and the interface's identity) and `grouping` of a masker (from the analyst scope). The
framework's maskers are always available. Only factories in the catalog can be instantiated; designs refer
to them by identity (`module:name`).
"""
from __future__ import annotations

import importlib
import inspect
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from multiecho_framework import EnergyMask, ManualMask, MonotonicSNRMask, NoMask, RepeatNoiseSNRMask

KINDS = ("world", "interface", "masker", "forward", "backward")
AUTOMATIC = {"interface": ("source", "target", "interface_id"), "masker": ("grouping",)}
BUILTIN = (("masker", NoMask), ("masker", MonotonicSNRMask), ("masker", RepeatNoiseSNRMask), ("masker", EnergyMask),
           ("masker", ManualMask))
#: ManualMask's `selected` (a boolean array over the grouping's locations) is entered as index ranges, e.g. "0-63, 70".
SELECTION = {("multiecho_framework.masking:ManualMask", "selected")}


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str          # bool, int, float, str, numbers, selection, auto, fixed
    default: object = None
    required: bool = False
    annotation: str = ""

    def record(self) -> dict:
        return {"name": self.name, "type": self.type, "default": _plain(self.default), "required": self.required,
                "annotation": self.annotation}


@dataclass(frozen=True)
class Entry:
    plugin_id: str
    kind: str
    factory: object = field(repr=False, compare=False)
    module: str = ""
    name: str = ""
    summary: str = ""
    doc: str = ""
    parameters: tuple[Parameter, ...] = ()
    builtin: bool = False

    def record(self) -> dict:
        return {"id": self.plugin_id, "kind": self.kind, "module": self.module, "name": self.name,
                "summary": self.summary, "doc": self.doc, "builtin": self.builtin,
                "parameters": [parameter.record() for parameter in self.parameters]}


def _plain(value):
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return repr(value)


def _type_of(plugin_id: str, kind: str, parameter: inspect.Parameter) -> tuple[str, object, bool]:
    if parameter.name in AUTOMATIC.get(kind, ()):
        return "auto", None, False
    if (plugin_id, parameter.name) in SELECTION:
        return "selection", "", True
    required = parameter.default is inspect.Parameter.empty
    default = None if required else parameter.default
    annotation = parameter.annotation if isinstance(parameter.annotation, str) else getattr(
        parameter.annotation, "__name__", str(parameter.annotation))
    text = str(annotation).replace("typing.", "")
    for name, kind_ in (("bool", "bool"), ("int", "int"), ("float", "float"), ("str", "str")):
        if text.split("|")[0].strip() == name:
            return kind_, default, required
    if isinstance(default, bool):
        return "bool", default, False
    if isinstance(default, int):
        return "int", default, False
    if isinstance(default, float):
        return "float", default, False
    if isinstance(default, str):
        return "str", default, False
    if isinstance(default, tuple) and default and all(isinstance(v, int | float) and not isinstance(v, bool)
                                                       for v in default):
        return "numbers", default, False
    if text.startswith("tuple[float") or text.startswith("tuple[int"):
        return "numbers", default, required
    return ("fixed", default, False) if not required else ("str", None, True)


def _entry(kind: str, factory, builtin: bool = False) -> Entry:
    if kind not in KINDS:
        raise CatalogError(f"Unknown plugin kind {kind!r}; use one of {KINDS}")
    if not callable(factory):
        raise CatalogError(f"A {kind} plugin must be a class or a function, not {factory!r}")
    module, name = factory.__module__, factory.__qualname__
    plugin_id = f"{module}:{name}"
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None
    parameters = []
    for parameter in (signature.parameters.values() if signature else ()):
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        type_, default, required = _type_of(plugin_id, kind, parameter)
        annotation = parameter.annotation
        parameters.append(Parameter(parameter.name, type_, default, required,
                                    "" if annotation is inspect.Parameter.empty else str(getattr(annotation, "__name__", annotation))))
    doc = inspect.getdoc(factory) or ""
    if not doc or doc.startswith(f"{factory.__name__}("):   # no docstring (a dataclass documents its signature)
        doc = (inspect.getdoc(sys.modules.get(module)) or "").split("\n\n")[0]
    return Entry(plugin_id, kind, factory, module, name, doc.split("\n\n")[0].replace("\n", " "), doc,
                 tuple(parameters), builtin)


def _selection(text: str, groups: int) -> np.ndarray:
    """Index ranges ("0-63, 70") → a boolean selection over `groups` locations."""
    selected = np.zeros(groups, dtype=bool)
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        try:
            start, stop = int(low), int(high or low)
        except ValueError as error:
            raise CatalogError(f"Location ranges are integers such as '0-63, 70', not {part!r}") from error
        if not 0 <= start <= stop < groups:
            raise CatalogError(f"Location range {part!r} lies outside 0-{groups - 1}")
        selected[start:stop + 1] = True
    if not selected.any():
        raise CatalogError("Select at least one location")
    return selected


class Catalog:
    """Plugins by identity, loaded from the workspace's plugin modules plus the framework's maskers."""

    def __init__(self, modules: tuple[str, ...] = (), paths: tuple[Path, ...] = ()):
        self.modules, self.paths = tuple(modules), tuple(paths)
        self.entries: dict[str, Entry] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            for path in self.paths:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            entries, errors = {}, {}
            for kind, factory in BUILTIN:
                entry = _entry(kind, factory, builtin=True)
                entries[entry.plugin_id] = entry
            for module_name in self.modules:
                try:
                    module = importlib.import_module(module_name)
                    listed = getattr(module, "MULTIECHO_PLUGINS", None)
                    if listed is None:
                        raise CatalogError(f"{module_name} defines no MULTIECHO_PLUGINS")
                    for item in listed:
                        if not isinstance(item, tuple) or len(item) != 2:
                            raise CatalogError(f"{module_name}: MULTIECHO_PLUGINS holds (kind, factory) pairs")
                        entry = _entry(*item)
                        entries[entry.plugin_id] = entry
                except Exception as error:  # noqa: BLE001 - a broken plugin module is reported, not fatal
                    errors[module_name] = f"{type(error).__name__}: {error}"
            self.entries, self.errors = entries, errors

    def records(self) -> dict:
        return {"plugins": [entry.record() for entry in self.entries.values()], "errors": self.errors,
                "modules": list(self.modules)}

    def entry(self, plugin_id: str, kind: str | None = None) -> Entry:
        entry = self.entries.get(plugin_id)
        if entry is None:
            raise CatalogError(f"Plugin {plugin_id!r} is not in the workspace catalog")
        if kind is not None and entry.kind != kind:
            raise CatalogError(f"Plugin {plugin_id!r} is a {entry.kind}, not a {kind}")
        return entry

    def build(self, plugin_id: str, kind: str, params: dict | None = None, **automatic):
        """Instantiate a catalog plugin with the given editable parameters and the workbench's automatic ones."""
        entry = self.entry(plugin_id, kind)
        params = dict(params or {})
        arguments = {}
        for parameter in entry.parameters:
            if parameter.type == "auto":
                if parameter.name in automatic:
                    arguments[parameter.name] = automatic[parameter.name]
                continue
            if parameter.type == "fixed":
                if parameter.name in params:
                    raise CatalogError(f"{entry.name}.{parameter.name} cannot be set from the workbench")
                continue
            if parameter.name not in params:
                if parameter.required:
                    raise CatalogError(f"{entry.name} needs a value for {parameter.name}")
                continue
            value = params.pop(parameter.name)
            arguments[parameter.name] = self._value(entry, parameter, value, automatic)
        if params:
            raise CatalogError(f"{entry.name} has no parameter {', '.join(sorted(params))}")
        try:
            return entry.factory(**arguments)
        except Exception as error:  # noqa: BLE001 - the plugin's own validation message is the useful one
            raise CatalogError(f"{entry.name}: {error}") from error

    @staticmethod
    def _value(entry: Entry, parameter: Parameter, value, automatic: dict):
        name = f"{entry.name}.{parameter.name}"
        try:
            if parameter.type == "bool":
                if not isinstance(value, bool):
                    raise TypeError
                return value
            if parameter.type == "int":
                if isinstance(value, bool) or float(value) != int(float(value)):
                    raise TypeError
                return int(float(value))
            if parameter.type == "float":
                if isinstance(value, bool):
                    raise TypeError
                return float(value)
            if parameter.type == "str":
                if not isinstance(value, str):
                    raise TypeError
                return value
            if parameter.type == "numbers":
                items = value if isinstance(value, list | tuple) else [v for v in str(value).replace(",", " ").split()]
                return tuple(float(item) for item in items)
            if parameter.type == "selection":
                grouping = automatic.get("grouping")
                if grouping is None:
                    raise CatalogError(f"{name} needs the analyst scope's grouping")
                return _selection(value, grouping.groups)
        except (TypeError, ValueError) as error:
            raise CatalogError(f"{name} expects a {parameter.type}, not {value!r}") from error
        raise CatalogError(f"{name} cannot be set from the workbench")
