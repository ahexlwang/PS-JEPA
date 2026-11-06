"""Compatibility import path for PS-JEPA; new code should import psjepa."""
import importlib
import importlib.abc
import importlib.util
import sys
from psjepa import __version__

class _AliasLoader(importlib.abc.Loader):
    def __init__(self, name, target):
        self.name, self.target = name, target
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        sys.modules[self.name] = importlib.import_module(self.target)

class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("biojepa."):
            return None
        canonical = "psjepa" + fullname[len("biojepa"):]
        canonical = canonical.replace("psjepa.models.biojepa", "psjepa.models.psjepa")
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(fullname, canonical),
            is_package=spec.submodule_search_locations is not None)

sys.meta_path.insert(0, _AliasFinder())
