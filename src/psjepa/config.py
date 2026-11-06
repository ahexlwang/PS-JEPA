'Loading a run configuration, and turning the paths in it into absolute paths.'
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from psjepa.paths import resolve_cache, resolve_output, resolve_weights

# Config keys that name a file or directory, and the root each is read against.
CHECKPOINT_KEYS = (
    "upstream_checkpoint",
    "decoder_checkpoint",
    "prior_checkpoint",
    "codec_checkpoint",
    "init_checkpoint",
)
OUTPUT_KEYS = ("output_dir",)
CACHE_KEYS = (("data", "cache"),)


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the known path keys of ``config`` in place as absolute paths."""
    for key in CHECKPOINT_KEYS:
        if isinstance(config.get(key), str):
            config[key] = str(resolve_weights(config[key]))
    for key in OUTPUT_KEYS:
        if isinstance(config.get(key), str):
            config[key] = str(resolve_output(config[key]))
    for section, key in CACHE_KEYS:
        block = config.get(section)
        if isinstance(block, dict) and isinstance(block.get(key), str):
            block[key] = str(resolve_cache(block[key]))
    return config


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    include = config.pop("include", None)
    if include is None:
        return resolve_paths(config)
    includes = [include] if isinstance(include, str) else include
    merged: dict[str, Any] = {}
    for item in includes:
        merged = _merge(merged, load_config(path.parent / item))
    return resolve_paths(_merge(merged, config))


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

