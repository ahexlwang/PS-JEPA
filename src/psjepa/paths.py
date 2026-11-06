'Where this checkout keeps its weights, data caches and run outputs.'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = [
    "project_root",
    "weights_root",
    "cache_root",
    "outputs_root",
    "resolve_weights",
    "resolve_cache",
    "resolve_output",
    "file_sha256",
    "resolve_recorded_checkpoint",
]


def project_root() -> Path:
    """The checkout root, found from this file rather than from the process's cwd."""
    return Path(__file__).resolve().parents[2]


def _root(variable: str, default: str) -> Path:
    value = os.environ.get(variable) or os.environ.get(variable.replace("PSJEPA_", "BIOJEPA_", 1))
    return Path(value).expanduser().resolve() if value else project_root() / default


def weights_root() -> Path:
    return _root("PSJEPA_WEIGHTS", "weights")


def cache_root() -> Path:
    return _root("PSJEPA_CACHE", "cache")


def outputs_root() -> Path:
    return _root("PSJEPA_OUTPUTS", "outputs")


def _resolve(root: Path, path: str | os.PathLike[str]) -> Path:
    """An absolute path is taken as given; a relative one is read under ``root``."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def resolve_weights(path: str | os.PathLike[str]) -> Path:
    return _resolve(weights_root(), path)


def resolve_cache(path: str | os.PathLike[str]) -> Path:
    return _resolve(cache_root(), path)


def resolve_output(path: str | os.PathLike[str]) -> Path:
    return _resolve(outputs_root(), path)


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_recorded_checkpoint(
    recorded: str | os.PathLike[str], expected_sha256: str | None = None
) -> Path:
    'Find a checkpoint that another checkpoint recorded a path to.'
    recorded_path = Path(recorded).expanduser()
    if recorded_path.is_file():
        return recorded_path

    candidates = [weights_root() / recorded_path.name]
    candidates += sorted(weights_root().glob(f"*/{recorded_path.name}"))
    found = [candidate for candidate in candidates if candidate.is_file()]

    if not found:
        raise FileNotFoundError(
            f"the checkpoint recorded its upstream as {recorded_path}, which does not exist, "
            f"and no file named {recorded_path.name} is under {weights_root()}. "
            "Set PSJEPA_WEIGHTS, or pass the upstream explicitly."
        )
    if expected_sha256 is None:
        raise RuntimeError(
            f"the checkpoint recorded its upstream as {recorded_path}, which does not exist. "
            f"{found[0]} has the same name, but the checkpoint records no sha256 to check it "
            "against, so it is not used: an encoder that merely shares a filename would give "
            "the decoder tokens it never saw. Pass the upstream explicitly."
        )
    for candidate in found:
        if file_sha256(candidate) == expected_sha256:
            return candidate
    raise RuntimeError(
        f"the checkpoint recorded its upstream as {recorded_path} with sha256 "
        f"{expected_sha256}. Nothing under {weights_root()} named {recorded_path.name} "
        f"matches that hash (checked {len(found)}), so none of them is the encoder this "
        "model was trained on."
    )
