'Export model tensors and configuration from a training checkpoint.'
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch

# Every key a released file may carry. Anything else in the source checkpoint is a
# resume artefact and is dropped, with its name recorded in `psjepa_export.dropped`.
WEIGHT_KEYS = ("model", "decoder", "prior", "codec", "bottleneck")
METADATA_KEYS = ("config", "prior_config", "step", "best_validation", "status")

# A run's terminal status lives in `status.json` beside its checkpoints. It is folded into the
# released file because that file is distributed on its own.
STATUS_FILE = "status.json"


LEGACY_FAMILIES = ("biojepa", "flexjepa", "bjepa")
CURRENT_FAMILY = "psjepa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def release_config(config: dict) -> dict:
    """Keep inference settings without local paths or training-run annotations."""
    result = deepcopy(config)
    for key in ("data", "training", "output_dir", "experiment"):
        result.pop(key, None)
    for key, value in list(result.items()):
        if isinstance(value, str) and key.endswith("_checkpoint"):
            result[key] = Path(value).name
    if result.get("model_family") in LEGACY_FAMILIES:
        result["model_family"] = CURRENT_FAMILY
    return result


def export(source: Path, dest: Path, keep: tuple[str, ...] = ()) -> dict:
    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise SystemExit(f"{source} is not a checkpoint dictionary")

    allowed = set(WEIGHT_KEYS) | set(METADATA_KEYS) | set(keep)
    released = {key: value for key, value in state.items() if key in allowed}
    previous_export = state.get("psjepa_export", state.get("biojepa_export", {}))
    dropped = sorted(
        {key for key in set(state) - set(released) if not key.endswith("_export")}
        | set(previous_export.get("dropped", []))
    )

    weights_present = [key for key in WEIGHT_KEYS if key in released]
    if not weights_present:
        raise SystemExit(
            f"{source} has none of {WEIGHT_KEYS}; it carries {sorted(state)}"
        )

    if "status" not in released:
        status_path = source.parent / STATUS_FILE
        if status_path.is_file():
            released["status"] = json.loads(status_path.read_text()).get("status", "")

    config = released.get("config")
    if isinstance(config, dict):
        released["config"] = release_config(config)

    parameters = sum(
        int(tensor.numel())
        for key in weights_present
        for tensor in released[key].values()
        if torch.is_tensor(tensor)
    )
    released["psjepa_export"] = {
        "source_sha256": sha256(source),
        "source_bytes": source.stat().st_size,
        "weight_keys": weights_present,
        "parameters": parameters,
        "dropped": dropped,
        "license": "MIT",
        "status": released.get("status"),
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(released, dest)
    return {
        "dest": str(dest),
        "dest_bytes": dest.stat().st_size,
        "dest_sha256": sha256(dest),
        **released["psjepa_export"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument(
        "--keep-key",
        action="append",
        default=[],
        help="an extra top-level key to carry into the released file",
    )
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.dest, tuple(args.keep_key)), indent=2))


if __name__ == "__main__":
    main()
