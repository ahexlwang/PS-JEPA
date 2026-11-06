#!/usr/bin/env python3
"""Encode a protein backbone into one latent per residue, and show that it is invariant.

This is the smallest useful thing the released encoder does. It takes N/CA/C/O coordinates
and returns a vector per residue that downstream tasks read directly, with no residue
identity anywhere in the pipeline: the model never sees the sequence.

    python examples/encode_backbone.py                    # a synthetic helix
    python examples/encode_backbone.py --pdb 1ABC.pdb     # a real structure

The second half of the output is the check worth reading. The featuriser is invariant by
construction, so rotating and translating the input must not move the latents at all. If
that line ever reports a difference larger than floating-point noise, something is wrong
with the installation rather than with the protein.

Runs on CPU. The Mamba kernels prefer a GPU but do not require one for a single chain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from psjepa.models import PSJEPA
from psjepa.sota import encoder_checkpoint

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def load_encoder(path: Path, device: torch.device) -> PSJEPA:
    if not path.is_file():
        raise SystemExit(
            f"no weights at {path}\n"
            "The weights are distributed separately from the code; see README.md."
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    model = PSJEPA.from_dict(config["model"], config["masking"], config["loss"])
    model.load_state_dict(state["model"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def synthetic_helix(length: int = 60) -> np.ndarray:
    """An idealised alpha helix: 100 degrees and 1.5 A of rise per residue."""
    index = np.arange(length, dtype=np.float32)
    angle = np.deg2rad(100.0) * index
    ca = np.stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * index], axis=-1)
    offsets = np.array(
        [[-1.2, 0.0, -0.5], [0.0, 0.0, 0.0], [1.2, 0.3, 0.4], [1.3, 1.2, 1.1]],
        dtype=np.float32,
    )
    return ca[:, None, :] + offsets[None, :, :]


def read_pdb_backbone(path: Path) -> np.ndarray:
    """N/CA/C/O of the first chain, keeping only residues where all four are resolved."""
    residues: dict[tuple[str, int], dict[str, list[float]]] = {}
    order: list[tuple[str, int]] = []
    chain_wanted = None
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        chain = line[21]
        if chain_wanted is None:
            chain_wanted = chain
        if chain != chain_wanted:
            continue
        key = (chain, int(line[22:26]))
        if key not in residues:
            residues[key] = {}
            order.append(key)
        residues[key][atom] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    complete = [k for k in order if len(residues[k]) == 4]
    if not complete:
        raise SystemExit(f"no residue in {path} has all of N, CA, C and O")
    return np.array(
        [[residues[k][a] for a in BACKBONE_ATOMS] for k in complete], dtype=np.float32
    )


def encode(model: PSJEPA, coords: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(coords).unsqueeze(0).to(device)
    atom_mask = torch.ones(tensor.shape[:3], dtype=torch.bool, device=device)
    residue_mask = torch.ones(tensor.shape[:2], dtype=torch.bool, device=device)
    with torch.inference_mode():
        out = model.tokenize(tensor, atom_mask, residue_mask, None)
    return out["continuous_semantic"][0].float().cpu()


def random_rigid_motion(coords: np.ndarray, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:          # keep it a rotation, not a reflection
        rotation[:, 0] *= -1
    translation = generator.normal(scale=25.0, size=3)
    return (coords @ rotation.T + translation).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, help="a PDB file; omit for a synthetic helix")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    path = encoder_checkpoint()
    model = load_encoder(path, device)

    coords = read_pdb_backbone(args.pdb) if args.pdb else synthetic_helix()
    latents = encode(model, coords, device)

    source = str(args.pdb) if args.pdb else "synthetic alpha helix"
    print(f"input     {source}")
    print(f"weights   {path.name}")
    print(f"residues  {coords.shape[0]}")
    print(f"latents   {tuple(latents.shape)}  (one {latents.shape[1]}-d vector per residue)")
    print(f"norm      mean {latents.norm(dim=-1).mean():.3f}")

    moved = encode(model, random_rigid_motion(coords), device)
    difference = (latents - moved).abs().max().item()
    print()
    print("rotation and translation of the input:")
    print(f"  largest change in any latent component  {difference:.3e}")
    print("  the featuriser is invariant by construction, so this is floating-point noise")


if __name__ == "__main__":
    main()
