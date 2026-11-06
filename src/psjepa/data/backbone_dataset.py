from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import center_on_ca, random_rotation


ATOM_INDEX = {"N": 0, "CA": 1, "C": 2, "O": 3}


def chain_split_indices(
    pdb_ids,
    chain_ids,
    selection_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    'Stable chain-grouped selection/final split for downstream evaluation.'
    fraction = float(selection_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("selection_fraction must be strictly between zero and one")
    if len(pdb_ids) != len(chain_ids):
        raise ValueError("pdb_ids and chain_ids must have the same length")
    selection, final = [], []
    denominator = float(1 << 64)
    for index, (pdb_id, chain_id) in enumerate(zip(pdb_ids, chain_ids)):
        key = f"{int(seed)}\0{str(pdb_id)}\0{str(chain_id)}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / denominator
        (selection if value < fraction else final).append(index)
    if not selection or not final:
        raise ValueError("chain hash split produced an empty partition")
    return selection, final


class BackboneDataset(Dataset):
    """Memory-mapped compact CATH backbone cache, one item per protein chain."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        max_residues: int = 512,
        augment: bool = False,
        limit: int | None = None,
        length_range: tuple[int, int] | None = None,
    ) -> None:
        root = Path(cache_root) / split
        required = [root / "coords.npy", root / "atom_mask.npy", root / "offsets.npy", root / "metadata.npz"]
        if not all(path.exists() for path in required):
            raise FileNotFoundError(
                f"Missing backbone cache under {root}. Run scripts/build_cache.py first."
            )
        self.coords = np.load(root / "coords.npy", mmap_mode="r")
        self.atom_mask = np.load(root / "atom_mask.npy", mmap_mode="r")
        residue_type_path = root / "residue_type.npy"
        self.residue_type = np.load(residue_type_path, mmap_mode="r") if residue_type_path.exists() else None
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r")
        meta = np.load(root / "metadata.npz", allow_pickle=False)
        # Two cache layouts exist. The PDB/CATH builders store `pdb_id` and
        # `chain_id`; the domain cache builder stores a single AlphaFold model
        # `name` instead. Identity is only carried through for reporting, so a
        # cache with either layout is usable.
        if "pdb_id" in meta:
            self.pdb_id = meta["pdb_id"]
            self.chain_id = meta["chain_id"]
        else:
            self.pdb_id = meta["name"]
            self.chain_id = np.full(len(self.pdb_id), "A")
        self.label = meta["label"]
        self.max_residues = max_residues
        self.augment = augment
        total = len(self.offsets) - 1


        self.index: np.ndarray | None = None
        if length_range is not None:
            low, high = int(length_range[0]), int(length_range[1])
            lengths = np.diff(np.asarray(self.offsets))
            keep = np.nonzero((lengths >= low) & (lengths <= high))[0]
            if keep.size == 0:
                raise ValueError(f"no chain in {root} has {low}-{high} residues")
            self.index = keep
            total = int(keep.size)
        self.length = min(total, limit) if limit is not None else total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.index is not None:
            index = int(self.index[index])
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        coords = torch.from_numpy(np.array(self.coords[start:end], copy=True))
        atom_mask = torch.from_numpy(np.array(self.atom_mask[start:end], copy=True)).bool()
        residue_type = (
            torch.from_numpy(np.array(self.residue_type[start:end], copy=True)).long()
            if self.residue_type is not None
            else torch.full((end - start,), 20, dtype=torch.long)
        )
        n = len(coords)
        if n > self.max_residues:
            crop = int(torch.randint(n - self.max_residues + 1, (1,))) if self.augment else (n - self.max_residues) // 2
            coords = coords[crop : crop + self.max_residues]
            atom_mask = atom_mask[crop : crop + self.max_residues]
            residue_type = residue_type[crop : crop + self.max_residues]
        coords = center_on_ca(coords, atom_mask)
        if self.augment:
            coords = torch.einsum("rad,df->raf", coords, random_rotation())
        return {
            "coords": coords.float(),
            "atom_mask": atom_mask,
            "residue_mask": torch.ones(len(coords), dtype=torch.bool),
            "residue_type": residue_type,
            "pdb_id": str(self.pdb_id[index]),
            "chain_id": str(self.chain_id[index]),
            "label": str(self.label[index]),
        }


def collate_backbones(items: list[dict[str, Any]]) -> dict[str, Any]:
    length = max(len(item["coords"]) for item in items)
    batch = len(items)
    atom_count = int(items[0]["coords"].shape[-2])
    if any(int(item["coords"].shape[-2]) != atom_count for item in items):
        raise ValueError("Cannot collate mixed atom layouts")
    coords = torch.zeros(batch, length, atom_count, 3)
    atom_mask = torch.zeros(batch, length, atom_count, dtype=torch.bool)
    residue_mask = torch.zeros(batch, length, dtype=torch.bool)
    residue_type = torch.full((batch, length), 20, dtype=torch.long)
    for i, item in enumerate(items):
        n = len(item["coords"])
        coords[i, :n] = item["coords"]
        atom_mask[i, :n] = item["atom_mask"]
        residue_mask[i, :n] = item["residue_mask"]
        residue_type[i, :n] = item["residue_type"]
    return {
        "coords": coords,
        "atom_mask": atom_mask,
        "residue_mask": residue_mask,
        "residue_type": residue_type,
        "pdb_id": [item["pdb_id"] for item in items],
        "chain_id": [item["chain_id"] for item in items],
        "label": [item["label"] for item in items],
    }
