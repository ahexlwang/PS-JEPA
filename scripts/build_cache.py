#!/usr/bin/env python3
"""Build compact memory-mapped N/CA/C/O residue caches from CATH parquet."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ATOM_INDEX = {"N": 0, "CA": 1, "C": 2, "O": 3}
COLUMNS = ["pdb_id", "chain_id", "CATH", "atom_long", "residue_ids", "residue_types", "structure", "unknown_structure"]
AMINO_ACIDS = {name: index for index, name in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def batches(files: list[Path], batch_size: int = 128):
    for path in files:
        yield from pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=COLUMNS)


def ca_label(labels) -> str:
    if not labels:
        return "unknown"
    candidates = sorted(".".join(str(item).split(".")[:2]) for item in labels)
    return candidates[0]


def residue_count(residue_ids) -> int:
    return len(dict.fromkeys(int(item) for item in residue_ids))


def convert_row(row: dict, drop_unresolved: bool = False,
                split_at_breaks: bool = False) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    'Backbone coordinates for one chain, one row per residue.'
    residue_ids = np.asarray(row["residue_ids"], dtype=np.int64)
    ordered_ids = list(dict.fromkeys(int(item) for item in residue_ids))
    lookup = {residue_id: i for i, residue_id in enumerate(ordered_ids)}
    coords = np.zeros((len(ordered_ids), 4, 3), dtype=np.float32)
    mask = np.zeros((len(ordered_ids), 4), dtype=np.uint8)
    residue_type = np.full(len(ordered_ids), 20, dtype=np.uint8)
    xyz = np.asarray(row["structure"], dtype=np.float32).reshape(-1, 3)
    unknown = np.asarray(row["unknown_structure"], dtype=bool)
    for atom_i, (name, residue_id) in enumerate(zip(row["atom_long"], residue_ids)):
        target = lookup[int(residue_id)]
        raw_type = str(row["residue_types"][atom_i])
        residue_type[target] = AMINO_ACIDS.get(raw_type.removeprefix("AA_"), 20)
        slot = ATOM_INDEX.get(str(name).strip())
        if slot is None or unknown[atom_i] or not np.isfinite(xyz[atom_i]).all():
            continue
        if not mask[target, slot]:
            coords[target, slot] = xyz[atom_i]
            mask[target, slot] = 1
    if not drop_unresolved:
        return [(coords, mask, residue_type)]
    keep = mask.all(axis=1)
    if not split_at_breaks:
        return [(coords[keep], mask[keep], residue_type[keep])]
    # Every maximal run of kept residues is contiguous in the crystal by construction:
    # the only thing separating two runs is a residue that was removed.
    segments = []
    start = None
    for index in range(len(keep) + 1):
        inside = index < len(keep) and bool(keep[index])
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            piece = slice(start, index)
            segments.append((coords[piece], mask[piece], residue_type[piece]))
            start = None
    return segments


MINIMUM_RESIDUES = 4


def build_split(source: Path, destination: Path, force: bool, drop_unresolved: bool = False,
                split_at_breaks: bool = False) -> None:
    files = sorted(source.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {source}")
    if (destination / "metadata.npz").exists() and not force:
        print(f"[skip] {destination}")
        return
    rows = 0
    residues = 0
    for batch in batches(files):
        data = batch.to_pydict()
        rows += batch.num_rows
        residues += sum(residue_count(item) for item in data["residue_ids"])
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    coords_out = np.lib.format.open_memmap(temporary / "coords.npy", mode="w+", dtype=np.float32, shape=(residues, 4, 3))
    mask_out = np.lib.format.open_memmap(temporary / "atom_mask.npy", mode="w+", dtype=np.uint8, shape=(residues, 4))
    residue_type_out = np.lib.format.open_memmap(temporary / "residue_type.npy", mode="w+", dtype=np.uint8, shape=(residues,))
    # One row per segment, so the row count is only known once the splitting is done.
    offsets: list[int] = [0]
    pdb_ids: list[str] = []
    chain_ids: list[str] = []
    labels: list[str] = []
    cursor = 0
    row_index = 0
    dropped_chains = 0
    dropped_residues = 0
    split_chains = 0
    for batch in batches(files):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            row = {key: value[i] for key, value in data.items()}
            segments = convert_row(row, drop_unresolved, split_at_breaks)
            kept = 0
            for coords, mask, residue_type in segments:
                if len(coords) < MINIMUM_RESIDUES:
                    # `sample_context_and_targets` already refuses a chain this short,
                    # so a segment below the floor is dropped rather than carried as a
                    # row the sampler will skip anyway.
                    dropped_residues += len(coords)
                    continue
                end = cursor + len(coords)
                coords_out[cursor:end] = coords
                mask_out[cursor:end] = mask
                residue_type_out[cursor:end] = residue_type
                cursor = end
                row_index += 1
                offsets.append(cursor)
                # A segment inherits its chain's identity and label: it is the same
                # chain, cut where the crystal stopped resolving.
                pdb_ids.append(str(row["pdb_id"]))
                chain_ids.append(str(row["chain_id"]))
                labels.append(ca_label(row["CATH"]))
                kept += 1
            if kept == 0:
                dropped_chains += 1
            elif kept > 1:
                split_chains += 1
    coords_out.flush()
    mask_out.flush()
    residue_type_out.flush()
    del coords_out, mask_out, residue_type_out
    if cursor < residues or row_index < rows:
        # Truncate to what was written. The allocation used the pre-drop count, which
        # is only an upper bound once residues and chains can leave.
        for name, shape in (("coords", (cursor, 4, 3)), ("atom_mask", (cursor, 4)),
                            ("residue_type", (cursor,))):
            path = temporary / f"{name}.npy"
            data = np.load(path, mmap_mode="r")[:cursor]
            np.save(temporary / f"{name}.trim.npy", np.ascontiguousarray(data))
            del data
            os.replace(temporary / f"{name}.trim.npy", path)
    np.save(temporary / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.savez_compressed(
        temporary / "metadata.npz",
        pdb_id=np.asarray(pdb_ids),
        chain_id=np.asarray(chain_ids),
        label=np.asarray(labels),
    )
    destination.mkdir(parents=True, exist_ok=True)
    for path in temporary.iterdir():
        os.replace(path, destination / path.name)
    temporary.rmdir()
    print(f"[built] {destination}: {row_index} rows, {cursor} residues"
          + (f" (dropped {dropped_chains} chains entirely and {dropped_residues} residues"
             f" in segments below {MINIMUM_RESIDUES}, {residues - cursor} residues gone"
             f" in total; {split_chains} chains split into several rows)"
             if drop_unresolved else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", default="cache/cath_backbone")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--drop-unresolved", action="store_true",
        help="Drop residues with any unresolved backbone atom",
    )
    parser.add_argument(
        "--split-at-breaks", action="store_true",
        help="emit one row per contiguous run of surviving residues, so removing a "
             "residue cannot make its neighbours adjacent (requires --drop-unresolved)",
    )
    args = parser.parse_args()
    if args.split_at_breaks and not args.drop_unresolved:
        parser.error("--split-at-breaks only means anything with --drop-unresolved")
    for split in args.splits:
        build_split(Path(args.data_root) / split, Path(args.cache_root) / split,
                    args.force, args.drop_unresolved, args.split_at_breaks)


if __name__ == "__main__":
    main()
