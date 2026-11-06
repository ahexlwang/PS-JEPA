#!/usr/bin/env python3
'Build N/CA/C/O memory-mapped backbone caches from AFDB domain parquet.'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BOND = 1.231
ANGLE = np.deg2rad(120.546)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)


def with_oxygen(coords: np.ndarray) -> tuple[np.ndarray, int]:
    """coords [L, 3, 3] of N, CA, C -> [L, 4, 3] with O appended."""
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    oxygen = np.zeros_like(c)
    exact = 0
    if len(c) > 1:
        direction = unit(-(unit(n[1:] - c[:-1]) + unit(ca[:-1] - c[:-1])))
        oxygen[:-1] = c[:-1] + BOND * direction
        exact = len(c) - 1

    axis = unit(c[-1] - ca[-1])
    normal = unit(np.cross(ca[-1] - n[-1], axis))
    transverse = np.cross(normal, axis)
    oxygen[-1] = c[-1] + BOND * (transverse * np.sin(ANGLE) - axis * np.cos(ANGLE))
    return np.concatenate([coords, oxygen[:, None, :]], axis=1), exact


def convert(files: list[Path], destination: Path, split: str, limit: int = 0) -> dict:
    coords_out, mask_out, offsets, names, labels, plddt_out = [], [], [0], [], [], []
    cursor = exact_total = residues = 0
    dropped = 0
    for path in files:
        table = pq.ParquetFile(path)
        for group in range(table.num_row_groups):
            # The pretraining corpus carries no label; a missing column is not an error
            # there, and defaulting it to -1 keeps one converter for both cases.
            available = set(table.schema_arrow.names)
            wanted = [c for c in ("name", "coords", "label", "plddt") if c in available]
            batch = table.read_row_group(group, columns=wanted)
            for row in batch.to_pylist():
                coords = np.asarray(row["coords"], dtype=np.float32)
                if coords.ndim != 3 or coords.shape[1] != 3 or len(coords) < 4:
                    dropped += 1
                    continue
                if not np.isfinite(coords).all():
                    dropped += 1
                    continue
                full, exact = with_oxygen(coords.astype(np.float64))
                coords_out.append(full.astype(np.float32))
                mask_out.append(np.ones((len(full), 4), dtype=np.uint8))
                cursor += len(full)
                offsets.append(cursor)
                names.append(str(row["name"]))
                labels.append(int(row["label"]) if row.get("label") is not None else -1)
                plddt_out.append(float(np.mean(row["plddt"])) if row.get("plddt") else -1.0)
                exact_total += exact
                residues += len(full)
                if limit and len(names) >= limit:
                    break
            if limit and len(names) >= limit:
                break
        if limit and len(names) >= limit:
            break
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "coords.npy", np.concatenate(coords_out))
    np.save(destination / "atom_mask.npy", np.concatenate(mask_out))
    np.save(destination / "offsets.npy", np.asarray(offsets, dtype=np.int64))


    np.savez_compressed(destination / "metadata.npz",
                        name=np.asarray(names), pdb_id=np.asarray(names),
                        chain_id=np.asarray(["_"] * len(names)),
                        label=np.asarray(labels, dtype=np.int64),
                        plddt=np.asarray(plddt_out, dtype=np.float32))
    lengths = np.diff(offsets)
    summary = {
        "split": split, "domains": len(names), "residues": int(residues),
        "dropped": dropped,
        "oxygen_exact_share": round(exact_total / max(residues, 1), 5),
        "classes_present": len({v for v in labels if v >= 0}),
        "unlabelled": int(sum(1 for v in labels if v < 0)),
        "length_min": int(lengths.min()), "length_median": int(np.median(lengths)),
        "length_mean": round(float(lengths.mean()), 1), "length_max": int(lengths.max()),
        "plddt_mean": round(float(np.mean(plddt_out)), 2),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HuggingFace dataset repository of domain parquet shards")
    parser.add_argument("--split", required=True, help="train, test, validation ...")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="downloads/parquet",
                        help="where the downloaded parquet shards are kept")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    wanted = sorted(f for f in api.list_repo_files(args.repo, repo_type="dataset")
                    if f.startswith(f"data/{args.split}-") and f.endswith(".parquet"))
    if not wanted:
        raise SystemExit(f"no {args.split} parquet in {args.repo}")
    print(f"{args.repo}/{args.split}: {len(wanted)} shards", flush=True)
    local = []
    for name in wanted:
        local.append(Path(hf_hub_download(args.repo, name, repo_type="dataset",
                                          local_dir=f"{args.cache}/{args.repo.split('/')[-1]}")))
        print(f"  have {name}", flush=True)
    summary = convert(local, Path(args.out), args.split, args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
