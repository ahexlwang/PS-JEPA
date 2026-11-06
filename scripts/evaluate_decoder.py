#!/usr/bin/env python3
"""Evaluate a coordinate decoder with per-chain metrics over a dataset split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from psjepa.config import load_config
from psjepa.data import BackboneDataset, collate_backbones
from psjepa.layers.flow_matching import sample_flow
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer
from psjepa.training.structure_metrics import structure_metrics
from evaluate_decoder_arms import reflected_rmsd
from train_decoder import (aligned_backbone_rmsd, autocast_context, build_decoder,
                                       ca_pairwise_distance_mae, device_batch, representation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None, help="default <run>/per_chain_<split>_<ckpt>_s<steps>.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run)
    config = load_config(run / "resolved_config.yaml")
    tokenizer = SemanticTokenizer.from_checkpoint(config["upstream_checkpoint"], device)
    encoder = tokenizer.model.requires_grad_(False).eval().to(device)
    upstream_config = torch.load(config["upstream_checkpoint"], map_location="cpu", weights_only=False)["config"]
    cache = args.cache or config["data"]["cache"]
    dataset = BackboneDataset(cache, args.split, int(config["max_residues"]), augment=False, limit=args.limit)
    meta = np.load(Path(cache) / args.split / "metadata.npz", allow_pickle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_backbones)

    state = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    _kind, decoder = build_decoder(config, upstream_config, device)
    decoder.load_state_dict(state["decoder"], strict=True)
    decoder.eval()

    out_path = Path(args.output) if args.output else run / f"per_chain_{args.split}_{args.checkpoint[:-3]}_s{args.steps}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                done.add(int(json.loads(line)["index"]))
    torch.manual_seed(args.seed)
    started = time.time()
    with out_path.open("a") as handle:
        for batch_index, batch in enumerate(loader):
            first = batch_index * args.batch_size
            if all((first + i) in done for i in range(int(batch["residue_mask"].shape[0]))):
                continue
            batch = device_batch(batch, device)
            with torch.no_grad(), autocast_context(device, config["training"]["precision"]):
                condition = representation(encoder, batch, config["representation"])
                generated = sample_flow(
                    decoder, condition, batch["atom_mask"], batch["residue_mask"],
                    float(config["decoder"]["coordinate_scale"]), int(args.steps),
                    bool(config.get("predict_x1", False)))
            generated = generated.float()
            for b in range(generated.shape[0]):
                index = first + b
                if index in done:
                    continue
                sl = slice(b, b + 1)
                g, c, m = generated[sl], batch["coords"][sl], batch["atom_mask"][sl]
                record = {
                    "index": index,
                    "pdb_id": str(meta["pdb_id"][index]),
                    "chain_id": str(meta["chain_id"][index]),
                    "label": str(meta["label"][index]),
                    "length": int(batch["residue_mask"][b].sum()),
                    "rmsd_proper": aligned_backbone_rmsd(g, c, m),
                    "rmsd_reflect_ok": reflected_rmsd(g, c, m),
                    "ca_distance_mae": ca_pairwise_distance_mae(g, c, m),
                }
                record.update({k: float(v) for k, v in structure_metrics(g, c, m).items()})
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            if batch_index % 10 == 0:
                print(f"{run.name} {args.split} batch {batch_index}/{len(loader)} {time.time()-started:.0f}s", flush=True)
    rows = [json.loads(l) for l in out_path.open() if l.strip()]
    r = np.array([x["rmsd_proper"] for x in rows])
    print(json.dumps({"run": run.name, "split": args.split, "checkpoint": args.checkpoint, "step": int(state.get("step", -1)),
                      "chains": len(rows), "rmsd_mean": float(np.nanmean(r)), "rmsd_median": float(np.nanmedian(r)),
                      "lddt_mean": float(np.nanmean([x["lddt"] for x in rows]))}), flush=True)


if __name__ == "__main__":
    main()
