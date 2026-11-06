#!/usr/bin/env python3
"""Compare coordinate decoders using matched inputs and sampling settings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from psjepa.config import load_config
from psjepa.data import BackboneDataset, collate_backbones
from psjepa.layers.flow_matching import sample_flow
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer
from psjepa.training.structure_metrics import structure_metrics
from train_decoder import (aligned_backbone_rmsd, autocast_context, build_decoder,
                                       ca_pairwise_distance_mae, device_batch, representation)


def reflected_rmsd(prediction: torch.Tensor, target: torch.Tensor, atom_mask: torch.Tensor) -> float:
    values = []
    for p, q, m in zip(prediction.float(), target.float(), atom_mask):
        p, q = p[m], q[m]
        p = p - p.mean(0, keepdim=True)
        q = q - q.mean(0, keepdim=True)
        u, _, vh = torch.linalg.svd(p.T @ q)
        values.append(((p @ (u @ vh) - q).square().sum(-1).mean()).sqrt())
    return float(torch.stack(values).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=["last.pt", "best.pt"])
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--steps", type=int, nargs="+", default=[100])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="JSON lines; default <run>/rollout_eval.jsonl")
    parser.add_argument("--zero-condition", action="store_true",
                        help="replace the condition with zeros at inference: what the trained decoder draws with no token at all")
    parser.add_argument("--shuffle-condition", action="store_true",
                        help="permute each chain's condition along the residue axis: tests whether order carries the fold")
    parser.add_argument("--cache", default=None, help="override the validation cache so arms trained on different caches are scored on the same chains")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders: dict[str, object] = {}
    for run_name in args.runs:
        run = Path(run_name)
        config = load_config(run / "resolved_config.yaml")
        upstream_path = config["upstream_checkpoint"]
        if upstream_path not in encoders:
            tokenizer = SemanticTokenizer.from_checkpoint(upstream_path, device)
            encoders[upstream_path] = tokenizer.model.requires_grad_(False).eval().to(device)
        encoder = encoders[upstream_path]
        upstream_config = torch.load(upstream_path, map_location="cpu", weights_only=False)["config"]
        dataset = BackboneDataset(args.cache or config["data"]["cache"], "validation",
                                  int(config["max_residues"]), augment=False, limit=args.chains)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_backbones)
        out_path = Path(args.output) if args.output else run / "rollout_eval.jsonl"
        for checkpoint in args.checkpoints:
            path = run / checkpoint
            if not path.is_file():
                print(f"{run.name}: no {checkpoint}", flush=True)
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            _kind, decoder = build_decoder(config, upstream_config, device)
            decoder.load_state_dict(state["decoder"], strict=True)
            decoder.eval()
            for steps in args.steps:
                sums: dict[str, float] = {}
                count = 0
                torch.manual_seed(args.seed)
                for batch in loader:
                    batch = device_batch(batch, device)
                    with torch.no_grad(), autocast_context(device, config["training"]["precision"]):
                        condition = representation(encoder, batch, config["representation"])
                        if args.zero_condition:
                            condition = torch.zeros_like(condition)
                        if args.shuffle_condition:
                            # per-chain permutation of the valid residues only, padding stays in place
                            shuffled = condition.clone()
                            for b in range(condition.shape[0]):
                                idx = batch["residue_mask"][b].nonzero(as_tuple=True)[0]
                                shuffled[b, idx] = condition[b, idx[torch.randperm(len(idx), device=idx.device)]]
                            condition = shuffled
                        generated = sample_flow(
                            decoder, condition, batch["atom_mask"], batch["residue_mask"],
                            float(config["decoder"]["coordinate_scale"]), int(steps),
                            bool(config.get("predict_x1", False)))
                    generated = generated.float()
                    metrics = {
                        "rmsd_proper": aligned_backbone_rmsd(generated, batch["coords"], batch["atom_mask"]),
                        "rmsd_reflect_ok": reflected_rmsd(generated, batch["coords"], batch["atom_mask"]),
                        "ca_distance_mae": ca_pairwise_distance_mae(generated, batch["coords"], batch["atom_mask"]),
                    }
                    metrics.update(structure_metrics(generated, batch["coords"], batch["atom_mask"]))
                    n = int(batch["residue_mask"].shape[0])
                    for key, value in metrics.items():
                        if value == value:  # skip NaN
                            sums[key] = sums.get(key, 0.0) + float(value) * n
                    count += n
                record = {"run": run.name + ("_SHUFFLED" if args.shuffle_condition else "") + ("_ZERO" if args.zero_condition else ""), "checkpoint": checkpoint, "step": int(state.get("step", -1)),
                          "cache": args.cache or config["data"]["cache"],
                          "sample_steps": int(steps), "chains": count,
                          **{k: v / count for k, v in sums.items()}}
                record["mirror_gap"] = record["rmsd_proper"] - record["rmsd_reflect_ok"]
                with out_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in record.items()}), flush=True)


if __name__ == "__main__":
    main()
