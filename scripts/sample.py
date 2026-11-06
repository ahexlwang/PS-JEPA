#!/usr/bin/env python3
'Sample protein backbones and evaluate reference-free geometric metrics.'
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psjepa.analysis.generation_metrics import chain_metrics
from psjepa.config import load_config
from psjepa.paths import resolve_recorded_checkpoint
from psjepa.data import BackboneDataset, collate_backbones
from psjepa.layers.flow_matching import sample_flow
from psjepa.models.token_prior import (
    load_prior_state_dict,
    prior_from_checkpoint,
    sample_tokens,
)
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer
from reconstruct_pdb import write_pdb

ARMS = (
    "data", "real", "prior", "gaussian", "shuffled",
    "perturb0p25", "perturb0p50", "interpolate",
)


def kabsch_rmsd(prediction: torch.Tensor, target: torch.Tensor) -> float:
    p = prediction - prediction.mean(dim=0, keepdim=True)
    q = target - target.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.transpose(0, 1) @ q)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return float((p @ rotation - q).square().sum(dim=-1).mean().sqrt())


def load_frozen(
    prior_dir: Path,
    checkpoint: str,
    device: torch.device,
    decoder_checkpoint: str | None = None,
):
    state = torch.load(prior_dir / checkpoint, map_location="cpu", weights_only=False)
    prior = prior_from_checkpoint(state)
    load_prior_state_dict(prior, state["prior"])
    prior.eval().requires_grad_(False).to(device)
    config = state["config"]
    if config.get("bottleneck_checkpoint"):
        raise ValueError("This sampler requires full-width encoder representations")

    # The raw module: the wrapper's `encode_batch` runs under `inference_mode`,
    # and such a tensor cannot be fed back in as a conditioning input.
    encoder = SemanticTokenizer.from_checkpoint(config["upstream_checkpoint"], device=device).model
    encoder.requires_grad_(False).eval().to(device)

    from train_decoder import build_decoder


    decoder_path = Path(decoder_checkpoint or config["decoder_checkpoint"])
    decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=False)
    decoder_config = decoder_state["config"]
    # The decoder recorded an absolute path to its frozen encoder, which is the right answer
    # on the machine that trained it and a missing directory anywhere else. Resolve it under
    # PSJEPA_WEIGHTS when it has moved, checked against the sha256 the decoder recorded, so a
    # same-named file cannot quietly supply tokens this decoder never saw.
    upstream_state = torch.load(
        resolve_recorded_checkpoint(
            decoder_config["upstream_checkpoint"],
            decoder_config.get("resolved_upstream_sha256"),
        ),
        map_location="cpu",
        weights_only=False,
    )
    _, decoder = build_decoder(decoder_config, upstream_state["config"], device)
    decoder.load_state_dict(decoder_state["decoder"], strict=True)
    decoder.eval().requires_grad_(False)
    print(
        f"prior step {state.get('step')} ({sum(p.numel() for p in prior.parameters())/1e6:.1f}M), "
        f"decoder step {decoder_state.get('step')}",
        flush=True,
    )
    return prior, encoder, decoder, decoder_config, config


@torch.no_grad()
def decode(decoder, decoder_config, tokens, atom_mask, residue_mask, steps):
    return sample_flow(
        decoder,
        tokens.to(next(decoder.parameters()).dtype),
        atom_mask,
        residue_mask,
        float(decoder_config["decoder"]["coordinate_scale"]),
        int(steps),
        bool(decoder_config.get("predict_x1", False)),
    ).float()


def masks_for_lengths(lengths, device) -> tuple[torch.Tensor, torch.Tensor]:
    longest = int(max(lengths))
    residue_mask = torch.zeros(len(lengths), longest, dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        residue_mask[row, : int(length)] = True
    atom_mask = residue_mask.unsqueeze(-1).expand(-1, -1, 4).contiguous()
    return residue_mask, atom_mask


@torch.no_grad()
def real_token_batches(encoder, dataset, batch_size, count, device):
    """Real validation chains with their tokens, in batches, exactly ``count`` chains.

    The last batch is cut to the remainder. Yielding it whole instead would give
    the real arms up to ``batch_size - 1`` chains more than the prior arm, whose
    count is exact, and the arms of one table would then rest on different
    numbers of chains.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_backbones,
    )
    taken = 0
    for batch in loader:
        if taken >= count:
            return
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        remaining = count - taken
        if batch["coords"].shape[0] > remaining:
            batch = {key: value[:remaining] for key, value in batch.items()}
        tokens = encoder.tokenize(
            batch["coords"], batch["atom_mask"], batch["residue_mask"]
        )["continuous_semantic"].detach()
        yield batch, tokens
        taken += batch["coords"].shape[0]


@torch.no_grad()
def refine(
    encoder, decoder, decoder_config, coordinates, residue_mask, atom_mask, steps, rounds
):
    'Push a generated structure onto the encoder/decoder fixed point.'
    moved = []
    for _ in range(int(rounds)):
        tokens = encoder.tokenize(
            coordinates, atom_mask, residue_mask
        )["continuous_semantic"].detach()
        nxt = decode(decoder, decoder_config, tokens, atom_mask, residue_mask, steps)
        step_rmsd = []
        for a, b, mask in zip(nxt, coordinates, residue_mask):
            if int(mask.sum()) >= 3:
                step_rmsd.append(
                    kabsch_rmsd(a[mask].reshape(-1, 3), b[mask].reshape(-1, 3))
                )
        moved.append(float(np.mean(step_rmsd)) if step_rmsd else float("nan"))
        coordinates = nxt
    return coordinates, moved


@torch.no_grad()
def self_consistency(
    encoder, decoder, decoder_config, coordinates, residue_mask, atom_mask, steps
) -> tuple[dict[str, float], torch.Tensor]:
    """Re-encode a generated backbone and decode it again.

    ``self_consistency_rmsd`` is the distance between the structure and its own
    round trip. It needs no ground truth, so it is defined for a de novo sample,
    and it is small exactly when the sample sits where the encoder and decoder
    agree.
    """
    tokens = encoder.tokenize(coordinates, atom_mask, residue_mask)["continuous_semantic"].detach()
    again = decode(decoder, decoder_config, tokens, atom_mask, residue_mask, steps)
    values = []
    for prediction, target, mask in zip(again, coordinates, residue_mask):
        keep = mask
        if int(keep.sum()) < 3:
            continue
        values.append(
            kabsch_rmsd(prediction[keep].reshape(-1, 3), target[keep].reshape(-1, 3))
        )
    return {
        "self_consistency_rmsd": float(np.mean(values)) if values else float("nan")
    }, tokens


def interpolate_pairs(
    tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    coordinates: torch.Tensor,
):
    """Midpoint of two real token sequences, over the residues both of them have.

    Chains are paired by length rank, so the two halves of a pair are as close in
    length as the batch allows, and both are then cut to the shorter one. The cut
    is what makes the arm measure what it claims. Pairing at random and keeping
    the longer chain's tail leaves that tail at its own real tokens, so the
    decoded chain is a chimera: an interpolated head joined to a real tail. On
    this validation set that left only 73% of residues interpolated on average,
    126 of 256 chains partly interpolated, and one chain at 2%, and the arm's
    mean was a mixture of two different quantities.

    Returns the interpolated tokens with the masks and the source coordinates cut
    to the same length, so every downstream metric sees one consistent chain.
    """
    lengths = residue_mask.sum(dim=1)
    order = torch.argsort(lengths, descending=True)
    partner = order.clone()
    for index in range(0, len(order), 2):
        if index + 1 < len(order):
            partner[order[index]] = order[index + 1]
            partner[order[index + 1]] = order[index]
        else:
            # Odd batch: the shortest chain borrows the one above it, which is
            # already paired. A batch of one interpolates with itself, which is
            # the `real` arm -- harmless, and it cannot happen at the sizes used.
            partner[order[index]] = order[index - 1]
    shared = torch.minimum(lengths, lengths[partner])
    positions = torch.arange(residue_mask.shape[1], device=residue_mask.device)
    keep = positions.unsqueeze(0) < shared.unsqueeze(1)
    condition = 0.5 * (tokens + tokens[partner]) * keep.unsqueeze(-1)
    return (
        condition,
        keep,
        atom_mask & keep.unsqueeze(-1),
        coordinates * keep.reshape(*keep.shape, 1, 1),
        float((shared.float() / lengths.clamp_min(1).float()).mean()),
    )  # the last value is the fraction of each source chain that survives the cut


def token_cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    x, y = a[mask], b[mask]
    return float(
        torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=-1).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", required=True, help="training output directory")
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--decoder-checkpoint", default=None,
                        help="score with a decoder other than the one the prior was trained beside")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--token-steps", type=int, default=100)
    parser.add_argument("--decoder-steps", type=int, default=100)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--write-pdb", type=int, default=16, help="PDBs per arm")
    parser.add_argument("--refine", type=int, default=0,
                        help="alternating encode/decode rounds applied to the prior arm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()


    arms = [arm for arm in args.arms.split(",") if arm]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_dir = Path(args.prior)
    out_dir = Path(args.out_dir or prior_dir / "samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    prior, encoder, decoder, decoder_config, config = load_frozen(
        prior_dir, args.checkpoint, device, decoder_checkpoint=args.decoder_checkpoint
    )


    validation_split = str(config["data"].get("validation_split", "validation"))
    validation = BackboneDataset(
        config["data"]["cache"], validation_split,
        max_residues=int(config["max_residues"]), augment=False,
        length_range=config["data"].get("length_range"),
    )
    print(f"validation data {config['data']['cache']}/{validation_split}: "
          f"{len(validation)} chains", flush=True)

    records: dict[str, list[dict[str, float]]] = {arm: [] for arm in arms}
    extra: dict[str, list[dict[str, float]]] = {arm: [] for arm in arms}
    written: dict[str, int] = {arm: 0 for arm in arms}

    def emit(arm: str, coordinates: torch.Tensor, residue_mask: torch.Tensor, tag: str) -> None:
        for index, (structure, mask) in enumerate(zip(coordinates, residue_mask)):
            record = chain_metrics(structure[mask].cpu())
            if record:
                records[arm].append(record)
            if written[arm] < args.write_pdb:
                keep = structure[mask].cpu().numpy()
                path = out_dir / "pdb" / arm
                path.mkdir(parents=True, exist_ok=True)
                write_pdb(
                    path / f"{tag}_{index:03d}.pdb",
                    keep,
                    np.ones((len(keep), 4), dtype=bool),
                    ["GLY"] * len(keep),
                    "A",
                    f"{arm} sample from {prior_dir}",
                )
                written[arm] += 1

    # -- the prior, the only arm that needs no real data ---------------------
    if "prior" in arms:
        values = np.array(prior.config.length_values)
        weights = np.array(prior.config.length_counts, dtype=np.float64)
        rng = np.random.RandomState(args.seed)
        lengths = rng.choice(values, size=args.count, p=weights / weights.sum())
        for start in range(0, args.count, args.batch_size):
            chunk = lengths[start : start + args.batch_size]
            residue_mask, atom_mask = masks_for_lengths(chunk, device)
            tokens = sample_tokens(prior, residue_mask, steps=args.token_steps)
            coordinates = decode(
                decoder, decoder_config, tokens, atom_mask, residue_mask, args.decoder_steps
            )
            if args.refine:
                coordinates, moved = refine(
                    encoder, decoder, decoder_config, coordinates,
                    residue_mask, atom_mask, args.decoder_steps, args.refine,
                )
                print(f"  refine movement per round (A): "
                      f"{', '.join(f'{v:.2f}' for v in moved)}", flush=True)
            emit("prior", coordinates, residue_mask, f"b{start:04d}")
            consistency, again = self_consistency(
                encoder, decoder, decoder_config, coordinates,
                residue_mask, atom_mask, args.decoder_steps,
            )


            key = "token_cosine_prior_to_final" if args.refine else "token_cosine_round_trip"
            consistency[key] = token_cosine(tokens, again, residue_mask)
            extra["prior"].append(consistency)
            print(f"prior {start + len(chunk)}/{args.count} {consistency}", flush=True)

    # -- everything that starts from a real chain ----------------------------
    real_arms = [arm for arm in arms if arm != "prior"]
    if real_arms:


        for chunk, (batch, tokens) in enumerate(real_token_batches(
            encoder, validation, args.batch_size, args.count, device
        )):
            tag = f"real{chunk:02d}"
            residue_mask = batch["residue_mask"]
            atom_mask = batch["atom_mask"]
            standard = prior.token_std.to(tokens.dtype)
            for arm in real_arms:
                if arm == "data":
                    emit("data", batch["coords"], residue_mask, tag)
                    continue


                arm_residue_mask, arm_atom_mask = residue_mask, atom_mask
                source = batch["coords"]
                interpolated_fraction = None
                if arm == "real":
                    condition = tokens
                elif arm == "gaussian":
                    condition = (
                        prior.token_mean + torch.randn_like(tokens) * standard
                    ) * residue_mask.unsqueeze(-1)
                elif arm == "shuffled":
                    condition = tokens.clone()
                    for row, mask in enumerate(residue_mask):
                        count = int(mask.sum())
                        condition[row, :count] = tokens[row][mask][
                            torch.randperm(count, device=tokens.device)
                        ]
                elif arm.startswith("perturb"):
                    sigma = float(arm.replace("perturb", "").replace("p", "."))
                    condition = (
                        tokens + torch.randn_like(tokens) * standard * sigma
                    ) * residue_mask.unsqueeze(-1)
                elif arm == "interpolate":
                    (
                        condition, arm_residue_mask, arm_atom_mask, source,
                        interpolated_fraction,
                    ) = interpolate_pairs(tokens, residue_mask, atom_mask, batch["coords"])
                else:
                    raise SystemExit(f"unhandled arm {arm}")
                coordinates = decode(
                    decoder, decoder_config, condition, arm_atom_mask,
                    arm_residue_mask, args.decoder_steps,
                )
                emit(arm, coordinates, arm_residue_mask, tag)
                consistency, again = self_consistency(
                    encoder, decoder, decoder_config, coordinates,
                    arm_residue_mask, arm_atom_mask, args.decoder_steps,
                )
                consistency["token_cosine_round_trip"] = token_cosine(
                    condition, again, arm_residue_mask
                )
                if interpolated_fraction is not None:
                    # Every residue that is decoded is a midpoint, by construction.
                    # This says how much of each source chain survived the cut to
                    # its partner's length, so a value far below 1 means the batch
                    # paired chains of very different lengths and the arm is
                    # reporting on noticeably shorter chains than `data`.
                    consistency["source_length_kept"] = interpolated_fraction
                if arm in ("real", "perturb0p25", "perturb0p50", "interpolate", "shuffled"):


                    values = []
                    for prediction, target, mask in zip(
                        coordinates, source, arm_residue_mask
                    ):
                        if int(mask.sum()) < 3:
                            continue
                        values.append(
                            kabsch_rmsd(
                                prediction[mask].reshape(-1, 3),
                                target[mask].reshape(-1, 3),
                            )
                        )
                    consistency["rmsd_to_source_chain"] = (
                        float(np.mean(values)) if values else float("nan")
                    )
                extra[arm].append(consistency)
            print(f"real arms: {len(records[real_arms[0]])} chains", flush=True)

    summary: dict[str, dict[str, float]] = {}
    for arm in arms:
        if not records[arm]:
            continue
        keys = sorted({key for record in records[arm] for key in record})
        row = {
            key: float(
                np.mean([r[key] for r in records[arm] if key in r and np.isfinite(r[key])])
            )
            for key in keys
        }
        row["n_chains"] = len(records[arm])


        for key in sorted({k for block in extra[arm] for k in block}):
            values = [b[key] for b in extra[arm] if key in b and np.isfinite(b[key])]
            row[key] = float(np.mean(values)) if values else float("nan")
        summary[arm] = row

    (out_dir / "generation_metrics.json").write_text(
        json.dumps(
            {
                "prior_dir": str(prior_dir),
                "checkpoint": args.checkpoint,
                # Record the decoder used to generate these structures.
                "decoder_checkpoint": args.decoder_checkpoint or config["decoder_checkpoint"],
                "count": args.count,
                "token_steps": args.token_steps,
                "decoder_steps": args.decoder_steps,
                "seed": args.seed,
                "refine_rounds": args.refine,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    for arm, row in summary.items():
        (out_dir / f"chains_{arm}.json").write_text(json.dumps(records[arm], indent=1))

    headline = [
        "n_chains", "peptide_c_n_mean", "chain_break_fraction", "ca_ca_mean",
        "omega_trans_fraction", "phi_negative_fraction", "helix_fraction",
        "sheet_fraction", "clash_fraction", "radius_of_gyration_ratio",
        "contact_density", "self_consistency_rmsd", "rmsd_to_source_chain",
    ]
    print("\narm            " + " ".join(f"{key[:11]:>12s}" for key in headline))
    for arm in arms:
        if arm not in summary:
            continue
        row = summary[arm]
        cells = " ".join(
            f"{row[key]:12.3f}" if key in row and np.isfinite(row[key]) else f"{'-':>12s}"
            for key in headline
        )
        print(f"{arm:15s}{cells}")
    print(f"\nwrote {out_dir}/generation_metrics.json", flush=True)


if __name__ == "__main__":
    main()
