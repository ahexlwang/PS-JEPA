#!/usr/bin/env python3
'Train a coordinate flow decoder conditioned on a frozen PS-JEPA encoder.'
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from psjepa.config import load_config, save_config
from psjepa.data import BackboneDataset, collate_backbones
from psjepa.data.masking import sample_motif_mask
from psjepa.layers.flow_attention import AttentionFlowDecoder
from psjepa.layers.flow_frame import AttentionFrameDecoder
from psjepa.layers.flow_ipa import IPAFlowDecoder
from psjepa.layers.flow_matching import (
    GVPFlowDecoder,
    conditional_flow_matching_loss,
    sample_flow,
)
from psjepa.models import PSJEPA
from psjepa.tokenization.semantic_tokenizer import PSJEPA_MODEL_FAMILIES
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer
from psjepa.tokenization.motif_tokens import motif_tokens


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing after this optimizer step.", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def learning_rate(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def append_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def atomic_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representation(encoder, batch: dict[str, Any], kind: str) -> torch.Tensor:
    # Do not pass residue_type: sequence identity is forbidden for this probe.
    tokens = encoder.tokenize(batch["coords"], batch["atom_mask"], batch["residue_mask"])
    if kind == "continuous_semantic":
        value = tokens["continuous_semantic"]
    elif kind == "target_normalized_semantic":


        target_norm = getattr(encoder, "target_norm", None)
        if target_norm is None:
            raise ValueError("target_normalized_semantic requires an encoder with target_norm")
        value = target_norm(tokens["continuous_semantic"])
    elif kind == "zero_condition":
        value = torch.zeros_like(tokens["continuous_semantic"])
    else:
        raise ValueError(f"Unknown downstream representation: {kind}")
    # Detached, never a gradient path into the encoder: this is a probe of a frozen
    # representation. Both tokenizers run under `no_grad`; `PSJEPA.tokenize` used
    # `inference_mode`, whose mark is permanent and survives cloning, so its tokens could
    # not be conditioned on at all until that was changed.
    return value.detach()


def motif_mask_for(batch: dict[str, Any], config: dict[str, Any]):
    """The residues handed to the decoder as given geometry, or None.

    Off by default, so every existing flow arm keeps its behaviour byte for byte.
    """
    motif = config.get("motif") or {}
    if not motif.get("enabled", False):
        return None


    if float(motif.get("probability", 1.0)) < 1.0 and \
            float(torch.rand(())) >= float(motif["probability"]):
        return None
    return sample_motif_mask(
        batch["coords"], batch["atom_mask"], batch["residue_mask"],
        min_fraction=float(motif.get("min_fraction", 0.05)),
        max_fraction=float(motif.get("max_fraction", 0.30)),
        spatial_probability=float(motif.get("spatial_probability", 0.7)),
        max_segments=int(motif.get("max_segments", 3)),
    )


def hold_motif_coordinates(config: dict[str, Any]) -> bool:
    """Whether a drawn motif is also handed to the decoder as clean coordinates.

    True keeps what the scaffolding arms did: the site is exact by construction and only the rest
    is scored. False draws the motif solely to mark which residues carry a site-only token, and
    the decoder is asked to produce every residue including the site -- which is the task when no
    coordinates may be imposed.
    """
    return bool((config.get("motif") or {}).get("hold_coordinates", True))


def motif_condition_for(config: dict[str, Any]) -> str:
    "Which dialect the motif's tokens are written in during training."
    return str((config.get("motif") or {}).get("condition", "context"))


def apply_motif_condition(condition, encoder, batch, motif, config):
    kind = motif_condition_for(config)
    if motif is None or kind == "context":
        return condition
    held = motif_tokens(encoder, batch["coords"], batch["atom_mask"],
                        batch["residue_mask"], motif, kind)
    return torch.where(motif.unsqueeze(-1), held.to(condition.dtype), condition)


def drop_scaffold_tokens(
    condition: torch.Tensor,
    config: dict[str, Any],
    motif: torch.Tensor | None,
) -> torch.Tensor:
    """Blank the token condition everywhere except the site, on a fraction of updates.

    This is the condition a designer actually has once the token prior is taken out of the loop:
    the site's own tokens, and nothing for the chain that has to be built around it. The decoder
    already starts from Gaussian noise, so with this it becomes the generative model itself and the
    tokens are guidance rather than a blueprint it must render.

    The target stays the true chain, which cannot be recovered from a site alone -- and does not
    need to be. Flow matching feeds the network `(1-t)*noise + t*x1`, so at larger times the input
    already carries the structure and the update is well posed; at small times nothing determines
    which protein was meant, and the irreducible part of the loss is exactly the pressure that stops
    the decoder from treating the condition as a blueprint. What it learns instead is to draw a
    chain that holds the site.

    Zeroes, not a learned token: the condition is zero at padding everywhere else in this codebase,
    and `sample_flow` is given a zero tensor when a caller has no tokens, so the two agree.
    """
    rate = float((config.get("motif") or {}).get("scaffold_dropout", 0.0))
    if rate <= 0.0 or motif is None:
        return condition
    drop = torch.rand(condition.shape[0], device=condition.device) < rate
    keep = motif | ~drop.reshape(-1, 1)
    return condition * keep.unsqueeze(-1).to(condition.dtype)


def drop_condition(
    condition: torch.Tensor,
    config: dict[str, Any],
    preserve_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blank the token condition for a random part of the batch.

    Without this the motif channel is redundant during scaffolding training: the
    condition is the *true* full-chain token sequence, which already determines
    the structure to about 1 A, so the cheapest solution is to read the tokens
    and ignore the given coordinates. At sampling time the tokens come from the
    prior and the motif is the only thing that is certainly true, so a decoder
    trained that way follows the wrong one of the two. Dropping the condition
    outright on some steps forces it to build a structure around the motif with
    no token guidance at all.
    """
    motif = config.get("motif") or {}
    rate = float(motif.get("condition_dropout", 0.0))
    mismatch = float(motif.get("condition_mismatch", 0.0))
    corrupt = float(motif.get("condition_corrupt", 0.0))
    shape = (-1,) + (1,) * (condition.dim() - 1)
    if corrupt > 0.0 and condition.shape[0] > 1:


        rolled = torch.roll(condition, shifts=1, dims=0)
        spoil = torch.rand(condition.shape[:2], device=condition.device) < corrupt
        if preserve_mask is not None:
            spoil = spoil & ~preserve_mask
        condition = torch.where(spoil.unsqueeze(-1), rolled, condition)
    if mismatch > 0.0 and condition.shape[0] > 1:


        swap = torch.rand(condition.shape[0], device=condition.device) < mismatch
        rolled = torch.roll(condition, shifts=1, dims=0)
        if preserve_mask is not None:
            rolled = torch.where(preserve_mask.unsqueeze(-1), condition, rolled)
        condition = torch.where(swap.reshape(shape), rolled, condition)
    if rate <= 0.0:
        return condition
    keep = torch.rand(condition.shape[0], device=condition.device) >= rate
    return condition * keep.reshape(shape).to(condition.dtype)


def aligned_backbone_rmsd(
    prediction: torch.Tensor, target: torch.Tensor, atom_mask: torch.Tensor
) -> float:
    values: list[torch.Tensor] = []
    for predicted, truth, mask in zip(prediction.float(), target.float(), atom_mask):
        p = predicted[mask]
        q = truth[mask]
        if p.shape[0] < 3:
            continue
        p = p - p.mean(dim=0, keepdim=True)
        q = q - q.mean(dim=0, keepdim=True)
        u, _, vh = torch.linalg.svd(p.transpose(0, 1) @ q)
        rotation = u @ vh
        if torch.linalg.det(rotation) < 0:
            u = u.clone()
            u[:, -1] *= -1
            rotation = u @ vh
        values.append(((p @ rotation - q).square().sum(dim=-1).mean()).sqrt())
    return float(torch.stack(values).mean()) if values else float("nan")


def ca_pairwise_distance_mae(
    prediction: torch.Tensor, target: torch.Tensor, atom_mask: torch.Tensor
) -> float:
    values: list[torch.Tensor] = []
    for predicted, truth, mask in zip(prediction.float(), target.float(), atom_mask[:, :, 1]):
        p = predicted[:, 1][mask]
        q = truth[:, 1][mask]
        if p.shape[0] < 2:
            continue
        keep = torch.triu(torch.ones(p.shape[0], p.shape[0], device=p.device, dtype=torch.bool), 1)
        values.append((torch.cdist(p, p)[keep] - torch.cdist(q, q)[keep]).abs().mean())
    return float(torch.stack(values).mean()) if values else float("nan")


def make_loader(dataset, config: dict[str, Any], epoch: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_backbones,
        drop_last=True,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )


@torch.no_grad()
def validate(
    encoder: PSJEPA,
    decoder,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    step: int,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    loss_sum = target_rms_sum = prediction_rms_sum = 0.0
    batches = 0
    sampled = False
    sampling_metrics: dict[str, float] = {}
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1729 + step)
        for batch in loader:
            batch = device_batch(batch, device)
            motif = motif_mask_for(batch, config)
            hold = motif if hold_motif_coordinates(config) else None
            predict_on = batch["residue_mask"] if hold is None else (
                batch["residue_mask"] & ~hold)
            with autocast_context(device, config["training"]["precision"]):
                condition = representation(encoder, batch, config["representation"])
                condition = apply_motif_condition(
                    condition, encoder, batch, motif, config)
                loss, metrics = conditional_flow_matching_loss(
                    decoder,
                    condition,
                    batch["coords"],
                    batch["atom_mask"],
                    predict_on,
                    float(config["decoder"]["coordinate_scale"]),
                    bool(config.get("align_target", False)),
                    bool(config.get("predict_x1", False)),
                    float(config.get("pair_weight", 0.0)),
                    float(config.get("pair_min_time", 0.5)),
                    bool(config.get("canonical_target", False)),
                    float(config.get("fape_weight", 0.0)),
                    float(config.get("adjacent_weight", 0.0)),
                    hold,
                )
            loss_sum += float(loss)
            target_rms_sum += float(metrics["flow_velocity_target_rms"])
            prediction_rms_sum += float(metrics["flow_velocity_prediction_rms"])
            batches += 1
            if not sampled and int(config["evaluation"]["sample_count"]) > 0:
                count = min(int(config["evaluation"]["sample_count"]), condition.shape[0])
                with autocast_context(device, config["training"]["precision"]):
                    generated = sample_flow(
                        decoder,
                        condition[:count],
                        batch["atom_mask"][:count],
                        predict_on[:count],
                        float(config["decoder"]["coordinate_scale"]),
                        int(config["evaluation"]["sample_steps"]),
                        bool(config.get("predict_x1", False)),
                        None if hold is None else hold[:count],
                        None if hold is None else batch["coords"][:count],
                    )
                sampling_metrics = {
                    "val_aligned_backbone_rmsd": aligned_backbone_rmsd(
                        generated, batch["coords"][:count], batch["atom_mask"][:count]
                    ),
                    "val_ca_pairwise_distance_mae": ca_pairwise_distance_mae(
                        generated, batch["coords"][:count], batch["atom_mask"][:count]
                    ),
                }
                if motif is not None:
                    # Report the site and scaffold separately. With coordinate holding the
                    # site number is a wiring check and should be zero; without it this is the
                    # actual placement error and must not be hidden by validation-time clamping.
                    built = batch["atom_mask"][:count] & (~motif[:count]).unsqueeze(-1)
                    site = batch["atom_mask"][:count] & motif[:count].unsqueeze(-1)
                    sampling_metrics["val_scaffold_rmsd"] = aligned_backbone_rmsd(
                        generated, batch["coords"][:count], built
                    )
                    sampling_metrics["val_motif_rmsd"] = aligned_backbone_rmsd(
                        generated, batch["coords"][:count], site
                    )
                    sampling_metrics["val_motif_coordinates_held"] = float(hold is not None)
                    sampling_metrics["val_motif_fraction"] = float(
                        motif[:count][batch["residue_mask"][:count]].float().mean()
                    )
                    # The mode that matters at design time: the motif is true and
                    # the tokens are not available. If this does not come down, the
                    # decoder is reading the answer off the condition instead.
                    with autocast_context(device, config["training"]["precision"]):
                        blind = sample_flow(
                            decoder,
                            torch.zeros_like(condition[:count]),
                            batch["atom_mask"][:count],
                            predict_on[:count],
                            float(config["decoder"]["coordinate_scale"]),
                            int(config["evaluation"]["sample_steps"]),
                            bool(config.get("predict_x1", False)),
                            None if hold is None else hold[:count],
                            None if hold is None else batch["coords"][:count],
                        )
                    sampling_metrics["val_scaffold_rmsd_no_tokens"] = aligned_backbone_rmsd(
                        blind, batch["coords"][:count], built
                    )
                    # The mode this decoder is being taught once `scaffold_dropout` is on: the
                    # site's tokens are present and the chain around them is unconditioned. No
                    # coordinates are imposed, so this motif RMSD says whether the decoder can hold
                    # a site while inventing the rest -- the whole task, with the prior out of it.
                    with autocast_context(device, config["training"]["precision"]):
                        site_only = sample_flow(
                            decoder,
                            condition[:count] * motif[:count].unsqueeze(-1).to(condition.dtype),
                            batch["atom_mask"][:count],
                            predict_on[:count],
                            float(config["decoder"]["coordinate_scale"]),
                            int(config["evaluation"]["sample_steps"]),
                            bool(config.get("predict_x1", False)),
                            None if hold is None else hold[:count],
                            None if hold is None else batch["coords"][:count],
                        )
                    sampling_metrics["val_motif_rmsd_site_only"] = aligned_backbone_rmsd(
                        site_only, batch["coords"][:count], site
                    )
                sampled = True
            if batches >= int(config["evaluation"]["validation_batches"]):
                break
    decoder.train()
    result = {
        "val_flow_loss": loss_sum / max(batches, 1),
        "val_flow_velocity_target_rms": target_rms_sum / max(batches, 1),
        "val_flow_velocity_prediction_rms": prediction_rms_sum / max(batches, 1),
        **sampling_metrics,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"Non-finite downstream validation metric: {result}")
    return result


def checkpoint_payload(
    decoder,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    epoch: int,
    batch_in_epoch: int,
    best_validation: float,
) -> dict[str, Any]:
    return {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "best_validation": best_validation,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def restore_rng(rng: dict[str, Any]) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def build_decoder(config: dict[str, Any], upstream_config: dict[str, Any],
                  device: torch.device):
    """The decoder a resolved config names, so evaluation scripts build the same object."""
    kind = str(config["decoder"].get("kind", "attention"))
    if config["representation"] in ("discrete_ids", "quantized_semantic"):
        raise ValueError("This release supports continuous encoder representations only")
    code_book = 0
    if kind in ("frame", "ipa") and not bool(config.get("predict_x1", False)):
        # These heads emit a structure, not a velocity, so the loss has to read it as x1.
        raise ValueError(f"decoder kind {kind!r} requires predict_x1: true")
    # An explicit "a latent was provided here" input, distinct from the latent's value.
    # Absent by default, so every existing checkpoint loads with `strict=True` unchanged;
    # a run that asks for it trains new parameters and cannot be initialised from one that
    # did not. Only the two attention trunks support this input.
    condition_mask_input = bool(config["decoder"].get("condition_mask_input", False))
    if condition_mask_input and kind not in ("attention", "frame"):
        raise ValueError(f"condition_mask_input is not wired for decoder kind {kind!r}")
    if kind == "frame":
        decoder = AttentionFrameDecoder(
            d_model=int(upstream_config["model"]["d_model"]), code_book=code_book,
            layers=int(config["decoder"]["layers"]),
            width=int(config["decoder"].get("width", 384)),
            heads=int(config["decoder"].get("heads", 6)),
            max_residues=int(config["max_residues"]),
            condition_mask_input=condition_mask_input,
            coordinate_scale=float(config["decoder"]["coordinate_scale"]))
    elif kind == "ipa":
        decoder = IPAFlowDecoder(
            d_model=int(upstream_config["model"]["d_model"]), code_book=code_book,
            trunk_layers=int(config["decoder"].get("layers", 4)),
            width=int(config["decoder"].get("width", 256)),
            heads=int(config["decoder"].get("heads", 8)),
            iterations=int(config["decoder"].get("iterations", 4)),
            max_residues=int(config["max_residues"]),
            dropout=float(config["decoder"].get("dropout", 0.0)),
            coordinate_scale=float(config["decoder"]["coordinate_scale"]),
            stop_rotation_gradient=bool(config["decoder"].get("stop_rotation_gradient", True)))
    elif kind == "attention":
        decoder = AttentionFlowDecoder(
            d_model=int(upstream_config["model"]["d_model"]), code_book=code_book,
            layers=int(config["decoder"]["layers"]),
            width=int(config["decoder"].get("width", 384)),
            heads=int(config["decoder"].get("heads", 6)),
            max_residues=int(config["max_residues"]),
            condition_mask_input=condition_mask_input,
            dropout=float(config["decoder"].get("dropout", 0.0)))
    elif kind == "gvp":
        decoder = GVPFlowDecoder(
            d_model=int(upstream_config["model"]["d_model"]),
            neighbours=int(config["decoder"]["neighbours"]),
            layers=int(config["decoder"]["layers"]),
            scalar_hidden=int(config["decoder"]["scalar_hidden"]),
            vector_hidden=int(config["decoder"]["vector_hidden"]))
    else:
        raise ValueError(f"unknown decoder kind: {kind!r}")
    return kind, decoder.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.validation_batches is not None:
        config["evaluation"]["validation_batches"] = args.validation_batches
    if args.sample_count is not None:
        config["evaluation"]["sample_count"] = args.sample_count
    if args.sample_steps is not None:
        config["evaluation"]["sample_steps"] = args.sample_steps

    upstream_path = Path(config["upstream_checkpoint"])
    if not upstream_path.exists():
        raise FileNotFoundError(f"Missing frozen upstream checkpoint: {upstream_path}")
    config["resolved_upstream_sha256"] = sha256(upstream_path)
    init_path = Path(config["init_checkpoint"]) if config.get("init_checkpoint") else None
    if init_path is not None:
        if not init_path.is_file():
            raise FileNotFoundError(f"Missing decoder initialization checkpoint: {init_path}")
        config["resolved_init_sha256"] = sha256(init_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = torch.load(upstream_path, map_location="cpu", weights_only=False)
    upstream_config = upstream.get("config")
    if not isinstance(upstream_config, dict):
        raise RuntimeError("Frozen upstream checkpoint has no resolved config")
    upstream_family = upstream_config.get("model_family")
    input_mode = upstream_config["model"].get("input_mode")
    if upstream_family not in PSJEPA_MODEL_FAMILIES:
        raise RuntimeError(
            f"the frozen upstream declares model_family={upstream_family!r}; this release's "
            f"decoder conditions on a continuous PSJEPA encoder only {PSJEPA_MODEL_FAMILIES}"
        )
    # `gvp_message` is `gvp` plus edge message passing; both leave only invariant scalars,
    # so the conditioning story below applies unchanged to either.
    if input_mode not in ("gvp", "gvp_message", "backbone"):
        raise RuntimeError(f"Flow reconstruction cannot condition on input_mode={input_mode!r}")
    if input_mode == "backbone":


        print("conditioning on a frame-carrying (non-invariant) encoder", flush=True)
    if upstream_config["model"].get("use_residue_identity", False):
        raise RuntimeError("Residue-identity checkpoints are forbidden")
    upstream_step = int(upstream.get("step", -1))
    # A released weights file carries the run's terminal status inside it, because it is
    # distributed on its own; a training checkpoint has it in status.json beside the weights.
    upstream_status = str(upstream.get("status") or "")
    if not upstream_status:
        status_file = upstream_path.parent / "status.json"
        if status_file.is_file():
            upstream_status = str(json.loads(status_file.read_text()).get("status", ""))


    if (upstream_step < int(upstream_config["training"]["max_steps"])
            and upstream_status not in ("plateau", "complete")):
        raise RuntimeError(f"Upstream checkpoint is incomplete at step {upstream_step} "
                           f"with status {upstream_status!r}")
    del upstream
    # `randomise_upstream: true` keeps the architecture and throws the trained weights away.
    # It is the control that makes a reconstruction number readable: the `backbone` input
    # path is a Linear applied to raw coordinates, so its embedding is a near-linear image
    # of the coordinates whatever the weights are -- a ridge probe recovered CA at 2.751 A
    # from the trained encoder and 3.263 A from an untrained one, a difference of 0.51 A.
    # Without this arm a good score cannot be attributed to pretraining.
    if bool(config.get("randomise_upstream", False)):
        model = PSJEPA.from_dict(upstream_config["model"], upstream_config["masking"],
                                   upstream_config["loss"])
        model.readout = "student"
        tokenizer = SemanticTokenizer(model, device)
        print("UPSTREAM WEIGHTS DISCARDED: architecture only", flush=True)
    else:
        tokenizer = SemanticTokenizer.from_checkpoint(upstream_path, device)
    encoder = tokenizer.model
    encoder.requires_grad_(False)
    encoder.eval().to(device)


    kind, decoder = build_decoder(config, upstream_config, device)
    print(f"decoder kind={kind}  parameters="
          f"{sum(p.numel() for p in decoder.parameters()):,}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.95),
    )

    last_path = output_dir / "last.pt"
    step = epoch = batch_in_epoch = 0
    micro = 0
    accumulation = int(config["training"].get("accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("accumulation_steps must be at least 1")
    best_validation = float("inf")
    prior_peak_memory = 0.0
    status_path = output_dir / "status.json"
    if status_path.exists():
        try:
            prior_peak_memory = float(json.loads(status_path.read_text()).get("cuda_peak_memory_gb", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if args.resume == "auto" and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            raise RuntimeError("Refusing downstream resume because the resolved config changed")
        decoder.load_state_dict(state["decoder"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, epoch = int(state["step"]), int(state["epoch"])
        batch_in_epoch = int(state.get("batch_in_epoch", 0))
        best_validation = float(state.get("best_validation", float("inf")))
        restore_rng(state["rng"])
        print(f"Resumed downstream decoder at step={step}, epoch={epoch}, batch={batch_in_epoch}")
    elif init_path is not None:
        # Fine-tuning starts a new schedule and optimizer but inherits only the decoder.
        # This is deliberately distinct from resume: its step zero and resolved config make
        # the resulting experiment independently reproducible.
        state = torch.load(init_path, map_location="cpu", weights_only=False)
        if "decoder" not in state:
            raise RuntimeError(f"Initialization checkpoint has no decoder state: {init_path}")
        decoder.load_state_dict(state["decoder"], strict=True)
        print(
            f"Initialized downstream decoder from {init_path} at source step={state.get('step')}",
            flush=True,
        )

    # The cache's split names are configurable, as they are for the token prior: the encoder's own
    # corpus lives at `cache/afdb/{train,validation}`, not at the default `train`.
    train_split = str(config["data"].get("train_split", "train"))
    validation_split = str(config["data"].get("validation_split", "validation"))
    train_dataset = BackboneDataset(
        config["data"]["cache"], train_split, int(config["max_residues"]), True,
        config["data"].get("train_limit")
    )
    validation_dataset = BackboneDataset(
        config["data"]["cache"], validation_split, int(config["max_residues"]), False,
        config["data"].get("validation_limit")
    )
    print(f"data {config['data']['cache']} splits {train_split}/{validation_split}: "
          f"{len(train_dataset)} train, {len(validation_dataset)} validation chains", flush=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_backbones,
    )

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    decoder.train()
    max_steps = int(config["training"]["max_steps"])
    warmup_steps = int(config["training"]["warmup_steps"])
    if warmup_steps != max(1, round(0.05 * max_steps)) and args.max_steps is None:
        raise RuntimeError("Downstream training must use exactly 5% warm-up")
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    train_log = output_dir / "train_metrics.jsonl"
    validation_log = output_dir / "validation_metrics.jsonl"
    while step < max_steps and not STOP_REQUESTED:
        loader = make_loader(train_dataset, config, epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_in_epoch:
                continue
            batch = device_batch(batch, device)
            # `no_grad`, not `inference_mode`: the latter marks every tensor it produces,
            # the mark is permanent -- `.detach().clone()` does not clear it -- and the
            # decoder's backward pass then refuses the condition. The encoder is frozen and
            # `representation` detaches, so nothing here reaches the encoder either way.
            with torch.no_grad(), autocast_context(device, config["training"]["precision"]):
                condition = representation(encoder, batch, config["representation"])
            motif = motif_mask_for(batch, config)
            condition = apply_motif_condition(condition, encoder, batch, motif, config)
            # A motif is drawn for two separable reasons: to hold its coordinates clean and score
            # only the rest, or purely to mark which residues carry a site-only token.
            # `hold_coordinates: false` does the second alone, which is the setting for a decoder
            # that has to place the site itself.
            hold = motif if hold_motif_coordinates(config) else None
            predict_on = batch["residue_mask"] if hold is None else (
                batch["residue_mask"] & ~hold)
            if motif is not None:
                condition = drop_condition(condition, config, preserve_mask=motif)
            condition = drop_scaffold_tokens(condition, config, motif)
            with autocast_context(device, config["training"]["precision"]):
                loss, metrics = conditional_flow_matching_loss(
                    decoder,
                    condition,
                    batch["coords"],
                    batch["atom_mask"],
                    predict_on,
                    float(config["decoder"]["coordinate_scale"]),
                    bool(config.get("align_target", False)),
                    bool(config.get("predict_x1", False)),
                    float(config.get("pair_weight", 0.0)),
                    float(config.get("pair_min_time", 0.5)),
                    bool(config.get("canonical_target", False)),
                    float(config.get("fape_weight", 0.0)),
                    float(config.get("adjacent_weight", 0.0)),
                    hold,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite downstream flow loss at step {step}: {loss}")
            # Gradient accumulation splits one optimiser step over several forward passes, so a
            # recipe fitted at batch 48 can run unchanged on a GPU that only holds 24 at a time:
            # the loss is scaled so the accumulated gradient equals the large-batch gradient, and
            # `step` -- which drives the learning-rate schedule, logging, validation and
            # `max_steps` -- still counts optimiser steps, not forward passes. At the default of 1
            # every line below behaves exactly as it did.
            (loss / accumulation).backward()
            if any(parameter.grad is not None for parameter in encoder.parameters()):
                raise RuntimeError("Frozen upstream encoder unexpectedly received a gradient")
            micro += 1
            batch_in_epoch = batch_index + 1
            stepped = micro % accumulation == 0
            if not stepped:
                continue
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(config["training"]["grad_clip"]))
            lr = learning_rate(
                step, max_steps, warmup_steps, float(config["training"]["learning_rate"])
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % int(config["training"]["log_every"]) == 0 or step == 1:
                record = {
                    "step": step,
                    "loss_flow": float(loss.detach()),
                    # The total hides which term is moving. An arm exists to change one of them
                    # -- pair distances for v5, the junction for v4 -- and without its own column
                    # there is no way to tell a term that did not help from one that never
                    # trained. Every component the loss already returns is written out.
                    **{k: float(v) for k, v in metrics.items() if k.startswith("loss_")},
                    "flow_velocity_target_rms": float(metrics["flow_velocity_target_rms"]),
                    "flow_velocity_prediction_rms": float(metrics["flow_velocity_prediction_rms"]),
                    "lr": lr,
                    "elapsed_seconds": time.time() - started,
                    "upstream_frozen": True,
                }
                if device.type == "cuda":
                    record.update(
                        cuda_memory_allocated_gb=torch.cuda.memory_allocated(device) / (1024 ** 3),
                        cuda_peak_memory_gb=torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    )
                append_json(train_log, record)
                print(json.dumps(record, sort_keys=True), flush=True)

            should_validate = step % int(config["training"]["validate_every"]) == 0 or step == max_steps
            if should_validate:
                validation = validate(encoder, decoder, validation_loader, config, device, step)
                validation["step"] = step
                validation["upstream_frozen"] = True
                append_json(validation_log, validation)
                print(json.dumps(validation, sort_keys=True), flush=True)
                if validation["val_flow_loss"] < best_validation:
                    best_validation = validation["val_flow_loss"]
                    atomic_checkpoint(
                        checkpoint_payload(decoder, optimizer, config, step, epoch, batch_in_epoch, best_validation),
                        output_dir / "best.pt",
                    )
            if (
                step % int(config["training"]["checkpoint_every"]) == 0
                or should_validate
                or STOP_REQUESTED
            ):
                atomic_checkpoint(
                    checkpoint_payload(decoder, optimizer, config, step, epoch, batch_in_epoch, best_validation),
                    last_path,
                )
            if step >= max_steps or STOP_REQUESTED:
                break
        else:
            epoch += 1
            batch_in_epoch = 0
            continue
        break

    atomic_checkpoint(
        checkpoint_payload(decoder, optimizer, config, step, epoch, batch_in_epoch, best_validation),
        last_path,
    )
    status = {
        "status": "complete" if step >= max_steps else "preempted",
        "step": step,
        "max_steps": max_steps,
        "representation": config["representation"],
        "upstream_checkpoint": str(upstream_path),
        "upstream_sha256": config["resolved_upstream_sha256"],
        "upstream_frozen": True,
    }
    if device.type == "cuda":
        status["cuda_peak_memory_gb"] = max(
            prior_peak_memory, torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
