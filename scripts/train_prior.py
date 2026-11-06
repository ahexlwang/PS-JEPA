#!/usr/bin/env python3
'Train a rectified-flow prior over frozen per-residue representations.'
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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psjepa.analysis.generation_metrics import batch_metrics
from psjepa.config import load_config, save_config
from psjepa.paths import resolve_recorded_checkpoint
from psjepa.data import BackboneDataset, collate_backbones
from psjepa.layers.flow_matching import sample_flow
from psjepa.data.masking import sample_motif_mask
from psjepa.tokenization.motif_tokens import motif_tokens
from psjepa.models.token_prior import (
    TokenFlowPrior,
    TokenPriorConfig,
    load_prior_state_dict,
    prior_flow_loss,
    sample_tokens,
)
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motif_mask_for(batch, config):
    """A motif to hold during training, drawn like the decoder's, or None.

    `motif.probability` sets how often a batch is an inpainting batch. The rest are ordinary
    unconditional steps, so one model serves both: draw a whole protein, or complete one
    around a site that is already fixed.
    """
    settings = config.get("motif") or {}
    if not settings.get("enabled", False):
        return None
    if float(settings.get("probability", 1.0)) < 1.0 and \
            float(torch.rand(())) >= float(settings["probability"]):
        return None
    return sample_motif_mask(
        batch["coords"], batch["atom_mask"], batch["residue_mask"],
        min_fraction=float(settings.get("min_fraction", 0.05)),
        max_fraction=float(settings.get("max_fraction", 0.30)),
        spatial_probability=float(settings.get("spatial_probability", 0.7)),
        max_segments=int(settings.get("max_segments", 3)),
    )

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


@torch.no_grad()
def encode_tokens(encoder, batch: dict[str, Any]) -> torch.Tensor:
    """The frozen encoder's continuous per-residue stream, detached.

    Residue identity is never passed: the generator must not be able to copy a
    sequence it was told about.
    """
    tokens = encoder.tokenize(batch["coords"], batch["atom_mask"], batch["residue_mask"])
    return tokens["continuous_semantic"].detach()


def make_loader(dataset, config: dict[str, Any], epoch: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_backbones,
        drop_last=shuffle,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )


@torch.no_grad()
def measure_token_statistics(
    encoder, dataset, config: dict[str, Any], device: torch.device, batches: int
) -> tuple[torch.Tensor, torch.Tensor]:
    'Per-dimension mean and standard deviation over valid residues.'
    loader = make_loader(dataset, config, epoch=0, shuffle=True)
    total = torch.zeros(int(config["model"]["d_token"]), dtype=torch.float64, device=device)
    total_square = torch.zeros_like(total)
    count = 0
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = device_batch(batch, device)
        tokens = encode_tokens(encoder, batch).float()
        selected = tokens[batch["residue_mask"]]
        total += selected.sum(dim=0).double()
        total_square += selected.square().sum(dim=0).double()
        count += selected.shape[0]
    if count < 2:
        raise RuntimeError("token statistics need at least two residues")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1e-12)
    print(
        f"token statistics over {count} residues: "
        f"mean |.| {float(mean.abs().mean()):.4f}, std mean {float(variance.sqrt().mean()):.4f}",
        flush=True,
    )
    return mean.float(), variance.sqrt().float()


def length_histogram(cache_root: str, split: str, max_residues: int,
                     length_range: tuple[int, int] | None = None) -> tuple[list[int], list[int]]:
    """Chain lengths of the training split, as the prior actually saw them.

    Sampling draws from this, so it has to describe the *filtered* pool. With a
    `length_range` the out-of-range chains were dropped, not cropped, and
    clipping them to the window here would put mass at exactly `max_residues`
    that the model never trained on.
    """
    offsets = np.load(Path(cache_root) / split / "offsets.npy")
    lengths = np.diff(offsets).astype(np.int64)
    if length_range is not None:
        low, high = int(length_range[0]), int(length_range[1])
        lengths = lengths[(lengths >= low) & (lengths <= high)]
    else:
        lengths = np.clip(lengths, 4, max_residues)
    values, counts = np.unique(lengths, return_counts=True)
    return [int(v) for v in values], [int(c) for c in counts]


def build_frozen_decoder(config: dict[str, Any], device: torch.device):
    """The trained flow decoder named by the config, in eval mode.

    Imported lazily so a prior can be trained (or a smoke run) on a machine where
    the decoder checkpoint is absent.
    """
    from train_decoder import build_decoder

    path = Path(config["decoder_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen decoder checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    decoder_config = state["config"]
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
    kind, decoder = build_decoder(decoder_config, upstream_state["config"], device)
    decoder.load_state_dict(state["decoder"], strict=True)
    decoder.eval().requires_grad_(False)
    print(
        f"frozen decoder {kind} from {path} at step {state.get('step')}, "
        f"predict_x1={decoder_config.get('predict_x1')}",
        flush=True,
    )
    return decoder, decoder_config


@torch.no_grad()
def decode_samples(
    prior: TokenFlowPrior,
    decoder,
    decoder_config: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    count: int,
    seed: int,
) -> dict[str, float]:
    """Sample lengths, sample tokens, decode, and score the geometry."""
    was_training = prior.training
    prior.eval()
    values = np.array(prior.config.length_values)
    weights = np.array(prior.config.length_counts, dtype=np.float64)
    rng = np.random.RandomState(seed)
    lengths = rng.choice(values, size=count, p=weights / weights.sum())
    longest = int(lengths.max())
    residue_mask = torch.zeros(count, longest, dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        residue_mask[row, : int(length)] = True
    atom_mask = residue_mask.unsqueeze(-1).expand(-1, -1, 4).contiguous()

    tokens = sample_tokens(
        prior, residue_mask, steps=int(config["evaluation"]["token_steps"])
    )
    coordinates = sample_flow(
        decoder,
        tokens.to(next(decoder.parameters()).dtype),
        atom_mask,
        residue_mask,
        float(decoder_config["decoder"]["coordinate_scale"]),
        int(config["evaluation"]["decoder_steps"]),
        bool(decoder_config.get("predict_x1", False)),
    )
    summary, _ = batch_metrics(coordinates.float().cpu(), residue_mask.cpu())
    if was_training:
        prior.train()
    return {f"gen_{key}": value for key, value in summary.items()}


@torch.no_grad()
def validate(
    encoder,
    prior: TokenFlowPrior,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    step: int,
) -> dict[str, float]:
    prior.eval()
    arm_sampling = str(config["training"].get("time_sampling", "uniform"))


    samplings = {"val_prior_loss": arm_sampling}
    if arm_sampling != "uniform":
        samplings["val_prior_loss_uniform"] = "uniform"
    motif_settings = config.get("motif") or {}
    motif_enabled = bool(motif_settings.get("enabled", False))
    isolated_condition = bool(motif_settings.get("isolated", False))
    motif_encoding = str(motif_settings.get("encoding", "in_place"))
    if motif_enabled:
        samplings["val_motif_prior_loss"] = arm_sampling
    loss_sums = {key: 0.0 for key in samplings}
    batches = 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1729 + step)
        for batch in loader:
            batch = device_batch(batch, device)
            with autocast_context(device, config["training"]["precision"]):
                tokens = encode_tokens(encoder, batch)
                for key, sampling in samplings.items():


                    given_mask = None
                    given_tokens = None
                    if key == "val_motif_prior_loss":
                        # Validation must exercise the same inpainting problem as training.
                        # Previously motif priors were selected solely by unconditional loss,
                        # which can improve while the quantity used by scaffolding regresses.
                        given_mask = sample_motif_mask(
                            batch["coords"], batch["atom_mask"], batch["residue_mask"],
                            min_fraction=float(motif_settings.get("min_fraction", 0.05)),
                            max_fraction=float(motif_settings.get("max_fraction", 0.30)),
                            spatial_probability=float(
                                motif_settings.get("spatial_probability", 0.7)),
                            max_segments=int(motif_settings.get("max_segments", 3)),
                        )
                        if isolated_condition:
                            given_tokens = motif_tokens(
                                encoder, batch["coords"], batch["atom_mask"],
                                batch["residue_mask"], given_mask, motif_encoding,
                            )
                    loss, _ = prior_flow_loss(
                        prior, tokens, batch["residue_mask"], sampling, 0.0,
                        given_mask=given_mask, given_tokens=given_tokens,
                    )
                    loss_sums[key] += float(loss)
            batches += 1
            if batches >= int(config["evaluation"]["validation_batches"]):
                break
    prior.train()
    result = {key: value / max(batches, 1) for key, value in loss_sums.items()}
    result.setdefault("val_prior_loss_uniform", result["val_prior_loss"])
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"Non-finite prior validation metric: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--statistics-batches", type=int)
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("bottleneck_checkpoint"):
        raise ValueError("This trainer requires full-width encoder representations")
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.train_limit is not None:
        config["data"]["train_limit"] = args.train_limit
    if args.statistics_batches is not None:
        config["training"]["statistics_batches"] = args.statistics_batches

    upstream_path = Path(config["upstream_checkpoint"])
    if not upstream_path.exists():
        raise FileNotFoundError(f"Missing frozen upstream checkpoint: {upstream_path}")
    config["resolved_upstream_sha256"] = sha256(upstream_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The raw module, not the SemanticTokenizer wrapper: the wrapper's
    # `encode_batch` runs under `inference_mode`, whose mark is permanent, and a
    # tensor carrying it cannot be used as a conditioning input afterwards.
    encoder = SemanticTokenizer.from_checkpoint(config["upstream_checkpoint"], device=device).model
    encoder.requires_grad_(False).eval().to(device)
    d_model = int(getattr(encoder.config, "d_model", 0))
    if d_model and d_model != int(config["model"]["d_token"]):
        raise RuntimeError(
            f"config d_token={config['model']['d_token']} but the encoder emits {d_model}"
        )

    max_residues = int(config["max_residues"])
    # The cache's split names are configurable because the encoder's own corpus
    # lives under `cache/afdb/{train,validation}` while CATH uses plain
    # `train`/`validation`. `length_range` drops chains longer than the window
    # instead of cropping them: a generator fitted to cropped domains learns to
    # emit truncated folds.
    train_split = str(config["data"].get("train_split", "train"))
    validation_split = str(config["data"].get("validation_split", "validation"))
    length_range = config["data"].get("length_range")
    if length_range is not None:
        length_range = (int(length_range[0]), int(length_range[1]))
    train_set = BackboneDataset(
        config["data"]["cache"], train_split, max_residues=max_residues,
        augment=False, limit=config["data"].get("train_limit"), length_range=length_range,
    )
    validation_set = BackboneDataset(
        config["data"]["cache"], validation_split, max_residues=max_residues,
        augment=False, limit=config["data"].get("validation_limit"), length_range=length_range,
    )
    print(f"cache {config['data']['cache']} splits {train_split}/{validation_split}: "
          f"train chains {len(train_set)}, validation chains {len(validation_set)}", flush=True)

    prior_config = TokenPriorConfig(
        d_token=int(config["model"]["d_token"]),
        width=int(config["model"]["width"]),
        layers=int(config["model"]["layers"]),
        heads=int(config["model"]["heads"]),
        mlp_ratio=float(config["model"].get("mlp_ratio", 4.0)),
        max_residues=max_residues,
        dropout=float(config["model"].get("dropout", 0.0)),
        self_conditioning=bool(config["model"].get("self_conditioning", False)),
        rotary_position=bool(config["model"].get("rotary_position", False)),
        rotary_base=float(config["model"].get("rotary_base", 10000.0)),
        distogram_weight=float(config["model"].get("distogram_weight", 0.0)),
        distogram_bins=int(config["model"].get("distogram_bins", 32)),
        distogram_max=float(config["model"].get("distogram_max", 24.0)),
        distogram_width=int(config["model"].get("distogram_width", 64)),
    )


    known_model_keys = {
        "d_token", "width", "layers", "heads", "mlp_ratio", "dropout",
        "self_conditioning", "rotary_position", "rotary_base",
        "distogram_weight", "distogram_bins", "distogram_max", "distogram_width",
    }
    unknown = set(config["model"]) - known_model_keys
    if unknown:
        raise ValueError(
            f"config model section has keys this trainer does not read: {sorted(unknown)}"
        )
    prior = TokenFlowPrior(prior_config).to(device)
    parameters = sum(p.numel() for p in prior.parameters())
    print(f"prior parameters {parameters/1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(
        prior.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.95),
    )

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    step = epoch = 0
    best_validation = float("inf")
    if args.resume == "auto" and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        load_prior_state_dict(prior, state["prior"])
        prior.config.token_mean = state["prior_config"]["token_mean"]
        prior.config.token_std = state["prior_config"]["token_std"]
        prior.config.length_values = state["prior_config"]["length_values"]
        prior.config.length_counts = state["prior_config"]["length_counts"]
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        epoch = int(state["epoch"])
        best_validation = float(state.get("best_validation", float("inf")))
        prior.to(device)
        print(f"resumed from {last_path} at step {step}", flush=True)
    else:
        mean, std = measure_token_statistics(
            encoder, train_set, config, device,
            int(config["training"].get("statistics_batches", 200)),
        )
        prior.set_token_statistics(mean, std)
        values, counts = length_histogram(
            config["data"]["cache"], train_split, max_residues, length_range)
        prior.config.length_values = values
        prior.config.length_counts = counts

    decoder = decoder_config = None
    if config.get("decoder_checkpoint"):
        decoder, decoder_config = build_frozen_decoder(config, device)

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    total_steps = int(config["training"]["max_steps"])
    warmup = int(config["training"]["warmup_steps"])
    peak = float(config["training"]["learning_rate"])
    grad_clip = float(config["training"]["grad_clip"])
    log_every = int(config["training"]["log_every"])
    validate_every = int(config["training"]["validate_every"])
    checkpoint_every = int(config["training"]["checkpoint_every"])
    decode_every = int(config["evaluation"].get("decode_every", 1))
    time_sampling = str(config["training"].get("time_sampling", "uniform"))

    metrics_path = output_dir / "train_metrics.jsonl"
    validation_path = output_dir / "validation_metrics.jsonl"
    validation_loader = make_loader(validation_set, config, epoch=0, shuffle=False)

    prior.train()
    isolated_condition = bool((config.get("motif") or {}).get("isolated", False))
    motif_encoding = str((config.get("motif") or {}).get("encoding", "in_place"))
    started = time.time()
    validations = 0
    while step < total_steps and not STOP_REQUESTED:
        loader = make_loader(train_set, config, epoch)
        for batch in loader:
            if step >= total_steps or STOP_REQUESTED:
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps, warmup, peak)
            batch = device_batch(batch, device)
            with autocast_context(device, config["training"]["precision"]):
                tokens = encode_tokens(encoder, batch)
                motif_mask = motif_mask_for(batch, config)
                held = None
                if motif_mask is not None and isolated_condition:
                    held = motif_tokens(encoder, batch["coords"], batch["atom_mask"],
                                        batch["residue_mask"], motif_mask, motif_encoding)
                loss, metrics = prior_flow_loss(
                    prior, tokens, batch["residue_mask"], time_sampling,
                    float(config["training"].get("self_conditioning_probability", 0.5)),
                    motif_mask,
                    batch["coords"],
                    given_tokens=held,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), grad_clip)
            optimizer.step()
            step += 1

            if step % log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss),
                    "grad_norm": float(norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": round(time.time() - started, 1),
                    **metrics,
                }
                append_json(metrics_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
                if not math.isfinite(float(loss)):
                    raise RuntimeError(f"Non-finite training loss at step {step}")

            if step % validate_every == 0 or step == total_steps:
                record = {"step": step, **validate(encoder, prior, validation_loader, config, device, step)}
                validations += 1
                if decoder is not None and validations % decode_every == 0:
                    record.update(
                        decode_samples(
                            prior, decoder, decoder_config, config, device,
                            int(config["evaluation"]["sample_count"]), seed=step,
                        )
                    )
                append_json(validation_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)
                selection_metric = (
                    "val_motif_prior_loss"
                    if bool((config.get("motif") or {}).get("enabled", False))
                    else "val_prior_loss"
                )
                if record[selection_metric] < best_validation:
                    best_validation = record[selection_metric]
                    atomic_checkpoint(
                        {
                            "prior": prior.state_dict(),
                            "prior_config": prior.to_dict(),
                            "config": config,
                            "step": step,
                            "epoch": epoch,
                            "best_validation": best_validation,
                        },
                        best_path,
                    )

            if step % checkpoint_every == 0 or step == total_steps:
                atomic_checkpoint(
                    {
                        "prior": prior.state_dict(),
                        "prior_config": prior.to_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "step": step,
                        "epoch": epoch,
                        "best_validation": best_validation,
                    },
                    last_path,
                )
        epoch += 1

    atomic_checkpoint(
        {
            "prior": prior.state_dict(),
            "prior_config": prior.to_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "epoch": epoch,
            "best_validation": best_validation,
        },
        last_path,
    )
    status = "complete" if step >= total_steps else "interrupted"
    (output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": status,
                "step": step,
                "best_validation": best_validation,
                "parameters": parameters,
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print(f"{status} at step {step}", flush=True)


if __name__ == "__main__":
    main()
