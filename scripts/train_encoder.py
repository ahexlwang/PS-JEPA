#!/usr/bin/env python3
'Train the continuous PS-JEPA encoder with masked latent prediction.'
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from psjepa.config import load_config
from psjepa.data import BackboneDataset, collate_backbones
from psjepa.models import PSJEPA
from psjepa.tokenization.semantic_tokenizer import PSJEPA_MODEL_FAMILIES
from psjepa.training.ema import cosine_ema_momentum, resolve_ema_start
from psjepa.training.plateau import PlateauStopper

REPORT = ("loss", "loss_s2s", "loss_target_seen", "loss_target_unseen",
          "loss_target_shuffled", "loss_prediction_rms", "effective_rank",
          "feature_std_mean", "target_coverage", "unseen_coverage",
          "target_seen_fraction")
STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"received signal {signum}; checkpointing after this optimizer step", flush=True)


def collapsed(metrics, training) -> bool:
    """Continuous-representation collapse gate for checkpoint/plateau decisions."""
    rank_floor = float(training.get("collapse_rank_floor", 8.0))
    std_floor = float(training.get("collapse_std_floor", 0.05))
    return (
        float(metrics.get("val_effective_rank", float("inf"))) < rank_floor
        or float(metrics.get("val_feature_std_mean", float("inf"))) < std_floor
    )


def learning_rate(step, total, warmup, peak, schedule="plateau", level=1.0):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if schedule == "plateau":
        return peak * level
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def rewind_log(path: Path, step: int) -> int:
    """Drop rows a resumed run is about to write again.

    Both metric logs are opened in append mode, so a run that dies at step 900 and
    resumes from its step-624 checkpoint would leave rows 655..900 in the file and then
    write 655.. after them. The file is no longer monotonic in `step` and every reader
    here -- the plateau history, the aggregators, the notebook -- takes the last row for
    a step as the current one. Truncating to the checkpoint is what makes the log agree
    with the weights that were actually restored.
    """
    if not path.exists():
        return 0
    kept, dropped = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            if int(json.loads(line)["step"]) <= step:
                kept.append(line)
            else:
                dropped += 1
        except (ValueError, KeyError):
            dropped += 1
    path.write_text("".join(f"{line}\n" for line in kept))
    return dropped


def loader(cache, split, config, shuffle, epoch=0):
    dataset = BackboneDataset(cache, split, config["model"]["max_residues"],
                              augment=shuffle, limit=config["data"].get(f"{split}_limit"))
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(int(config.get("seed", 42)) + int(epoch))
    return DataLoader(dataset, batch_size=config["training"]["batch_size"],
                      shuffle=shuffle, num_workers=config["data"].get("num_workers", 4),
                      collate_fn=collate_backbones, drop_last=shuffle, pin_memory=True,
                      generator=generator)


def checkpoint_payload(model, optimizer, config, step, epoch, batch_in_epoch, stopper):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "stopper": {
            "best": stopper.best,
            "waited": stopper.waited,
            "reductions": stopper.reductions,
            "level": stopper.level,
            "history": list(stopper.history),
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def atomic_checkpoint(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(model, batches, device, limit):
    model.eval()
    sums, seen = {}, 0
    for index, batch in enumerate(batches):
        if index >= limit:
            break
        output = model({k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}, step=10 ** 9)
        for key in REPORT:
            if key in output:
                sums[key] = sums.get(key, 0.0) + float(output[key])
        seen += 1
    model.train()
    return {f"val_{k}": v / max(seen, 1) for k, v in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("model_family") not in PSJEPA_MODEL_FAMILIES:
        raise SystemExit(
            f"this trainer only runs model_family: psjepa "
            f"(earlier spellings {PSJEPA_MODEL_FAMILIES[1:]} are also accepted)"
        )
    if config["model"].get("quantize"):
        raise SystemExit("PSJEPA has no quantiser; set quantize: false")
    training = config["training"]
    if args.max_steps:
        training["max_steps"] = args.max_steps
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Path(out / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PSJEPA.from_dict(config["model"], config["masking"], config["loss"]).to(device)
    encoder_parameters = sum(p.numel() for p in model.context_encoder.parameters())
    print(f"PSJEPA: total {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M, "
          f"deployed encoder {encoder_parameters / 1e6:.3f}M", flush=True)


    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters,
                                  lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"],
                                  betas=(0.9, 0.95))
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    orphans = sorted(name for name, p in model.named_parameters()
                     if p.requires_grad and id(p) not in owned)
    if orphans:
        raise SystemExit(f"these trainable parameters are not being optimised: {orphans}")
    print(f"optimising {sum(p.numel() for p in parameters) / 1e6:.3f}M parameters "
          f"in {len(parameters)} tensors", flush=True)
    stopper = PlateauStopper(**training["plateau"])
    validation_batches = list(loader(config["data"]["cache"], "validation", config, False))
    accumulation = int(training.get("accumulation_steps", 1))
    total = int(training["max_steps"])
    horizon = int(training.get("ema_horizon", total))
    ema_max = float(training.get("ema_max", 0.9999))
    window_fraction = training.get("ema_window_fraction")
    ema_start = (
        resolve_ema_start(horizon, float(window_fraction))
        if window_fraction is not None else float(training["ema_start"])
    )
    scaler_dtype = torch.bfloat16 if training.get("precision") == "bf16" else torch.float32

    step = 0
    epoch = 0
    batch_in_epoch = 0
    micro = 0
    started = time.time()
    status = "complete"
    last_path = out / "last.pt"
    if args.resume == "auto" and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            raise RuntimeError("refusing PSJEPA resume with config drift")
        if "optimizer" not in state:
            raise RuntimeError("legacy PSJEPA checkpoint cannot resume; use a new output directory")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        epoch = int(state.get("epoch", 0))
        batch_in_epoch = int(state.get("batch_in_epoch", 0))
        saved_stopper = state.get("stopper", {})
        for key in ("best", "waited", "reductions", "level"):
            if key in saved_stopper:
                setattr(stopper, key, saved_stopper[key])
        stopper.history.extend(saved_stopper.get("history", []))
        rng = state.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        removed = sum(rewind_log(out / name, step) for name in
                      ("train_metrics.jsonl", "validation_metrics.jsonl"))
        print(f"resumed {last_path} at optimizer step {step}, epoch {epoch}, "
              f"batch {batch_in_epoch}; dropped {removed} log rows past the checkpoint",
              flush=True)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    model.train()


    optimizer.zero_grad(set_to_none=True)
    while step < total and not STOP_REQUESTED:
        train_loader = loader(config["data"]["cache"], "train", config, True, epoch)
        for batch_index, batch in enumerate(train_loader):
            if batch_index < batch_in_epoch:
                continue
            if step >= total or (STOP_REQUESTED and micro % accumulation == 0):
                break
            moved = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=scaler_dtype, enabled=device == "cuda"):
                output = model(moved, step=step)
            (output["loss"] / accumulation).backward()
            micro += 1
            batch_in_epoch = batch_index + 1
            if micro % accumulation:
                continue
            rate = learning_rate(step, total, training["warmup_steps"],
                                 training["learning_rate"], training.get("schedule", "plateau"),
                                 stopper.level)
            for group in optimizer.param_groups:
                group["lr"] = rate
            torch.nn.utils.clip_grad_norm_(parameters, training["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            # One teacher update per optimizer step, not per micro-batch: the cosine
            # schedule is written in optimizer steps and 128x4 was moving the teacher
            # four times as fast as `ema_horizon` intends.
            momentum = min(cosine_ema_momentum(min(step, horizon), horizon,
                                               ema_start), ema_max)
            model.update_target(momentum)

            if step % training["log_every"] == 0:
                append(out / "train_metrics.jsonl", {
                    "step": step, "lr": rate, "ema_momentum": momentum,
                    "elapsed_seconds": time.time() - started,
                    "cuda_peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                            if device == "cuda" else 0.0),
                    **{k: float(output[k]) for k in REPORT if k in output}})
            if step % training["validate_every"]:
                continue
            metrics = validate(model, validation_batches, device,
                               training["validation_batches"])
            is_collapsed = collapsed(metrics, training)
            decision = stopper.update(step, metrics["val_loss"], is_collapsed)
            record = {"step": step, "lr_level": stopper.level,
                      "collapsed": is_collapsed,
                      "plateau_waited": stopper.waited,
                      "plateau_reductions": stopper.reductions, **metrics}
            append(out / "validation_metrics.jsonl", record)
            print(f"step {step:6} " + "  ".join(
                f"{k.replace('val_', '')} {v:.4f}" for k, v in metrics.items()), flush=True)
            atomic_checkpoint(
                checkpoint_payload(model, optimizer, config, step, epoch, batch_in_epoch, stopper),
                last_path,
            )
            if decision == "stop":
                status = "plateau"
                break
            if step % int(training.get("checkpoint_every", training["validate_every"])) == 0:
                atomic_checkpoint(
                    checkpoint_payload(model, optimizer, config, step, epoch, batch_in_epoch, stopper),
                    last_path,
                )
            if STOP_REQUESTED:
                break
        else:
            epoch += 1
            batch_in_epoch = 0
            continue
        if status == "plateau" or STOP_REQUESTED or step >= total:
            break

    atomic_checkpoint(
        checkpoint_payload(model, optimizer, config, step, epoch, batch_in_epoch, stopper),
        last_path,
    )
    if STOP_REQUESTED:
        status = "preempted"
    (out / "status.json").write_text(json.dumps({
        "status": status, "step": step, "max_steps": total,
        "schedule": training.get("schedule"), "lr_level": stopper.level,
        "plateau_best": stopper.best if stopper.best != float("inf") else None,
        "plateau_reductions": stopper.reductions,
        "batch_size": training["batch_size"], "accumulation_steps": accumulation,
        "encoder_parameters": encoder_parameters,
        "cuda_peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                if device == "cuda" else 0.0),
    }, indent=2))
    print(f"done: {status} at step {step}", flush=True)


if __name__ == "__main__":
    main()
