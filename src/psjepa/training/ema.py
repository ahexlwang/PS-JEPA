from __future__ import annotations

import math
import torch
from torch import nn


def cosine_ema_momentum(step: int, total_steps: int, start: float = 0.996) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return 1.0 - (1.0 - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def resolve_ema_start(max_steps: int, window_fraction: float) -> float:
    """Return the initial momentum whose averaging window is a fixed share of the run.

    The starting momentum sets a teacher averaging window of ``1 / (1 - start)``
    optimizer steps. That window is meaningful only relative to the run length:
    the screening default of 0.996 spans 250 steps, which is 1.25% of a
    20,000-step run but 40% of a 620-step large-batch run, so a teacher tuned for
    the former barely departs from initialization in the latter. Deriving the
    momentum from ``window_fraction`` keeps the ratio fixed when the physical
    batch size changes the optimizer-step count at a constant example budget.
    """
    if not 0.0 < window_fraction <= 1.0:
        raise ValueError(f"EMA window fraction must be in (0, 1]: {window_fraction}")
    window = window_fraction * max(int(max_steps), 1)
    if window <= 1.0:
        return 0.0
    return 1.0 - 1.0 / window


@torch.no_grad()
def update_ema(target: nn.Module, context: nn.Module, momentum: float) -> None:
    context_params = dict(context.named_parameters())
    for name, target_value in target.named_parameters():
        target_value.mul_(momentum).add_(context_params[name], alpha=1.0 - momentum)
    context_buffers = dict(context.named_buffers())
    for name, target_value in target.named_buffers():
        source = context_buffers[name]
        if target_value.is_floating_point():
            target_value.mul_(momentum).add_(source, alpha=1.0 - momentum)
        else:
            target_value.copy_(source)


@torch.no_grad()
def copy_encoder(target: nn.Module, context: nn.Module) -> None:
    target.load_state_dict(context.state_dict(), strict=True)
    target.requires_grad_(False)

