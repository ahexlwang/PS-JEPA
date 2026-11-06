'Attention-based flow decoder for Cartesian backbone coordinates.'
from __future__ import annotations

import math

import torch
from torch import nn


def _centred(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove the mean over all present atoms, so the flow is translation free."""
    weight = mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=(-3, -2), keepdim=True).clamp_min(1.0)
    centre = (values * weight).sum(dim=(-3, -2), keepdim=True) / count
    return (values - centre) * weight


class TimestepEmbedding(nn.Module):
    """Sinusoidal time features into an MLP, as in DiT."""

    def __init__(self, width: int, frequencies: int = 128) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.mlp = nn.Sequential(
            nn.Linear(frequencies, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.frequencies // 2
        scale = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = time.float().unsqueeze(-1) * scale * 1000.0
        features = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.mlp(features.to(self.mlp[0].weight.dtype))


class Block(nn.Module):
    """Pre-norm self-attention and MLP, both modulated by the timestep (adaLN-Zero)."""

    def __init__(self, width: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        # six vectors: shift/scale/gate for the attention and for the MLP
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        parts = self.modulation(t).unsqueeze(1).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = parts
        h = self.norm1(x) * (1 + scale_a) + shift_a
        attended, _ = self.attention(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + gate_a * attended
        h = self.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(h)


class AttentionFlowDecoder(nn.Module):
    """Velocity for four backbone atoms per residue, from state + time + condition only."""

    def __init__(
        self,
        d_model: int,
        atoms: int = 4,
        layers: int = 6,
        width: int = 384,
        heads: int = 6,
        max_residues: int = 512,
        dropout: float = 0.0,
        code_book: int = 0,
        condition_mask_input: bool = False,
    ) -> None:
        super().__init__()
        self.atoms = int(atoms)
        self.width = int(width)
        self.state_in = nn.Linear(self.atoms * 3, width)
        # A nonzero `code_book` means the condition arrives as integers rather than
        # vectors: the decoder is handed N symbols and learns its own table for them, so
        # nothing of the encoder's continuous space reaches it.
        self.code_book = int(code_book)
        if self.code_book:
            self.condition_in = nn.Embedding(self.code_book + 1, width)   # +1 for padding
        else:
            self.condition_in = nn.Linear(d_model, width)
        self.target_in = nn.Embedding(2, width)          # is this residue supervised
        # Whether a latent was PROVIDED at this residue, which is not the same question as
        # what the latent says. A blank condition is written as zeros everywhere in this
        # codebase, so "no latent here" and "a latent that happens to be near zero" look
        # alike; with a site-only condition that ambiguity covers most of the chain. This
        # embedding is new state, so a checkpoint trained without it cannot be loaded into a
        # decoder that has it -- which is why it is off by default and opt-in per run.
        self.condition_mask_in = (
            nn.Embedding(2, width) if bool(condition_mask_input) else None)
        self.time_embed = TimestepEmbedding(width)
        self.position = nn.Parameter(torch.zeros(1, max_residues, width))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(int(layers)))
        self.norm_out = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.state_out = nn.Linear(width, self.atoms * 3)
        # Zero output, so the decoder starts by predicting no motion at all and every
        # nonzero velocity it later produces was learned rather than an init artefact.
        for module in (self.modulation_out[1], self.state_out):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        condition: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        time: torch.Tensor,
        prediction_mask: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = condition.shape[:2]
        if length > self.position.shape[1]:
            raise ValueError(f"{length} residues exceeds max_residues {self.position.shape[1]}")
        present = atom_mask.unsqueeze(-1).to(coordinates.dtype)
        state = (coordinates * present).reshape(batch, length, self.atoms * 3)

        if self.code_book:
            condition = condition.long().clamp_max(self.code_book)
        x = self.state_in(state) + self.condition_in(condition)
        x = x + self.target_in(prediction_mask.long()) + self.position[:, :length]
        if self.condition_mask_in is not None:
            if condition_mask is None:
                raise ValueError(
                    "this decoder was built with condition_mask_input: true and needs a "
                    "condition_mask saying where a latent was provided")
            x = x + self.condition_mask_in(condition_mask.long())
        t = self.time_embed(time).to(x.dtype)


        pad = ~atom_mask.any(dim=-1)
        pad = pad & ~pad.all(dim=-1, keepdim=True)
        for block in self.blocks:
            x = block(x, t, pad)
        shift, scale = self.modulation_out(t).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale) + shift
        velocity = self.state_out(x).reshape(batch, length, self.atoms, 3)
        return _centred(velocity, atom_mask)
