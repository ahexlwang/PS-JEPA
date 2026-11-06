from __future__ import annotations

import torch
from torch import nn

from psjepa.models.mamba_encoder import BiMambaStack, MambaEncoderConfig


class ContextPredictor(nn.Module):
    'Mask token + bidirectional Mamba + linear read-out, at an optional narrower width.'

    def __init__(
        self, d_model: int, n_layer: int = 2, backend: str = "mamba", width: int | None = None
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.width = int(width) if width is not None else self.d_model
        self.input = nn.Linear(self.d_model, self.width) if self.width != self.d_model else None
        self.mask_token = nn.Parameter(torch.zeros(self.width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.stack = BiMambaStack(
            MambaEncoderConfig(d_model=self.width, n_layer=n_layer, bidirectional=True, backend=backend)
        )
        self.output = nn.Linear(self.width, self.d_model)

    def forward(
        self, x: torch.Tensor, prediction_mask: torch.Tensor, residue_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.input is not None:
            x = self.input(x)
        x = x + prediction_mask.unsqueeze(-1).to(x.dtype) * self.mask_token
        return self.output(self.stack(x, ~residue_mask))


class TransformerContextPredictor(nn.Module):
    """Predict target residues from visible context without reusing target states.

    The context encoder still emits an aligned length-L tensor, but only entries in
    ``context_mask`` are admitted to attention.  Entries in ``prediction_mask`` are
    replaced by learned mask tokens carrying an absolute residue-position embedding;
    every other entry is padding for this predictor call.  Consequently one target
    block cannot read another block or a context-encoder state at its own position.
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int = 2,
        max_residues: int = 512,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.max_residues = int(max_residues)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        self.position = nn.Parameter(torch.empty(self.max_residues, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=int(round(mlp_ratio * d_model)),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=n_layer, norm=nn.LayerNorm(d_model))
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        prediction_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        length = x.shape[1]
        if length > self.max_residues:
            raise ValueError(
                f"sequence length {length} exceeds predictor table {self.max_residues}"
            )
        target = prediction_mask & residue_mask
        context = context_mask & residue_mask & ~target
        valid = context | target
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any() and length:
            # PyTorch attention returns NaNs when every key in a row is padded.
            # Admit one zero placeholder internally, then remove it from the output.
            safe_valid[empty, 0] = True
        position = self.position[:length].unsqueeze(0).to(x.dtype)
        mask = self.mask_token.view(1, 1, -1).to(x.dtype)
        hidden = torch.where(target.unsqueeze(-1), mask, x) + position
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        predicted = self.stack(hidden, src_key_padding_mask=~safe_valid)
        return self.output(predicted) * valid.unsqueeze(-1).to(predicted.dtype)

    def forward_sparse(
        self,
        context: torch.Tensor,
        context_positions: torch.Tensor,
        context_valid: torch.Tensor,
        target_positions: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Predict packed target positions from packed crop-context tokens."""
        if context_positions.numel():
            maximum = int(context_positions[context_valid].max()) if context_valid.any() else -1
            if maximum >= self.max_residues:
                raise ValueError(f"context position {maximum} exceeds table {self.max_residues}")
        if target_positions.numel():
            maximum = int(target_positions[target_valid].max()) if target_valid.any() else -1
            if maximum >= self.max_residues:
                raise ValueError(f"target position {maximum} exceeds table {self.max_residues}")

        context_pos = self.position[context_positions.clamp(0, self.max_residues - 1)].to(context.dtype)
        target_pos = self.position[target_positions.clamp(0, self.max_residues - 1)].to(context.dtype)
        context_input = (context + context_pos) * context_valid.unsqueeze(-1).to(context.dtype)
        target_input = (
            self.mask_token.view(1, 1, -1).to(context.dtype) + target_pos
        ) * target_valid.unsqueeze(-1).to(context.dtype)
        hidden = torch.cat([context_input, target_input], dim=1)
        valid = torch.cat([context_valid, target_valid], dim=1)
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any() and safe_valid.shape[1]:
            safe_valid[empty, 0] = True
        predicted = self.stack(hidden, src_key_padding_mask=~safe_valid)
        target = predicted[:, context.shape[1]:]
        return self.output(target) * target_valid.unsqueeze(-1).to(target.dtype)
