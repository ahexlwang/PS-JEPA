"""Sketched isotropic-Gaussian regularizer for per-residue JEPA tokens.

Uses the Epps-Pulley characteristic-function statistic from LeJEPA. Random 1D
projections test the *vector* distribution, including dependencies between its
channels. This does not constrain the joint distribution across residues.
"""
from __future__ import annotations

import torch


def sigreg_tokens(tokens: torch.Tensor, valid: torch.Tensor, *,
                  slices: int = 64, per_chain: int = 16) -> torch.Tensor:
    """Compare equally sampled chains' valid token vectors to N(0, I)."""
    if tokens.ndim != 3 or valid.shape != tokens.shape[:2]:
        raise ValueError("expected tokens [B,L,D] and valid [B,L]")
    picked = []
    for row, mask in zip(tokens, valid):
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if len(indices):
            if len(indices) > per_chain:
                indices = indices[torch.randperm(len(indices), device=indices.device)[:per_chain]]
            picked.append(row[indices])
    if not picked:
        return tokens.float().sum() * 0.0
    device = tokens.device
    with torch.autocast(device_type=device.type, enabled=False):
        x = torch.cat(picked, dim=0).float()
        directions = torch.randn(x.shape[-1], slices, device=device, dtype=torch.float32)
        directions = torch.nn.functional.normalize(directions, dim=0)
        t = torch.linspace(-5, 5, 17, device=device, dtype=torch.float32)
        phase = (x @ directions).unsqueeze(-1) * t
        expected = torch.exp(-0.5 * t.square())
        real_error = phase.cos().mean(0) - expected
        imag_error = phase.sin().mean(0)
        statistic = torch.trapezoid(
            (real_error.square() + imag_error.square()) * expected, t, dim=-1)
        # LeJEPA's statistic multiplies by N so the Gaussian null stays O(1).
        return statistic.mean() * x.shape[0]
