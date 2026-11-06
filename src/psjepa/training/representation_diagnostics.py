'Batch-level checks on per-residue tokens that a falling loss cannot vouch for.'
from __future__ import annotations

import torch


def _rbf(values: torch.Tensor, centres: torch.Tensor, width: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((values[:, None] - centres[None]) / width).square())


def position_basis(length: int, device=None) -> torch.Tensor:
    """[length, 115] features of (i, L): RBFs over i, L-1-i and L, Fourier terms of i/(L-1)."""
    i = torch.arange(length, device=device, dtype=torch.float64)
    j = length - 1 - i
    ticks = torch.arange(0, 513, 16, device=device, dtype=torch.float64)
    frequencies = torch.arange(1, 9, device=device, dtype=torch.float64)
    phase = torch.pi * (i / max(length - 1, 1))[:, None] * frequencies[None]
    return torch.cat([_rbf(i, ticks, 8.0), _rbf(j, ticks, 8.0),
                      _rbf(torch.full_like(i, float(length)), ticks, 24.0),
                      phase.sin(), phase.cos()], dim=1)


def ridge_r2(train_x, train_y, test_x=None, test_y=None, penalty: float = 1e-2) -> float:
    """Variance of `test_y` explained by a ridge fit on standardised `train_x`."""
    if test_x is None:
        test_x, test_y = train_x, train_y
    # Columns that are constant on the fit set (an RBF centred past every training length)
    # carry no information and, standardised, would blow up on held-out rows.
    scale = train_x.std(0)
    keep = scale > 1e-6
    train_x, test_x, scale = train_x[:, keep], test_x[:, keep], scale[keep]
    mean_x, mean_y = train_x.mean(0), train_y.mean(0)
    a, b = (train_x - mean_x) / scale, (test_x - mean_x) / scale
    gram = a.T @ a + penalty * len(a) * torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    weights = torch.linalg.solve(gram, a.T @ (train_y - mean_y))
    residual = (test_y - mean_y) - b @ weights
    total = (test_y - test_y.mean(0)).square().sum()
    return float(1.0 - residual.square().sum() / total.clamp_min(1e-12))


def effective_rank(matrix: torch.Tensor) -> torch.Tensor:
    centred = matrix - matrix.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centred.double())
    share = (singular / singular.sum().clamp_min(1e-12)).clamp_min(1e-12)
    return torch.exp(-(share * share.log()).sum())


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


@torch.no_grad()
def token_diagnostics(tokens: torch.Tensor, residue_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """In-sample positional R^2 and chain-level spread of one batch of [B, L, D] tokens."""
    tokens = tokens.detach().double()
    lengths = residue_mask.sum(1)
    rows = [k for k in range(len(tokens)) if int(lengths[k]) > 1]
    zero = tokens.new_zeros(())
    if len(rows) < 2:
        return {"positional_r2": zero, "between_chain_share": zero,
                "chain_effective_rank": zero}
    flat = torch.cat([tokens[k][residue_mask[k]] for k in rows])
    basis = torch.cat([position_basis(int(lengths[k]), tokens.device) for k in rows])
    pooled = masked_mean(tokens[rows], residue_mask[rows])
    within = torch.cat([tokens[k][residue_mask[k]] - pooled[n] for n, k in enumerate(rows)])
    total = (flat - flat.mean(0)).square().sum().clamp_min(1e-12)
    return {
        "positional_r2": tokens.new_tensor(ridge_r2(basis, flat)),
        "between_chain_share": 1.0 - within.square().sum() / total,
        "chain_effective_rank": effective_rank(pooled),
    }
