'The motif constraint has to be exact, and the old behaviour has to be untouched.'
from __future__ import annotations

import torch

from psjepa.data.masking import sample_motif_mask
from psjepa.layers.flow_attention import AttentionFlowDecoder
from psjepa.layers.flow_matching import conditional_flow_matching_loss, sample_flow


def _decoder(d_model: int = 16, width: int = 32) -> AttentionFlowDecoder:
    torch.manual_seed(0)
    return AttentionFlowDecoder(
        d_model=d_model, code_book=0, layers=2, width=width, heads=2, max_residues=64
    )


def _batch(batch: int = 2, length: int = 24, d_model: int = 16):
    torch.manual_seed(1)
    coords = torch.randn(batch, length, 4, 3) * 5.0
    atom_mask = torch.ones(batch, length, 4, dtype=torch.bool)
    residue_mask = torch.ones(batch, length, dtype=torch.bool)
    condition = torch.randn(batch, length, d_model)
    return coords, atom_mask, residue_mask, condition


def kabsch_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    p = a - a.mean(0, keepdim=True)
    q = b - b.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.T @ q)
    r = u @ vh
    if torch.linalg.det(r) < 0:
        u = u.clone()
        u[:, -1] *= -1
        r = u @ vh
    return float((p @ r - q).square().sum(-1).mean().sqrt())


def test_sampler_reproduces_the_given_motif_exactly():
    coords, atom_mask, residue_mask, condition = _batch()
    motif = torch.zeros_like(residue_mask)
    motif[:, 4:10] = True
    motif[0, 15:18] = True  # a second, discontiguous segment
    generated = sample_flow(
        _decoder(), condition, atom_mask, residue_mask & ~motif,
        coordinate_scale=10.0, steps=8, predict_x1=False,
        given_mask=motif, given_coordinates=coords,
    )
    for item in range(coords.shape[0]):
        got = generated[item][motif[item]].reshape(-1, 3)
        want = coords[item][motif[item]].reshape(-1, 3)
        # Both are centred on the motif, so they coincide without superposition.
        assert kabsch_rmsd(got, want) < 1e-3
        assert (got - (want - want.mean(0, keepdim=True))).abs().max() < 1e-2


def test_the_scaffold_is_not_zeroed_by_motif_centring():
    coords, atom_mask, residue_mask, condition = _batch()
    motif = torch.zeros_like(residue_mask)
    motif[:, :6] = True
    generated = sample_flow(
        _decoder(), condition, atom_mask, residue_mask & ~motif,
        coordinate_scale=10.0, steps=4, predict_x1=False,
        given_mask=motif, given_coordinates=coords,
    )
    built = generated[:, 6:]
    assert built.abs().max() > 1e-3, "centring on the motif deleted the scaffold"
    assert torch.isfinite(generated).all()


def test_loss_holds_the_motif_and_scores_only_the_rest():
    coords, atom_mask, residue_mask, condition = _batch()
    motif = torch.zeros_like(residue_mask)
    motif[:, 8:14] = True
    decoder = _decoder()
    loss, metrics = conditional_flow_matching_loss(
        decoder, condition, coords, atom_mask, residue_mask & ~motif,
        coordinate_scale=10.0, given_mask=motif,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())


def test_a_given_motif_refuses_the_other_gauge_fixes():
    coords, atom_mask, residue_mask, condition = _batch()
    motif = torch.zeros_like(residue_mask)
    motif[:, :6] = True
    for kwargs in ({"align_target": True}, {"canonical_target": True}):
        try:
            conditional_flow_matching_loss(
                _decoder(), condition, coords, atom_mask, residue_mask & ~motif,
                given_mask=motif, **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should be refused alongside a motif")


def test_unconditioned_behaviour_is_unchanged():
    coords, atom_mask, residue_mask, condition = _batch()
    decoder = _decoder()
    torch.manual_seed(7)
    a, _ = conditional_flow_matching_loss(
        decoder, condition, coords, atom_mask, residue_mask,
        coordinate_scale=10.0, align_target=True, predict_x1=False,
    )
    torch.manual_seed(7)
    b, _ = conditional_flow_matching_loss(
        decoder, condition, coords, atom_mask, residue_mask,
        coordinate_scale=10.0, align_target=True, predict_x1=False, given_mask=None,
    )
    assert torch.allclose(a, b)


def test_motif_sampler_shape_and_coverage():
    coords, atom_mask, residue_mask, _ = _batch(batch=8, length=60)
    torch.manual_seed(3)
    motif = sample_motif_mask(
        coords, atom_mask, residue_mask, min_fraction=0.1, max_fraction=0.3
    )
    assert motif.shape == residue_mask.shape and motif.dtype == torch.bool
    fraction = motif.float().sum(dim=1) / residue_mask.float().sum(dim=1)
    assert (fraction > 0.0).all() and (fraction < 0.6).all(), fraction
    assert (motif & ~residue_mask).sum() == 0
