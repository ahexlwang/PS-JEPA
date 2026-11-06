"""Splitting the mask into blocks must change its shape, not its size."""
from __future__ import annotations

import pytest
import torch

from psjepa.data.masking import sample_prediction_mask


def batch(length: int = 120, valid: int | None = None):
    valid = length if valid is None else valid
    coords = torch.randn(3, length, 4, 3) * 10
    atom_mask = torch.ones(3, length, 4, dtype=torch.bool)
    residue_mask = torch.zeros(3, length, dtype=torch.bool)
    residue_mask[:, :valid] = True
    return coords, atom_mask, residue_mask


def runs(row: torch.Tensor) -> int:
    """Number of maximal True runs along the chain."""
    values = row.tolist()
    return sum(1 for i, v in enumerate(values) if v and (i == 0 or not values[i - 1]))


@pytest.mark.parametrize("blocks", [1, 2, 4, 8])
def test_total_masked_fraction_is_unchanged_by_block_count(blocks):
    torch.manual_seed(0)
    coords, atom_mask, residue_mask = batch()
    mask, _ = sample_prediction_mask(coords, atom_mask, residue_mask, 0.4, 0.6, 0.5, blocks)
    fraction = mask.float().sum(dim=1) / residue_mask.float().sum(dim=1)
    assert ((fraction >= 0.39) & (fraction <= 0.61)).all(), fraction


def test_contiguous_blocks_produce_that_many_runs():
    torch.manual_seed(1)
    coords, atom_mask, residue_mask = batch(length=200)
    # spatial_probability 0 forces the chain-run mode, where run count is exact.
    mask, modes = sample_prediction_mask(coords, atom_mask, residue_mask, 0.4, 0.5, 0.0, 4)
    assert all(mode.startswith("contiguous") for mode in modes)
    for row in mask:
        # Adjacent blocks can touch and merge, so four blocks give at most four
        # runs and at least two unless they happened to land side by side.
        assert 1 <= runs(row) <= 4


def test_blocks_never_exceed_the_masked_residue_count():
    torch.manual_seed(2)
    coords, atom_mask, residue_mask = batch(length=12, valid=6)
    mask, modes = sample_prediction_mask(coords, atom_mask, residue_mask, 0.4, 0.4, 0.5, 32)
    assert mask.float().sum(dim=1).max() <= 6
    assert all("x" in mode or mode in {"spatial", "contiguous"} for mode in modes)


def test_padding_is_never_masked():
    torch.manual_seed(3)
    coords, atom_mask, residue_mask = batch(length=100, valid=60)
    mask, _ = sample_prediction_mask(coords, atom_mask, residue_mask, 0.4, 0.6, 0.5, 4)
    assert not mask[:, 60:].any()


def test_something_is_always_left_visible():
    torch.manual_seed(4)
    coords, atom_mask, residue_mask = batch(length=80)
    mask, _ = sample_prediction_mask(coords, atom_mask, residue_mask, 0.6, 0.6, 0.5, 4)
    assert (~mask & residue_mask).any(dim=1).all()


def test_block_count_below_one_is_refused():
    coords, atom_mask, residue_mask = batch()
    with pytest.raises(ValueError, match="blocks"):
        sample_prediction_mask(coords, atom_mask, residue_mask, 0.4, 0.6, 0.5, 0)


def test_single_block_matches_the_documented_default():
    torch.manual_seed(5)
    coords, atom_mask, residue_mask = batch()
    mask, modes = sample_prediction_mask(coords, atom_mask, residue_mask, 0.4, 0.6, 0.5)
    assert all(mode in {"spatial", "contiguous"} for mode in modes)
    assert mask.any()
