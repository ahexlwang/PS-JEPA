"""Context and targets are independent regions, so overlap is a result not a rule."""
from __future__ import annotations

import pytest
import torch

from psjepa.data.masking import sample_context_and_targets


def batch(length: int = 160, count: int = 6):
    coords = torch.randn(count, length, 4, 3) * 10
    atom_mask = torch.ones(count, length, 4, dtype=torch.bool)
    residue_mask = torch.ones(count, length, dtype=torch.bool)
    return coords, atom_mask, residue_mask


def runs(row: torch.Tensor) -> int:
    values = row.tolist()
    return sum(1 for i, v in enumerate(values) if v and (i == 0 or not values[i - 1]))


def test_target_is_not_the_complement_of_the_context():
    torch.manual_seed(0)
    out = sample_context_and_targets(*batch(), context_ratio=0.5, target_ratio=0.18, blocks=4)
    covered = out["context_mask"] | out["target_mask"]
    assert not covered.all(), "some residues must sit out the step"


def test_overlap_varies_instead_of_being_fixed():
    torch.manual_seed(1)
    fractions = []
    for _ in range(12):
        out = sample_context_and_targets(*batch(count=4), blocks=1, spatial_probability=1.0)
        seen = out["target_seen"].float().sum(1)
        total = out["target_mask"].float().sum(1).clamp_min(1)
        fractions += (seen / total).tolist()
    spread = max(fractions) - min(fractions)
    assert spread > 0.4, f"overlap should range widely, saw {spread:.2f}"


def test_seen_and_unseen_partition_the_target():
    torch.manual_seed(2)
    out = sample_context_and_targets(*batch(), blocks=4)
    assert torch.equal(out["target_seen"] | out["target_unseen"], out["target_mask"])
    assert not (out["target_seen"] & out["target_unseen"]).any()


def test_context_size_follows_its_ratio():
    torch.manual_seed(3)
    out = sample_context_and_targets(*batch(), context_ratio=0.5)
    fraction = out["context_mask"].float().sum(1) / 160
    assert ((fraction > 0.48) & (fraction < 0.52)).all(), fraction


def test_sequence_mode_grows_chain_runs():
    torch.manual_seed(4)
    out = sample_context_and_targets(*batch(), blocks=1, spatial_probability=0.0)
    for row in out["context_mask"]:
        assert runs(row) == 1, "a chain-mode context is one stretch"


def test_more_blocks_cover_more_of_the_structure():
    torch.manual_seed(5)
    one = sample_context_and_targets(*batch(), blocks=1)["target_mask"].float().sum(1).mean()
    four = sample_context_and_targets(*batch(), blocks=4)["target_mask"].float().sum(1).mean()
    assert four > one


def test_padding_is_never_selected():
    torch.manual_seed(6)
    coords, atom_mask, residue_mask = batch(length=200)
    residue_mask[:, 120:] = False
    out = sample_context_and_targets(coords, atom_mask, residue_mask, blocks=4)
    assert not out["context_mask"][:, 120:].any()
    assert not out["target_mask"][:, 120:].any()


def test_block_count_below_one_is_refused():
    with pytest.raises(ValueError, match="blocks"):
        sample_context_and_targets(*batch(), blocks=0)
