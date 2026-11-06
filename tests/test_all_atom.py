import torch

from psjepa.data.backbone_dataset import collate_backbones
from psjepa.layers.residue_projection import ResidueProjection


def test_all_atom_projection_pools_to_one_residue_feature():
    torch.manual_seed(5)
    coords = torch.randn(2, 7, 37, 3)
    mask = torch.rand(2, 7, 37) > 0.25
    output = ResidueProjection(16, input_mode="all_atom")(coords, mask)
    assert output.shape == (2, 7, 16)
    assert torch.isfinite(output).all()


def test_collate_supports_atom37_without_residue_identity():
    items = []
    for length in (3, 5):
        items.append({
            "coords": torch.randn(length, 37, 3),
            "atom_mask": torch.ones(length, 37, dtype=torch.bool),
            "residue_mask": torch.ones(length, dtype=torch.bool),
            "residue_type": torch.full((length,), 20),
            "pdb_id": "x",
            "chain_id": "A",
            "label": "unknown",
        })
    batch = collate_backbones(items)
    assert batch["coords"].shape == (2, 5, 37, 3)
    assert torch.equal(batch["residue_type"], torch.full((2, 5), 20))
