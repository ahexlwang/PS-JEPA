"""Fixtures small enough that the whole suite runs on a CPU in seconds."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from psjepa.models import PSJEPA, PSJEPAConfig  # noqa: E402


def tiny_psjepa(**overrides):
    """The released architecture at a size a unit test can afford."""
    settings = dict(
        d_model=16,
        n_layer=1,
        predictor_layers=1,
        max_residues=16,
        mixer_backend="gru",
        input_mode="gvp_message",
        gvp_neighbours=4,
        gvp_layers=1,
        gvp_message_layers=1,
        gvp_node_vectors=4,
        gvp_scalar_hidden=16,
        context_ratio=0.5,
        target_ratio=0.25,
        target_blocks=2,
    )
    settings.update(overrides)
    return PSJEPA(PSJEPAConfig(**settings))


def tiny_batch():
    """Two short chains, the second one padded, with coordinates zeroed under the mask."""
    torch.manual_seed(7)
    coords = torch.randn(2, 7, 4, 3)
    atom_mask = torch.ones(2, 7, 4, dtype=torch.bool)
    residue_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    atom_mask &= residue_mask.unsqueeze(-1)
    coords *= atom_mask.unsqueeze(-1)
    return {"coords": coords, "atom_mask": atom_mask, "residue_mask": residue_mask}


@pytest.fixture
def batch():
    return tiny_batch()


@pytest.fixture
def model():
    return tiny_psjepa()
