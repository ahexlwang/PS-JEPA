import torch

from psjepa.data.masking import sample_prediction_mask
from psjepa.models.encoder import masked_mse
from conftest import tiny_batch


def test_prediction_and_padding_masks_are_separate():
    batch = tiny_batch()
    prediction, modes = sample_prediction_mask(
        batch["coords"], batch["atom_mask"], batch["residue_mask"], 0.4, 0.4, 0.5
    )
    assert not torch.any(prediction & ~batch["residue_mask"])
    assert prediction[0].sum() == 2
    assert prediction[1].sum() == 3
    assert set(modes) <= {"contiguous", "spatial"}


def test_masked_mse_ignores_unselected_and_padding_positions():
    prediction = torch.zeros(1, 4, 2)
    target = torch.zeros_like(prediction)
    target[:, 0] = 2
    target[:, 1:] = 100
    mask = torch.tensor([[True, False, False, False]])
    assert torch.isclose(masked_mse(prediction, target, mask), torch.tensor(4.0))

