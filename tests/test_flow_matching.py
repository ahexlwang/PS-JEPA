import pytest
import torch

from psjepa.layers.flow_matching import GVPFlowDecoder, conditional_flow_matching_loss, sample_flow

from conftest import tiny_batch


def rotation(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def test_flow_velocity_is_rotation_equivariant_and_translation_free():
    torch.manual_seed(2)
    decoder = GVPFlowDecoder(d_model=16, neighbours=3, layers=2, scalar_hidden=24, vector_hidden=8)
    decoder.eval()
    batch = tiny_batch()
    condition = torch.randn(2, 7, 16)
    time = torch.tensor([0.2, 0.8])
    prediction_mask = batch["residue_mask"].clone()
    matrix = rotation(4)

    base = decoder(condition, batch["coords"], batch["atom_mask"], time, prediction_mask)
    turned = decoder(
        condition,
        batch["coords"] @ matrix,
        batch["atom_mask"],
        time,
        prediction_mask,
    )
    shifted = decoder(
        condition,
        (batch["coords"] + torch.tensor([3.0, -5.0, 11.0]))
        * batch["atom_mask"].unsqueeze(-1),
        batch["atom_mask"],
        time,
        prediction_mask,
    )
    assert torch.allclose(base @ matrix, turned, atol=2e-5)
    assert torch.allclose(base, shifted, atol=2e-5)


def test_flow_velocity_is_explicitly_conditioned_on_the_semantic_token():
    torch.manual_seed(12)
    decoder = GVPFlowDecoder(d_model=16, neighbours=3, layers=1, scalar_hidden=16, vector_hidden=6)
    decoder.eval()
    batch = tiny_batch()
    condition = torch.randn(2, 7, 16)
    prediction_mask = batch["residue_mask"].clone()
    first = decoder(
        condition, batch["coords"], batch["atom_mask"], torch.tensor([0.4, 0.6]), prediction_mask
    )
    second = decoder(
        condition.flip(1), batch["coords"], batch["atom_mask"], torch.tensor([0.4, 0.6]), prediction_mask
    )
    valid = batch["atom_mask"]
    assert not torch.allclose(first[valid], second[valid])


def test_flow_loss_uses_only_selected_present_atoms():
    torch.manual_seed(3)
    decoder = GVPFlowDecoder(d_model=16, neighbours=3, layers=1, scalar_hidden=16, vector_hidden=6)
    batch = tiny_batch()
    condition = torch.randn(2, 7, 16, requires_grad=True)
    nothing = torch.zeros_like(batch["residue_mask"])
    loss, metrics = conditional_flow_matching_loss(
        decoder, condition, batch["coords"], batch["atom_mask"], nothing
    )
    assert loss.item() == pytest.approx(0.0)
    assert metrics["flow_velocity_target_rms"].item() == pytest.approx(0.0)


def test_flow_loss_detaches_frozen_representation_but_trains_decoder():
    torch.manual_seed(5)
    decoder = GVPFlowDecoder(d_model=16, neighbours=3, layers=1, scalar_hidden=16, vector_hidden=6)
    batch = tiny_batch()
    condition = torch.randn(2, 7, 16, requires_grad=True)
    loss, _ = conditional_flow_matching_loss(
        decoder,
        condition,
        batch["coords"],
        batch["atom_mask"],
        batch["residue_mask"],
    )
    loss.backward()
    assert condition.grad is None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in decoder.parameters()
    )


def test_flow_sampler_returns_centred_masked_coordinates():
    torch.manual_seed(7)
    decoder = GVPFlowDecoder(d_model=16, neighbours=3, layers=1, scalar_hidden=16, vector_hidden=6)
    batch = tiny_batch()
    generated = sample_flow(
        decoder,
        torch.randn(2, 7, 16),
        batch["atom_mask"],
        batch["residue_mask"],
        steps=2,
    )
    assert generated.shape == batch["coords"].shape
    assert torch.equal(generated[~batch["atom_mask"]], torch.zeros_like(generated[~batch["atom_mask"]]))
    for item, mask in zip(generated, batch["atom_mask"]):
        assert torch.allclose(item[mask].mean(dim=0), torch.zeros(3), atol=2e-5)
