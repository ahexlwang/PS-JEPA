"The bidirectional stack must not let one chain's tokens depend on its batch."
import pytest
import torch

from psjepa.models.mamba_encoder import (
    BiMambaStack,
    MambaEncoderConfig,
    _reverse_valid,
    set_padding_invariant,
)

LENGTHS = [11, 7, 4]


def stack(backend: str, padding_invariant: bool, d_model: int = 16, n_layer: int = 2):
    if backend == "mamba":
        pytest.importorskip("mamba_ssm")
    torch.manual_seed(11)
    return BiMambaStack(
        MambaEncoderConfig(
            d_model=d_model,
            n_layer=n_layer,
            bidirectional=True,
            backend=backend,
            fused_add_norm=False,
            padding_invariant=padding_invariant,
        )
    ).eval()


def padded_batch(d_model: int = 16, total: int = 14):
    torch.manual_seed(5)
    rows = [torch.randn(length, d_model) for length in LENGTHS]
    hidden = torch.zeros(len(rows), total, d_model)
    mask = torch.ones(len(rows), total, dtype=torch.bool)
    for index, row in enumerate(rows):
        hidden[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = False
    return rows, hidden, mask


def alone(model, row):
    return model(
        row.unsqueeze(0), torch.zeros(1, row.shape[0], dtype=torch.bool)
    )[0]


@pytest.mark.parametrize("backend", ["gru", "mamba"])
def test_padded_batch_matches_chains_encoded_alone(backend):
    model = stack(backend, padding_invariant=True)
    rows, hidden, mask = padded_batch()
    with torch.no_grad():
        batched = model(hidden, mask)
        for index, row in enumerate(rows):
            reference = alone(model, row)
            assert torch.allclose(batched[index, : row.shape[0]], reference, atol=1e-5), (
                f"row {index} moved by "
                f"{(batched[index, : row.shape[0]] - reference).abs().max():.2e}"
            )
        assert batched[1, LENGTHS[1]:].abs().max() == 0.0


@pytest.mark.parametrize("backend", ["gru", "mamba"])
def test_legacy_path_still_depends_on_padding(backend):
    """The default path is the one every stored checkpoint was trained under."""
    model = stack(backend, padding_invariant=False)
    rows, hidden, mask = padded_batch()
    with torch.no_grad():
        batched = model(hidden, mask)
        invariant = stack(backend, padding_invariant=True)(hidden, mask)
        shortest = rows[-1]
        moved = (batched[-1, : shortest.shape[0]] - alone(model, shortest)).abs().max()


    assert moved > 1e-5, "legacy behaviour changed; stored results are no longer reproducible"
    assert not torch.allclose(batched, invariant, atol=1e-6)


@pytest.mark.parametrize("backend", ["gru", "mamba"])
def test_flag_is_a_no_op_without_padding(backend):
    """So switching a frozen model over cannot be read as switching model."""
    rows, _, _ = padded_batch()
    row = rows[0].unsqueeze(0)
    empty = torch.zeros(1, row.shape[1], dtype=torch.bool)
    with torch.no_grad():
        legacy = stack(backend, padding_invariant=False)(row, empty)
        invariant = stack(backend, padding_invariant=True)(row, empty)
    assert torch.allclose(legacy, invariant, atol=1e-6)


def test_set_padding_invariant_switches_a_built_model():
    model = stack("gru", padding_invariant=False)
    _, hidden, mask = padded_batch()
    with torch.no_grad():
        before = model(hidden, mask)
        assert set_padding_invariant(model) == 1
        after = model(hidden, mask)
        assert set_padding_invariant(model, False) == 1
        again = model(hidden, mask)
    assert not torch.allclose(before, after, atol=1e-5)
    assert torch.allclose(before, again, atol=1e-7)


def test_reverse_valid_reverses_only_the_chain_and_is_its_own_inverse():
    value = torch.arange(2 * 5 * 1, dtype=torch.float32).reshape(2, 5, 1)
    lengths = torch.tensor([3, 5])
    reversed_value = _reverse_valid(value, lengths)
    assert reversed_value[0, :, 0].tolist() == [2.0, 1.0, 0.0, 3.0, 4.0]
    assert reversed_value[1, :, 0].tolist() == [9.0, 8.0, 7.0, 6.0, 5.0]
    assert torch.equal(_reverse_valid(reversed_value, lengths), value)
