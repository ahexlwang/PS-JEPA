'`predictor_width` narrows (or widens) the Mamba predictor without touching the encoder.'
from __future__ import annotations

import pytest
import torch

from psjepa.layers.predictors import ContextPredictor
from psjepa.models.psjepa import PSJEPA

MODEL = {
    "d_model": 16, "n_layer": 1, "predictor_layers": 1, "max_residues": 32,
    "mixer_backend": "gru", "input_mode": "backbone", "encoder_layout": "residue_only",
}
MASKING = {"context_ratio": 0.5, "target_ratio": 0.2, "target_blocks": 2,
           "spatial_probability": 0.0, "exclude_context": True}


def _count(module):
    return sum(p.numel() for p in module.parameters())


def test_default_width_preserves_parameter_keys():
    narrow_free = ContextPredictor(16, 1, backend="gru")
    explicit = ContextPredictor(16, 1, backend="gru", width=16)
    assert narrow_free.input is None and explicit.input is None
    assert set(narrow_free.state_dict()) == set(explicit.state_dict())
    assert not any(k.startswith("input.") for k in narrow_free.state_dict())
    assert narrow_free.mask_token.shape == (16,)


def test_narrow_predictor_projects_in_and_out():
    predictor = ContextPredictor(16, 1, backend="gru", width=8)
    assert predictor.input.weight.shape == (8, 16)
    assert predictor.mask_token.shape == (8,)
    assert predictor.output.weight.shape == (16, 8)
    x = torch.randn(2, 5, 16)
    prediction = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]], dtype=torch.bool)
    residue = torch.ones(2, 5, dtype=torch.bool)
    out = predictor(x, prediction, residue)
    assert out.shape == (2, 5, 16)
    assert _count(predictor) < _count(ContextPredictor(16, 1, backend="gru"))


@pytest.mark.parametrize("width", [8, 32])
def test_psjepa_threads_predictor_width_and_trains_it(width):
    model = PSJEPA.from_dict({**MODEL, "predictor_width": width}, MASKING, {"s2s": 1.0})
    assert model.predictor.width == width
    assert model.predictor.input is not None
    predictor_ids = {id(p) for p in model.predictor.parameters()}
    trainable_ids = {id(p) for p in model.trainable_parameters()}
    assert predictor_ids <= trainable_ids, "the predictor must be in the optimiser's list"
    default = PSJEPA.from_dict(MODEL, MASKING, {"s2s": 1.0})
    assert default.predictor.input is None and default.predictor.width == MODEL["d_model"]


def test_predictor_width_rejected_for_transformer_backend():
    with pytest.raises(ValueError, match="predictor_width"):
        PSJEPA.from_dict({**MODEL, "predictor_backend": "transformer", "predictor_heads": 4,
                            "predictor_width": 8}, MASKING, {"s2s": 1.0})
