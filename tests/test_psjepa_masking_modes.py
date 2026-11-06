from __future__ import annotations

import pytest
import torch

from psjepa.layers.predictors import TransformerContextPredictor
from psjepa.models.psjepa import PSJEPA, ResidueOnlyStructureEncoder


MODEL = {
    "d_model": 16,
    "n_layer": 1,
    "predictor_layers": 1,
    "max_residues": 32,
    "mixer_backend": "gru",
    "input_mode": "backbone",
    "encoder_layout": "residue_only",
    "predictor_backend": "transformer",
    "predictor_heads": 4,
}
MASKING = {
    "context_ratio": 0.5,
    "target_ratio": 0.2,
    "target_blocks": 2,
    "spatial_probability": 0.0,
    "exclude_context": True,
}

CROP_MASKING = {
    "scheme": "crop_flanks_v1_1",
    "crop_views": 3,
    "crop_ratio_min": 0.5,
    "crop_ratio_max": 0.5,
    "trim_ratio": 0.2,
    "prediction_ratio": 0.3,
    "exclude_context": True,
}

CROP30_TRIM10_MASKING = {
    **CROP_MASKING,
    "crop_ratio_min": 0.3,
    "crop_ratio_max": 0.3,
    "trim_ratio": 0.1,
}


def batch(length: int = 20) -> dict[str, torch.Tensor]:
    return {
        "coords": torch.randn(2, length, 4, 3),
        "atom_mask": torch.ones(2, length, 4, dtype=torch.bool),
        "residue_mask": torch.ones(2, length, dtype=torch.bool),
    }


def test_v1_1_uses_exactly_one_residue_stream_and_transformer_predictor():
    model = PSJEPA.from_dict(MODEL, MASKING, {"s2s": 1.0})
    assert isinstance(model.context_encoder, ResidueOnlyStructureEncoder)
    assert isinstance(model.predictor, TransformerContextPredictor)
    seen = {}
    original = model.context_encoder.encoder.forward

    def spy(x, padding_mask=None):
        seen["length"] = x.shape[1]
        return original(x, padding_mask)

    model.context_encoder.encoder.forward = spy
    model(batch(length=20))["loss"]
    assert seen["length"] == 20


def test_transformer_target_does_not_depend_on_encoder_target_state():
    torch.manual_seed(0)
    predictor = TransformerContextPredictor(16, 1, 32, 4)
    predictor.eval()
    x = torch.randn(2, 12, 16)
    residue = torch.ones(2, 12, dtype=torch.bool)
    context = torch.zeros_like(residue)
    context[:, :5] = True
    target = torch.zeros_like(residue)
    target[:, 7:9] = True
    first = predictor(x, target, residue, context)
    changed = x.clone()
    changed[:, 7:9] = torch.randn_like(changed[:, 7:9]) * 1000
    second = predictor(changed, target, residue, context)
    assert torch.equal(first, second)


def test_transformer_ignores_other_target_blocks_and_handles_empty_rows():
    torch.manual_seed(1)
    predictor = TransformerContextPredictor(16, 1, 32, 4)
    predictor.eval()
    x = torch.randn(2, 12, 16)
    residue = torch.ones(2, 12, dtype=torch.bool)
    residue[1] = False
    context = torch.zeros_like(residue)
    context[0, :5] = True
    target = torch.zeros_like(residue)
    target[0, 7:9] = True
    first = predictor(x, target, residue, context)
    changed = x.clone()
    changed[0, 9:] = torch.randn_like(changed[0, 9:]) * 1000
    second = predictor(changed, target, residue, context)
    assert torch.equal(first[0, 7:9], second[0, 7:9])
    assert torch.isfinite(first).all()
    assert not first[1].any()


def test_optimizer_owns_and_updates_v1_1_predictor():
    model = PSJEPA.from_dict(MODEL, MASKING, {"s2s": 1.0})
    parameters = model.trainable_parameters()
    owned = {id(parameter) for parameter in parameters}
    assert all(id(parameter) in owned for parameter in model.predictor.parameters())
    before = [parameter.detach().clone() for parameter in model.predictor.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    model(batch())["loss"].backward()
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.predictor.parameters()))
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_crop_flanks_encodes_fifty_then_keeps_thirty_and_predicts_thirty():
    torch.manual_seed(4)
    model = PSJEPA.from_dict(MODEL, CROP_MASKING, {"s2s": 1.0})
    mask = torch.ones(2, 100, dtype=torch.bool)
    for _ in range(20):
        rows = model._sample_crop_view(mask)
        for item in rows:
            assert item["crop"].numel() == 50
            assert item["core"].numel() == 30
            assert item["target"].numel() == 30
            overlap = torch.isin(item["crop"], item["target"])
            assert int(overlap.sum()) == 20
            assert not torch.isin(item["core"], item["target"]).any()


def test_crop30_trim10_encodes_thirty_keeps_twenty_and_predicts_thirty():
    torch.manual_seed(6)
    model = PSJEPA.from_dict(MODEL, CROP30_TRIM10_MASKING, {"s2s": 1.0})
    mask = torch.ones(2, 100, dtype=torch.bool)
    for _ in range(20):
        rows = model._sample_crop_view(mask)
        for item in rows:
            assert item["crop"].numel() == 30
            assert item["core"].numel() == 20
            assert item["target"].numel() == 30
            assert int(torch.isin(item["crop"], item["target"]).sum()) == 10
            assert int(item["target_inside"].sum()) == 10
            assert not torch.isin(item["core"], item["target"]).any()


def test_crop_flanks_runs_three_student_mamba_views_and_one_full_teacher():
    torch.manual_seed(5)
    model = PSJEPA.from_dict(MODEL, CROP_MASKING, {"s2s": 1.0})
    student_lengths = []
    teacher_lengths = []
    student_forward = model.context_encoder.encoder.forward
    teacher_forward = model.target_encoder.encoder.forward

    def student_spy(x, padding_mask=None):
        student_lengths.append(x.shape[1])
        return student_forward(x, padding_mask)

    def teacher_spy(x, padding_mask=None):
        teacher_lengths.append(x.shape[1])
        return teacher_forward(x, padding_mask)

    model.context_encoder.encoder.forward = student_spy
    model.target_encoder.encoder.forward = teacher_spy
    item = batch(length=20)
    output = model(item)
    assert teacher_lengths == [20]
    assert len(student_lengths) == 3
    assert all(length == 10 for length in student_lengths)
    assert torch.isfinite(output["loss"])


    assert float(output["target_seen_fraction"]) == pytest.approx(2.0 / 3.0)
    assert torch.isfinite(output["loss_target_seen"])
    assert torch.isfinite(output["loss_target_unseen"])
