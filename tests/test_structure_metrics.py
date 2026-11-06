import pytest
import torch

from psjepa.training.structure_metrics import backbone_bond_metrics


def three_residue_backbone() -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.zeros(1, 3, 4, 3)
    for index in range(3):
        start = 3.8 * index
        coords[0, index, 0, 0] = start
        coords[0, index, 1, 0] = start + 1.46
        coords[0, index, 2, 0] = start + 2.47
        coords[0, index, 3, 0] = start + 2.80
    return coords, torch.ones(1, 3, 4, dtype=torch.bool)


def test_backbone_bond_metrics_are_zero_for_an_exact_reconstruction():
    target, atom_mask = three_residue_backbone()
    metrics = backbone_bond_metrics(target.clone(), target, atom_mask)

    assert metrics["n_ca_mae"] == pytest.approx(0.0)
    assert metrics["ca_c_mae"] == pytest.approx(0.0)
    assert metrics["peptide_c_n_mae"] == pytest.approx(0.0)
    assert metrics["peptide_c_n_within_tolerance"] == pytest.approx(1.0)


def test_peptide_metric_detects_disconnected_residues_without_blaming_local_bonds():
    target, atom_mask = three_residue_backbone()
    prediction = target.clone()
    prediction[:, 1:] += torch.tensor([10.0, 0.0, 0.0])
    metrics = backbone_bond_metrics(prediction, target, atom_mask)

    assert metrics["n_ca_mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["ca_c_mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["peptide_c_n_mae"] > 4.9
    assert metrics["peptide_c_n_within_tolerance"] == pytest.approx(0.5)
