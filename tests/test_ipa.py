import pytest
import torch

from psjepa.layers.ipa import (
    StructureModule,
    adjacent_frame_loss,
    aligned_coordinate_loss,
    apply_frames,
    ca_distance_loss,
    canonical_coordinate_loss,
    canonicalize_backbone,
    compose_local_frame_update,
    fape_loss,
    gram_schmidt_frames,
    invert_frames,
    peptide_angle_loss,
    peptide_bond_loss,
    six_d_to_rotation,
)


def random_rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value, upper = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    value = value * torch.sign(torch.diagonal(upper)).unsqueeze(0)
    if torch.det(value) < 0:
        value[:, 0] *= -1
    return value


def test_row_frame_round_trip_and_equivariance():
    torch.manual_seed(0)
    coordinates = torch.randn(2, 7, 4, 3)
    rotation, translation = gram_schmidt_frames(coordinates)
    local = torch.randn(2, 7, 5, 3)
    recovered = invert_frames(rotation, translation, apply_frames(rotation, translation, local))
    torch.testing.assert_close(recovered, local, atol=2e-5, rtol=2e-5)

    motion = random_rotation(1)
    shift = torch.tensor([7.0, -2.0, 11.0])
    moved_rotation, moved_translation = gram_schmidt_frames(coordinates @ motion + shift)
    torch.testing.assert_close(moved_rotation, rotation @ motion, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(moved_translation, translation @ motion + shift,
                               atol=2e-5, rtol=2e-5)


def test_local_frame_composition_uses_the_preupdate_frame_for_translation():
    current_rotation = random_rotation(2).reshape(1, 1, 3, 3)
    current_translation = torch.tensor([[[3.0, -4.0, 2.0]]])
    local_rotation = random_rotation(3).reshape(1, 1, 3, 3)
    local_translation = torch.tensor([[[0.5, 1.0, -0.25]]])
    rotation, translation = compose_local_frame_update(
        current_rotation, current_translation, local_rotation, local_translation)

    points = torch.randn(1, 1, 6, 3)
    sequential = apply_frames(
        current_rotation, current_translation,
        apply_frames(local_rotation, local_translation, points))
    composed = apply_frames(rotation, translation, points)
    torch.testing.assert_close(composed, sequential, atol=2e-6, rtol=2e-6)


def test_six_d_rotation_is_orthogonal_and_identity_at_its_bias():
    bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rotation = six_d_to_rotation(bias)
    torch.testing.assert_close(rotation, torch.eye(3))
    value = torch.randn(3, 7, 6)
    rotation = six_d_to_rotation(value)
    torch.testing.assert_close(
        rotation @ rotation.transpose(-1, -2), torch.eye(3).expand_as(rotation),
        atol=2e-5, rtol=2e-5)


def test_fape_ignores_unobserved_atoms_and_frames():
    torch.manual_seed(4)
    model = StructureModule(width=32, heads=4, iterations=2, coordinate_scale=1.0)
    residue_mask = torch.ones(1, 6, dtype=torch.bool)
    pad = ~residue_mask
    prediction, _, _ = model(torch.randn(1, 6, 32), pad)
    target = prediction.detach() + torch.randn_like(prediction) * 0.05
    atom_mask = torch.ones(1, 6, 4, dtype=torch.bool)
    atom_mask[:, 2] = False
    first = fape_loss(prediction, target, residue_mask, atom_mask, clamp=None)
    changed = target.clone()
    changed[:, 2] = 10_000.0
    second = fape_loss(prediction, changed, residue_mask, atom_mask, clamp=None)
    torch.testing.assert_close(first, second)


def test_structure_and_auxiliary_losses_are_exact_for_an_exact_chain():
    torch.manual_seed(5)
    model = StructureModule(width=32, heads=4, iterations=2, coordinate_scale=1.0,
                            initial_frames="line", pair_width=8)
    residue_mask = torch.ones(2, 8, dtype=torch.bool)
    pair = torch.randn(2, 8, 8, 8)
    coordinates, trajectory, rotation = model(torch.randn(2, 8, 32), ~residue_mask, pair)
    atom_mask = torch.ones(2, 8, 4, dtype=torch.bool)
    assert coordinates.shape == (2, 8, 4, 3)
    assert trajectory.shape == (2, 2, 8, 4, 3)
    assert rotation.shape == (2, 8, 3, 3)
    torch.testing.assert_close(
        rotation @ rotation.transpose(-1, -2), torch.eye(3).expand_as(rotation),
        atol=2e-5, rtol=2e-5)


    assert float(fape_loss(coordinates, coordinates, residue_mask, atom_mask)) == pytest.approx(1e-5, rel=1e-3)
    assert float(adjacent_frame_loss(coordinates, coordinates, residue_mask, atom_mask)) == pytest.approx(1e-4, rel=1e-3)
    assert float(ca_distance_loss(coordinates, coordinates, residue_mask, atom_mask)) < 1e-7
    assert float(peptide_bond_loss(coordinates, coordinates, residue_mask, atom_mask)) < 1e-7
    assert float(peptide_angle_loss(
        coordinates, coordinates, residue_mask, atom_mask
    )) < 1e-7
    assert float(aligned_coordinate_loss(
        coordinates, coordinates, residue_mask, atom_mask)) < 5e-6


def test_peptide_bond_loss_defaults_to_l1_and_optionally_uses_huber():
    target = torch.zeros(1, 2, 4, 3)
    target[:, 1, 0, 0] = 2.0
    predicted = target.clone()
    predicted[:, 1, 0, 0] = 4.0
    residue_mask = torch.ones(1, 2, dtype=torch.bool)
    atom_mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    atom_mask[:, 0, 2] = True
    atom_mask[:, 1, 0] = True


    torch.testing.assert_close(
        peptide_bond_loss(predicted, target, residue_mask, atom_mask),
        torch.tensor(2.0),
    )
    # Smooth L1 at error=2 and beta=1 is 2 - beta/2 = 1.5.
    torch.testing.assert_close(
        peptide_bond_loss(
            predicted, target, residue_mask, atom_mask, beta=1.0
        ),
        torch.tensor(1.5),
    )


def test_aligned_coordinate_loss_removes_only_a_proper_global_motion():
    torch.manual_seed(8)
    coordinates = torch.randn(2, 12, 4, 3)
    motion = random_rotation(9)
    moved = coordinates @ motion + torch.tensor([8.0, -3.0, 12.0])
    residue_mask = torch.ones(2, 12, dtype=torch.bool)
    atom_mask = torch.ones(2, 12, 4, dtype=torch.bool)
    assert float(aligned_coordinate_loss(
        moved, coordinates, residue_mask, atom_mask)) < 2e-6
    reflected = coordinates.clone()
    reflected[..., 0] *= -1
    assert float(aligned_coordinate_loss(
        reflected, coordinates, residue_mask, atom_mask)) > 0.1


def test_canonical_target_is_rigid_motion_invariant_and_supplies_a_fixed_gauge():
    torch.manual_seed(10)
    coordinates = torch.randn(2, 12, 4, 3, dtype=torch.float64)
    residue_mask = torch.ones(2, 12, dtype=torch.bool)
    atom_mask = torch.ones(2, 12, 4, dtype=torch.bool)
    motion = random_rotation(11).double()
    moved = coordinates @ motion + torch.tensor([8.0, -3.0, 12.0], dtype=torch.float64)

    canonical, live = canonicalize_backbone(coordinates, residue_mask, atom_mask)
    moved_canonical, moved_live = canonicalize_backbone(moved, residue_mask, atom_mask)
    assert live.all() and moved_live.all()

    torch.testing.assert_close(moved_canonical, canonical, atol=1e-6, rtol=1e-6)
    assert float(canonical_coordinate_loss(
        canonical, moved, residue_mask, atom_mask)) < 2e-6
    assert float(canonical_coordinate_loss(
        canonical + 0.1, moved, residue_mask, atom_mask)) > 0.1


def test_learned_origin_starts_every_frame_at_the_origin():
    model = StructureModule(width=32, heads=4, iterations=0, coordinate_scale=10.0,
                            initial_frames="learned_origin")
    residue_mask = torch.ones(2, 9, dtype=torch.bool)
    coordinates, _, rotation = model(torch.randn(2, 9, 32), ~residue_mask)
    torch.testing.assert_close(rotation, torch.eye(3).expand_as(rotation))
    torch.testing.assert_close(
        coordinates[..., 1, :], torch.zeros_like(coordinates[..., 1, :]))


def test_learned_line_seed_receives_a_direct_gradient():
    torch.manual_seed(6)
    model = StructureModule(width=32, heads=4, iterations=2, coordinate_scale=10.0,
                            initial_frames="learned_line")
    single = torch.randn(1, 10, 32)
    residue_mask = torch.ones(1, 10, dtype=torch.bool)
    coordinates, _, _ = model(single, ~residue_mask)
    loss = coordinates[..., 1, :].square().mean()
    loss.backward()
    assert model.seed_update is not None
    assert model.seed_update.weight.grad is not None
    assert float(model.seed_update.weight.grad.abs().sum()) > 0.0


def test_learned_seed_is_a_valid_decoder_without_ipa_refinement():
    model = StructureModule(width=32, heads=4, iterations=0, coordinate_scale=10.0,
                            initial_frames="learned_line")
    residue_mask = torch.ones(2, 9, dtype=torch.bool)
    coordinates, trajectory, rotation = model(torch.randn(2, 9, 32), ~residue_mask)
    assert coordinates.shape == (2, 9, 4, 3)
    assert trajectory.shape == (1, 2, 9, 4, 3)
    assert rotation.shape == (2, 9, 3, 3)
    coordinates.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_external_frames_are_required_and_used_as_the_ipa_start():
    torch.manual_seed(15)
    model = StructureModule(
        width=32, heads=4, iterations=1, coordinate_scale=10.0,
        initial_frames="external",
    )
    single = torch.randn(2, 7, 32)
    pad = torch.zeros(2, 7, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires rotation and translation"):
        model(single, pad)

    rotation = random_rotation(16).expand(2, 7, 3, 3).clone()
    translation = torch.randn(2, 7, 3)
    coordinates, trajectory, recovered_rotation = model(
        single, pad, rotation=rotation, translation=translation
    )


    torch.testing.assert_close(coordinates[..., 1, :], translation)
    torch.testing.assert_close(recovered_rotation, rotation)
    assert trajectory.shape == (1, 2, 7, 4, 3)


def test_external_frame_shape_is_checked_before_attention():
    model = StructureModule(
        width=16, heads=2, iterations=1, initial_frames="external"
    )
    single = torch.randn(1, 5, 16)
    with pytest.raises(ValueError, match="rotation must have shape"):
        model(
            single, torch.zeros(1, 5, dtype=torch.bool),
            rotation=torch.eye(3).expand(1, 4, 3, 3),
            translation=torch.zeros(1, 5, 3),
        )
