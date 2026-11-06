import torch

from psjepa.layers.flow_ipa import IPAFlowDecoder
from psjepa.layers.flow_matching import conditional_flow_matching_loss
from psjepa.layers.ipa import StructureModule, apply_frames, gram_schmidt_frames


def random_rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value, upper = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    value = value * torch.sign(torch.diagonal(upper)).unsqueeze(0)
    if torch.det(value) < 0:
        value[:, 0] *= -1
    return value


def backbone_batch(seed: int = 0, batch: int = 2, length: int = 9):
    torch.manual_seed(seed)
    coords = torch.randn(batch, length, 4, 3) * 0.6
    coords[..., 1, :] += torch.arange(length).view(1, -1, 1) * 0.38
    atom_mask = torch.ones(batch, length, 4, dtype=torch.bool)
    atom_mask[1, -2:] = False
    residue_mask = atom_mask.any(dim=-1)
    return coords, atom_mask, residue_mask


def test_ipa_flow_decoder_is_exactly_equivariant():
    coords, atom_mask, residue_mask = backbone_batch()
    decoder = IPAFlowDecoder(d_model=16, trunk_layers=1, width=32, heads=4, iterations=2,
                             max_residues=16, coordinate_scale=1.0).double().eval()
    for parameter in decoder.parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=0.05)
    condition = torch.randn(2, 9, 16, dtype=torch.float64)
    time = torch.tensor([0.3, 0.9], dtype=torch.float64)
    coords = coords.double()
    motion = random_rotation(3).double()
    shift = torch.tensor([4.0, -2.0, 7.0], dtype=torch.float64)
    with torch.no_grad():
        base = decoder(condition, coords, atom_mask, time, residue_mask)
        moved = decoder(condition, (coords @ motion + shift) * atom_mask.unsqueeze(-1),
                        atom_mask, time, residue_mask)
    assert base.shape == (2, 9, 4, 3)
    torch.testing.assert_close(moved, base @ motion, atol=1e-4, rtol=1e-4)
    # padded residues carry no coordinates
    assert float(base[1, -2:].abs().max()) == 0.0


def test_ipa_flow_decoder_places_ideal_geometry_and_trains():
    coords, atom_mask, residue_mask = backbone_batch(seed=1)
    decoder = IPAFlowDecoder(d_model=16, trunk_layers=1, width=32, heads=4, iterations=1,
                             max_residues=16, coordinate_scale=10.0)
    condition = torch.randn(2, 9, 16)
    time = torch.rand(2)
    out = decoder(condition, coords / 10.0, atom_mask, time, residue_mask) * 10.0
    n, ca, c = out[0, :, 0], out[0, :, 1], out[0, :, 2]
    torch.testing.assert_close((n - ca).norm(dim=-1), torch.full((9,), 1.4606), atol=1e-3, rtol=0)  # template N: (-0.525, 1.363)
    torch.testing.assert_close((ca - c).norm(dim=-1), torch.full((9,), 1.526), atol=2e-3, rtol=0)
    loss, _ = conditional_flow_matching_loss(
        decoder, condition, coords, atom_mask, residue_mask, coordinate_scale=10.0,
        predict_x1=True, align_target=True)
    loss.backward()
    grads = [p.grad for p in decoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_structure_module_accepts_given_frames():
    torch.manual_seed(5)
    coords, atom_mask, residue_mask = backbone_batch(seed=2)
    rotation, translation = gram_schmidt_frames(coords[..., :3, :])
    module = StructureModule(width=16, heads=2, iterations=0, coordinate_scale=1.0,
                             initial_frames="learned_origin")
    placed, trajectory, out_rotation = module(torch.randn(2, 9, 16), ~residue_mask,
                                              None, rotation, translation)
    # zero-initialised seed: the template sits exactly on the supplied frames
    torch.testing.assert_close(out_rotation, rotation)
    torch.testing.assert_close(placed[..., 1, :], translation, atol=1e-6, rtol=0)
    torch.testing.assert_close(placed[..., :3, :],
                               apply_frames(rotation, translation, module.template.expand(2, 9, 3, 3)),
                               atol=1e-6, rtol=0)


def test_canonical_target_loss_is_rigid_motion_invariant():
    coords, atom_mask, residue_mask = backbone_batch(seed=3)
    decoder = IPAFlowDecoder(d_model=16, trunk_layers=1, width=32, heads=4, iterations=1,
                             max_residues=16, coordinate_scale=10.0).double().eval()
    condition = torch.randn(2, 9, 16, dtype=torch.float64)
    motion = random_rotation(7).double()
    moved = (coords.double() @ motion + torch.tensor([3.0, 1.0, -2.0], dtype=torch.float64))
    moved = moved * atom_mask.unsqueeze(-1)
    values = []
    for c in (coords.double(), moved):
        torch.manual_seed(11)  # same noise and time draw
        loss, _ = conditional_flow_matching_loss(
            decoder, condition, c, atom_mask, residue_mask, coordinate_scale=10.0,
            predict_x1=True, canonical_target=True)
        values.append(loss)
    torch.testing.assert_close(values[0], values[1], atol=1e-5, rtol=1e-5)
