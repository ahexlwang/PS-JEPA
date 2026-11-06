import pytest
import torch

from psjepa.layers.gvp import GVP, GVPFeaturizer, radial_basis, signed_triple_products
from psjepa.layers.gvp_message import GVPMessageFeaturizer, paper_node_features

from conftest import tiny_batch


def rotation(seed=0):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q.float()


REFLECTION = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))


def message_featurizer(node_features="atom", global_radial=0, edge_positional=False):
    torch.manual_seed(13)
    return GVPMessageFeaturizer(
        d_model=16, atoms=4, neighbours=3, layers=2, message_layers=2,
        scalar_hidden=32, vector_hidden=8, node_vectors=8, radial=4,
        global_radial=global_radial, edge_positional=edge_positional,
        node_features=node_features,
    )


@pytest.mark.parametrize(
    "mode,global_radial,raw_scalars,raw_vectors",
    [("atom", 6, 17, 4), ("atom_clean", 6, 11, 4),
     ("paper", 0, 6, 3), ("paper", 6, 13, 3),
     ("paper_scalar_atom4", 0, 6, 4)],
)
def test_message_node_feature_modes_have_the_declared_raw_dimensions(
    mode, global_radial, raw_scalars, raw_vectors
):
    model = message_featurizer(mode, global_radial)
    assert model.node_in.vector_hidden.in_features == raw_vectors
    vector_hidden = model.node_in.vector_hidden.out_features
    assert model.node_in.scalar_out.in_features - vector_hidden == raw_scalars


def test_paper_node_features_match_original_shapes_and_are_rotation_equivariant():
    batch = tiny_batch()
    visible = batch["residue_mask"]
    scalar, vector = paper_node_features(batch["coords"], batch["atom_mask"], visible)
    turned_scalar, turned_vector = paper_node_features(
        batch["coords"] @ rotation(17), batch["atom_mask"], visible)
    assert scalar.shape == (2, 7, 6)
    assert vector.shape == (2, 7, 3, 3)
    assert torch.allclose(scalar, turned_scalar, atol=1e-4)
    assert torch.allclose(vector @ rotation(17), turned_vector, atol=1e-4)


@pytest.mark.parametrize("global_radial,edge_positional", [(0, False), (6, True)])
def test_paper_message_output_is_rotation_and_translation_invariant(
    global_radial, edge_positional
):
    batch = tiny_batch()
    model = message_featurizer("paper", global_radial, edge_positional).eval()
    base = model(batch["coords"], batch["atom_mask"], batch["residue_mask"])
    turned = model(batch["coords"] @ rotation(19), batch["atom_mask"],
                   batch["residue_mask"])
    shifted_coords = ((batch["coords"] + torch.tensor([3.0, -5.0, 8.0]))
                      * batch["atom_mask"].unsqueeze(-1))
    shifted = model(shifted_coords, batch["atom_mask"], batch["residue_mask"])
    assert torch.allclose(base, turned, atol=1e-4)
    assert torch.allclose(base, shifted, atol=1e-4)


def test_paper_node_neighbour_features_do_not_leak_hidden_coordinates():
    batch = tiny_batch()
    visible = batch["residue_mask"].clone()
    visible[:, 2:4] = False
    moved = batch["coords"].clone()
    moved[:, 2:4] += 25.0
    moved *= batch["atom_mask"].unsqueeze(-1)
    model = message_featurizer("paper", global_radial=6).eval()
    before = model(batch["coords"], batch["atom_mask"], visible)
    after = model(moved, batch["atom_mask"], visible)
    keep = visible & batch["residue_mask"]
    assert torch.allclose(before[keep], after[keep], atol=1e-4)


def test_ca_anchor_and_paper_edge_features_match_the_declared_spec():
    batch = tiny_batch()
    coords, atom_mask = batch["coords"], batch["atom_mask"]
    model = GVPMessageFeaturizer(
        d_model=16, atoms=4, neighbours=3, layers=1, message_layers=2,
        scalar_hidden=32, vector_hidden=8, node_vectors=8,
        node_features="paper", global_radial=0, edge_positional=True,
        anchor_mode="ca", rbf_width_mode="paper", positional_order="cos_sin",
    ).eval()
    assert model.node_in.scalar_out.in_features - model.node_in.vector_hidden.out_features == 6
    with torch.no_grad():
        output = model(coords, atom_mask, batch["residue_mask"])
        rotated = model(coords @ rotation(20), atom_mask, batch["residue_mask"])
    assert output.shape == (2, 7, 16)
    assert torch.allclose(output, rotated, atol=1e-4)

    # The original implementation's RBF uses 20/16, not center spacing 20/15.
    distance = torch.tensor([1.25, 7.5])
    centres = torch.linspace(0, 20, 16)
    expected = torch.exp(-((distance[:, None] - centres) / (20 / 16)) ** 2)
    assert torch.allclose(radial_basis(distance, 16, 20, "paper"), expected)
    assert not torch.allclose(radial_basis(distance, 16, 20), expected)

    index = torch.tensor([[[0, 2], [0, 2], [0, 1]]])
    encoded = model._separation(index)
    # At zero sequence separation the paper's cosine half is one, sine half zero.
    assert torch.allclose(encoded[0, 0, 0, :8], torch.ones(8))
    assert torch.allclose(encoded[0, 0, 0, 8:], torch.zeros(8))


def test_ca_anchor_uses_ca_coordinates_for_neighbour_selection(monkeypatch):
    import psjepa.layers.gvp_message as module

    batch = tiny_batch()
    recorded = []
    original = module.neighbour_indices

    def capture(anchors, visible, count):
        recorded.append(anchors.detach().clone())
        return original(anchors, visible, count)

    monkeypatch.setattr(module, "neighbour_indices", capture)
    for mode in ("mean", "ca"):
        model = GVPMessageFeaturizer(
            d_model=16, atoms=4, neighbours=3, layers=1, message_layers=2,
            scalar_hidden=32, vector_hidden=8, node_vectors=8,
            node_features="paper", anchor_mode=mode,
        ).eval()
        model(batch["coords"], batch["atom_mask"], batch["residue_mask"])
    assert torch.allclose(recorded[1], batch["coords"][..., 1, :])
    assert not torch.allclose(recorded[0], recorded[1])


def test_paper_scalar_atom4_reads_all_four_atom_offsets():
    batch = tiny_batch()
    model = message_featurizer("paper_scalar_atom4", global_radial=0,
                               edge_positional=True).eval()
    seen = {}
    original = model.node_in.forward

    def capture(scalars, vectors):
        seen["scalars"], seen["vectors"] = scalars.detach(), vectors.detach()
        return original(scalars, vectors)

    model.node_in.forward = capture
    model(batch["coords"], batch["atom_mask"], batch["residue_mask"])
    coords, atom_mask = batch["coords"], batch["atom_mask"]
    weights = atom_mask.unsqueeze(-1)
    anchor = (coords * weights).sum(-2) / atom_mask.sum(-1, keepdim=True).clamp_min(1)
    expected_vectors = (coords - anchor.unsqueeze(-2)) * weights
    expected_scalars, _ = paper_node_features(coords, atom_mask, batch["residue_mask"])
    assert seen["scalars"].shape[-1] == 6
    assert seen["vectors"].shape[-2] == 4
    assert torch.allclose(seen["scalars"], expected_scalars)
    assert torch.allclose(seen["vectors"], expected_vectors)
