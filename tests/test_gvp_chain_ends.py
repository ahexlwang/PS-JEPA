'`chain_ends: bonded` for the paper GVP node features (2026-10-13 input audit).'
import torch

from psjepa.layers.gvp_message import GVPMessageFeaturizer, paper_node_features


def _backbone(length, seed=0):
    """N/CA/C/O with consecutive C(i)-N(i+1) at peptide distance."""
    g = torch.Generator().manual_seed(seed)
    steps = torch.nn.functional.normalize(torch.randn(length, 3, generator=g), dim=-1) * 3.8
    ca = steps.cumsum(0)
    x = torch.zeros(length, 4, 3)
    x[:, 1] = ca
    nxt = torch.cat([ca[1:], ca[-1:] + steps[-1:]])
    prv = torch.cat([ca[:1] - steps[:1], ca[:-1]])
    x[:, 2] = ca + 0.4 * (nxt - ca) + 0.3 * torch.randn(length, 3, generator=g)  # C toward i+1
    x[:, 0] = ca + 0.4 * (prv - ca) + 0.3 * torch.randn(length, 3, generator=g)  # N toward i-1
    x[:, 3] = x[:, 2] + 0.5 * torch.randn(length, 3, generator=g)
    # Close every peptide link: N(i+1) sits 1.33 A from C(i).
    for i in range(length - 1):
        d = torch.nn.functional.normalize(x[i + 1, 1] - x[i, 2], dim=0)
        x[i + 1, 0] = x[i, 2] + 1.33 * d
    return x


def _pad(chains):
    width = max(len(c) for c in chains)
    coords = torch.zeros(len(chains), width, 4, 3)
    mask = torch.zeros(len(chains), width, 4, dtype=torch.bool)
    for row, c in enumerate(chains):
        coords[row, :len(c)] = c
        mask[row, :len(c)] = True
    return coords, mask


def test_bonded_c_terminus_does_not_depend_on_batch_padding():
    short, longer = _backbone(30, 1), _backbone(41, 2)
    alone = paper_node_features(short[None], torch.ones(1, 30, 4, dtype=torch.bool),
                                torch.ones(1, 30, dtype=torch.bool), chain_ends="bonded")
    coords, mask = _pad([short, longer])
    padded = paper_node_features(coords, mask, mask.any(-1), chain_ends="bonded")
    torch.testing.assert_close(alone[0][0], padded[0][0, :30])
    torch.testing.assert_close(alone[1][0], padded[1][0, :30])
    # Both ends read the reference terminus: psi/omega (and phi at the N end) as angle 0.
    assert torch.equal(alone[0][0, -1, [1, 2, 4, 5]], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(alone[0][0, 0, [0, 3]], torch.tensor([1.0, 0.0]))

    legacy_alone = paper_node_features(short[None], torch.ones(1, 30, 4, dtype=torch.bool),
                                       torch.ones(1, 30, dtype=torch.bool))
    legacy_padded = paper_node_features(coords, mask, mask.any(-1))
    assert not torch.allclose(legacy_alone[0][0, -1], legacy_padded[0][0, 29])
    featurizer = GVPMessageFeaturizer(16, 4, neighbours=6, layers=1, node_features="paper",
                                      global_radial=16, edge_positional=True,
                                      chain_ends="bonded").eval()
    with torch.no_grad():
        out_alone = featurizer(short[None], torch.ones(1, 30, 4, dtype=torch.bool))
        out_padded = featurizer(coords, mask)
    torch.testing.assert_close(out_alone[0], out_padded[0, :30], atol=1e-5, rtol=1e-5)


def test_bonded_break_reads_as_two_chain_ends():
    chain = _backbone(40, 3)
    full = paper_node_features(chain[None], torch.ones(1, 40, 4, dtype=torch.bool),
                               torch.ones(1, 40, dtype=torch.bool), chain_ends="bonded")
    keep = torch.cat([torch.arange(0, 20), torch.arange(26, 40)])      # residues 20-25 missing
    gapped = chain[keep]
    assert (gapped[20, 0] - gapped[19, 2]).norm() > 2.0
    out = paper_node_features(gapped[None], torch.ones(1, 34, 4, dtype=torch.bool),
                              torch.ones(1, 34, dtype=torch.bool), chain_ends="bonded")
    scalars, vectors = out[0][0], out[1][0]
    # Slot 19 (residue 19) loses psi/omega and its forward direction; slot 20 (residue 26)
    # loses phi and its backward direction. Both read as termini.
    assert torch.equal(scalars[19, [1, 2, 4, 5]], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(scalars[20, [0, 3]], torch.tensor([1.0, 0.0]))
    assert torch.all(vectors[19, 0] == 0) and torch.all(vectors[20, 1] == 0)
    # Away from the break, every feature equals the unbroken chain's.
    for slot, residue in [(5, 5), (18, 18), (21, 27), (33, 39)]:
        torch.testing.assert_close(scalars[slot], full[0][0, residue])
        torch.testing.assert_close(vectors[slot], full[1][0, residue])
    # The legacy mode computes straight across the gap.
    legacy = paper_node_features(gapped[None], torch.ones(1, 34, 4, dtype=torch.bool),
                                 torch.ones(1, 34, dtype=torch.bool))
    assert torch.any(legacy[1][0, 19, 0] != 0)


def test_bonded_hidden_neighbour_is_zero_and_never_read():
    chain = _backbone(30, 4)
    mask = torch.ones(1, 30, 4, dtype=torch.bool)
    visible = torch.ones(1, 30, dtype=torch.bool)
    visible[0, 12] = False
    base = paper_node_features(chain[None], mask, visible, chain_ends="bonded")
    # Neighbours of a hidden residue read (0, 0), not a terminus.
    assert torch.all(base[0][0, 11, [1, 2, 4, 5]] == 0)
    assert torch.all(base[0][0, 13, [0, 3]] == 0)
    moved = chain.clone()
    moved[12] += torch.tensor([30.0, -20.0, 10.0])                    # also breaks its links
    shifted = paper_node_features(moved[None], mask, visible, chain_ends="bonded")
    others = torch.arange(30) != 12
    torch.testing.assert_close(base[0][0, others], shifted[0][0, others])
    torch.testing.assert_close(base[1][0, others], shifted[1][0, others])


def test_bonded_gathered_view_gap_reads_as_chain_end_and_config_plumbs():
    from psjepa.models.psjepa import PSJEPAConfig
    chain = _backbone(20, 5)
    adjacent = torch.ones(1, 19, dtype=torch.bool)
    adjacent[0, 9] = False                       # slots 9 and 10 are not consecutive
    out = paper_node_features(chain[None], torch.ones(1, 20, 4, dtype=torch.bool),
                              torch.ones(1, 20, dtype=torch.bool), adjacent, "bonded")
    assert torch.equal(out[0][0, 9, [1, 2, 4, 5]], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(out[0][0, 10, [0, 3]], torch.tensor([1.0, 0.0]))
    base = {"d_model": 16, "n_layer": 1, "max_residues": 64, "predictor_layers": 1}
    config = PSJEPAConfig.from_dict({**base, "input_mode": "gvp_message",
                                       "gvp_node_features": "paper",
                                       "gvp_chain_ends": "bonded"}, {}, {})
    assert config.gvp_chain_ends == "bonded"
    assert PSJEPAConfig.from_dict(base, {}, {}).gvp_chain_ends == "legacy"
