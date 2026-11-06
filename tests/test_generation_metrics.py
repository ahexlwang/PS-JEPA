"""The dihedral convention and the reference-free metrics, pinned to real data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from psjepa.analysis.generation_metrics import chain_metrics, torsion

CACHE = Path("cache/cath_backbone_split/validation")


def reference_dihedral(p0, p1, p2, p3) -> float:
    """The standard ('praxeolitic') IUPAC dihedral, written out independently."""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def test_torsion_matches_an_independent_implementation():
    generator = np.random.RandomState(0)
    for _ in range(64):
        points = generator.normal(size=(4, 3))
        expected = reference_dihedral(*points)
        measured = float(torsion(*torch.from_numpy(points).float()))
        assert abs(((measured - expected + 180) % 360) - 180) < 1e-2, (expected, measured)


def test_torsion_is_cis_and_trans_the_right_way_round():
    # Coplanar points: d on the same side of the b-c axis as a is cis (0), on
    # the opposite side is trans (180).
    a = torch.tensor([0.0, 1.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([1.0, 0.0, 0.0])
    assert abs(float(torsion(a, b, c, torch.tensor([1.0, 1.0, 0.0])))) < 1e-3
    assert abs(abs(float(torsion(a, b, c, torch.tensor([1.0, -1.0, 0.0])))) - 180) < 1e-3


def test_torsion_sign_flips_under_reflection():
    points = torch.randn(4, 3)
    direct = float(torsion(*points))
    mirrored = points.clone()
    mirrored[:, 2] *= -1
    assert abs(float(torsion(*mirrored)) + direct) < 1e-3


@pytest.mark.skipif(not CACHE.exists(), reason="CATH backbone cache not present")
def test_real_backbones_are_trans_and_left_handed():
    coords = np.load(CACHE / "coords.npy", mmap_mode="r")
    offsets = np.load(CACHE / "offsets.npy")
    records = []
    for index in range(24):
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end - start < 20:
            continue
        records.append(chain_metrics(torch.from_numpy(np.array(coords[start:end]))))
    assert records, "no chain long enough in the first 24 cache entries"

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in records]))


    assert mean("omega_trans_fraction") > 0.95, mean("omega_trans_fraction")
    assert mean("phi_negative_fraction") > 0.85, mean("phi_negative_fraction")
    assert mean("chain_break_fraction") < 0.02, mean("chain_break_fraction")
    assert 1.30 < mean("peptide_c_n_mean") < 1.36, mean("peptide_c_n_mean")
    assert 3.75 < mean("ca_ca_mean") < 3.85, mean("ca_ca_mean")
    assert mean("clash_fraction") < 0.01, mean("clash_fraction")
    # CATH is a fold library, so both basins have to be populated; an assertion
    # that only one is would pass on a helical bundle and fail on a beta barrel.
    assert mean("helix_fraction") + mean("sheet_fraction") > 0.45, (
        mean("helix_fraction"), mean("sheet_fraction")
    )


@pytest.mark.skipif(not CACHE.exists(), reason="CATH backbone cache not present")
def test_a_scrambled_backbone_scores_badly():
    coords = np.load(CACHE / "coords.npy", mmap_mode="r")
    offsets = np.load(CACHE / "offsets.npy")
    start, end = int(offsets[0]), int(offsets[1])
    real = torch.from_numpy(np.array(coords[start:end]))
    scrambled = real[torch.randperm(real.shape[0])]
    assert chain_metrics(scrambled)["chain_break_fraction"] > 0.5
    assert chain_metrics(real)["chain_break_fraction"] < 0.05
