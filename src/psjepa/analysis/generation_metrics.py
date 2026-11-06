'Reference-free quality metrics for generated backbones.'
from __future__ import annotations

import math

import torch

# Engh & Huber ideal backbone geometry.
IDEAL_N_CA = 1.458
IDEAL_CA_C = 1.525
IDEAL_C_N = 1.329
IDEAL_CA_CA = 3.80


def _norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, dim=-1)


def torsion(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Signed IUPAC dihedral in degrees for four consecutive points.

    The first bond vector points *backwards* (a - b). Getting that wrong offsets
    every angle by 180 degrees, which turns an all-trans backbone into an
    all-cis one, and an extra negation mirrors the Ramachandran plot so real
    L-amino acids read as D. The first version of this function had both faults
    and reported real CATH chains as 0.2% trans with 5.6% negative phi.
    `tests/test_generation_metrics.py` pins the convention against an
    independent implementation and against real CATH chains.
    """
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.rad2deg(torch.atan2(y, x))


def _chain_torsions(coords: torch.Tensor) -> dict[str, torch.Tensor]:
    """phi, psi and omega of one chain, in degrees, as 1-D tensors."""
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    return {
        "phi": torsion(c[:-1], n[1:], ca[1:], c[1:]),
        "psi": torsion(n[:-1], ca[:-1], c[:-1], n[1:]),
        "omega": torsion(ca[:-1], c[:-1], n[1:], ca[1:]),
    }


def chain_metrics(coords: torch.Tensor) -> dict[str, float]:
    """Every reference-free metric for a single chain of shape [L, 4, 3]."""
    coords = coords.float()
    length = coords.shape[0]
    if length < 4:
        return {}
    n, ca, c = coords[:, 0], coords[:, 1], coords[:, 2]
    torsions = _chain_torsions(coords)
    phi, psi, omega = torsions["phi"], torsions["psi"], torsions["omega"]

    # Bonds. `*_rmse` is the deviation from ideal geometry, not from a reference
    # structure, so it is defined for a de novo sample.
    bonds = {
        "n_ca": (_norm(n, ca), IDEAL_N_CA),
        "ca_c": (_norm(ca, c), IDEAL_CA_C),
        "peptide_c_n": (_norm(c[:-1], n[1:]), IDEAL_C_N),
        "ca_ca": (_norm(ca[:-1], ca[1:]), IDEAL_CA_CA),
    }
    result: dict[str, float] = {}
    for name, (values, ideal) in bonds.items():
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_rmse_vs_ideal"] = float((values - ideal).square().mean().sqrt())
        result[f"{name}_within_0p1"] = float(((values - ideal).abs() < 0.1).float().mean())

    # A chain break is a peptide bond far from ideal; one break ruins a fold even
    # if the average bond looks fine, so count them as well as average them.
    peptide = bonds["peptide_c_n"][0]
    result["chain_break_fraction"] = float((peptide > 2.0).float().mean())

    # Torsions. Real backbones are essentially all-trans and L-handed, so
    # `omega_trans` and `phi_negative` are chirality and planarity checks.
    result["omega_trans_fraction"] = float((omega.abs() > 150.0).float().mean())
    result["phi_negative_fraction"] = float((phi < 0).float().mean())

    # Coarse Ramachandran occupancy. The boxes are deliberately wide: the point
    # is the balance between the two basins, not a validation-grade count.
    helix = (phi > -100) & (phi < -30) & (psi > -80) & (psi < -5)
    sheet = (phi > -180) & (phi < -45) & ((psi > 90) | (psi < -150))
    result["helix_fraction"] = float(helix.float().mean())
    result["sheet_fraction"] = float(sheet.float().mean())
    result["coil_fraction"] = float((~helix & ~sheet).float().mean())

    # Packing. A generated chain that fails to collapse looks fine locally and
    # is not a protein; radius of gyration against length is the cheapest test.
    centre = ca.mean(dim=0, keepdim=True)
    radius = float((ca - centre).square().sum(dim=-1).mean().sqrt())
    result["radius_of_gyration"] = radius
    # Compact globular proteins follow Rg ~ 2.2 * N^0.38 Angstrom.
    result["radius_of_gyration_ratio"] = radius / (2.2 * length**0.38)

    distances = torch.cdist(ca, ca)
    index = torch.arange(length, device=coords.device)
    separation = (index.reshape(-1, 1) - index.reshape(1, -1)).abs()
    far = separation >= 3
    upper = torch.triu(torch.ones_like(far), diagonal=1).bool()
    nonlocal_pairs = far & upper
    if nonlocal_pairs.any():
        result["clash_fraction"] = float(
            (distances[nonlocal_pairs] < 3.5).float().mean()
        )
        contacts = nonlocal_pairs & (distances < 8.0)
        count = int(contacts.sum())
        result["contact_density"] = count / length
        result["relative_contact_order"] = (
            float(separation[contacts].float().mean()) / length if count else 0.0
        )
    result["length"] = float(length)
    return result


def batch_metrics(
    coords: torch.Tensor, residue_mask: torch.Tensor
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Mean over a padded batch plus the per-chain records behind it."""
    records: list[dict[str, float]] = []
    for structure, mask in zip(coords, residue_mask):
        record = chain_metrics(structure[mask])
        if record:
            records.append(record)
    if not records:
        return {}, []
    keys = sorted(records[0])
    summary = {
        key: float(
            torch.tensor([r[key] for r in records if key in r and math.isfinite(r[key])]).mean()
        )
        for key in keys
    }
    return summary, records


# -- diversity and memorisation -------------------------------------------
#
# Plausibility and diversity fail in opposite directions, and a generator can
# score perfectly on the geometry above while emitting one fold every time. The
# functions below are the other half of the test: how different are two samples
# of the same length, and how close is a sample to the training distribution it
# was fitted to.


def kabsch_superpose(mobile: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """`mobile` rotated and translated onto `reference` (proper rotations only)."""
    p = mobile - mobile.mean(dim=0, keepdim=True)
    q = reference - reference.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.transpose(0, 1) @ q)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return p @ rotation


def tm_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """TM-score of two equal-length CA traces under residue-index correspondence.

    Real TM-score searches over alignments; this fixes the 1-to-1 residue
    correspondence and superposes with Kabsch, so it is a lower bound. That is
    the conservative direction for a diversity claim: a *low* value here really
    means the two structures differ, while a high value should be confirmed with
    a proper alignment tool before it is called near-identity.
    """
    if a.shape != b.shape or a.shape[0] < 4:
        return float("nan")
    length = a.shape[0]
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8 if length > 21 else 0.5
    d0 = max(d0, 0.5)
    aligned = kabsch_superpose(a.float(), b.float())
    target = b.float() - b.float().mean(dim=0, keepdim=True)
    distance = torch.linalg.vector_norm(aligned - target, dim=-1)
    return float((1.0 / (1.0 + (distance / d0) ** 2)).mean())


def pairwise_diversity(
    structures: list[torch.Tensor], max_pairs: int = 4000, seed: int = 0
) -> dict[str, float]:
    """Similarity between every pair of equal-length CA traces in a set.

    `mean_pairwise_tm` near 1 is mode collapse; on real CATH chains of the same
    length it is low, and that value is the reference this should be read
    against rather than any absolute threshold.
    """
    by_length: dict[int, list[torch.Tensor]] = {}
    for structure in structures:
        by_length.setdefault(int(structure.shape[0]), []).append(structure)
    scores: list[float] = []
    generator = torch.Generator().manual_seed(seed)
    for group in by_length.values():
        if len(group) < 2:
            continue
        pairs = [(i, j) for i in range(len(group)) for j in range(i + 1, len(group))]
        if len(pairs) > max_pairs:
            chosen = torch.randperm(len(pairs), generator=generator)[:max_pairs]
            pairs = [pairs[int(k)] for k in chosen]
        for i, j in pairs:
            value = tm_score(group[i], group[j])
            if value == value:
                scores.append(value)
    if not scores:
        return {"mean_pairwise_tm": float("nan"), "pairs_compared": 0.0}
    values = torch.tensor(scores)
    return {
        "mean_pairwise_tm": float(values.mean()),
        "max_pairwise_tm": float(values.max()),
        "fraction_pairs_tm_above_0p5": float((values > 0.5).float().mean()),
        "pairs_compared": float(len(scores)),
    }


def nearest_reference_similarity(
    samples: list[torch.Tensor], reference: list[torch.Tensor]
) -> dict[str, float]:
    """For each sample, the best TM-score against any equal-length reference chain.

    High values would mean the generator is reproducing chains it was fitted to.
    Only equal-length pairs are compared, so this is a memorisation screen and
    not a novelty measurement against a fold database.
    """
    grouped: dict[int, list[torch.Tensor]] = {}
    for structure in reference:
        grouped.setdefault(int(structure.shape[0]), []).append(structure)
    best: list[float] = []
    for sample in samples:
        candidates = grouped.get(int(sample.shape[0]), [])
        values = [tm_score(sample, other) for other in candidates]
        values = [v for v in values if v == v]
        if values:
            best.append(max(values))
    if not best:
        return {"mean_nearest_reference_tm": float("nan"), "samples_matched": 0.0}
    values = torch.tensor(best)
    return {
        "mean_nearest_reference_tm": float(values.mean()),
        "max_nearest_reference_tm": float(values.max()),
        "samples_matched": float(len(best)),
    }
