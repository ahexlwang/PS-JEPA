"""Reconstruction metrics that separate local geometry from global arrangement.

Aligned backbone RMSD answers one question — how far apart are two structures
after the best rigid superposition — and it answers it with a single number that
grows with chain length and is dominated by the worst-placed residues. At the
19-21 Angstrom operating point this project reached, that number cannot
distinguish a model with correct local geometry and scrambled packing from one
that is wrong everywhere, and those are exactly the two hypotheses the token
diagnostics leave open: the tokens carry backbone dihedral information at
NMI 0.22 and carry almost nothing about global arrangement.

Every metric here is chosen to break that conflation.

``lddt``            superposition-free, so no alignment can be blamed
``fragment_rmsd``   the same RMSD over sliding windows, which is the direct test
``dihedral_error``  ties reconstruction to what the tokens provably encode
``contact_precision`` topology alone, at the length-normalized scale folds use
"""
from __future__ import annotations

import math

import torch


def _kabsch_rmsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """RMSD after the optimal proper rotation. Reflections are excluded."""
    p = p - p.mean(dim=0, keepdim=True)
    q = q - q.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.transpose(0, 1) @ q)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return (p @ rotation - q).square().sum(dim=-1).mean().sqrt()


@torch.no_grad()
def lddt(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ca_mask: torch.Tensor,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Local distance difference test on CA atoms, averaged over chains.

    No superposition is involved: the score is the fraction of reference
    distances shorter than ``inclusion_radius`` that the prediction reproduces
    within each tolerance, averaged over tolerances. A rotation-invariant
    representation cannot be penalised by an unlucky alignment here, which
    removes the objection that Kabsch-based scores invite.
    """
    values: list[torch.Tensor] = []
    for predicted, truth, mask in zip(prediction.float(), target.float(), ca_mask):
        p = predicted[:, 1][mask]
        q = truth[:, 1][mask]
        n = p.shape[0]
        if n < 2:
            continue
        reference = torch.cdist(q, q)
        keep = torch.triu(torch.ones(n, n, device=p.device, dtype=torch.bool), 1)
        keep &= reference < inclusion_radius
        if not keep.any():
            continue
        deviation = (torch.cdist(p, p)[keep] - reference[keep]).abs()
        preserved = torch.stack([(deviation < t).float().mean() for t in thresholds])
        values.append(preserved.mean())
    return float(torch.stack(values).mean()) if values else float("nan")


@torch.no_grad()
def fragment_rmsd(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ca_mask: torch.Tensor,
    window: int,
) -> float:
    """CA RMSD over sliding windows, each superposed on its own.

    Read as a ladder across window sizes this decomposes the global number.
    A model whose local geometry is right and whose packing is wrong stays low
    at small windows and climbs; a model that is wrong everywhere is flat.
    """
    values: list[torch.Tensor] = []
    for predicted, truth, mask in zip(prediction.float(), target.float(), ca_mask):
        p = predicted[:, 1][mask]
        q = truth[:, 1][mask]
        n = p.shape[0]
        if n < window or window < 3:
            continue
        # Stride with the window so fragments do not overlap: overlapping
        # windows correlate, and the mean over them would understate spread
        # without saying so.
        for start in range(0, n - window + 1, window):
            values.append(_kabsch_rmsd(p[start : start + window], q[start : start + window]))
    return float(torch.stack(values).mean()) if values else float("nan")


def _torsion(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Signed dihedral of four consecutive points, in radians."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    m = torch.cross(n1, b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8), dim=-1)
    return torch.atan2((m * n2).sum(-1), (n1 * n2).sum(-1))


@torch.no_grad()
def dihedral_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
    tolerance_degrees: float = 30.0,
) -> dict[str, float]:
    """Backbone phi/psi recovery, as circular mean absolute error and a hit rate.

    phi needs C(i-1), N(i), CA(i), C(i); psi needs N(i), CA(i), C(i), N(i+1).
    Atom order in this project's cache is N, CA, C, O.
    """
    errors: list[torch.Tensor] = []
    for predicted, truth, mask in zip(prediction.float(), target.float(), atom_mask):
        complete = mask[:, 0] & mask[:, 1] & mask[:, 2]
        index = torch.nonzero(complete, as_tuple=False).flatten()
        if index.numel() < 2:
            continue
        # Only consecutive complete residues carry a defined torsion pair.
        consecutive = index[1:][(index[1:] - index[:-1]) == 1]
        if consecutive.numel() == 0:
            continue
        previous = consecutive - 1
        phi_p = _torsion(predicted[previous, 2], predicted[consecutive, 0],
                         predicted[consecutive, 1], predicted[consecutive, 2])
        phi_q = _torsion(truth[previous, 2], truth[consecutive, 0],
                         truth[consecutive, 1], truth[consecutive, 2])
        psi_p = _torsion(predicted[previous, 0], predicted[previous, 1],
                         predicted[previous, 2], predicted[consecutive, 0])
        psi_q = _torsion(truth[previous, 0], truth[previous, 1],
                         truth[previous, 2], truth[consecutive, 0])
        both_p = torch.cat([phi_p, psi_p])
        both_q = torch.cat([phi_q, psi_q])
        difference = both_p - both_q
        # Wrap to (-pi, pi]: an error of 359 degrees is an error of 1 degree.
        errors.append((difference + math.pi).remainder(2 * math.pi) - math.pi)
    if not errors:
        return {"dihedral_mae_degrees": float("nan"), "dihedral_within_tolerance": float("nan")}
    stacked = torch.cat(errors).abs()
    return {
        "dihedral_mae_degrees": float(stacked.mean() * 180.0 / math.pi),
        "dihedral_within_tolerance": float(
            (stacked < tolerance_degrees * math.pi / 180.0).float().mean()
        ),
    }


@torch.no_grad()
def contact_precision(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ca_mask: torch.Tensor,
    cutoff: float = 8.0,
    minimum_separation: int = 6,
    top_multiple: float = 1.0,
) -> float:
    """Precision of the top-L predicted non-local CA contacts against the truth.

    Fold identity is a contact-map property, and the top-L convention makes the
    number comparable across chain lengths. Contacts within
    ``minimum_separation`` residues along the chain are excluded because they
    follow from the chain itself and would inflate every model equally.
    """
    values: list[torch.Tensor] = []
    for predicted, truth, mask in zip(prediction.float(), target.float(), ca_mask):
        p = predicted[:, 1][mask]
        q = truth[:, 1][mask]
        n = p.shape[0]
        if n < minimum_separation + 2:
            continue
        offset = torch.arange(n, device=p.device)
        separation = (offset[None, :] - offset[:, None]).abs()
        eligible = torch.triu(torch.ones(n, n, device=p.device, dtype=torch.bool), 1)
        eligible &= separation >= minimum_separation
        if not eligible.any():
            continue
        truth_contact = (torch.cdist(q, q) < cutoff) & eligible
        distance = torch.cdist(p, p).masked_fill(~eligible, float("inf"))
        count = min(int(top_multiple * n), int(eligible.sum()))
        if count < 1:
            continue
        flat = torch.topk(distance.flatten(), count, largest=False).indices
        values.append(truth_contact.flatten()[flat].float().mean())
    return float(torch.stack(values).mean()) if values else float("nan")


@torch.no_grad()
def backbone_bond_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
    tolerance: float = 0.1,
) -> dict[str, float]:
    """Bond diagnostics that distinguish a backbone from an atom cloud.

    The cache atom order is N, CA, C, O.  Within-residue N--CA and CA--C bonds
    test local residue geometry; C(i)--N(i+1) is the indispensable peptide-chain
    continuity test that the original Cartesian flow probe omitted.
    """

    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "n_ca": [],
        "ca_c": [],
        "peptide_c_n": [],
    }
    for predicted, truth, mask in zip(prediction.float(), target.float(), atom_mask):
        n_ca = mask[:, 0] & mask[:, 1]
        ca_c = mask[:, 1] & mask[:, 2]
        peptide = mask[:-1, 2] & mask[1:, 0]
        if n_ca.any():
            groups["n_ca"].append(
                (
                    torch.linalg.vector_norm(predicted[n_ca, 0] - predicted[n_ca, 1], dim=-1),
                    torch.linalg.vector_norm(truth[n_ca, 0] - truth[n_ca, 1], dim=-1),
                )
            )
        if ca_c.any():
            groups["ca_c"].append(
                (
                    torch.linalg.vector_norm(predicted[ca_c, 1] - predicted[ca_c, 2], dim=-1),
                    torch.linalg.vector_norm(truth[ca_c, 1] - truth[ca_c, 2], dim=-1),
                )
            )
        if peptide.any():
            groups["peptide_c_n"].append(
                (
                    torch.linalg.vector_norm(
                        predicted[:-1, 2][peptide] - predicted[1:, 0][peptide], dim=-1
                    ),
                    torch.linalg.vector_norm(
                        truth[:-1, 2][peptide] - truth[1:, 0][peptide], dim=-1
                    ),
                )
            )

    result: dict[str, float] = {}
    for name, records in groups.items():
        if not records:
            result[f"{name}_mae"] = float("nan")
            result[f"{name}_predicted_mean"] = float("nan")
            result[f"{name}_within_tolerance"] = float("nan")
            continue
        predicted = torch.cat([record[0] for record in records])
        truth = torch.cat([record[1] for record in records])
        error = (predicted - truth).abs()
        result[f"{name}_mae"] = float(error.mean())
        result[f"{name}_predicted_mean"] = float(predicted.mean())
        result[f"{name}_within_tolerance"] = float((error < tolerance).float().mean())
    return result


@torch.no_grad()
def structure_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
    fragment_windows: tuple[int, ...] = (4, 8, 16, 32, 64),
) -> dict[str, float]:
    """Every metric in this module, on one batch of generated backbones."""
    ca_mask = atom_mask[:, :, 1]
    result: dict[str, float] = {
        "lddt": lddt(prediction, target, ca_mask),
        "contact_precision_at_l": contact_precision(prediction, target, ca_mask),
    }
    for window in fragment_windows:
        result[f"fragment_rmsd_{window}"] = fragment_rmsd(prediction, target, ca_mask, window)
    result.update(dihedral_error(prediction, target, atom_mask))
    result.update(backbone_bond_metrics(prediction, target, atom_mask))
    return result
