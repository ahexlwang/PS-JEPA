"""Geometric vector perceptron featurisation, following Jing et al. 2021.

A GVP maps a (scalar, vector) pair to a new pair such that the scalars are
invariant under rotation of the vectors and the vectors are equivariant. Scalars
receive geometry only through vector norms, and vectors are only ever rescaled by
scalar gains, so no operation mixes the three spatial components.

Only the scalar channel leaves this module. That is what makes the guarantee
usable here: the Mamba stack downstream is an ordinary sequence model that treats
its input as plain features, so handing it an equivariant vector would destroy the
invariance. Everything vector-valued stops at this boundary.

Nothing in the construction is specific to proteins. A site is described by its
own atom cloud and by its nearest neighbouring sites, so the same code applies to
nucleotides or to the atoms of a ligand — no backbone frame, no dihedral, and no
residue identity is used.

Norms alone would be invariant to reflection as well as rotation, which would make
a structure and its mirror image indistinguishable. That is unacceptable for
ligands, where enantiomers routinely differ in activity, so signed triple products
are included; those are invariant under rotation and change sign under reflection.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GVP(nn.Module):
    """One geometric vector perceptron: invariant scalars, equivariant vectors."""

    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        vector_hidden: int | None = None,
    ) -> None:
        super().__init__()
        hidden = vector_hidden or max(vector_in, vector_out)
        self.vector_hidden = nn.Linear(vector_in, hidden, bias=False)
        self.scalar_out = nn.Linear(scalar_in + hidden, scalar_out)
        self.vector_out = nn.Linear(hidden, vector_out, bias=False)
        self.gate = nn.Linear(scalar_out, vector_out)

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # vectors: (..., vector_in, 3). Mixing channels with one weight shared
        # across x, y, and z is what keeps the output equivariant.
        hidden = self.vector_hidden(vectors.transpose(-1, -2)).transpose(-1, -2)
        norms = torch.linalg.vector_norm(hidden, dim=-1)
        updated = self.scalar_out(torch.cat([scalars, norms], dim=-1))
        out_vectors = self.vector_out(hidden.transpose(-1, -2)).transpose(-1, -2)
        out_vectors = out_vectors * torch.sigmoid(self.gate(updated)).unsqueeze(-1)
        return F.silu(updated), out_vectors


def signed_triple_products(vectors: torch.Tensor, count: int = 4) -> torch.Tensor:
    """Chirality-sensitive scalars: rotation invariant, sign-flipping under reflection."""
    channels = vectors.shape[-2]
    if channels < 3:
        return vectors.new_zeros((*vectors.shape[:-2], 0))
    products = []
    for start in range(min(count, channels - 2)):
        a, b, c = vectors[..., start, :], vectors[..., start + 1, :], vectors[..., start + 2, :]
        scale = (
            torch.linalg.vector_norm(a, dim=-1)
            * torch.linalg.vector_norm(b, dim=-1)
            * torch.linalg.vector_norm(c, dim=-1)
        )
        products.append((torch.cross(a, b, dim=-1) * c).sum(dim=-1) / scale.clamp_min(1e-6))
    return torch.stack(products, dim=-1)


def radial_basis(distances: torch.Tensor, count: int, cutoff: float,
                 width_mode: str = "spacing") -> torch.Tensor:
    centres = torch.linspace(0.0, cutoff, count, device=distances.device, dtype=distances.dtype)
    if width_mode == "spacing":
        width = cutoff / max(count - 1, 1)
    elif width_mode == "paper":
        # drorlab/gvp-pytorch ProteinGraphDataset._rbf uses D_sigma = 20 / 16.
        width = cutoff / count
    else:
        raise ValueError(f"unknown radial basis width mode: {width_mode!r}")
    return torch.exp(-(((distances.unsqueeze(-1) - centres) / width) ** 2))


def neighbour_indices(
    anchors: torch.Tensor, visible: torch.Tensor, count: int, chunk: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices of the nearest visible sites, excluding the site itself.

    The graph is built over visible sites only. Including a masked site would let
    its position reach a visible site's features through the neighbour list, and
    even selecting it as a neighbour would reveal that it lies nearby — so the
    prediction target would be readable from the context.

    Distances carry no gradient, so they are computed under no_grad in batch
    chunks; a full pairwise matrix at batch 576 and length 512 would otherwise
    cost about 600 MB on its own.
    """
    batch, length, _ = anchors.shape
    picks, valid = [], []
    with torch.no_grad():
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            block = anchors[start:stop]
            distance = torch.cdist(block.float(), block.float())
            usable = visible[start:stop].unsqueeze(1).expand(-1, length, -1).clone()
            eye = torch.eye(length, dtype=torch.bool, device=anchors.device)
            usable &= ~eye.unsqueeze(0)
            distance = distance.masked_fill(~usable, float("inf"))
            taken = min(count, length)
            chosen = distance.topk(taken, dim=-1, largest=False)
            index = chosen.indices
            present = torch.isfinite(chosen.values)
            if taken < count:
                # The slot count has to stay static or it stops matching the layer
                # widths, so short structures get padded with unusable slots.
                pad = count - taken
                index = F.pad(index, (0, pad))
                present = F.pad(present, (0, pad), value=False)
            picks.append(index)
            valid.append(present)
    return torch.cat(picks), torch.cat(valid)


def global_radial_features(
    anchors: torch.Tensor,
    visible: torch.Tensor,
    count: int,
    cutoff: float,
    reference: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Where a site sits in the structure as a whole, as invariant scalars.

    Every other feature here is strictly local — a site's own atoms and its
    twelve nearest neighbours — so nothing tells the encoder whether a site is
    buried or on the surface. Absolute coordinates carried that implicitly,
    because the dataset centres each structure on its centroid, and dropping
    them is the most likely cause of the binding regression: the token geometry
    study found distance from the centre to be the strongest genuine geometric
    correlate of the token (NMI 0.0875, against 0.0094 for backbone dihedrals),
    and it separated core tokens from surface ones — 255 at r/Rg 0.49 against
    247 at 1.11.

    Both returned scalars are norms of differences from a centroid computed here,
    so they are unchanged by any rotation or translation of the input and the
    structural invariance guarantee still holds.

    The centroid and the radius of gyration are taken over VISIBLE sites only,
    for the same reason `neighbour_indices` builds its graph that way: a
    reference point computed over masked sites too would let their positions
    reach every visible site's features.
    """
    weight = visible.unsqueeze(-1).to(anchors.dtype)
    total = weight.sum(dim=-2).clamp_min(1.0)
    if reference is None:
        centroid = (anchors * weight).sum(dim=-2) / total
        offset = anchors - centroid.unsqueeze(-2)
        distance = torch.linalg.vector_norm(offset, dim=-1)
        gyration = ((distance**2 * weight.squeeze(-1)).sum(dim=-1)
                    / total.squeeze(-1)).clamp_min(1e-6).sqrt()
    else:


        centroid, gyration = reference
        offset = anchors - centroid.unsqueeze(-2)
        distance = torch.linalg.vector_norm(offset, dim=-1)
        gyration = gyration.clamp_min(1e-6)
    relative = (distance / gyration.unsqueeze(-1)).unsqueeze(-1)
    return torch.cat([radial_basis(distance, count, cutoff), relative], dim=-1)


class GVPFeaturizer(nn.Module):
    """Invariant per-site features from raw coordinates, with no protein priors."""

    def __init__(
        self,
        d_model: int,
        atoms: int,
        neighbours: int = 12,
        layers: int = 2,
        scalar_hidden: int = 128,
        vector_hidden: int = 16,
        radial: int = 8,
        cutoff: float = 20.0,
        global_radial: int = 0,
        global_cutoff: float = 40.0,
    ) -> None:
        super().__init__()
        self.neighbours = neighbours
        self.radial = radial
        self.cutoff = cutoff
        self.global_radial = global_radial
        self.global_cutoff = global_cutoff
        vector_channels = atoms + neighbours
        scalar_channels = (
            atoms  # distance from the site anchor to each of its own atoms
            + atoms  # which of those atoms are present
            + neighbours * radial  # neighbour distances, expanded
            + neighbours  # which neighbours exist
            + min(4, max(vector_channels - 2, 0))  # chirality
            # distance from the visible centroid, expanded, plus that distance
            # divided by the radius of gyration
            + (global_radial + 1 if global_radial else 0)
        )
        self.blocks = nn.ModuleList()
        scalars, vectors = scalar_channels, vector_channels
        for _ in range(layers):
            self.blocks.append(GVP(scalars, vectors, scalar_hidden, vector_hidden))
            scalars, vectors = scalar_hidden, vector_hidden
        self.project = nn.Sequential(nn.LayerNorm(scalars), nn.Linear(scalars, d_model))

    def forward(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        visible: torch.Tensor | None = None,
        radial_reference: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        present = atom_mask.unsqueeze(-1).to(coords.dtype)
        counts = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(coords.dtype)
        anchors = (coords * present).sum(dim=-2) / counts
        site_valid = atom_mask.any(dim=-1)
        if visible is None:
            visible = site_valid
        else:
            visible = visible & site_valid

        own = (coords - anchors.unsqueeze(-2)) * present
        own_distance = torch.linalg.vector_norm(own, dim=-1)

        index, neighbour_valid = neighbour_indices(anchors, visible, self.neighbours)
        gathered = torch.gather(
            anchors.unsqueeze(1).expand(-1, anchors.shape[1], -1, -1),
            2,
            index.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        keep = neighbour_valid.unsqueeze(-1).to(coords.dtype)
        neighbour = (gathered - anchors.unsqueeze(-2)) * keep
        neighbour_distance = torch.linalg.vector_norm(neighbour, dim=-1)

        vectors = torch.cat([own, neighbour], dim=-2)
        parts = [
            own_distance,
            atom_mask.to(coords.dtype),
            radial_basis(neighbour_distance, self.radial, self.cutoff).flatten(-2)
            * keep.expand(-1, -1, -1, self.radial).flatten(-2),
            neighbour_valid.to(coords.dtype),
            signed_triple_products(vectors),
        ]
        if self.global_radial:
            parts.append(
                global_radial_features(anchors, visible, self.global_radial, self.global_cutoff,
                                       radial_reference)
            )
        scalars = torch.cat(parts, dim=-1)
        for block in self.blocks:
            scalars, vectors = block(scalars, vectors)
        return self.project(scalars) * site_valid.unsqueeze(-1)
