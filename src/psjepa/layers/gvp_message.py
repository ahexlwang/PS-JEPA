'Geometric vector perceptron message passing (Jing et al., 2021).'
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from psjepa.layers.gvp import (GVP, global_radial_features, neighbour_indices,
                                radial_basis, signed_triple_products)


def _normalise(vector: torch.Tensor) -> torch.Tensor:
    """Unit vectors with missing or degenerate directions represented by zero."""
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)


# A C(i)-N(i+1) distance above this is a chain break, not a peptide bond (1.33 A).
PEPTIDE_BREAK = 2.0


def paper_node_features(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    visible: torch.Tensor,
    adjacent: torch.Tensor | None = None,
    chain_ends: str = "legacy",
) -> tuple[torch.Tensor, torch.Tensor]:
    'Original GVP protein node inputs, made safe for masked JEPA contexts.'
    if chain_ends not in ("legacy", "bonded"):
        raise ValueError(f"unknown chain_ends mode: {chain_ends!r}")
    if coords.shape[-2] < 3:
        raise ValueError("paper GVP node features require N, CA and C atoms")
    batch, length = coords.shape[:2]
    backbone = coords[..., :3, :]
    point_valid = (atom_mask[..., :3] & visible.unsqueeze(-1)).reshape(batch, 3 * length)

    # This is the original implementation's flattened N-CA-C construction. After
    # padding and reshaping the three angles at a site are phi, psi and omega.
    points = backbone.reshape(batch, 3 * length, 3)
    bonds = _normalise(points[:, 1:] - points[:, :-1])
    previous, middle, following = bonds[:, :-2], bonds[:, 1:-1], bonds[:, 2:]
    normal_previous = _normalise(torch.cross(previous, middle, dim=-1))
    normal_following = _normalise(torch.cross(middle, following, dim=-1))
    cosine = (normal_previous * normal_following).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.sign((previous * normal_following).sum(-1)) * torch.acos(cosine)
    angle_valid = (point_valid[:, :-3] & point_valid[:, 1:-2]
                   & point_valid[:, 2:-1] & point_valid[:, 3:])
    # The reference implementation represents unavailable terminal angles as angle zero.
    angle = F.pad(angle, (1, 2)).reshape(batch, length, 3)
    angle_valid = F.pad(angle_valid, (1, 2), value=True).reshape(batch, length, 3)
    ca = coords[..., 1, :]
    ca_valid = visible & atom_mask[..., 1]
    ca_step = ca[:, 1:] - ca[:, :-1]
    step_valid = ca_valid[:, 1:] & ca_valid[:, :-1]
    if chain_ends == "bonded":
        # One state per pair of slots (i, i+1): linked, hidden neighbour, or chain end.
        present = atom_mask.any(-1)
        own = visible & present
        slots = adjacent if adjacent is not None else torch.ones_like(present[:, 1:])
        both_present = present[:, :-1] & present[:, 1:] & slots
        measurable = (own[:, :-1] & own[:, 1:] & atom_mask[:, :-1, 2] & atom_mask[:, 1:, 0]
                      & slots)
        peptide = torch.linalg.vector_norm(coords[:, 1:, 0] - coords[:, :-1, 2], dim=-1)
        broken = measurable & (peptide > PEPTIDE_BREAK)
        linked = measurable & ~broken
        end = ~both_present | broken
        # phi(i) needs the pair (i-1, i); psi(i) and omega(i) need (i, i+1).
        links = torch.stack([F.pad(linked, (1, 0), value=False),
                             F.pad(linked, (0, 1), value=False),
                             F.pad(linked, (0, 1), value=False)], -1)
        ends = torch.stack([F.pad(end, (1, 0), value=True),
                            F.pad(end, (0, 1), value=True),
                            F.pad(end, (0, 1), value=True)], -1)
        computed = angle_valid & links
        terminal = (ends & own.unsqueeze(-1)).to(angle.dtype)
        node_scalars = torch.cat([torch.where(computed, torch.cos(angle), terminal),
                                  torch.where(computed, torch.sin(angle),
                                              torch.zeros_like(angle))], dim=-1)
        step_valid = step_valid & linked
    else:
        if adjacent is not None:
            # phi(i) needs residue i-1; psi(i) and omega(i) need residue i+1.
            previous_link = F.pad(adjacent, (1, 0), value=True)
            next_link = F.pad(adjacent, (0, 1), value=True)
            angle_valid = angle_valid & torch.stack([previous_link, next_link, next_link], -1)
        node_scalars = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
        node_scalars = node_scalars * torch.cat([angle_valid, angle_valid], dim=-1)
        if adjacent is not None:
            step_valid = step_valid & adjacent
    forward = F.pad(_normalise(ca_step) * step_valid.unsqueeze(-1), (0, 0, 0, 1))
    backward = F.pad(_normalise(-ca_step) * step_valid.unsqueeze(-1), (0, 0, 1, 0))

    nitrogen = _normalise(coords[..., 0, :] - ca)
    carbon = _normalise(coords[..., 2, :] - ca)
    bisector = _normalise(carbon + nitrogen)
    perpendicular = _normalise(torch.cross(carbon, nitrogen, dim=-1))
    sidechain = (-bisector * (1 / 3) ** 0.5
                 - perpendicular * (2 / 3) ** 0.5)
    own_valid = visible & atom_mask[..., :3].all(-1)
    sidechain = sidechain * own_valid.unsqueeze(-1)
    node_vectors = torch.stack([forward, backward, sidechain], dim=-2)
    return node_scalars, node_vectors


class VectorLayerNorm(nn.Module):
    """LayerNorm for a (scalar, vector) pair.

    Scalars take an ordinary `nn.LayerNorm`. Vectors cannot: subtracting a mean would not be
    equivariant and normalising each channel separately would erase the relative lengths that
    carry the geometry. Instead the whole set is divided by the root-mean-square of its
    channel norms, which is a single invariant scalar per site, so direction is untouched.
    """

    def __init__(self, scalars: int) -> None:
        super().__init__()
        self.scalar_norm = nn.LayerNorm(scalars)

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        squared = vectors.float().square().sum(dim=-1).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(squared.clamp_min(1e-8)).unsqueeze(-1).to(vectors.dtype)
        return self.scalar_norm(scalars), vectors * scale


class GVPScalarOut(nn.Module):
    """The original's terminal `GVP(node_h_dim, (ns, 0))`: consumes vectors, emits scalars.

    Without this the last layer's vector output has no consumer -- `project` reads scalars
    only -- so its `vector_out` and `gate` weights receive no gradient and the geometry the
    final round of message passing computed is thrown away. Here those vectors are mixed and
    entered as norms, which is invariant, so the boundary the Mamba stack relies on holds.
    """

    def __init__(self, scalar_in: int, vector_in: int, scalar_out: int,
                 vector_hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(vector_hidden or vector_in)
        self.vector_hidden = nn.Linear(vector_in, hidden, bias=False)
        self.scalar_out = nn.Linear(scalar_in + hidden, scalar_out)

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        hidden = self.vector_hidden(vectors.transpose(-1, -2)).transpose(-1, -2)
        norms = torch.linalg.vector_norm(hidden, dim=-1)
        return F.silu(self.scalar_out(torch.cat([scalars, norms], dim=-1)))


class GVPConv(nn.Module):
    """One round of message passing over the neighbour graph.

    A message from j to i is built from both endpoints' features and the edge between them,
    then averaged over i's valid neighbours. Averaging vectors is linear, so the aggregate is
    still equivariant; averaging rather than summing keeps the scale independent of how many
    neighbours a site happens to have, which varies at chain ends and under masking.
    """

    def __init__(
        self,
        node_scalars: int,
        node_vectors: int,
        edge_scalars: int,
        edge_vectors: int,
        hidden_scalars: int,
        hidden_vectors: int,
        layers: int = 2,
    ) -> None:
        super().__init__()
        blocks = []
        scalars = 2 * node_scalars + edge_scalars
        vectors = 2 * node_vectors + edge_vectors
        for index in range(int(layers)):
            last = index == int(layers) - 1
            blocks.append(GVP(
                scalars, vectors,
                node_scalars if last else hidden_scalars,
                node_vectors if last else hidden_vectors,
            ))
            scalars = node_scalars if last else hidden_scalars
            vectors = node_vectors if last else hidden_vectors
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        node_scalars: torch.Tensor,
        node_vectors: torch.Tensor,
        edge_scalars: torch.Tensor,
        edge_vectors: torch.Tensor,
        index: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = node_scalars.shape
        neighbours = index.shape[-1]

        gathered_scalars = torch.gather(
            node_scalars.unsqueeze(1).expand(-1, length, -1, -1), 2,
            index.unsqueeze(-1).expand(-1, -1, -1, node_scalars.shape[-1]))
        channels = node_vectors.shape[-2]
        flat = node_vectors.reshape(batch, length, channels * 3)
        gathered_vectors = torch.gather(
            flat.unsqueeze(1).expand(-1, length, -1, -1), 2,
            index.unsqueeze(-1).expand(-1, -1, -1, channels * 3)
        ).reshape(batch, length, neighbours, channels, 3)

        own_scalars = node_scalars.unsqueeze(-2).expand(-1, -1, neighbours, -1)
        own_vectors = node_vectors.unsqueeze(-3).expand(-1, -1, neighbours, -1, -1)

        scalars = torch.cat([own_scalars, gathered_scalars, edge_scalars], dim=-1)
        vectors = torch.cat([own_vectors, gathered_vectors, edge_vectors], dim=-2)
        for block in self.blocks:
            scalars, vectors = block(scalars, vectors)


        keep = valid.unsqueeze(-1).to(scalars.dtype)
        count = keep.sum(dim=-2).clamp_min(1.0)
        message_scalars = (scalars * keep).sum(dim=-2) / count
        message_vectors = ((vectors * keep.unsqueeze(-1)).sum(dim=-3)
                           / count.unsqueeze(-1))
        return message_scalars, message_vectors


class GVPConvLayer(nn.Module):
    """Message passing then a pointwise update, each residual and normalised."""

    def __init__(
        self,
        node_scalars: int,
        node_vectors: int,
        edge_scalars: int,
        edge_vectors: int,
        hidden_scalars: int,
        hidden_vectors: int,
        message_layers: int = 2,
    ) -> None:
        super().__init__()
        self.conv = GVPConv(node_scalars, node_vectors, edge_scalars, edge_vectors,
                            hidden_scalars, hidden_vectors, message_layers)
        self.norm_message = VectorLayerNorm(node_scalars)
        self.feedforward = nn.ModuleList([
            GVP(node_scalars, node_vectors, hidden_scalars, hidden_vectors),
            GVP(hidden_scalars, hidden_vectors, node_scalars, node_vectors),
        ])
        self.norm_feedforward = VectorLayerNorm(node_scalars)

    def forward(self, s, v, edge_scalars, edge_vectors, index, valid):
        ms, mv = self.conv(s, v, edge_scalars, edge_vectors, index, valid)
        s, v = self.norm_message(s + ms, v + mv)
        hs, hv = s, v
        for block in self.feedforward:
            hs, hv = block(hs, hv)
        return self.norm_feedforward(s + hs, v + hv)


class GVPMessageFeaturizer(nn.Module):
    """Invariant per-site features from coordinates, with learned inter-site exchange."""

    def __init__(
        self,
        d_model: int,
        atoms: int,
        neighbours: int = 12,
        layers: int = 3,
        message_layers: int = 2,
        scalar_hidden: int = 128,
        vector_hidden: int = 16,
        node_vectors: int = 16,
        radial: int = 16,
        cutoff: float = 20.0,
        global_radial: int = 0,
        global_cutoff: float = 40.0,
        edge_positional: bool = False,
        positional: int = 16,
        node_features: str = "atom",
        anchor_mode: str = "mean",
        rbf_width_mode: str = "spacing",
        positional_order: str = "sin_cos",
        chain_ends: str = "legacy",
    ) -> None:
        super().__init__()
        self.neighbours = int(neighbours)
        self.radial = int(radial)
        self.cutoff = float(cutoff)
        self.global_radial = int(global_radial)
        self.global_cutoff = float(global_cutoff)
        self.edge_positional = bool(edge_positional)
        self.positional = int(positional)
        self.atoms = int(atoms)
        self.node_features = str(node_features)
        self.anchor_mode = str(anchor_mode)
        self.rbf_width_mode = str(rbf_width_mode)
        self.positional_order = str(positional_order)
        if self.anchor_mode not in ("mean", "ca"):
            raise ValueError(f"unknown GVP anchor mode: {self.anchor_mode!r}")
        if self.rbf_width_mode not in ("spacing", "paper"):
            raise ValueError(f"unknown GVP RBF width mode: {self.rbf_width_mode!r}")
        if self.positional_order not in ("sin_cos", "cos_sin"):
            raise ValueError(f"unknown GVP positional order: {self.positional_order!r}")
        if self.node_features not in ("atom", "atom_clean", "paper", "paper_scalar_atom4"):
            raise ValueError(f"unknown GVP node feature mode: {self.node_features!r}")
        # How paper node features treat chain ends and breaks; see `paper_node_features`.
        self.chain_ends = str(chain_ends)
        if self.chain_ends not in ("legacy", "bonded"):
            raise ValueError(f"unknown GVP chain_ends mode: {self.chain_ends!r}")
        if self.chain_ends != "legacy" and self.node_features not in ("paper",
                                                                      "paper_scalar_atom4"):
            raise ValueError("chain_ends applies to the paper node features only")
        # Training-time augmentation: the share of neighbour edges dropped at random
        # (GearNet's random edge masking). Not a parameter or buffer, so checkpoints are
        # unchanged; a caller sets it, and it is ignored in eval mode.
        self.edge_drop = 0.0
        # Keep only the dimensionless r/Rg of the global radial features and zero the absolute
        # distance basis. Set by callers that measure every view against its parent chain:
        # with both, Rg = d / (d/Rg) -- the chain's size -- is readable from any one residue.
        self.radial_relative_only = False


        raw_vectors = 3 if self.node_features == "paper" else self.atoms
        if self.node_features in ("paper", "paper_scalar_atom4"):
            raw_scalars = 6                               # cos/sin(phi, psi, omega)
        elif self.node_features == "atom_clean":
            raw_scalars = self.atoms                      # own-atom distances only
        else:
            raw_scalars = (
                self.atoms                                # own-atom distances
                + self.atoms                              # which atoms are present
                + min(4, max(raw_vectors - 2, 0))         # signed triple products
            )
        raw_scalars += self.global_radial + 1 if self.global_radial else 0
        edge_scalars = self.radial + (self.positional if self.edge_positional else 0)
        self.edge_vectors = 1

        self.node_in = GVP(raw_scalars, raw_vectors, int(scalar_hidden), int(node_vectors))
        self.layers = nn.ModuleList([
            GVPConvLayer(int(scalar_hidden), int(node_vectors), edge_scalars,
                         self.edge_vectors, int(scalar_hidden), int(vector_hidden),
                         int(message_layers))
            for _ in range(int(layers))
        ])


        self.output = GVPScalarOut(int(scalar_hidden), int(node_vectors), int(scalar_hidden))
        self.project = nn.Sequential(nn.LayerNorm(int(scalar_hidden)),
                                     nn.Linear(int(scalar_hidden), d_model))

    def _separation(self, index: torch.Tensor,
                    positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is None:
            length = index.shape[1]
            slots = torch.arange(length, device=index.device)
            delta = (index - slots.view(1, -1, 1)).float()
        else:
            # Parent-chain separation for views whose slots are not consecutive residues.
            gathered = positions.gather(1, index.flatten(1)).view_as(index)
            delta = (gathered - positions.unsqueeze(-1)).float()
        frequency = torch.exp(
            torch.arange(0, self.positional, 2, device=index.device).float()
            * -(9.2103 / max(self.positional, 2))
        )
        angle = delta.unsqueeze(-1) * frequency
        sine, cosine = torch.sin(angle), torch.cos(angle)
        return (torch.cat([cosine, sine], dim=-1)
                if self.positional_order == "cos_sin"
                else torch.cat([sine, cosine], dim=-1))

    def forward(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        visible: torch.Tensor | None = None,
        radial_reference: tuple[torch.Tensor, torch.Tensor] | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`positions` [B, L]: each slot's residue number in its parent chain. None means the
        slots are the chain itself, in order (a whole chain or a contiguous window)."""
        present = atom_mask.unsqueeze(-1).to(coords.dtype)
        counts = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(coords.dtype)
        if self.anchor_mode == "ca":
            anchors = coords[..., 1, :]
            site_valid = atom_mask[..., 1]
        else:
            anchors = (coords * present).sum(dim=-2) / counts
            site_valid = atom_mask.any(dim=-1)
        visible = site_valid if visible is None else (visible & site_valid)

        own = (coords - anchors.unsqueeze(-2)) * present
        own_distance = torch.linalg.vector_norm(own, dim=-1)

        # The graph is built over visible sites only, for the reason `gvp.neighbour_indices`
        # documents: with message passing the stakes are higher, because a masked site would
        # now hand its features to every visible neighbour, not merely its distance.
        index, valid = neighbour_indices(anchors, visible, self.neighbours)
        if self.training and self.edge_drop > 0:
            valid = valid & (torch.rand(valid.shape, device=valid.device) >= self.edge_drop)
        gathered = torch.gather(
            anchors.unsqueeze(1).expand(-1, anchors.shape[1], -1, -1), 2,
            index.unsqueeze(-1).expand(-1, -1, -1, 3))
        keep = valid.unsqueeze(-1).to(coords.dtype)
        offset = (gathered - anchors.unsqueeze(-2)) * keep
        distance = torch.linalg.vector_norm(offset, dim=-1)

        if self.node_features in ("paper", "paper_scalar_atom4"):
            adjacent = (None if positions is None
                        else (positions[:, 1:] - positions[:, :-1]) == 1)
            node_scalars, paper_vectors = paper_node_features(coords, atom_mask, visible,
                                                               adjacent, self.chain_ends)
            node_vectors = (paper_vectors if self.node_features == "paper" else own)
            parts = [node_scalars]
        elif self.node_features == "atom_clean":
            node_vectors = own
            parts = [own_distance]
        else:
            node_vectors = own
            parts = [own_distance, atom_mask.to(coords.dtype), signed_triple_products(own)]
        if self.global_radial:
            radial = global_radial_features(anchors, visible, self.global_radial,
                                            self.global_cutoff, radial_reference)
            if self.radial_relative_only:
                radial = torch.cat([torch.zeros_like(radial[..., :-1]), radial[..., -1:]], -1)
            parts.append(radial)
        node_scalars = torch.cat(parts, dim=-1)

        edge_parts = [radial_basis(distance, self.radial, self.cutoff, self.rbf_width_mode)
                      * keep.expand(-1, -1, -1, self.radial)]
        if self.edge_positional:
            edge_parts.append(self._separation(index, positions)
                              * keep.expand(-1, -1, -1, self.positional))
        edge_scalars = torch.cat(edge_parts, dim=-1)
        # A unit vector, so length lives in the scalar channel and direction in the vector
        # channel rather than both being tangled in one quantity.
        unit = offset / distance.unsqueeze(-1).clamp_min(1e-6)
        edge_vectors = (unit * keep).unsqueeze(-2)

        s, v = self.node_in(node_scalars, node_vectors)
        for layer in self.layers:
            s, v = layer(s, v, edge_scalars, edge_vectors, index, valid)
        return self.project(self.output(s, v)) * site_valid.unsqueeze(-1)
