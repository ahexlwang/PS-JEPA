from __future__ import annotations

import torch
from torch import nn

from psjepa.layers.gvp import GVPFeaturizer
from psjepa.layers.gvp_message import GVPMessageFeaturizer


class ResidueProjection(nn.Module):
    """Project backbone coordinates or pool an atom37 set per residue."""

    def __init__(
        self,
        d_model: int,
        use_residue_identity: bool = False,
        input_mode: str = "backbone",
        gvp_neighbours: int = 12,
        gvp_global_radial: int = 0,
        gvp_layers: int = 3,
        gvp_message_layers: int = 2,
        gvp_node_vectors: int = 16,
        gvp_edge_positional: bool = False,
        gvp_node_features: str = "atom",
        gvp_anchor_mode: str = "mean",
        gvp_rbf_width_mode: str = "spacing",
        gvp_positional_order: str = "sin_cos",
        gvp_chain_ends: str = "legacy",
        gvp_message_scalar_hidden: int = 128,
        gvp_message_vector_hidden: int = 16,
        # Separate from `gvp_layers`/`gvp_node_vectors` above, which belong to
        # `gvp_message`. These two exist so a capacity-matched POINTWISE control can be
        # built: the message-passing featuriser is 6x the parameters of the default
        # pointwise one, so a bare comparison would confound capacity with the mechanism.
        # Defaults reproduce `GVPFeaturizer`'s own defaults, leaving every existing arm
        # byte-identical.
        gvp_featurizer_layers: int = 2,
        gvp_scalar_hidden: int = 128,


        backbone_presence_bits: bool = False,
    ) -> None:
        super().__init__()
        self.use_residue_identity = use_residue_identity
        self.input_mode = input_mode
        self.backbone_presence_bits = bool(backbone_presence_bits)
        if input_mode == "gvp":
            # Invariant by construction rather than by augmentation. Because the
            # neighbour graph must exclude masked sites, this mode needs the
            # prediction mask, which the other modes ignore.
            self.net = GVPFeaturizer(
                d_model,
                atoms=4,
                neighbours=gvp_neighbours,
                layers=gvp_featurizer_layers,
                scalar_hidden=gvp_scalar_hidden,
                global_radial=gvp_global_radial,
            )
            self.atom_embedding = self.atom_coord = self.ca_projection = None
        elif input_mode == "gvp_message":
            # Same invariance guarantee as `gvp`, but the vector channels survive every
            # layer and neighbours exchange features instead of only distances.
            self.net = GVPMessageFeaturizer(
                d_model,
                atoms=4,
                neighbours=gvp_neighbours,
                layers=gvp_layers,
                message_layers=gvp_message_layers,
                node_vectors=gvp_node_vectors,
                global_radial=gvp_global_radial,
                edge_positional=gvp_edge_positional,
                node_features=gvp_node_features,
                anchor_mode=gvp_anchor_mode,
                rbf_width_mode=gvp_rbf_width_mode,
                positional_order=gvp_positional_order,
                chain_ends=gvp_chain_ends,
                scalar_hidden=gvp_message_scalar_hidden,
                vector_hidden=gvp_message_vector_hidden,
            )
            self.atom_embedding = self.atom_coord = self.ca_projection = None
        elif input_mode == "backbone":
            self.net = nn.Sequential(
                nn.Linear(16 if self.backbone_presence_bits else 12, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            )
            self.atom_embedding = self.atom_coord = self.ca_projection = None
        elif input_mode == "all_atom":
            self.net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.atom_embedding = nn.Embedding(37, d_model)
            self.atom_coord = nn.Linear(3, d_model, bias=False)
            self.ca_projection = nn.Linear(3, d_model)
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        self.residue_embedding = nn.Embedding(21, d_model) if use_residue_identity else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):


        key = prefix + "net.0.weight"
        if self.input_mode == "backbone" and key in state_dict:
            width = int(state_dict[key].shape[1])
            if width != self.net[0].in_features:
                old = self.net[0]
                self.net[0] = nn.Linear(width, old.out_features).to(old.weight.device, old.weight.dtype)
                self.backbone_presence_bits = width == 16
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_type: torch.Tensor | None = None,
        visible: torch.Tensor | None = None,
        radial_reference: tuple[torch.Tensor, torch.Tensor] | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked = coords * atom_mask.unsqueeze(-1)
        if self.input_mode == "gvp_message":
            # Parent-chain residue numbers; only the message featuriser reads sequence order.
            output = self.net(coords, atom_mask, visible, radial_reference, positions)
        elif self.input_mode == "gvp":
            output = self.net(coords, atom_mask, visible, radial_reference)
        elif self.input_mode == "backbone":
            if coords.shape[-2] != 4:
                raise ValueError(f"backbone mode requires 4 atoms, got {coords.shape[-2]}")
            features = masked.flatten(-2)
            if self.backbone_presence_bits:
                features = torch.cat([features, atom_mask.to(coords.dtype)], dim=-1)
            output = self.net(features)
        else:
            if coords.shape[-2] != 37:
                raise ValueError(f"all_atom mode requires atom37 input, got {coords.shape[-2]}")
            ca = masked[..., 1, :]
            relative = (coords - ca.unsqueeze(-2)) * atom_mask.unsqueeze(-1)
            atom_ids = torch.arange(37, device=coords.device)
            features = self.atom_coord(relative) + self.atom_embedding(atom_ids)
            features = features * atom_mask.unsqueeze(-1)
            pooled = features.sum(dim=-2) / atom_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            output = self.net(pooled + self.ca_projection(ca))
        if self.residue_embedding is not None:
            if residue_type is None:
                raise ValueError("residue_type is required when use_residue_identity=true")
            output = output + self.residue_embedding(residue_type)
        return output
