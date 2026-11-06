'Backbone feature projection and residue sequence encoding.'
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from psjepa.data.transforms import random_rotation
from psjepa.layers.residue_projection import ResidueProjection
from psjepa.models.mamba_encoder import ResidueMambaEncoder


@dataclass
class EncoderConfig:
    d_model: int = 128
    n_layer: int = 4
    predictor_layers: int = 2
    max_residues: int = 512
    levels: tuple[int, ...] = (4, 4, 4, 4, 4, 4)
    vq_codebook_size: int = 512
    vq_decay: float = 0.99
    vq_dead_code_threshold: float = 0.25
    quantize: bool = True
    mixer_backend: str = "mamba"
    independent_rotation: bool = False
    query_mode: str = "content_conditioned"
    content_query_rank: int = 16
    p2s_target: str = "continuous"


    p2s_masked: bool = True


    semantic_stride: int = 1
    shared_predictor: bool = False
    use_residue_identity: bool = False
    input_mode: str = "backbone"
    gvp_neighbours: int = 12
    gvp_global_radial: int = 0
    # `input_mode: gvp_message` only. Depth of the message-passing stack, the number of GVPs
    # inside one message, the width of the node vector channel, and whether an edge carries
    # a sinusoidal encoding of sequence separation.
    gvp_layers: int = 3
    gvp_message_layers: int = 2
    gvp_node_vectors: int = 16
    gvp_edge_positional: bool = False
    gvp_node_features: str = "atom"
    # Pointwise `input_mode: gvp` only; defaults match `GVPFeaturizer`'s own.
    gvp_featurizer_layers: int = 2
    gvp_scalar_hidden: int = 128


    backbone_presence_bits: bool = False


    global_slots: int = 0


    semantic_slots_only: bool = False
    mask_ratio_min: float = 0.4
    mask_ratio_max: float = 0.6
    spatial_probability: float = 0.5


    overlapping_targets: bool = False


    exclude_context: bool = False
    context_ratio: float = 0.5
    target_ratio: float = 0.18
    target_blocks: int = 4


    normalize_prediction: bool = True


    per_block_prediction: bool = False
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "s2p": 1.0,
            "p2s": 1.0,
            "p2p": 1.0,
            "code": 0.0,
            "variance": 0.0,
            "covariance": 0.0,
            "w2p": 0.0,
            "p2w": 0.0,
            "torsion": 0.0,
            "pair_distance": 0.0,
        }
    )
    code_warmup_steps: int = 2000
    pair_distance_warmup_steps: int = 0
    pair_distance_ramp_steps: int = 1
    variance_target: float = 1.0


class StructureEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.projection = ResidueProjection(
            config.d_model,
            config.use_residue_identity,
            config.input_mode,
            gvp_neighbours=config.gvp_neighbours,
            gvp_global_radial=config.gvp_global_radial,
            gvp_layers=config.gvp_layers,
            gvp_message_layers=config.gvp_message_layers,
            gvp_node_vectors=config.gvp_node_vectors,
            gvp_edge_positional=config.gvp_edge_positional,
            gvp_node_features=config.gvp_node_features,
            gvp_featurizer_layers=config.gvp_featurizer_layers,
            gvp_scalar_hidden=config.gvp_scalar_hidden,
            backbone_presence_bits=config.backbone_presence_bits,
        )
        self.mask_x = nn.Parameter(torch.zeros(config.d_model))
        nn.init.normal_(self.mask_x, std=0.02)
        self.encoder = ResidueMambaEncoder(
            config.d_model,
            config.n_layer,
            config.max_residues,
            config.mixer_backend,
            config.query_mode,
            config.content_query_rank,
            config.global_slots,
            config.semantic_slots_only,
        )

    def forward(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        prediction_mask: torch.Tensor | None = None,
        residue_type: torch.Tensor | None = None,
        radial_reference: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:


        visible = None if prediction_mask is None else ~prediction_mask
        residues = self.projection(coords, atom_mask, residue_type, visible, radial_reference)
        if prediction_mask is not None:
            residues = torch.where(prediction_mask.unsqueeze(-1), self.mask_x, residues)
        residues = residues * residue_mask.unsqueeze(-1)
        return self.encoder(residues, residue_mask)


def _rotate_batch(coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    rotated = []
    for item in coords:
        rotated.append(torch.einsum("rad,df->raf", item, random_rotation(item.device, item.dtype)))
    return torch.stack(rotated) * atom_mask.unsqueeze(-1)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_residue = (prediction - target).float().square().mean(dim=-1)
    return per_residue[mask].mean() if mask.any() else per_residue.sum() * 0.0


__all__ = ["EncoderConfig", "StructureEncoder", "masked_mse"]
