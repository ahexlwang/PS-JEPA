"""An SE(3)-equivariant flow decoder built from the IPA structure module.

`AttentionFlowDecoder` and `AttentionFrameDecoder` read the noisy state as twelve plain
numbers per residue, so a rotation of `x_t` is just a different input to them: with an
invariant condition they have to learn SO(3) from augmentation, and the first sweep gave
them 200k samples to do it. This decoder does not have to learn it. The noisy structure is
read only through the residue frames it defines, `InvariantPointAttention` sees geometry
only as distances between frame-placed points, and the predicted structure is a local
update composed onto those frames. Rotate `x_t` and the output rotates with it, exactly,
whatever the weights are. That is the FrameFlow / FoldFlow arrangement: an invariant
condition, frames as the state, IPA as the network.

Conditioning is invariant too: the semantic token, the time, and residue position enter an
ordinary transformer trunk whose output is the `single` representation the structure module
refines. The trunk never sees coordinates, so nothing in it can break the symmetry.

The head predicts the CLEAN structure (use with ``predict_x1: true``): the structure module
places an idealised template on frames, and a template is a structure, not a velocity.
"""
from __future__ import annotations

import torch
from torch import nn

from psjepa.layers.flow_attention import TimestepEmbedding, _centred
from psjepa.layers.ipa import StructureModule, gram_schmidt_frames


class IPAFlowDecoder(nn.Module):
    """Clean backbone from an invariant condition and the frames of a noisy backbone."""

    def __init__(
        self,
        d_model: int,
        atoms: int = 4,
        trunk_layers: int = 4,
        width: int = 256,
        heads: int = 8,
        iterations: int = 4,
        max_residues: int = 512,
        dropout: float = 0.0,
        code_book: int = 0,
        coordinate_scale: float = 10.0,
        stop_rotation_gradient: bool = True,
    ) -> None:
        super().__init__()
        if int(atoms) != 4:
            raise ValueError("the structure module places N, CA, C and O, so atoms must be 4")
        self.atoms = 4
        self.width = int(width)
        self.code_book = int(code_book)
        if self.code_book:
            self.condition_in = nn.Embedding(self.code_book + 1, width)
        else:
            self.condition_in = nn.Linear(d_model, width)
        self.target_in = nn.Embedding(2, width)
        self.time_embed = TimestepEmbedding(width)
        self.position = nn.Parameter(torch.zeros(1, max_residues, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True,
                                           norm_first=True, activation="gelu",
                                           dropout=float(dropout))
        self.trunk = nn.TransformerEncoder(layer, int(trunk_layers), enable_nested_tensor=False)


        self.structure = StructureModule(
            width=width, heads=heads, iterations=int(iterations), dropout=float(dropout),
            coordinate_scale=float(coordinate_scale), initial_frames="learned_origin",
            stop_rotation_gradient=bool(stop_rotation_gradient))

    def forward(
        self,
        condition: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        time: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = condition.shape[:2]
        if length > self.position.shape[1]:
            raise ValueError(f"{length} residues exceeds max_residues {self.position.shape[1]}")
        if self.code_book:
            condition = condition.long().clamp_max(self.code_book)
        x = self.condition_in(condition) + self.target_in(prediction_mask.long())
        x = x + self.position[:, :length]
        x = x + self.time_embed(time).to(x.dtype).unsqueeze(1)
        pad = ~atom_mask.any(dim=-1)
        safe = pad & ~pad.all(dim=-1, keepdim=True)
        single = self.trunk(x, src_key_padding_mask=safe)

        # Frames read off the noisy backbone. At t=0 these are the frames of Gaussian
        # noise, which is fine: they are a random but equivariant gauge for the update.
        rotation, translation = gram_schmidt_frames(coordinates[..., :3, :].float())
        # Match the structure module's parameter dtype (float64 in the equivariance tests;
        # under autocast the Linear inside casts for itself).
        single = single.to(self.structure.update.weight.dtype)
        structure, _trajectory, _rotation = self.structure(
            single, safe, None, rotation, translation)
        return _centred(structure, atom_mask)
