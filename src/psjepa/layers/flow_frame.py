"""A frame-predicting decoder head: rigid transforms onto an idealised backbone template.

`AttentionFlowDecoder` regresses twelve numbers per residue, so bond lengths and angles have
to be learned. They are not learned well when the condition is invariant: the GVP arm's
generated C(i)--N(i+1) came out at 2.03 +/- 1.60 A against a true 1.33 +/- 0.004, and the
whole failure was local geometry on top of a wrong global arrangement.

GCP-VQVAE (BioMatrix, arXiv 2606.22138) does not regress coordinates at all. Its decoder
"predicts a rigid frame for each residue, parameterized by a translation and a rotation
represented with a continuous 6D rotation head. These predicted residue frames are applied
to an idealized local backbone template containing the N, C-alpha, and C atoms." Bond
lengths and angles are then correct by construction and every parameter is spent on where
the frames go, which is exactly the part our measurements say is missing.

This module keeps the attention trunk unchanged and replaces only the output head. It
predicts the CLEAN structure, not a velocity, so it must be used with `predict_x1: true`;
the loss then derives the velocity as (x_hat_1 - x_t) / (1 - t).

O is not in the template. It belongs to the psi group -- its position rotates about the
CA->C axis with the backbone torsion -- so a fixed template entry would be wrong by up to
2.1 A. The head predicts sin(psi), cos(psi) alongside the frame and rotates the ideal
C->O offset by it, which is how AlphaFold places the same atom.
"""
from __future__ import annotations

import torch
from torch import nn

from psjepa.layers.flow_attention import Block, TimestepEmbedding, _centred

# Idealised backbone in the residue frame, in angstroms: CA at the origin, C on +x, N in the
# xy-plane. Standard bond geometry (N-CA 1.458, CA-C 1.526, angle N-CA-C 111.0 degrees).
TEMPLATE = ((-0.525, 1.363, 0.000), (0.000, 0.000, 0.000), (1.526, 0.000, 0.000))
# C->O at psi = 0, from C-O 1.229 A and angle CA-C-O 120.8 degrees. Rotated about +x by psi.
CARBONYL = (0.630, 1.055, 0.000)


def six_d_to_rotation(vector: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt on two 3-vectors (Zhou et al. 2019): a continuous SO(3) parameterisation.

    Quaternions and Euler angles are discontinuous as regression targets -- antipodal
    quaternions name the same rotation and the map has a seam. Six numbers orthonormalised
    have neither problem, which is why the frame heads in this literature all use them.
    """
    # The epsilon belongs inside the root, for the same reason as the psi normalisation below:
    # `vector_norm` has an undefined gradient at exactly zero and a later `clamp_min` cannot
    # undo that. These six numbers are predicted, so the model can put them at the origin.
    def _unit(v):
        return v / torch.sqrt(v.square().sum(dim=-1, keepdim=True) + 1e-12)

    first, second = vector[..., :3], vector[..., 3:6]
    e1 = _unit(first)
    second = second - (e1 * second).sum(dim=-1, keepdim=True) * e1
    e2 = _unit(second)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)


class AttentionFrameDecoder(nn.Module):
    """Predict the clean backbone as one rigid frame plus one psi angle per residue."""

    def __init__(
        self,
        d_model: int,
        atoms: int = 4,
        layers: int = 6,
        width: int = 384,
        heads: int = 6,
        max_residues: int = 512,
        dropout: float = 0.0,
        code_book: int = 0,
        coordinate_scale: float = 10.0,
        condition_mask_input: bool = False,
    ) -> None:
        super().__init__()
        if int(atoms) != 4:
            raise ValueError("the frame head places N, CA, C and O, so atoms must be 4")
        self.atoms = 4
        self.width = int(width)
        self.state_in = nn.Linear(self.atoms * 3, width)
        self.code_book = int(code_book)
        if self.code_book:
            self.condition_in = nn.Embedding(self.code_book + 1, width)
        else:
            self.condition_in = nn.Linear(d_model, width)
        self.target_in = nn.Embedding(2, width)
        # "A latent was provided here", separate from what the latent says. See
        # `AttentionFlowDecoder` for why the zero vector cannot carry both. New state, so
        # off by default: a decoder with it cannot load a checkpoint trained without it.
        self.condition_mask_in = (
            nn.Embedding(2, width) if bool(condition_mask_input) else None)
        self.time_embed = TimestepEmbedding(width)
        self.position = nn.Parameter(torch.zeros(1, max_residues, width))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(int(layers)))
        self.norm_out = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        # 3 translation + 6 rotation + 2 (sin psi, cos psi)
        self.frame_out = nn.Linear(width, 11)
        nn.init.zeros_(self.modulation_out[1].weight)
        nn.init.zeros_(self.modulation_out[1].bias)
        # Zero weights so nothing is predicted from the input at step 0, but a bias that
        # makes that zero output the IDENTITY frame. Leaving the bias at zero would hand
        # `six_d_to_rotation` two zero vectors, whose Gram-Schmidt is undefined.
        nn.init.zeros_(self.frame_out.weight)
        with torch.no_grad():
            self.frame_out.bias.copy_(torch.tensor(
                [0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0]))
        # The loss works in units of coordinates / coordinate_scale, so the template has to
        # be divided by the same number or the predicted structure is ten times too large.
        scale = float(coordinate_scale)
        self.register_buffer("template", torch.tensor(TEMPLATE) / scale, persistent=False)
        self.register_buffer("carbonyl", torch.tensor(CARBONYL) / scale, persistent=False)

    def forward(
        self,
        condition: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        time: torch.Tensor,
        prediction_mask: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = condition.shape[:2]
        if length > self.position.shape[1]:
            raise ValueError(f"{length} residues exceeds max_residues {self.position.shape[1]}")
        present = atom_mask.unsqueeze(-1).to(coordinates.dtype)
        state = (coordinates * present).reshape(batch, length, self.atoms * 3)

        if self.code_book:
            condition = condition.long().clamp_max(self.code_book)
        x = self.state_in(state) + self.condition_in(condition)
        x = x + self.target_in(prediction_mask.long()) + self.position[:, :length]
        if self.condition_mask_in is not None:
            if condition_mask is None:
                raise ValueError(
                    "this decoder was built with condition_mask_input: true and needs a "
                    "condition_mask saying where a latent was provided")
            x = x + self.condition_mask_in(condition_mask.long())
        t = self.time_embed(time).to(x.dtype)
        pad = ~atom_mask.any(dim=-1)
        pad = pad & ~pad.all(dim=-1, keepdim=True)
        if not torch.isfinite(x).all():
            # `state` and `condition` are finite here and the weights are small, so the overflow
            # is one of magnitude, not of a NaN arriving from upstream: bf16 tops out near 3.4e38
            # and a large coordinate times a linear layer reaches it. Report the sizes.
            raise RuntimeError(
                "the block stack's input is already non-finite. "
                f"state: max|.|={float(state.abs().max()):.4g} dtype={state.dtype}; "
                f"condition: max|.|={float(condition.abs().max()):.4g} dtype={condition.dtype}; "
                f"state_in(state): max|.|="
                f"{float(self.state_in(state.float()).abs().max()):.4g} in fp32; "
                f"condition_in(condition): max|.|="
                f"{float(self.condition_in(condition.float()).abs().max()):.4g} in fp32")
        if not torch.isfinite(t).all():
            raise RuntimeError("the time embedding is non-finite")
        if bool(pad.all(dim=-1).any()):
            # A row masked everywhere makes attention softmax over an all -inf row, which is
            # NaN for the whole row -- and block 0 came out with every one of its 14,090,240
            # entries non-finite, which is what a fully masked batch looks like rather than an
            # overflow that grows with depth.
            raise RuntimeError("a batch row is padding-masked at every position")
        for depth, block in enumerate(self.blocks):
            x = block(x, t, pad)
            if not torch.isfinite(x).all():
                # Name the block. Everything upstream has already been cleared: the condition,
                # the coordinates and the time are finite on entry, and the frame head is not
                # reached. A single block overflowing in bf16 is the remaining explanation, and
                # its depth says whether this is an accumulating drift or one bad layer.
                finite = x.isfinite()
                raise RuntimeError(
                    f"decoder block {depth} of {len(self.blocks)} produced a non-finite state: "
                    f"{int((~finite).sum())} of {finite.numel()} entries, "
                    f"largest finite magnitude {float(x[finite].abs().max()) if finite.any() else float('nan'):.3g}")
        shift, scale = self.modulation_out(t).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale) + shift
        if not torch.isfinite(x).all():
            # Locate the blow-up instead of letting a NaN reach the loss and be reported as
            # "the flow term is not finite". scaffold_v4 reached this point five times; the
            # per-term check narrowed it to the decoder, and this narrows it to the block stack
            # versus the frame head. bf16 overflows in an attention block look exactly like a
            # broken normalisation from the outside.
            raise RuntimeError(
                "decoder block stack produced a non-finite state before the frame head: "
                f"condition finite={bool(torch.isfinite(condition).all())}, "
                f"coordinates finite={bool(torch.isfinite(coordinates).all())}, "
                f"time finite={bool(torch.isfinite(time).all())}")
        frame = self.frame_out(x).float()

        translation = frame[..., :3]
        rotation = six_d_to_rotation(frame[..., 3:9])
        sin_psi, cos_psi = frame[..., 9], frame[..., 10]
        # The epsilon goes *inside* the root. `clamp_min` afterwards guards the division but
        # not the backward pass: d(sqrt)/dx at zero is infinite, and the predicted (sin, cos)
        # pair is free to reach the origin. This is where scaffold_v4's NaN came from -- four
        # restarts blamed the loss terms, but the per-term check said the decoder's own output
        # was non-finite while target and path were fine.
        norm = torch.sqrt(sin_psi.square() + cos_psi.square() + 1e-12)
        sin_psi, cos_psi = sin_psi / norm, cos_psi / norm

        # N, CA and C are the template rotated into place; the frame rows are the axes, so
        # a local point p maps to p @ rotation (rows) plus the translation.
        # Rows of `rotation` are the frame axes (`six_d_to_rotation` stacks them that
        # way), so a local point p maps to p-transpose times R, matching
        # `ipa.gram_schmidt_frames` and therefore FAPE.
        placed = torch.einsum("blid,ai->blad", rotation, self.template.to(rotation.dtype))
        # O rotates about the CA->C axis, which is +x in the residue frame, by psi.
        y, z = self.carbonyl[1].to(rotation.dtype), self.carbonyl[2].to(rotation.dtype)
        oxygen_local = torch.stack([
            self.carbonyl[0].to(rotation.dtype).expand_as(sin_psi),
            cos_psi * y - sin_psi * z,
            sin_psi * y + cos_psi * z,
        ], dim=-1)
        # Relative to C, which is template[2], then rotated by the same frame.
        oxygen = torch.einsum("blid,bli->bld",
                              rotation, oxygen_local + self.template[2].to(rotation.dtype))
        coordinates = torch.cat([placed, oxygen.unsqueeze(-2)], dim=-2) + translation.unsqueeze(-2)
        return _centred(coordinates, atom_mask)
