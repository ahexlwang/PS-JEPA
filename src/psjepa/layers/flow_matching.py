"""SE(3)-equivariant Cartesian conditional flow matching for backbones.

The semantic stream is rotation invariant, so it cannot directly regress an
absolute Cartesian frame.  Instead, isotropic noisy coordinates provide the
frame and the invariant semantic token conditions an equivariant vector field.
The target and noise are centred per protein, making the flow translation free.
"""
from __future__ import annotations

import math

import math
import torch
from torch import nn

from psjepa.layers.gvp import GVP, neighbour_indices, radial_basis
from psjepa.layers.ipa import adjacent_frame_loss, canonicalize_backbone, fape_loss


def _centred_on(
    values: torch.Tensor, centre_mask: torch.Tensor, keep_mask: torch.Tensor
) -> torch.Tensor:
    """Subtract the centroid of `centre_mask`, keep everything in `keep_mask`.

    `_centred` below zeroes every atom outside its mask, which is right when the
    mask is the set of real atoms and wrong when it is a motif: centring a
    structure on its motif must not delete the scaffold.
    """
    weight = centre_mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=(-3, -2), keepdim=True).clamp_min(1.0)
    centre = (values * weight).sum(dim=(-3, -2), keepdim=True) / count
    return (values - centre) * keep_mask.unsqueeze(-1).to(values.dtype)


def _centred(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=(-3, -2), keepdim=True).clamp_min(1.0)
    centre = (values * weight).sum(dim=(-3, -2), keepdim=True) / count
    return (values - centre) * weight


class GVPFlowDecoder(nn.Module):
    """Predict a translation-free equivariant velocity for four backbone atoms."""

    def __init__(
        self,
        d_model: int,
        atoms: int = 4,
        neighbours: int = 12,
        layers: int = 2,
        scalar_hidden: int | None = None,
        vector_hidden: int = 16,
        radial: int = 8,
        cutoff: float = 20.0,
    ) -> None:
        super().__init__()
        self.atoms = int(atoms)
        self.neighbours = int(neighbours)
        self.radial = int(radial)
        self.cutoff = float(cutoff)
        scalar_hidden = int(scalar_hidden or d_model)
        scalar_in = (
            d_model
            + 4  # t, t^2, sin(pi t), cos(pi t)
            + 1  # which residues are reconstruction targets
            + atoms  # current within-residue distances
            + atoms  # atom availability
            + neighbours * radial
            + neighbours
        )
        vector_in = atoms + neighbours
        self.vector_in = vector_in
        # An explicit invariant-scalar -> equivariant-vector path prevents the
        # decoder from treating the semantic token as an optional weak gate. The
        # token predicts scalar coefficients over the current relative vectors;
        # a scalar-weighted sum of equivariant vectors is still equivariant.
        self.conditional_vector_weights = nn.Sequential(
            nn.LayerNorm(d_model + 5),
            nn.Linear(d_model + 5, atoms * vector_in),
        )
        self.blocks = nn.ModuleList()
        scalars, vectors = scalar_in, vector_in
        for _ in range(int(layers)):
            self.blocks.append(GVP(scalars, vectors, scalar_hidden, vector_hidden))
            scalars, vectors = scalar_hidden, vector_hidden
        self.vector_out = nn.Linear(vectors, atoms, bias=False)

    def forward(
        self,
        condition: torch.Tensor,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        time: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        present = atom_mask.unsqueeze(-1).to(coordinates.dtype)
        count = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(coordinates.dtype)
        anchors = (coordinates * present).sum(dim=-2) / count
        site_valid = atom_mask.any(dim=-1)

        own = (coordinates - anchors.unsqueeze(-2)) * present
        own_distance = torch.linalg.vector_norm(own, dim=-1)
        index, neighbour_valid = neighbour_indices(anchors, site_valid, self.neighbours)
        gathered = torch.gather(
            anchors.unsqueeze(1).expand(-1, anchors.shape[1], -1, -1),
            2,
            index.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        keep = neighbour_valid.unsqueeze(-1).to(coordinates.dtype)
        neighbour = (gathered - anchors.unsqueeze(-2)) * keep
        neighbour_distance = torch.linalg.vector_norm(neighbour, dim=-1)

        batch, length = condition.shape[:2]
        t = time.reshape(batch, 1, 1).expand(-1, length, -1).to(condition.dtype)
        time_features = torch.cat([t, t.square(), torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)
        scalars = torch.cat(
            [
                condition,
                time_features,
                prediction_mask.unsqueeze(-1).to(condition.dtype),
                own_distance,
                atom_mask.to(condition.dtype),
                radial_basis(neighbour_distance, self.radial, self.cutoff).flatten(-2)
                * keep.expand(-1, -1, -1, self.radial).flatten(-2),
                neighbour_valid.to(condition.dtype),
            ],
            dim=-1,
        )
        vectors = torch.cat([own, neighbour], dim=-2)
        input_vectors = vectors
        for block in self.blocks:
            scalars, vectors = block(scalars, vectors)
        velocity = self.vector_out(vectors.transpose(-1, -2)).transpose(-1, -2)
        conditional_scalars = torch.cat(
            [condition, time_features, prediction_mask.unsqueeze(-1).to(condition.dtype)], dim=-1
        )
        weights = self.conditional_vector_weights(conditional_scalars).reshape(
            batch, length, self.atoms, self.vector_in
        )
        velocity = velocity + torch.einsum("braq,brqd->brad", weights, input_vectors) / math.sqrt(
            self.vector_in
        )
        # The source and target are both centred.  Projecting the prediction onto
        # the same translation-free subspace prevents an unidentifiable drift.
        return _centred(velocity, atom_mask)


def _align_to(mobile: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rotate each `mobile` structure onto its `reference` by Kabsch, per batch item.

    The loss below compares a predicted velocity with `target - source` in whatever frame
    the file happened to be written in. For an equivariant condition that is fine -- the
    condition names the frame. For an INVARIANT one it is unanswerable: every rotation of
    the target is equally consistent with the condition, so the model is scored against one
    arbitrary choice out of SO(3), and at t=0, where `x_t` is pure noise and carries no
    frame either, the best possible prediction is the mean over all rotations, which is
    nearly zero.

    Aligning the target onto the sampled noise removes the arbitrariness: the frame is then
    fixed by `source`, which the decoder sees inside `x_t` at every t. This is the SO(3)
    alignment step FoldFlow and MultiFlow use for the same reason.
    """
    weight = mask.unsqueeze(-1).to(torch.float32)
    out = []
    # Kabsch has to stay in float32: the loss runs under autocast, which would send the
    # matmul and the SVD to bfloat16, and cuSOLVER has no bfloat16 batched SVD at all.
    with torch.autocast(device_type=mobile.device.type, enabled=False):
        for item, target, keep in zip(mobile.float(), reference.float(), weight):
            p = (item * keep).reshape(-1, 3)
            q = (target * keep).reshape(-1, 3)
            u, _, vh = torch.linalg.svd(p.transpose(0, 1) @ q)
            rotation = u @ vh
            if torch.linalg.det(rotation) < 0:
                u = u.clone()
                u[:, -1] *= -1
                rotation = u @ vh
            out.append(item @ rotation)
    return torch.stack(out).to(mobile.dtype) * mask.unsqueeze(-1)

def _pair_distance_loss(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-item MSE between the CA-CA distance matrices of `predicted` and `target`.

    Distances are rotation and translation invariant, so this term asks for the fold and
    never for the frame. That is the whole point for an invariant condition: the Cartesian
    MSE scores one arbitrary member of SO(3), this scores the shape itself.
    """
    a = predicted[:, :, 1].float()
    b = target[:, :, 1].float()
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    pair = pair & ~torch.eye(valid.shape[1], dtype=torch.bool, device=valid.device)
    error = (torch.cdist(a, a) - torch.cdist(b, b)).square()
    count = pair.sum(dim=(-2, -1)).clamp_min(1).float()
    return (error * pair).sum(dim=(-2, -1)) / count


def conditional_flow_matching_loss(
    decoder: GVPFlowDecoder,
    condition: torch.Tensor,
    coordinates: torch.Tensor,
    atom_mask: torch.Tensor,
    prediction_mask: torch.Tensor,
    coordinate_scale: float = 10.0,
    align_target: bool = False,
    predict_x1: bool = False,
    pair_weight: float = 0.0,
    pair_min_time: float = 0.5,
    canonical_target: bool = False,
    fape_weight: float = 0.0,
    adjacent_weight: float = 0.0,
    given_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Downstream rectified-flow objective with a frozen representation.

    ``condition`` is detached here as a second line of defence.  Flow
    reconstruction is a probe of a pretrained representation, never an
    auxiliary PS-JEPA objective and never a route for gradients into its
    encoder or quantizer.

    ``given_mask`` marks residues whose true coordinates are *handed to* the
    decoder rather than generated: motif scaffolding. Those residues keep their
    clean coordinates at every time, so the network builds a structure around a
    fixed piece instead of denoising it too, and they are excluded from the loss.
    Supplying a motif also removes the gauge ambiguity that ``align_target``
    exists to solve -- the given coordinates *are* the frame -- so the structure
    is centred on the motif and no alignment is applied.

    ``canonical_target`` expresses the target in its first residue's N/CA/C frame before
    centring, so an invariant condition faces one fixed gauge instead of a uniformly
    random rotation of the same fold. ``align_target`` is the other way to remove that
    freedom: it lets the noise choose the gauge, per sample. They are exclusive.
    """
    if align_target and canonical_target:
        raise ValueError("align_target and canonical_target fix the gauge in different ways")
    condition = condition.detach()
    coordinates = coordinates.float()
    if canonical_target:
        coordinates, _ = canonicalize_backbone(
            coordinates, prediction_mask & atom_mask.any(dim=-1), atom_mask)
    if given_mask is None:
        target = _centred(coordinates, atom_mask) / float(coordinate_scale)
        source = _centred(torch.randn_like(target), atom_mask)
        if align_target:
            target = _align_to(target, source, atom_mask)
    else:
        if align_target or canonical_target:
            raise ValueError("a given motif already fixes the gauge")
        # Centre on the motif, not on every atom: at sampling time the scaffold
        # does not exist yet, so an all-atom centre is not computable and the
        # two would disagree. The motif centre is available in both settings.
        motif_atoms = atom_mask & given_mask.unsqueeze(-1)
        if not bool(motif_atoms.any()):
            raise ValueError("given_mask selected no resolved atoms")
        target = _centred_on(coordinates, motif_atoms, atom_mask) / float(coordinate_scale)
        source = _centred(torch.randn_like(target), atom_mask)
    time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
    t = time.reshape(-1, 1, 1, 1)
    path = ((1.0 - t) * source + t * target) * atom_mask.unsqueeze(-1)
    if given_mask is not None:
        # The motif is not on a trajectory: it is present, exactly, at every time.
        path = torch.where(given_mask.reshape(*given_mask.shape, 1, 1), target, path)
        path = path * atom_mask.unsqueeze(-1)
    target_velocity = target - source


    dtype = next(decoder.parameters()).dtype
    output = decoder(condition, path.to(dtype), atom_mask, time, prediction_mask).float()
    selected = atom_mask & prediction_mask.unsqueeze(-1)

    # `predict_x1` reads the same decoder output as the clean structure instead of the
    # velocity. The two are one substitution apart -- x_t + (1-t) * v == x_1 -- but
    # regressing x_1 keeps the target bounded as t -> 1, where the velocity target is
    # divided by a vanishing (1-t), and it gives the auxiliary term below a structure to
    # act on at every t rather than only at the end of the trajectory.
    if predict_x1:
        estimated_target = output
        residual = output - target
    else:
        estimated_target = path.float() + (1.0 - t) * output
        residual = output - target_velocity
    per_atom = residual.square().mean(dim=-1)
    loss = per_atom[selected].mean() if selected.any() else per_atom.sum() * 0.0
    flow_loss = loss

    pair_loss = loss.new_zeros(())
    if pair_weight > 0.0:
        valid = atom_mask.any(dim=-1) & prediction_mask
        per_item = _pair_distance_loss(estimated_target, target, valid)
        # Only score the shape once the trajectory is near the data. Early in the flow the
        # structure is still mostly noise and its distance matrix carries no fold to match.
        late = time >= float(pair_min_time)
        if late.any():
            pair_loss = per_item[late].mean()
            loss = loss + float(pair_weight) * pair_loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"the flow term itself is not finite: prediction finite="
            f"{bool(torch.isfinite(output).all())}, target finite={bool(torch.isfinite(target).all())}, "
            f"path finite={bool(torch.isfinite(path).all())}")
    fape_value = loss.new_zeros(())
    if fape_weight > 0.0:
        # Coordinate MSE barely sees a residue's orientation: N, C and O sit within 1.5 A of
        # CA, so a frame can be rotated far from the truth at small cost. The rollout scores
        # of the template heads showed exactly that -- CA distance maps close to the
        # attention decoder's, phi/psi errors of 75-89 degrees and C(i)-N(i+1) at 2-4 A.
        # FAPE measures every atom from every residue frame, so a misoriented frame is
        # charged for the whole structure it sees. Both structures are in one gauge already
        # (aligned or canonical), and FAPE is invariant anyway. Units are scaled
        # coordinates, so no clamp and scale 1.
        frame_mask = prediction_mask & atom_mask.any(dim=-1)
        fape_value = fape_loss(estimated_target.float(), target, frame_mask, atom_mask,
                               clamp=None, scale=1.0)
        loss = loss + float(fape_weight) * fape_value

    adjacent_value = loss.new_zeros(())
    if adjacent_weight > 0.0:


        adjacent_value = adjacent_frame_loss(
            estimated_target.float(), target, atom_mask.any(dim=-1), atom_mask, scale=1.0)
        if not torch.isfinite(adjacent_value):
            # Name the term instead of letting the trainer report a non-finite total. This one
            # took three restarts to place: the total was NaN at step 17,008, 18,699 and 18,653,
            # and only a per-term check says which of flow, FAPE, pair and adjacent produced it.
            raise RuntimeError(
                f"adjacent_frame_loss is not finite (flow={float(loss):.4g}); "
                f"estimated_target finite={bool(torch.isfinite(estimated_target).all())}, "
                f"target finite={bool(torch.isfinite(target).all())}")
        loss = loss + float(adjacent_weight) * adjacent_value

    velocity = output if not predict_x1 else (output - path.float()) / (1.0 - t).clamp_min(1e-3)
    return loss, {
        "loss_flow_velocity": flow_loss.detach(),
        "loss_flow_pair": pair_loss.detach(),
        "loss_flow_fape": fape_value.detach(),
        "loss_flow_adjacent": adjacent_value.detach(),
        "flow_velocity_target_rms": target_velocity[selected].square().mean().sqrt()
        if selected.any()
        else target_velocity.sum() * 0.0,
        "flow_velocity_prediction_rms": velocity[selected].square().mean().sqrt()
        if selected.any()
        else velocity.sum() * 0.0,
    }


@torch.inference_mode()
def sample_flow(
    decoder: GVPFlowDecoder,
    condition: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    coordinate_scale: float = 10.0,
    steps: int = 16,
    predict_x1: bool = False,
    given_mask: torch.Tensor | None = None,
    given_coordinates: torch.Tensor | None = None,
    churn: float = 0.0,
    motif_ramp: float = 0.0,
    jumps: int = 0,
) -> torch.Tensor:
    """Generate a centred backbone with midpoint Euler integration.

    With ``given_mask`` and ``given_coordinates`` the residues it marks are held
    at the supplied geometry for the whole trajectory, so the sampler builds a
    scaffold around a motif. The supplied coordinates are taken in the same
    units as the returned structure (Angstrom) and are centred on the motif
    here, matching how the training loss centred them.
    """
    if steps < 1:
        raise ValueError("Flow sampling requires at least one integration step")
    if (given_mask is None) != (given_coordinates is None):
        raise ValueError("given_mask and given_coordinates go together")
    condition = condition.detach()
    shape = (*atom_mask.shape, 3)
    dtype = next(decoder.parameters()).dtype
    coordinates = _centred(
        torch.randn(shape, device=condition.device, dtype=torch.float32), atom_mask
    ).to(dtype)
    given_scaled = None
    if given_mask is not None:
        motif_atoms = atom_mask & given_mask.unsqueeze(-1)
        given_scaled = _centred_on(
            given_coordinates.float(), motif_atoms, atom_mask
        ) / float(coordinate_scale)
        select = given_mask.reshape(*given_mask.shape, 1, 1)
        coordinates = torch.where(select, given_scaled.to(dtype), coordinates)
    delta = 1.0 / steps

    def integrate(state, index, time_value):
        """One integration step from `state` at `time_value`, motif re-imposed."""
        time = torch.full((condition.shape[0],), time_value,
                          device=condition.device, dtype=torch.float32)
        out = decoder(condition, state, atom_mask, time, residue_mask)
        if predict_x1:
            left = (1.0 - time).reshape(-1, 1, 1, 1).clamp_min(delta).to(out.dtype)
            clean = out
            out = (out - state) / left
        else:
            clean = state + (1.0 - time).reshape(-1, 1, 1, 1).to(out.dtype) * out
        moved = state + delta * out
        if given_scaled is not None:
            if motif_ramp > 0.0:


                weight = min(1.0, ((index + 1) * delta) / float(motif_ramp))
                blend = (1.0 - weight) * moved.float() + weight * given_scaled
                moved = torch.where(select, blend.to(dtype), moved)
            else:
                moved = torch.where(select, given_scaled.to(dtype), moved)
            moved = (moved.float() * atom_mask.unsqueeze(-1)).to(dtype)
        else:
            moved = _centred(moved.float(), atom_mask).to(dtype)
        return moved, clean

    delta = 1.0 / steps
    for index in range(steps):


        offset = 0.0 if predict_x1 else 0.5
        time = torch.full(
            (condition.shape[0],),
            (index + offset) * delta,
            device=condition.device,
            dtype=torch.float32,
        )
        output = decoder(condition, coordinates, atom_mask, time, residue_mask)
        if predict_x1:
            remaining = (1.0 - time).reshape(-1, 1, 1, 1).clamp_min(delta).to(output.dtype)
            predicted_clean = output
            output = (output - coordinates) / remaining
        else:
            predicted_clean = coordinates + (1.0 - time).reshape(-1, 1, 1, 1).to(output.dtype) * output
        coordinates = coordinates + delta * output
        if jumps > 0 and given_scaled is not None and index < steps - 1:


            back = max(0.0, (index + 1) * delta - delta)
            for _ in range(jumps):
                clean = torch.where(select, given_scaled.to(dtype), predicted_clean)
                noise = torch.randn_like(clean)
                state = ((1.0 - back) * noise.float() + back * clean.float()).to(dtype)
                state = torch.where(select, given_scaled.to(dtype), state)
                coordinates, predicted_clean = integrate(state, index, (index + offset) * delta)
        if churn > 0.0 and index < steps - 1:


            remaining = 1.0 - (index + 1) * delta
            sigma = churn * remaining * math.sqrt(delta)
            noise = torch.randn_like(coordinates) * sigma
            coordinates = coordinates + noise.to(dtype) * atom_mask.unsqueeze(-1)
        if given_scaled is None:
            coordinates = _centred(coordinates.float(), atom_mask).to(dtype)
        else:


            coordinates = torch.where(select, given_scaled.to(dtype), coordinates)
            coordinates = (coordinates.float() * atom_mask.unsqueeze(-1)).to(dtype)
    return coordinates.float() * float(coordinate_scale)
