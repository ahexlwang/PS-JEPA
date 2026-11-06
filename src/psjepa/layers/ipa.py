'Invariant point attention and frame-aligned point error (Jumper et al., 2021).'
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# Idealised residue geometry in the local frame, angstroms: CA at the origin, C on +x, N in
# the xy-plane. Bonds N-CA 1.458, CA-C 1.526, angle N-CA-C 111.0 degrees.
TEMPLATE = ((-0.525, 1.363, 0.000), (0.000, 0.000, 0.000), (1.526, 0.000, 0.000))
# C->O at psi = 0, from C-O 1.229 A and angle CA-C-O 120.8 degrees; rotated about +x by psi.
CARBONYL = (0.630, 1.055, 0.000)


def gram_schmidt_frames(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames from real N, CA, C positions, the same construction the template inverts.

    x runs CA->C, y is the CA->N component orthogonal to it, z completes a right-handed set.
    Returns rotation rows, so a local point p maps to `p @ rotation` plus the translation.
    """


    def _unit(vector):
        return vector / torch.sqrt(vector.square().sum(dim=-1, keepdim=True) + 1e-12)

    n, ca, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    e1 = _unit(c - ca)
    second = n - ca
    second = second - (e1 * second).sum(dim=-1, keepdim=True) * e1
    e2 = _unit(second)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2), ca


def apply_frames(rotation: torch.Tensor, translation: torch.Tensor,
                 points: torch.Tensor) -> torch.Tensor:
    """Map local row-vector points through frames whose basis vectors are matrix rows."""
    return torch.einsum("...id,...pi->...pd", rotation, points) + translation.unsqueeze(-2)


def invert_frames(rotation: torch.Tensor, translation: torch.Tensor,
                  points: torch.Tensor) -> torch.Tensor:
    """Map global points into the corresponding local row-vector frames."""
    return torch.einsum("...id,...pd->...pi", rotation, points - translation.unsqueeze(-2))


frames_from_backbone = gram_schmidt_frames


def compose_local_frame_update(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    local_rotation: torch.Tensor,
    local_translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a local rigid update with row-basis frames.

    If ``p -> p @ local_rotation + local_translation`` is followed by the current
    ``p -> p @ rotation + translation`` frame, the composed map has rotation
    ``local_rotation @ rotation`` and translation ``local_translation @ rotation + translation``.
    """
    updated_translation = translation + torch.einsum(
        "...id,...i->...d", rotation, local_translation)
    updated_rotation = local_rotation @ rotation
    return updated_rotation, updated_translation


def quaternion_update(vector: torch.Tensor) -> torch.Tensor:
    """AF2's frame update: three numbers as the imaginary part of an unnormalised quaternion.

    (1, b, c, d) is never the zero quaternion, so a zero-initialised head is the identity
    rotation and the module starts by leaving the frames alone.
    """
    b, c, d = vector[..., 0], vector[..., 1], vector[..., 2]
    one = torch.ones_like(b)
    norm = torch.sqrt(one + b * b + c * c + d * d)
    w, x, y, z = one / norm, b / norm, c / norm, d / norm
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def six_d_to_rotation(vector: torch.Tensor) -> torch.Tensor:
    """Continuous SO(3) representation from two predicted row-basis vectors."""
    # Same reason as `gram_schmidt_frames`: differentiable at zero, which a predicted 6-D vector
    # is free to reach.
    def _unit(v):
        return v / torch.sqrt(v.square().sum(dim=-1, keepdim=True) + 1e-12)

    first, second = vector[..., :3], vector[..., 3:6]
    e1 = _unit(first)
    second = second - (e1 * second).sum(dim=-1, keepdim=True) * e1
    e2 = _unit(second)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)


class InvariantPointAttention(nn.Module):
    """Attention whose geometric term is a distance between points carried in local frames."""

    def __init__(self, width: int, heads: int = 8, channels: int = 16,
                 query_points: int = 4, value_points: int = 8,
                 pair_width: int = 0, pair_channels: int = 4) -> None:
        super().__init__()
        self.heads, self.channels = int(heads), int(channels)
        self.query_points, self.value_points = int(query_points), int(value_points)
        self.pair_width, self.pair_channels = int(pair_width), int(pair_channels)
        h, c, q, v = self.heads, self.channels, self.query_points, self.value_points
        self.to_q = nn.Linear(width, h * c, bias=False)
        self.to_k = nn.Linear(width, h * c, bias=False)
        self.to_v = nn.Linear(width, h * c, bias=False)
        self.to_q_points = nn.Linear(width, h * q * 3, bias=False)
        self.to_k_points = nn.Linear(width, h * q * 3, bias=False)
        self.to_v_points = nn.Linear(width, h * v * 3, bias=False)
        # One learned weight per head on the geometric term, through softplus so it stays
        # positive: the distance penalty must never become a reward.
        self.head_weight = nn.Parameter(torch.full((h,), math.log(math.e - 1.0)))
        if self.pair_width:
            self.pair_bias = nn.Linear(self.pair_width, h, bias=False)
            self.pair_value = nn.Linear(self.pair_width, h * self.pair_channels, bias=False)
        else:
            self.pair_bias = None
            self.pair_value = None
        self.out = nn.Linear(h * (c + v * 4 + (self.pair_channels if self.pair_width else 0)),
                             width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, single: torch.Tensor, rotation: torch.Tensor,
                translation: torch.Tensor, pad: torch.Tensor,
                pair: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = single.shape
        h, c, q, v = self.heads, self.channels, self.query_points, self.value_points

        query = self.to_q(single).view(batch, length, h, c)
        key = self.to_k(single).view(batch, length, h, c)
        value = self.to_v(single).view(batch, length, h, c)
        local_q = self.to_q_points(single).view(batch, length, h * q, 3)
        local_k = self.to_k_points(single).view(batch, length, h * q, 3)
        local_v = self.to_v_points(single).view(batch, length, h * v, 3)

        def to_global(points):
            # Rows of `rotation` are the frame axes, so a local point p maps to p-transpose
            # times R. Writing this as `R @ p` costs the invariance the module exists for:
            # under a global rotation R -> R Q and (R Q) @ p is not (R @ p) Q.
            return torch.einsum("blid,blpi->blpd", rotation, points) + translation.unsqueeze(-2)
        global_q = to_global(local_q).view(batch, length, h, q, 3)
        global_k = to_global(local_k).view(batch, length, h, q, 3)
        global_v = to_global(local_v).view(batch, length, h, v, 3)

        logits = torch.einsum("bihc,bjhc->bhij", query, key) / math.sqrt(c)
        logit_terms = 2
        if self.pair_bias is not None:
            if pair is None or pair.shape[:3] != (batch, length, length):
                raise ValueError("pair must have shape (B, L, L, pair_width)")
            logits = logits + self.pair_bias(pair).permute(0, 3, 1, 2)
            logit_terms = 3
        elif pair is not None:
            raise ValueError("pair features were supplied to an IPA block with pair_width=0")


        live = (~pad).to(translation.dtype)
        centre = ((translation * live.unsqueeze(-1)).sum(dim=1)
                  / live.sum(dim=1, keepdim=True).clamp_min(1.0)).view(batch, 1, 1, 1, 3)
        geometry_dtype = torch.float64 if global_q.dtype == torch.float64 else torch.float32
        centred_q = global_q.to(geometry_dtype) - centre.to(geometry_dtype)
        centred_k = global_k.to(geometry_dtype) - centre.to(geometry_dtype)
        query_norm = centred_q.square().sum(dim=(-1, -2)).permute(0, 2, 1)
        key_norm = centred_k.square().sum(dim=(-1, -2)).permute(0, 2, 1)
        cross = torch.einsum("bihpd,bjhpd->bhij", centred_q, centred_k)
        squared = (query_norm.unsqueeze(-1) + key_norm.unsqueeze(-2) - 2.0 * cross).clamp_min(0.0)
        gamma = F.softplus(self.head_weight).view(1, h, 1, 1)
        weight = math.sqrt(2.0 / (9.0 * q))
        # Scalar, point, and (when present) pair logits should keep the same aggregate
        # variance.  The pair-free simplification has two terms; enabling pair bias restores
        # AF2's third term.
        logits = (logits - gamma * weight * 0.5 * squared.to(logits.dtype)) / math.sqrt(
            float(logit_terms)
        )
        # A row whose keys are all padded would make softmax return NaN, so keep one key
        # alive; the caller's residue mask removes that sample from every loss anyway.
        key_pad = pad & ~pad.all(dim=-1, keepdim=True)
        logits = logits.masked_fill(key_pad.reshape(batch, 1, 1, length), float("-inf"))
        attention = logits.softmax(dim=-1)

        out_scalar = torch.einsum("bhij,bjhc->bihc", attention, value).reshape(batch, length, h * c)
        out_points = torch.einsum("bhij,bjhpd->bihpd", attention, global_v)
        # Back into each residue's own frame: that is what makes the output invariant.
        local_out = torch.einsum("blid,blhpd->blhpi", rotation,
                                 out_points - translation.view(batch, length, 1, 1, 3))
        norms = torch.linalg.vector_norm(local_out, dim=-1)
        outputs = [out_scalar, local_out.reshape(batch, length, h * v * 3),
                   norms.reshape(batch, length, h * v)]
        if self.pair_value is not None:
            pair_value = self.pair_value(pair).view(
                batch, length, length, h, self.pair_channels)
            out_pair = torch.einsum("bhij,bijhc->bihc", attention, pair_value)
            outputs.append(out_pair.reshape(batch, length, h * self.pair_channels))
        features = torch.cat(outputs, dim=-1)
        return self.out(features)


class StructureModule(nn.Module):
    """Iterated IPA with a frame update, then the idealised template placed on the frames."""

    def __init__(self, width: int = 256, heads: int = 8, iterations: int = 8,
                 dropout: float = 0.0, coordinate_scale: float = 1.0,
                 pair_width: int = 0, pair_channels: int = 4,
                 initial_frames: str = "black_hole",
                 stop_rotation_gradient: bool = True) -> None:
        super().__init__()
        self.iterations = int(iterations)
        self.attention = InvariantPointAttention(
            width, heads, pair_width=pair_width, pair_channels=pair_channels)
        if initial_frames not in (
                "black_hole", "line", "learned_line", "learned_origin", "external"):
            raise ValueError(
                "initial_frames must be 'black_hole', 'line', 'learned_line', "
                "'learned_origin', or 'external'"
            )
        self.initial_frames = initial_frames
        self.stop_rotation_gradient = bool(stop_rotation_gradient)
        # Direct frame (translation + continuous 6D rotation) and carbonyl psi. The latter matters when
        # IPA refinement is disabled: fixing psi=0 leaves O in the wrong plane even when the
        # N/CA/C frame is exact.
        self.seed_update = (
            nn.Linear(width, 11)
            if initial_frames in ("learned_line", "learned_origin") else None
        )
        if self.seed_update is not None:
            nn.init.zeros_(self.seed_update.weight)
            nn.init.zeros_(self.seed_update.bias)
            with torch.no_grad():
                self.seed_update.bias.copy_(torch.tensor(
                    [0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0]))
        self.norm_attention = nn.LayerNorm(width)
        self.transition = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width))
        self.norm_transition = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        # 3 translation + 3 quaternion imaginary part + 2 (sin psi, cos psi)
        self.update = nn.Linear(width, 8)
        nn.init.zeros_(self.update.weight)
        with torch.no_grad():
            self.update.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        scale = float(coordinate_scale)
        self.coordinate_scale = scale
        self.register_buffer("template", torch.tensor(TEMPLATE) / scale, persistent=False)
        self.register_buffer("carbonyl", torch.tensor(CARBONYL) / scale, persistent=False)

    def forward(self, single: torch.Tensor, pad: torch.Tensor,
                pair: torch.Tensor | None = None,
                rotation: torch.Tensor | None = None,
                translation: torch.Tensor | None = None,
                external_local_backbone: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = single.shape
        # Frames live in float32 whatever autocast does to `single`; float64 only when the
        # caller runs the whole module in float64 (equivariance checks).
        device = single.device
        dtype = torch.float64 if single.dtype == torch.float64 else torch.float32
        if (rotation is None) != (translation is None):
            raise ValueError("rotation and translation must be given together")
        if self.initial_frames == "external" and rotation is None:
            raise ValueError(
                "initial_frames='external' requires rotation and translation"
            )
        if external_local_backbone is not None:
            if self.initial_frames != "external":
                raise ValueError(
                    "external_local_backbone requires initial_frames='external'"
                )
            expected_backbone = (batch, length, 4, 3)
            if external_local_backbone.shape != expected_backbone:
                raise ValueError(
                    "external_local_backbone must have shape "
                    f"{expected_backbone}, got {tuple(external_local_backbone.shape)}"
                )
            external_local_backbone = external_local_backbone.to(
                device=device, dtype=dtype
            )
        if rotation is not None:
            # Frames supplied by the caller, e.g. read off a noisy structure by a flow
            # decoder. Any learned seed then acts as a local update on top of them, so the
            # equivariance of the output under a rigid motion of the given frames holds.
            expected_rotation = (batch, length, 3, 3)
            expected_translation = (batch, length, 3)
            if rotation.shape != expected_rotation:
                raise ValueError(
                    f"rotation must have shape {expected_rotation}, got {tuple(rotation.shape)}"
                )
            if translation.shape != expected_translation:
                raise ValueError(
                    "translation must have shape "
                    f"{expected_translation}, got {tuple(translation.shape)}"
                )
            rotation = rotation.to(device=device, dtype=dtype)
            translation = translation.to(device=device, dtype=dtype)
        else:
            # AF2's "black hole" start: every frame the identity at the origin.
            rotation = torch.eye(3, device=device, dtype=dtype).expand(
                batch, length, 3, 3).contiguous()
            translation = torch.zeros(batch, length, 3, device=device, dtype=dtype)
            if self.initial_frames in ("line", "learned_line"):
                translation[..., 0] = (
                    torch.arange(length, device=device, dtype=dtype)
                    * (3.8 / self.coordinate_scale)
                )
        psi = torch.zeros(batch, length, 2, device=device, dtype=dtype)
        if self.seed_update is not None:
            seed = self.seed_update(single).to(dtype)
            rotation, translation = compose_local_frame_update(
                rotation, translation, six_d_to_rotation(seed[..., 3:9]), seed[..., :3])
            psi = seed[..., 9:11]
        # Every iteration's structure, not only the last. AF2 supervises each layer of the
        # structure module; with one shared set of weights and end-only supervision the
        # eight steps get a single, very indirect gradient.
        # A learned seed is already a complete direct-frame decoder. Keep it in the
        # supervised trajectory and let IPA be optional refinement rather than forcing every
        # reconstruction through a recurrent update that can destroy a good direct answer.
        trajectory = ([self.place(
            rotation, translation, psi, external_local_backbone
        )]
                      if self.seed_update is not None else [])
        for iteration in range(self.iterations):
            single = self.norm_attention(
                single + self.dropout(self.attention(single, rotation, translation, pad, pair)))
            single = self.norm_transition(single + self.dropout(self.transition(single)))
            delta = self.update(single).to(dtype)
            # Compose: the update is expressed in the current local frame.
            rotation, translation = compose_local_frame_update(
                rotation, translation, quaternion_update(delta[..., 3:6]), delta[..., :3])
            psi = delta[..., 6:8]
            trajectory.append(self.place(
                rotation, translation, psi, external_local_backbone
            ))
            if self.training and self.stop_rotation_gradient and iteration + 1 < self.iterations:
                rotation = rotation.detach()
        if not trajectory:
            raise ValueError(
                "iterations=0 requires initial_frames='learned_line' or 'learned_origin'"
            )
        return trajectory[-1], torch.stack(trajectory), rotation

    def place(self, rotation: torch.Tensor, translation: torch.Tensor,
              psi: torch.Tensor,
              external_local_backbone: torch.Tensor | None = None) -> torch.Tensor:
        if external_local_backbone is not None:
            return apply_frames(rotation, translation, external_local_backbone)
        sin_psi, cos_psi = psi[..., 0], psi[..., 1]
        norm = torch.sqrt((sin_psi.square() + cos_psi.square()).clamp_min(1e-12))
        sin_psi, cos_psi = sin_psi / norm, cos_psi / norm
        placed = torch.einsum("blid,ai->blad", rotation, self.template.to(rotation.dtype))
        y, z = self.carbonyl[1].to(rotation.dtype), self.carbonyl[2].to(rotation.dtype)
        oxygen_local = torch.stack([
            self.carbonyl[0].to(rotation.dtype).expand_as(sin_psi),
            cos_psi * y - sin_psi * z,
            sin_psi * y + cos_psi * z,
        ], dim=-1) + self.template[2].to(rotation.dtype)
        oxygen = torch.einsum("blid,bli->bld", rotation, oxygen_local)
        return torch.cat([placed, oxygen.unsqueeze(-2)], dim=-2) + translation.unsqueeze(-2)


def fape_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
    clamp: float | None = 10.0,
    scale: float = 10.0,
) -> torch.Tensor:
    'Frame-aligned point error (AF2 Algorithm 28).'
    batch, length, atoms, _ = predicted.shape

    def localise(points):
        rotation, translation = gram_schmidt_frames(points)
        flat = points.reshape(batch, length * atoms, 3)
        offset = flat.unsqueeze(1) - translation.unsqueeze(2)      # (b, i, j*a, 3)
        # rows of `rotation` are the frame axes, so the inverse map is a contraction over
        # the row index: local_d = sum_j rotation[d, j] * offset[j].
        return torch.einsum("bidj,bikj->bikd", rotation, offset)

    # The norm is taken through a square root, and the diagonal entries -- a residue's own CA
    # viewed in its own frame -- are exactly zero for both structures by construction. An
    # epsilon inside the root keeps that point differentiable; AF2 does the same.
    difference = localise(predicted) - localise(target)
    error = torch.sqrt(difference.square().sum(dim=-1) + 1e-8)
    # `clamp=None` is not a detail. From AF2's black-hole init every frame sits at the
    # origin, so the whole structure is collapsed and almost every pair's error exceeds a
    # 10 A clamp -- where `clamp_max` has zero gradient. Trained clamped-only, the frames
    # never had a reason to spread: FAPE moved 0.907 -> 0.898 over 3,125 steps and the
    # output stayed collapsed (pairwise MAE 16.24 A against a true mean near 20). AF2 mixes
    # unclamped FAPE into a fraction of batches for exactly this reason.
    if clamp is not None:
        error = error.clamp_max(clamp)
    error = error / scale
    if atom_mask is None:
        atom_mask = residue_mask.unsqueeze(-1).expand(batch, length, atoms)
    if atom_mask.shape != (batch, length, atoms):
        raise ValueError("atom_mask must have shape (B, L, atoms)")
    frame_valid = residue_mask & atom_mask[..., :3].all(dim=-1)
    atom_valid = (atom_mask & residue_mask.unsqueeze(-1)).reshape(batch, length * atoms)
    pair = frame_valid.unsqueeze(-1) & atom_valid.unsqueeze(1)
    # A 500-residue chain must not count 100x more than a 50-residue chain merely because
    # FAPE forms O(L^2) pairs. AF2 uses fixed crops; this decoder batches variable lengths.
    per_example = (error * pair).sum(dim=(1, 2)) / pair.sum(dim=(1, 2)).clamp_min(1)
    valid_example = pair.any(dim=(1, 2))
    return per_example[valid_example].mean() if valid_example.any() else error.sum() * 0.0


def adjacent_frame_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    scale: float = 1.0,
    pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    'FAPE restricted to i -> i+1, so polymer connectivity is not diluted by L^2 pairs.'
    pred_rotation, pred_translation = gram_schmidt_frames(predicted)
    true_rotation, true_translation = gram_schmidt_frames(target)
    pred_offset = predicted[:, 1:] - pred_translation[:, :-1, None]
    true_offset = target[:, 1:] - true_translation[:, :-1, None]
    pred_local = torch.einsum("blij,blaj->blai", pred_rotation[:, :-1], pred_offset)
    true_local = torch.einsum("blij,blaj->blai", true_rotation[:, :-1], true_offset)
    # Same guard as fape_loss: the norm goes through a square root, and this term is the one
    # that drives its own argument to zero, so the gradient at the optimum is exactly where a
    # bare `vector_norm` is undefined. scaffold_v4 diverged to NaN at step 17,008 without it.
    difference = pred_local - true_local
    error = torch.sqrt(difference.square().sum(dim=-1) + 1e-8) / scale
    frame_valid = residue_mask[:, :-1] & atom_mask[:, :-1, :3].all(dim=-1)
    if pairs is not None:
        if pairs.shape != frame_valid.shape:
            raise ValueError(
                f"pairs must have shape {tuple(frame_valid.shape)}, got {tuple(pairs.shape)}")
        frame_valid = frame_valid & pairs
    point_valid = residue_mask[:, 1:, None] & atom_mask[:, 1:]
    valid = frame_valid.unsqueeze(-1) & point_valid
    per_example = (error * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)
    live = valid.any(dim=(1, 2))
    return per_example[live].mean() if live.any() else error.sum() * 0.0


def ca_distance_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Invariant global-shape loss on CA distance maps, macro-averaged over proteins."""
    pred_distance = torch.cdist(predicted[..., 1, :].float(), predicted[..., 1, :].float())
    true_distance = torch.cdist(target[..., 1, :].float(), target[..., 1, :].float())
    ca_valid = residue_mask & atom_mask[..., 1]
    pair = ca_valid.unsqueeze(-1) & ca_valid.unsqueeze(-2)
    pair = torch.triu(pair, diagonal=1)
    error = (pred_distance - true_distance).abs()
    per_example = (error * pair).sum(dim=(1, 2)) / pair.sum(dim=(1, 2)).clamp_min(1)
    live = pair.any(dim=(1, 2))
    return per_example[live].mean() if live.any() else error.sum() * 0.0


def canonicalize_backbone(
    coordinates: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a backbone in the first observed residue frame.

    The encoder representation is invariant to a global rigid motion, so an absolute PDB
    pose is not a learnable target.  The first complete N/CA/C frame supplies a unique,
    deterministic gauge without an iterative SVD.  With row-basis rotations, the inverse
    map is ``(x - t_anchor) @ R_anchor.T``.

    Returns canonical coordinates and a boolean saying which examples have an anchor.
    """
    if coordinates.ndim != 4 or coordinates.shape[-2:] != (4, 3):
        raise ValueError("coordinates must have shape (B, L, 4, 3)")
    frame_valid = residue_mask & atom_mask[..., :3].all(dim=-1)
    has_anchor = frame_valid.any(dim=-1)
    # argmax returns zero for an empty row. Such a row is removed by every consuming loss,
    # while choosing an in-range index keeps this helper free of data-dependent Python.
    anchor_index = frame_valid.to(torch.int64).argmax(dim=-1)
    rotation, translation = gram_schmidt_frames(coordinates)
    batch_index = torch.arange(coordinates.shape[0], device=coordinates.device)
    anchor_rotation = rotation[batch_index, anchor_index]
    anchor_translation = translation[batch_index, anchor_index]
    canonical = torch.einsum(
        "bid,blad->blai", anchor_rotation,
        coordinates - anchor_translation[:, None, None, :],
    )
    return canonical, has_anchor


def canonical_coordinate_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    'Direct RMSD to an invariant, fixed-gauge target.'
    canonical_target, has_anchor = canonicalize_backbone(target, residue_mask, atom_mask)
    valid = atom_mask & residue_mask.unsqueeze(-1) & has_anchor[:, None, None]
    squared = (predicted.float() - canonical_target.float()).square().sum(dim=-1)
    per_example = torch.sqrt(
        (squared * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1) + 1e-12
    )
    live = valid.any(dim=(1, 2))
    return per_example[live].mean() if live.any() else squared.sum() * 0.0


def aligned_coordinate_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Proper-Kabsch-aligned backbone RMSD, differentiated with a fixed optimal gauge.

    FAPE is the physically meaningful frame loss, but from a line or black-hole start its
    gradient has to discover a global gauge and a fold simultaneously. A detached Kabsch
    solve chooses only the nuisance global pose; the coordinate residual then gives every
    atom a direct vector toward the target. Detaching the solve is the envelope-theorem
    gradient and avoids unstable SVD derivatives at the rank-one line initialization.
    """
    values = []
    valid_atoms = atom_mask & residue_mask.unsqueeze(-1)
    for prediction, truth, valid in zip(predicted, target, valid_atoms):
        p, q = prediction[valid].float(), truth[valid].float()
        if len(p) < 3:
            continue
        p = p - p.mean(dim=0, keepdim=True)
        q = q - q.mean(dim=0, keepdim=True)
        with torch.no_grad():
            u, _, vh = torch.linalg.svd(p.transpose(0, 1) @ q)
            sign = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
            diagonal = torch.ones(3, device=p.device, dtype=p.dtype)
            diagonal[-1] = sign
            rotation = u @ torch.diag(diagonal) @ vh
        values.append(((p @ rotation - q).square().sum(dim=-1).mean() + 1e-12).sqrt())
    return torch.stack(values).mean() if values else predicted.sum() * 0.0


def peptide_bond_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    beta: float | None = None,
) -> torch.Tensor:
    'Match C(i)-N(i+1) lengths on truly consecutive, observed residues.'
    pred = torch.linalg.vector_norm(predicted[:, :-1, 2] - predicted[:, 1:, 0], dim=-1)
    truth = torch.linalg.vector_norm(target[:, :-1, 2] - target[:, 1:, 0], dim=-1)
    valid = (residue_mask[:, :-1] & residue_mask[:, 1:]
             & atom_mask[:, :-1, 2] & atom_mask[:, 1:, 0])
    error = (
        (pred - truth).abs()
        if beta is None
        else F.smooth_l1_loss(
            pred, truth, reduction="none", beta=float(beta)
        )
    )
    per_example = (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    live = valid.any(dim=1)
    return per_example[live].mean() if live.any() else error.sum() * 0.0


def peptide_angle_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Match the two bond angles spanning each consecutive peptide link."""
    def angle(first: torch.Tensor, centre: torch.Tensor,
              third: torch.Tensor) -> torch.Tensor:
        left = first - centre
        right = third - centre
        denominator = (
            torch.linalg.vector_norm(left, dim=-1)
            * torch.linalg.vector_norm(right, dim=-1)
        ).clamp_min(1e-8)
        cosine = (left * right).sum(dim=-1) / denominator
        return torch.acos(cosine.clamp(-1.0, 1.0))

    predicted_angles = torch.stack((
        angle(predicted[:, :-1, 1], predicted[:, :-1, 2], predicted[:, 1:, 0]),
        angle(predicted[:, :-1, 2], predicted[:, 1:, 0], predicted[:, 1:, 1]),
    ), dim=-1)
    target_angles = torch.stack((
        angle(target[:, :-1, 1], target[:, :-1, 2], target[:, 1:, 0]),
        angle(target[:, :-1, 2], target[:, 1:, 0], target[:, 1:, 1]),
    ), dim=-1)
    valid_link = residue_mask[:, :-1] & residue_mask[:, 1:]
    valid = torch.stack((
        valid_link & atom_mask[:, :-1, 1] & atom_mask[:, :-1, 2]
        & atom_mask[:, 1:, 0],
        valid_link & atom_mask[:, :-1, 2] & atom_mask[:, 1:, 0]
        & atom_mask[:, 1:, 1],
    ), dim=-1)
    error = (predicted_angles - target_angles).square()
    per_example = (
        (error * valid).sum(dim=(1, 2))
        / valid.sum(dim=(1, 2)).clamp_min(1)
    )
    live = valid.any(dim=(1, 2))
    return per_example[live].mean() if live.any() else error.sum() * 0.0
