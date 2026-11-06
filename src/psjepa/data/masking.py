from __future__ import annotations

import torch


def _spatial_block(
    ca: torch.Tensor,
    available: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Grow one block outward from a seed by CA distance, within `available`."""
    index = torch.nonzero(available, as_tuple=False).flatten()
    if index.numel() == 0:
        return index
    seed = index[torch.randint(int(index.numel()), (), device=ca.device)]
    distance = (ca - ca[seed]).square().sum(dim=-1)
    distance[~available] = torch.inf
    return torch.argsort(distance)[: min(size, int(index.numel()))]


def _contiguous_block(
    valid: torch.Tensor,
    taken: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Take a run along the chain, preferring a start whose run is still free."""
    n_valid = int(valid.numel())
    span = min(size, n_valid)
    starts = max(1, n_valid - span + 1)
    order = torch.randperm(starts, device=valid.device)
    for start in order.tolist():
        window = valid[start : start + span]
        if not taken[window].any():
            return window


    free = valid[~taken[valid]]
    return free[:span]


def sample_prediction_mask(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    ratio_min: float = 0.4,
    ratio_max: float = 0.6,
    spatial_probability: float = 0.5,
    blocks: int = 1,
) -> tuple[torch.Tensor, list[str]]:
    'Mask a fraction of each protein as `blocks` separate regions.'
    if blocks < 1:
        raise ValueError("blocks must be at least 1")
    batch, _ = residue_mask.shape
    output = torch.zeros_like(residue_mask)
    modes: list[str] = []
    for b in range(batch):
        valid = torch.nonzero(residue_mask[b], as_tuple=False).flatten()
        n_valid = int(valid.numel())
        if n_valid == 0:
            modes.append("empty")
            continue
        ratio = float(torch.empty((), device=coords.device).uniform_(ratio_min, ratio_max))
        n_mask = max(1, min(n_valid - 1 if n_valid > 1 else 1, int(round(ratio * n_valid))))
        # One block per residue is the most this can mean; asking for more would
        # silently deliver fewer masked residues than the ratio requested.
        count = max(1, min(blocks, n_mask))
        spatial = bool(torch.rand((), device=coords.device) < spatial_probability)
        modes.append(("spatial" if spatial else "contiguous") + (f"x{count}" if count > 1 else ""))

        taken = torch.zeros_like(residue_mask[b])
        ca = coords[b, :, 1]
        ca_valid = residue_mask[b] & atom_mask[b, :, 1]
        for part in range(count):
            # Give the remainder to the earliest blocks so the sizes stay within
            # one residue of each other and the total is exact.
            size = n_mask // count + (1 if part < n_mask % count else 0)
            if size <= 0:
                continue
            if spatial:
                available = ca_valid & ~taken
                chosen = _spatial_block(ca, available, size)
                if chosen.numel() < size:
                    remaining = valid[~taken[valid]]
                    remaining = remaining[~torch.isin(remaining, chosen)]
                    chosen = torch.cat([chosen, remaining[: size - chosen.numel()]])
            else:
                chosen = _contiguous_block(valid, taken, size)
            taken[chosen] = True
        output[b] = taken
    return output, modes


def _grow(ca: torch.Tensor, pool: torch.Tensor, seed: int, size: int) -> torch.Tensor:
    """Take the `size` residues of `pool` nearest `seed` in 3D."""
    if size <= 0 or not pool.any():
        return torch.empty(0, dtype=torch.long, device=ca.device)
    distance = (ca - ca[seed]).square().sum(dim=-1)
    distance[~pool] = torch.inf
    return torch.argsort(distance)[: min(size, int(pool.sum()))]


def _region(
    ca: torch.Tensor,
    valid: torch.Tensor,
    live: torch.Tensor,
    size: int,
    spatial: bool,
) -> torch.Tensor:
    """One coherent region: a 3D neighbourhood, or a run along the chain."""
    if spatial:
        index = torch.nonzero(live, as_tuple=False).flatten()
        if index.numel() == 0:
            return valid[:size]
        seed = index[torch.randint(int(index.numel()), (), device=ca.device)]
        distance = (ca - ca[seed]).square().sum(dim=-1)
        distance[~live] = torch.inf
        return torch.argsort(distance)[: min(size, int(live.sum()))]
    n_valid = int(valid.numel())
    span = min(size, n_valid)
    start = int(torch.randint(max(1, n_valid - span + 1), (), device=ca.device))
    return valid[start : start + span]


def sample_context_and_targets(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    context_ratio: float = 0.5,
    target_ratio: float = 0.18,
    blocks: int = 4,
    spatial_probability: float = 0.5,
    exclude_context: bool = False,
) -> dict[str, torch.Tensor]:
    'A context region and several target regions, each grown where it lands.'
    if blocks < 1:
        raise ValueError("blocks must be at least 1")
    batch, length = residue_mask.shape
    context = torch.zeros_like(residue_mask)
    # One mask per block, kept apart rather than merged. I-JEPA runs its predictor
    # once per target block so that no block can be read off another; merging them
    # into one mask hands the predictor every target position at once, and a
    # sequence mixer will use one block's mask tokens while filling another's.
    blocks_mask = torch.zeros(batch, blocks, length, dtype=torch.bool, device=residue_mask.device)
    modes: list[str] = []
    for b in range(batch):
        valid = torch.nonzero(residue_mask[b], as_tuple=False).flatten()
        n_valid = int(valid.numel())
        if n_valid < 4:
            modes.append("empty")
            continue
        ca = coords[b, :, 1]
        live = residue_mask[b] & atom_mask[b, :, 1]
        if int(live.sum()) < 4:
            live = residue_mask[b].clone()
        spatial = bool(torch.rand((), device=coords.device) < spatial_probability)
        modes.append("spatial" if spatial else "contiguous")
        context[b, _region(ca, valid, live, max(1, int(round(context_ratio * n_valid))), spatial)] = True
        size = max(1, int(round(target_ratio * n_valid)))
        for part in range(blocks):
            blocks_mask[b, part, _region(ca, valid, live, size, spatial)] = True
    # `target_ratio` is per block and is deliberately not divided by `blocks`. With a
    # predictor call per block each block is its own prediction problem, so four
    # blocks pose four problems of the same size rather than one problem cut into
    # quarters — which is what I-JEPA does, and what a single merged mask cannot
    # express. The union is still reported, for the diagnostics that describe
    # coverage.
    target = blocks_mask.any(dim=1)
    if exclude_context:
        context = context & ~target
    return {
        "context_mask": context,
        "target_mask": target,
        "target_blocks": blocks_mask,
        "target_seen": target & context,
        "target_unseen": target & ~context,
        "modes": modes,
    }


def sample_window_and_flanks(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    context_ratio: float = 0.2,
    target_ratio: float = 0.2,
) -> dict[str, torch.Tensor]:
    """One contiguous window of chain, predicting the run on each side of it.

    `sample_context_and_targets` grows the context and the targets from independent
    seeds, so the context ends up perforated: at `exclude_context: true` the encoder's
    input is 28.2% real residues and 71.8% `mask_x`, scattered. Two costs follow. The
    predictor receives the student's own output at every target position and can pass it
    through instead of predicting, and the encoder is never once shown a whole protein
    even though that is exactly what a downstream probe feeds it.

    Both disappear if the context is a single run and the targets are its two flanks.
    The context is then a short but *intact* subchain, which is much closer to the probe
    regime, and the target positions hold nothing but a mask token, so there is nothing
    to pass through. This is also the one arrangement a state-space mixer can express
    without lying about position: interior holes cannot be dropped from the sequence
    without renumbering every residue after them, whereas a run has its neighbours in
    their true relative places and the recurrence supplies the rest. A transformer needs
    none of this care -- attention is permutation-equivariant and I-JEPA hands the
    predictor a mask token carrying the target's positional embedding -- but Mamba has no
    such table, because the recurrence *is* the position.

    Predicting only the chain's two termini would be a narrower task than this, so the
    window is placed uniformly rather than centred: its flanks are then interior runs
    almost everywhere. When the window lands near an end there is not enough room on
    that side, and the shortfall is handed to the other flank rather than dropped, so the
    supervised fraction is `2 * target_ratio` regardless of where the window fell and
    position stays unbiased.
    """
    batch, length = residue_mask.shape
    context = torch.zeros_like(residue_mask)
    blocks_mask = torch.zeros(batch, 2, length, dtype=torch.bool, device=residue_mask.device)
    modes: list[str] = []
    for b in range(batch):
        valid = torch.nonzero(residue_mask[b], as_tuple=False).flatten()
        n_valid = int(valid.numel())
        if n_valid < 4:
            modes.append("empty")
            continue
        modes.append("flanks")
        window = max(1, min(n_valid - 2, int(round(context_ratio * n_valid))))
        budget = min(n_valid - window, max(2, int(round(2.0 * target_ratio * n_valid))))
        start = int(torch.randint(n_valid - window + 1, (), device=residue_mask.device))
        room_left, room_right = start, n_valid - start - window
        left = min(room_left, budget - budget // 2)
        right = min(room_right, budget - left)
        # Whatever one side could not take, the other side absorbs if it has room, so
        # the supervised total does not depend on where the window landed.
        left = min(room_left, budget - right)
        context[b, valid[start : start + window]] = True
        if left > 0:
            blocks_mask[b, 0, valid[start - left : start]] = True
        if right > 0:
            blocks_mask[b, 1, valid[start + window : start + window + right]] = True
    target = blocks_mask.any(dim=1)
    return {
        "context_mask": context,
        "target_mask": target,
        "target_blocks": blocks_mask,
        "target_seen": target & context,
        "target_unseen": target & ~context,
        "modes": modes,
    }


def sample_motif_mask(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    min_fraction: float = 0.05,
    max_fraction: float = 0.30,
    spatial_probability: float = 0.7,
    max_segments: int = 3,
) -> torch.Tensor:
    """Residues to hand a scaffolding decoder as *given* geometry.

    A binding site is a spatial cluster whose residues usually come from several
    places in the sequence, so the default draw is spatial: a seed residue and
    its nearest neighbours in 3D, which lands on a discontiguous set of segments
    on its own. The contiguous draw covers motifs that really are one or a few
    runs of chain, such as a single helix.

    Returns a boolean mask of the given residues. Everything else, within
    `residue_mask`, is what the decoder has to build.
    """
    batch, length = residue_mask.shape
    motif = torch.zeros_like(residue_mask)
    ca = coords[:, :, 1]
    for item in range(batch):
        valid = torch.nonzero(residue_mask[item], as_tuple=False).flatten()
        if valid.numel() < 8:
            continue
        fraction = float(
            torch.empty((), device=coords.device).uniform_(min_fraction, max_fraction)
        )
        size = max(4, int(round(fraction * float(valid.numel()))))
        spatial = bool(torch.rand((), device=coords.device) < spatial_probability)
        if spatial:
            available = residue_mask[item].clone()
            chosen = _spatial_block(ca[item], available, size)
        else:
            segments = int(torch.randint(1, max_segments + 1, ()))
            taken = torch.zeros(length, dtype=torch.bool, device=coords.device)
            picked = []
            for _ in range(segments):
                block = _contiguous_block(valid, taken, max(2, size // segments))
                if block.numel():
                    taken[block] = True
                    picked.append(block)
            chosen = torch.cat(picked) if picked else valid[:size]
        motif[item, chosen] = True
    # A residue with no resolved atoms cannot be given as geometry.
    return motif & residue_mask & atom_mask.any(dim=-1)
