'Rectified-flow prior over standardized per-residue embeddings.'
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TokenPriorConfig:
    d_token: int = 128
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    max_residues: int = 512
    dropout: float = 0.0


    token_mean: list[float] = field(default_factory=list)
    token_std: list[float] = field(default_factory=list)
    # Empirical chain-length histogram of the training split, so sampling does
    # not need the cache. `length_values[i]` occurs `length_counts[i]` times.
    length_values: list[int] = field(default_factory=list)
    length_counts: list[int] = field(default_factory=list)


    self_conditioning: bool = False
    # Rotary position embedding on queries and keys. Absolute position embeddings
    # make a relative offset something the model has to infer; beta-strand pairing
    # is a specific |i-j| pattern, and sheet pairing is exactly what the generated
    # samples lose (ladder pairs per residue 0.227 against 0.318 for real chains).
    # RoPE encodes the offset directly and costs no attention-bias tensor.
    rotary_position: bool = False


    distogram_weight: float = 0.0
    distogram_bins: int = 32
    distogram_max: float = 24.0
    distogram_width: int = 64
    rotary_base: float = 10000.0


def timestep_embedding(time: torch.Tensor, width: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding of a continuous time in [0, 1]."""
    half = width // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / half
    )
    angles = time.float().reshape(-1, 1) * frequencies.reshape(1, -1) * 1000.0
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if width % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return values * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rotary_tables(length: int, head_dim: int, base: float, device, dtype):
    """cos/sin tables of shape [1, 1, length, head_dim] for rotary embedding."""
    half = head_dim // 2
    frequency = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    angle = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1) * frequency
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos().to(dtype)[None, None], angle.sin().to(dtype)[None, None]


def apply_rotary(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    rotated = torch.cat([-values[..., half:], values[..., :half]], dim=-1)
    return values * cos + rotated * sin


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional rotary position embedding.

    `nn.MultiheadAttention` cannot take a rotary transform between projection and
    the dot product, so the projections are written out. The padding mask stays a
    boolean broadcast over heads, which keeps the fused kernel eligible.
    """

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, values, key_padding_mask, rotary=None):
        batch, length, width = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if rotary is not None:
            cos, sin = rotary
            query = apply_rotary(query, cos[..., :length, :], sin[..., :length, :])
            key = apply_rotary(key, cos[..., :length, :], sin[..., :length, :])
        # True means "may attend"; the caller passes True for padded keys.
        mask = (~key_padding_mask).reshape(batch, 1, 1, length)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.projection(attended)


    _LEGACY_NAMES = {
        "in_proj_weight": "qkv.weight",
        "in_proj_bias": "qkv.bias",
        "out_proj.weight": "projection.weight",
        "out_proj.bias": "projection.bias",
    }

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for legacy, current in self._LEGACY_NAMES.items():
            if prefix + legacy in state_dict and prefix + current not in state_dict:
                state_dict[prefix + current] = state_dict.pop(prefix + legacy)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class DiTBlock(nn.Module):
    """Pre-norm self-attention + MLP, both modulated and gated by the time vector."""

    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(width, heads, dropout)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        # adaLN-Zero: the block contributes nothing until it learns to.
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(
        self, values: torch.Tensor, condition: torch.Tensor, pad_mask: torch.Tensor,
        rotary=None,
    ) -> torch.Tensor:
        parameters = self.modulation(condition).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = parameters
        normed = modulate(self.norm_attention(values), shift_a, scale_a)
        attended = self.attention(normed, pad_mask, rotary)
        values = values + gate_a.unsqueeze(1) * attended
        normed = modulate(self.norm_mlp(values), shift_m, scale_m)
        return values + gate_m.unsqueeze(1) * self.mlp(normed)


class TokenFlowPrior(nn.Module):
    """Velocity field over standardised token sequences.

    ``forward`` returns d(x)/dt for the straight path between Gaussian noise and a
    real token sequence; the trainer regresses it onto ``x1 - x0``.
    """

    config_class = TokenPriorConfig

    def __init__(self, config: TokenPriorConfig) -> None:
        super().__init__()
        self.config = config
        width = int(config.width)
        self.input = nn.Linear(int(config.d_token), width)
        self.position = nn.Parameter(torch.zeros(1, int(config.max_residues), width))
        nn.init.normal_(self.position, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        # The chain length is global information that padding alone conveys only
        # indirectly; a scalar feature makes it explicit to every block.
        self.length_mlp = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            DiTBlock(width, int(config.heads), float(config.mlp_ratio), float(config.dropout))
            for _ in range(int(config.layers))
        )
        self.norm_out = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, int(config.d_token))
        # Zero-initialised, so a self-conditioning model starts out identical to
        # one without it and the signal is learned rather than injected as noise.
        self.self_condition_input = (
            nn.Linear(int(config.d_token), width) if config.self_conditioning else None
        )
        if self.self_condition_input is not None:
            nn.init.zeros_(self.self_condition_input.weight)
            nn.init.zeros_(self.self_condition_input.bias)
        nn.init.zeros_(self.modulation_out[1].weight)
        nn.init.zeros_(self.modulation_out[1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Auxiliary only: it shapes the trunk through its gradient and is never
        # called at sampling time, so nothing downstream has to know about it.
        self.distogram = (
            DistogramHead(int(config.d_token), int(config.distogram_width),
                          int(config.distogram_bins))
            if float(config.distogram_weight) > 0.0 else None
        )

        self.rotary_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        mean = torch.tensor(config.token_mean or [0.0] * int(config.d_token))
        std = torch.tensor(config.token_std or [1.0] * int(config.d_token))
        self.register_buffer("token_mean", mean.float(), persistent=True)
        self.register_buffer("token_std", std.float().clamp_min(1e-6), persistent=True)

    # -- token scaling -------------------------------------------------------
    def standardise(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens - self.token_mean) / self.token_std

    def destandardise(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * self.token_std + self.token_mean

    def set_token_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.token_mean.copy_(mean.to(self.token_mean.device).float())
        self.token_std.copy_(std.to(self.token_std.device).float().clamp_min(1e-6))
        self.config.token_mean = [float(v) for v in mean.flatten()]
        self.config.token_std = [float(v) for v in std.flatten()]

    # -- network -------------------------------------------------------------
    def forward(
        self, tokens: torch.Tensor, time: torch.Tensor, residue_mask: torch.Tensor,
        self_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = tokens.shape[0], tokens.shape[1]
        if length > self.position.shape[1]:
            raise ValueError(
                f"sequence of {length} residues exceeds max_residues={self.position.shape[1]}"
            )
        values = self.input(tokens) + self.position[:, :length]
        if self.self_condition_input is not None:
            previous = (
                torch.zeros_like(tokens) if self_condition is None
                else self_condition.to(tokens.dtype)
            )
            values = values + self.self_condition_input(previous)
        elif self_condition is not None:
            raise ValueError("this prior was not built with self_conditioning")
        counts = residue_mask.sum(dim=-1).clamp_min(1).float()
        scale = float(self.position.shape[1])
        length_features = torch.stack(
            [counts / scale, torch.log(counts) / math.log(scale)], dim=-1
        )
        condition = self.time_mlp(
            timestep_embedding(time, self.position.shape[-1])
        ) + self.length_mlp(length_features.to(values.dtype))
        pad_mask = ~residue_mask


        pad_mask = pad_mask.clone()
        pad_mask[:, 0] = False
        rotary = self._rotary(values) if self.config.rotary_position else None
        for block in self.blocks:
            values = block(values, condition, pad_mask, rotary)
        shift, scale_out = self.modulation_out(condition).chunk(2, dim=-1)
        values = modulate(self.norm_out(values), shift, scale_out)
        return self.output(values) * residue_mask.unsqueeze(-1).to(values.dtype)

    def _rotary(self, values: torch.Tensor):
        """cos/sin for the full window, built once and reused across blocks."""
        head_dim = self.position.shape[-1] // int(self.config.heads)
        cached = self.rotary_cache
        if (
            cached is None
            or cached[0].device != values.device
            or cached[0].dtype != values.dtype
        ):
            cached = rotary_tables(
                int(self.config.max_residues), head_dim,
                float(self.config.rotary_base), values.device, values.dtype,
            )
            self.rotary_cache = cached
        return cached

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Every config field, so a checkpoint rebuilds the architecture it was trained as.

        This used to enumerate the fields by hand, and adding one without adding
        it here wrote checkpoints that silently rebuilt as a *different* model:
        the distogram arm's samples could not be loaded at all, because the head
        was missing and `strict=True` then rejected its weights.
        `test_every_config_field_is_serialised` pins this.
        """
        return asdict(self.config)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TokenFlowPrior":
        return cls(TokenPriorConfig(**values))


class DistogramHead(nn.Module):
    """Predict the CA-CA distance map from a clean-token estimate.

    A pair representation built as a sum of two per-residue projections is the
    cheapest form that still depends on both members of the pair; the map is
    symmetrised so the head cannot encode i<j order instead of geometry.
    """

    def __init__(self, d_token: int, width: int, bins: int) -> None:
        super().__init__()
        self.left = nn.Linear(d_token, width)
        self.right = nn.Linear(d_token, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, bins),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pair = self.left(tokens).unsqueeze(2) + self.right(tokens).unsqueeze(1)
        logits = self.mlp(pair)
        return 0.5 * (logits + logits.transpose(1, 2))


def distogram_targets(coordinates: torch.Tensor, bins: int, maximum: float) -> torch.Tensor:
    """Bin indices of the CA-CA distance map; the last bin absorbs everything beyond `maximum`."""
    alpha = coordinates[:, :, 1]
    distances = torch.cdist(alpha, alpha)
    edges = torch.linspace(0.0, maximum, bins, device=coordinates.device)
    return torch.bucketize(distances, edges).clamp_(max=bins - 1)


_LEGACY_ATTENTION_NAMES = {
    "in_proj_weight": "qkv.weight",
    "in_proj_bias": "qkv.bias",
    "out_proj.weight": "projection.weight",
    "out_proj.bias": "projection.bias",
}


def prior_from_checkpoint(state: dict[str, Any]) -> "TokenFlowPrior":
    """Rebuild the prior a checkpoint was trained as, filling fields it predates.

    `to_dict` used to enumerate config fields by hand, so checkpoints written
    before a field was added carry a `prior_config` without it and rebuild as a
    *different* architecture -- for the distogram arms, one with no head at all,
    which `strict=True` then rejects. The training config is stored in the same
    checkpoint and does have the value, so the missing field is recovered from
    there rather than by rewriting multi-gigabyte files (and a run still in
    flight would overwrite such a repair at its next checkpoint anyway).
    """
    values = dict(state["prior_config"])
    model_config = (state.get("config") or {}).get("model") or {}
    for entry in fields(TokenPriorConfig):
        if entry.name not in values and entry.name in model_config:
            values[entry.name] = model_config[entry.name]
    return TokenFlowPrior(TokenPriorConfig(**values))


def load_prior_state_dict(prior: "TokenFlowPrior", state: dict[str, Any]) -> None:
    """Load a prior checkpoint, translating pre-rotary attention parameter names."""
    # The distogram head is trained-only: it shapes the trunk through its gradient
    # and is never called at sampling time. A checkpoint that carries it must
    # still load into a model built for sampling, which has no such head.
    if prior.distogram is None:
        state = {k: v for k, v in state.items() if not k.startswith("distogram.")}
    migrated = {}
    for key, value in state.items():
        for old, new in _LEGACY_ATTENTION_NAMES.items():
            suffix = ".attention." + old
            if key.endswith(suffix):
                key = key[: -len(suffix)] + ".attention." + new
                break
        migrated[key] = value
    prior.load_state_dict(migrated, strict=True)


def prior_flow_loss(
    prior: TokenFlowPrior,
    tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    time_sampling: str = "uniform",
    self_conditioning_probability: float = 0.5,
    given_mask: torch.Tensor | None = None,
    coordinates: torch.Tensor | None = None,
    given_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rectified-flow regression on standardised tokens.

    ``logit_normal`` concentrates the sampled times near 0.5, where the velocity
    target of a straight path is hardest to predict; uniform weighting spends
    much of the budget on the two ends, which are nearly trivial.

    ``given_mask`` trains the model for inpainting: the marked residues sit at their clean
    tokens at *every* time instead of on a trajectory, and they are excluded from the loss.
    Sampling already imposes them that way, so without this the model is asked at inference
    for something it was never trained on -- it has only ever seen every residue equally
    noisy. This is the token-space twin of how the decoder is taught a motif.

    ``given_tokens`` decides *which* tokens the held residues sit at. Left out, they sit at their
    own clean tokens, read off the finished chain -- which no real scaffolding task can supply,
    because the rest of the chain is what is being designed. Supplying the motif's tokens as
    computed from the motif alone trains the model for the condition it will actually be given.
    The regression target stays the true in-context token either way: the point is to recover the
    chain the site belongs to, from the impoverished description of the site that is on offer.
    """
    target = prior.standardise(tokens.float()) * residue_mask.unsqueeze(-1)
    noise = torch.randn_like(target) * residue_mask.unsqueeze(-1)
    batch = target.shape[0]
    if time_sampling == "logit_normal":
        time = torch.sigmoid(torch.randn(batch, device=target.device))
    elif time_sampling == "uniform":
        time = torch.rand(batch, device=target.device)
    else:
        raise ValueError(f"unknown time_sampling: {time_sampling!r}")
    t = time.reshape(-1, 1, 1)
    path = (1.0 - t) * noise + t * target
    velocity = target - noise
    if given_mask is not None:
        select = given_mask.unsqueeze(-1)
        held = target if given_tokens is None else (
            prior.standardise(given_tokens.float()) * residue_mask.unsqueeze(-1))
        path = torch.where(select, held, path) * residue_mask.unsqueeze(-1)
    dtype = next(prior.parameters()).dtype


    self_condition = None
    if (
        prior.config.self_conditioning
        and float(self_conditioning_probability) > 0.0
        and bool(torch.rand(()) < float(self_conditioning_probability))
    ):
        with torch.no_grad():
            guess = prior(path.to(dtype), time, residue_mask, None).float()
            self_condition = (path + (1.0 - t) * guess) * residue_mask.unsqueeze(-1)
    prediction = prior(path.to(dtype), time, residue_mask, self_condition).float()
    residual = (prediction - velocity).square().mean(dim=-1)
    # Given residues are handed over, not predicted, so they are not scored either.
    selected = residue_mask if given_mask is None else (residue_mask & ~given_mask)
    loss = residual[selected].mean() if selected.any() else residual.sum() * 0.0
    metrics = {
        "prior_velocity_target_rms": float(velocity[selected].square().mean().sqrt()),
        "prior_velocity_prediction_rms": float(prediction[selected].square().mean().sqrt()),
    }
    if prior.distogram is not None and coordinates is not None:
        # The straight path gives the clean-token estimate in closed form.
        estimate = path + (1.0 - t) * prediction
        logits = prior.distogram(estimate.to(dtype)).float()
        target = distogram_targets(coordinates, prior.config.distogram_bins,
                                   prior.config.distogram_max)
        pair_mask = residue_mask.unsqueeze(1) & residue_mask.unsqueeze(2)
        if pair_mask.any():
            auxiliary = F.cross_entropy(
                logits[pair_mask], target[pair_mask], reduction="mean"
            )
            loss = loss + float(prior.config.distogram_weight) * auxiliary
            metrics["prior_distogram_loss"] = float(auxiliary)
            metrics["prior_distogram_accuracy"] = float(
                (logits[pair_mask].argmax(-1) == target[pair_mask]).float().mean()
            )
    return loss, metrics


@torch.no_grad()
def progressive_tokens(
    prior: "TokenFlowPrior",
    residue_mask: torch.Tensor,
    rounds: int = 8,
    steps: int = 100,
    given_mask: torch.Tensor | None = None,
    given_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build the sequence in chunks, each conditioned on what is already committed.

    The prior's conditionals work and its joint does not: given 90% of a real chain's tokens it
    inpaints the rest into something that folds, while a single unconditional draw of the whole
    sequence folds at zero. `refine_tokens` tried to exploit that by redrawing a share of a finished
    sequence over and over, and drifted -- it kept re-deciding tokens that were already fine, on a
    joint that is wrong, which is Gibbs sampling against a mis-specified model.

    This commits instead of revising. Each round draws the whole sequence conditioned on the tokens
    fixed so far, then freezes a further `1/rounds` of the remaining positions and never touches
    them again. The first round is a genuinely unconditional draw, but only a small slice of it
    survives, and every later slice is decided against a growing context -- which is the regime the
    measurement says the model is good in. A supplied `given_mask` (the motif) is committed from the
    start and never redrawn.
    """
    batch, length = residue_mask.shape
    committed = torch.zeros_like(residue_mask) if given_mask is None else given_mask.clone()
    tokens = torch.zeros(batch, length, prior.config.d_token,
                         device=residue_mask.device, dtype=torch.float32)
    if given_tokens is not None:
        tokens = torch.where(committed.unsqueeze(-1), given_tokens.float(), tokens)
    for index in range(int(rounds)):
        held = committed if bool(committed.any()) else None
        drawn = sample_tokens(prior, residue_mask, steps=steps, generator=generator,
                              given_mask=held, given_tokens=tokens if held is not None else None,
                              temperature=temperature)
        free = residue_mask & ~committed
        if not bool(free.any()):
            break
        # Freeze an equal share of what is still free, chosen at random: the prior has no
        # confidence signal to order by, so an arbitrary order is the honest default.
        share = 1.0 / float(int(rounds) - index)
        take = (torch.rand(residue_mask.shape, device=residue_mask.device,
                           generator=generator) < share) & free
        if index == int(rounds) - 1:
            take = free
        tokens = torch.where(take.unsqueeze(-1), drawn, tokens)
        committed = committed | take
    return tokens


def refine_tokens(
    prior: "TokenFlowPrior",
    tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    rounds: int = 8,
    fraction: float = 0.15,
    steps: int = 100,
    keep_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    'Redraw a small share of the tokens conditioned on the rest, repeatedly.'
    if rounds <= 0 or fraction <= 0.0:
        return tokens
    current = tokens
    for _ in range(int(rounds)):
        redraw = torch.rand(residue_mask.shape, device=residue_mask.device,
                            generator=generator) < float(fraction)
        redraw = redraw & residue_mask
        if keep_mask is not None:
            redraw = redraw & ~keep_mask
        given = residue_mask & ~redraw
        if not bool(redraw.any()) or not bool(given.any()):
            continue
        current = sample_tokens(prior, residue_mask, steps=steps, generator=generator,
                                given_mask=given, given_tokens=current, temperature=temperature)
    return current


def sample_tokens(
    prior: TokenFlowPrior,
    residue_mask: torch.Tensor,
    steps: int = 100,
    guidance: float = 0.0,
    generator: torch.Generator | None = None,
    given_mask: torch.Tensor | None = None,
    given_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Integrate the prior from Gaussian noise and return decoder-scale tokens.

    ``guidance`` sharpens the field by extrapolating away from the unconditional
    (all-padding-length) direction; 0 disables it, which is the default because
    the prior has no class label to drop.

    ``given_mask`` and ``given_tokens`` inpaint: the marked residues are held at the
    supplied tokens for the whole trajectory, so the rest is drawn conditioned on them.
    This is the token-space half of motif scaffolding -- holding the motif only in
    coordinate space leaves the decoder fighting a token sequence drawn for a different
    protein. Tokens are supplied in decoder scale and standardised here, to match the
    space the integration runs in.
    """
    if steps < 1:
        raise ValueError("token sampling requires at least one integration step")
    batch, length = residue_mask.shape
    device = residue_mask.device
    dtype = next(prior.parameters()).dtype
    values = torch.randn(
        batch, length, prior.config.d_token, device=device, dtype=torch.float32,
        generator=generator,
    )
    # `temperature` below 1 starts the trajectory closer to the mean of the token
    # distribution. Structure generators routinely trade a little diversity this way for
    # samples that land inside the decoder's competence instead of on its edges.
    values = values * float(temperature) * residue_mask.unsqueeze(-1)
    if (given_mask is None) != (given_tokens is None):
        raise ValueError("given_mask and given_tokens go together")
    given_standard = None
    if given_mask is not None:
        given_standard = prior.standardise(given_tokens.float()) * residue_mask.unsqueeze(-1)
        select = given_mask.unsqueeze(-1)
        values = torch.where(select, given_standard, values)
    delta = 1.0 / steps
    self_condition = None
    for index in range(steps):
        time = torch.full((batch,), (index + 0.5) * delta, device=device, dtype=torch.float32)
        velocity = prior(values.to(dtype), time, residue_mask, self_condition).float()
        if prior.config.self_conditioning:
            remaining = (1.0 - time).reshape(-1, 1, 1)
            self_condition = (
                values + remaining * velocity
            ) * residue_mask.unsqueeze(-1)
        if guidance:
            velocity = velocity * (1.0 + guidance)
        values = (values + delta * velocity) * residue_mask.unsqueeze(-1)
        if given_standard is not None:
            # Re-impose after every step, exactly as the decoder does with the motif's
            # coordinates: an inpainting constraint that is applied once at the start is
            # carried away by the field within a few steps.
            values = torch.where(select, given_standard, values)
    return prior.destandardise(values) * residue_mask.unsqueeze(-1)
