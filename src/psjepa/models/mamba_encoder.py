"""Bidirectional Mamba sequence encoding.

The GPU backend uses the separately installed mamba-ssm package. A PyTorch
reference recurrence supports CPU diagnostics, and a GRU backend supports tests.
Valid-residue reverse scans keep padded chains independent of batch lengths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def _normalise_reference(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for the LayerNorm/RMSNorm modules used by Mamba blocks."""
    if module.__class__.__name__ == "RMSNorm":
        output = value * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True)
                                     + float(module.eps)).to(value.dtype)
        output = output * module.weight
        bias = getattr(module, "bias", None)
        return output if bias is None else output + bias
    return F.layer_norm(value, value.shape[-1:], module.weight, module.bias, float(module.eps))


def _mamba1_reference(mixer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Kernel-free Mamba-1 scan used only when a frozen checkpoint is read on CPU.

    This is the recurrence in ``mamba_ssm.modules.mamba_simple.Mamba.step`` applied to a
    whole sequence. It is intentionally slow but makes diagnostics and token export possible
    on machines without CUDA/Triton; experiment training continues to use the fused path.
    """
    if mixer.__class__.__name__ != "Mamba":
        raise TypeError(f"CPU reference supports Mamba-1, got {type(mixer).__name__}")
    x, z = mixer.in_proj(hidden_states).chunk(2, dim=-1)
    x = F.silu(mixer.conv1d(x.transpose(1, 2))[..., : hidden_states.shape[1]])
    projected = mixer.x_proj(x.transpose(1, 2))
    dt, b_value, c_value = torch.split(
        projected, [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1)
    dt = F.linear(dt, mixer.dt_proj.weight).transpose(1, 2)
    x = x.contiguous()
    z = z.transpose(1, 2).contiguous()
    a_value = -torch.exp(mixer.A_log.float())
    state = torch.zeros(
        hidden_states.shape[0], mixer.d_inner, mixer.d_state,
        device=hidden_states.device, dtype=torch.float32)
    outputs = []
    for index in range(hidden_states.shape[1]):
        delta = F.softplus(dt[..., index].float() + mixer.dt_proj.bias.float())
        decay = torch.exp(delta.unsqueeze(-1) * a_value.unsqueeze(0))
        drive = delta.unsqueeze(-1) * b_value[:, index].float().unsqueeze(1)
        state = state * decay + x[..., index].float().unsqueeze(-1) * drive
        output = torch.einsum("bdn,bn->bd", state, c_value[:, index].float())
        output = output + mixer.D.float() * x[..., index].float()
        output = output * F.silu(z[..., index].float())
        outputs.append(output.to(hidden_states.dtype))
    return mixer.out_proj(torch.stack(outputs, dim=1))


def set_padding_invariant(module: nn.Module, enabled: bool = True) -> int:
    """Switch every Mamba stack under `module` between the two reverse-scan paths.

    Returns how many stacks were switched. This exists so a *frozen* checkpoint can be
    read both ways in one process: the flag changes no parameter, and on an unpadded batch
    the two paths agree exactly, so `enabled=True` reproduces batch-size-1 encoding for
    any batch rather than defining a different model.
    """
    switched = 0
    for child in module.modules():
        if isinstance(child, BiMambaStack):
            child.padding_invariant = bool(enabled)
            switched += 1
    return switched


@dataclass
class MambaEncoderConfig:
    d_model: int = 128
    n_layer: int = 4
    bidirectional: bool = True
    d_intermediate: int = 0
    ssm_cfg: dict = field(default_factory=dict)
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    backend: str = "mamba"
    # False is the behaviour every existing checkpoint was trained, evaluated and paired
    # with a decoder under, and it stays the default so that re-running a stored result
    # reproduces it. True reverses only each chain's own residues, which makes a padded
    # batch agree with the same chain encoded alone -- see `_reverse_valid`. Switch a
    # frozen model over at runtime with `set_padding_invariant`.
    padding_invariant: bool = False


class BiMambaStack(nn.Module):
    def __init__(self, config: MambaEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backend = config.backend
        self.bidirectional = config.bidirectional
        # getattr: configs pickled into older checkpoints have no such field.
        self.padding_invariant = bool(getattr(config, "padding_invariant", False))
        if self.backend == "gru":
            self.gru = nn.GRU(
                config.d_model,
                config.d_model // 2 if config.bidirectional else config.d_model,
                num_layers=config.n_layer,
                batch_first=True,
                bidirectional=config.bidirectional,
            )
            self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
            return
        if self.backend != "mamba":
            raise ValueError(f"Unknown mixer backend: {self.backend}")
        from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block
        try:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
        except ImportError:
            RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
        self._layer_norm_fn = layer_norm_fn
        self._rms_norm_type = RMSNorm
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("Mamba fused LayerNorm/RMSNorm kernels are unavailable")
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    d_intermediate=config.d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )
        norm_cls = RMSNorm if config.rms_norm else nn.LayerNorm
        self.norm_f = norm_cls(config.d_model, eps=config.norm_epsilon)
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                n_residuals_per_layer=1 if config.d_intermediate == 0 else 2,
            )
        )

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        valid = (~padding_mask).unsqueeze(-1).to(hidden_states.dtype)
        lengths = (~padding_mask).sum(dim=1)
        if self.backend == "gru":
            masked = hidden_states * valid
            if self.bidirectional and self.padding_invariant:
                # nn.GRU runs its own backward direction over whatever it is given, so here
                # the padding has to be hidden from it rather than reordered.
                sequence = nn.utils.rnn.pack_padded_sequence(
                    masked, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
                unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                    self.gru(sequence)[0], batch_first=True, total_length=masked.shape[1])
                return self.norm_f(unpacked) * valid
            hidden_states, _ = self.gru(masked)
            return self.norm_f(hidden_states) * valid
        if self.bidirectional and self.padding_invariant:
            def reverse(value: torch.Tensor) -> torch.Tensor:
                return _reverse_valid(value, lengths)
        else:
            def reverse(value: torch.Tensor) -> torch.Tensor:
                return value.flip(1)
        if hidden_states.device.type == "cpu":
            residual = None
            for layer in self.layers:
                hidden_states = hidden_states * valid
                if residual is not None:
                    residual = residual * valid

                def direction(value: torch.Tensor, previous: torch.Tensor | None):
                    updated_residual = value + previous if previous is not None else value
                    normalised = _normalise_reference(
                        layer.norm, updated_residual.to(layer.norm.weight.dtype))
                    if self.residual_in_fp32:
                        updated_residual = updated_residual.float()
                    return _mamba1_reference(layer.mixer, normalised), updated_residual

                forward, residual_forward = direction(hidden_states, residual)
                if self.bidirectional:
                    backward, residual_backward = direction(
                        reverse(hidden_states),
                        reverse(residual) if residual is not None else None)
                    hidden_states = forward + reverse(backward)
                    residual = residual_forward + reverse(residual_backward)
                else:
                    hidden_states, residual = forward, residual_forward
            residual = hidden_states + residual if residual is not None else hidden_states
            return _normalise_reference(
                self.norm_f, residual.to(self.norm_f.weight.dtype)) * valid
        residual = None
        for layer in self.layers:
            hidden_states = hidden_states * valid
            if residual is not None:
                residual = residual * valid
            forward, residual_forward = layer(hidden_states, residual)
            if self.bidirectional:
                backward, residual_backward = layer(
                    reverse(hidden_states),
                    reverse(residual) if residual is not None else None,
                )
                hidden_states = forward + reverse(backward)
                residual = residual_forward + reverse(residual_backward)
            else:
                hidden_states, residual = forward, residual_forward
        if not self.fused_add_norm:
            residual = hidden_states + residual if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(self.norm_f.weight.dtype))
        else:
            hidden_states = self._layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, self._rms_norm_type),
            )
        return hidden_states * valid


def _reverse_valid(value: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence over its own residues, leaving the padding where it is.

    `value.flip(1)` reverses the padded axis, so right padding arrives at the recurrent scan
    *before* the chain does and the state that reaches residue 1 depends on how much padding the
    batch happened to carry. Zeroing the padded positions does not prevent this: the mixer's
    projections and convolution have biases, so a zero input still moves the state. Measured on a
    57-residue chain, padding it to 65/89/185 changes its own latents by 5.1%, 10.3% and 14.9% on
    average, with single residues moving up to 57%. Every batched encode therefore depended on its
    neighbours' lengths, which is not a property an encoder may have -- and the token prior trains
    against these as targets.

    Reversing only `lengths[i]` entries per row keeps each chain's scan identical whatever it is
    batched with.
    """
    index = torch.arange(value.shape[1], device=value.device).unsqueeze(0)
    source = (lengths.unsqueeze(1) - 1 - index).clamp_min(0)
    source = torch.where(index < lengths.unsqueeze(1), source, index)
    return value.gather(1, source.unsqueeze(-1).expand_as(value))


class ResidueMambaEncoder(nn.Module):
    """Encode concatenated [semantic queries | residue features]."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        max_residues: int,
        backend: str = "mamba",
        query_mode: str = "content_conditioned",
        content_query_rank: int = 16,
        global_slots: int = 0,
        semantic_slots_only: bool = False,
    ) -> None:
        super().__init__()


        self.semantic_slots_only = bool(semantic_slots_only)
        if self.semantic_slots_only and not global_slots:
            raise ValueError("semantic_slots_only requires global_slots > 0")


        self.global_slots = int(global_slots)
        if self.global_slots:
            self.global_queries = nn.Parameter(torch.empty(self.global_slots, d_model))
            nn.init.trunc_normal_(self.global_queries, std=0.02)
        else:
            self.register_parameter("global_queries", None)
        if query_mode not in {"position_table", "shared_plus_position", "shared", "content_conditioned"}:
            raise ValueError(f"Unknown semantic query mode: {query_mode}")
        self.query_mode = query_mode
        self.content_query = None
        if query_mode == "position_table":
            self.semantic_queries = nn.Parameter(torch.empty(max_residues, d_model))
            self.register_parameter("shared_query", None)
            nn.init.trunc_normal_(self.semantic_queries, std=0.02)
        else:
            self.shared_query = nn.Parameter(torch.empty(1, d_model))
            nn.init.trunc_normal_(self.shared_query, std=0.02)
            if query_mode == "shared_plus_position":
                self.semantic_queries = nn.Parameter(torch.empty(max_residues, d_model))
                nn.init.trunc_normal_(self.semantic_queries, std=0.02)
            else:
                self.register_parameter("semantic_queries", None)
            if query_mode == "content_conditioned":
                # Derive each query slot from that residue's own features, so no
                # parameter is indexed by position and the quantised stream cannot
                # be a function of the residue index. A low-rank map keeps the slot
                # from simply copying the residue stream, which would make the two
                # streams near-duplicates and the cross-stream terms trivial. The
                # Rescaling to unit length is what stops the other escape: without
                # it, driving these weights toward zero returns exactly the
                # shared-query behaviour, and weight decay pushes that way. A
                # LayerNorm cannot do this job, because its additive eps dominates
                # once the variance falls below it and the output dies anyway; the
                # division below is relative, so content dependence survives at any
                # weight scale above the clamp.
                self.content_query = nn.Sequential(
                    nn.Linear(d_model, content_query_rank, bias=False),
                    nn.Linear(content_query_rank, d_model, bias=False),
                )
        self.stack = BiMambaStack(
            MambaEncoderConfig(d_model=d_model, n_layer=n_layer, bidirectional=True, backend=backend)
        )

    def forward(
        self, residues: torch.Tensor, residue_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, length, _ = residues.shape
        if self.semantic_queries is not None and length > len(self.semantic_queries):
            raise ValueError(f"Sequence length {length} exceeds query table {len(self.semantic_queries)}")
        if self.query_mode == "content_conditioned":
            # Already per protein, so it must not be broadcast from a single row.
            content = self.content_query(residues)
            content = content / content.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            queries = self.shared_query + content
        else:
            if self.query_mode == "position_table":
                queries = self.semantic_queries[:length]
            elif self.query_mode == "shared_plus_position":
                queries = self.shared_query + self.semantic_queries[:length]
            else:
                queries = self.shared_query.expand(length, -1)
            queries = queries.unsqueeze(0).expand(batch, -1, -1)
        parts, pads = [], []
        if self.global_slots:
            # Always present, never padded: a summary slot exists whether or not a
            # given position holds a residue.
            parts.append(self.global_queries.unsqueeze(0).expand(batch, -1, -1))
            pads.append(torch.zeros(batch, self.global_slots, dtype=torch.bool,
                                    device=residues.device))
        if not self.semantic_slots_only:
            parts.append(queries)
            pads.append(~residue_mask)
        parts.append(residues)
        pads.append(~residue_mask)
        encoded = self.stack(torch.cat(parts, dim=1), torch.cat(pads, dim=1))
        valid = residue_mask.unsqueeze(-1).to(encoded.dtype)
        if self.semantic_slots_only:
            # The slots are the semantic stream, so there is no separate world
            # stream to return and nothing to mask them against: they are not
            # indexed by residue.
            semantic = encoded[:, : self.global_slots]
            patch = encoded[:, self.global_slots :]
            return semantic, patch * valid, None
        offset = self.global_slots
        world = encoded[:, :offset] if offset else None
        semantic = encoded[:, offset:offset + length]
        patch = encoded[:, offset + length:]
        # Always a triple, with `world` None when the slots are off, so callers do
        # not have to branch on the arity of a return value.
        return semantic * valid, patch * valid, world
