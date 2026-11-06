"""Contracts of the generative token prior that a silent change would break."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from psjepa.models.token_prior import (
    TokenFlowPrior,
    TokenPriorConfig,
    apply_rotary,
    load_prior_state_dict,
    prior_flow_loss,
    rotary_tables,
    sample_tokens,
)


def build(**overrides) -> TokenFlowPrior:
    torch.manual_seed(0)
    config = TokenPriorConfig(
        d_token=32, width=64, layers=2, heads=4, max_residues=48, **overrides
    )
    prior = TokenFlowPrior(config)
    prior.config.length_values = [8, 16]
    prior.config.length_counts = [1, 1]
    return prior


def masks(batch: int = 3, length: int = 20) -> torch.Tensor:
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[0, :length] = True
    mask[1, : length // 2] = True
    mask[2, :4] = True
    return mask


@pytest.mark.parametrize("options", [{}, {"self_conditioning": True}, {"rotary_position": True}])
def test_starts_as_the_identity_map(options):


    prior = build(**options)
    mask = masks()
    out = prior(torch.randn(3, 20, 32), torch.rand(3), mask)
    assert float(out.abs().max()) == 0.0


@pytest.mark.parametrize("options", [{}, {"self_conditioning": True}, {"rotary_position": True}])
def test_padding_stays_zero_and_gradients_flow(options):
    prior = build(**options)
    mask = masks()
    tokens = torch.randn(3, 20, 32)
    prior.set_token_statistics(tokens[mask].mean(0), tokens[mask].std(0))
    loss, _ = prior_flow_loss(prior, tokens, mask, "uniform", 1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in prior.parameters())
    sampled = sample_tokens(prior, mask, steps=3)
    assert sampled.shape == tokens.shape
    assert float(sampled[2, 4:].abs().max()) == 0.0
    assert float(sampled[1, 10:].abs().max()) == 0.0


def test_padded_positions_do_not_change_real_ones():
    """A shorter chain padded out to a longer window must decode the same."""
    prior = build(rotary_position=True)
    # Non-zero output requires a trained-looking model; perturb the zero-init.
    with torch.no_grad():
        for parameter in prior.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    tokens = torch.randn(1, 20, 32)
    short_mask = torch.zeros(1, 20, dtype=torch.bool)
    short_mask[0, :12] = True
    time = torch.full((1,), 0.3)
    full = prior(tokens, time, short_mask)

    other = tokens.clone()
    other[0, 12:] = torch.randn(8, 32)
    again = prior(other, time, short_mask)
    assert torch.allclose(full[0, :12], again[0, :12], atol=1e-5)


def test_rotary_is_relative():
    """The same q and k at two places with one offset must score the same."""
    cos, sin = rotary_tables(32, 8, 10000.0, torch.device("cpu"), torch.float32)
    q = torch.randn(8)
    k = torch.randn(8)

    def score(position_q: int, position_k: int) -> torch.Tensor:
        field = torch.zeros(1, 1, 32, 8)
        field[0, 0, position_q] = q
        rotated_q = apply_rotary(field, cos, sin)[0, 0, position_q]
        field = torch.zeros(1, 1, 32, 8)
        field[0, 0, position_k] = k
        rotated_k = apply_rotary(field, cos, sin)[0, 0, position_k]
        return (rotated_q * rotated_k).sum()

    assert torch.allclose(score(5, 9), score(15, 19), atol=1e-4)
    assert torch.allclose(score(2, 30), score(0, 28), atol=1e-4)
    # A different offset must give a different score, or the embedding is inert.
    assert not torch.allclose(score(5, 9), score(5, 20), atol=1e-3)


def test_optional_features_change_the_parameter_count_only_when_asked():
    plain = sum(p.numel() for p in build().parameters())
    rotary = sum(p.numel() for p in build(rotary_position=True).parameters())
    conditioned = sum(p.numel() for p in build(self_conditioning=True).parameters())
    assert rotary == plain, "rotary adds no parameters"
    assert conditioned > plain, "self-conditioning adds an input projection"


def test_self_conditioning_is_refused_when_not_built():
    prior = build()
    with pytest.raises(ValueError):
        prior(torch.randn(3, 20, 32), torch.rand(3), masks(),
              self_condition=torch.randn(3, 20, 32))


def test_checkpoints_written_before_the_attention_rewrite_still_load():
    'Old names must map onto the new module *and give the same numbers*.'
    from psjepa.models.token_prior import SelfAttention

    torch.manual_seed(0)
    width, heads, length = 16, 4, 7
    reference = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True).eval()
    legacy = {
        "in_proj_weight": reference.in_proj_weight.detach().clone(),
        "in_proj_bias": reference.in_proj_bias.detach().clone(),
        "out_proj.weight": reference.out_proj.weight.detach().clone(),
        "out_proj.bias": reference.out_proj.bias.detach().clone(),
    }
    rewritten = SelfAttention(width, heads, 0.0).eval()
    rewritten.load_state_dict(legacy, strict=True)

    values = torch.randn(2, length, width)
    pad = torch.zeros(2, length, dtype=torch.bool)
    pad[1, 5:] = True  # True marks padding for both implementations
    with torch.no_grad():
        expected, _ = reference(values, values, values, key_padding_mask=pad,
                                need_weights=False)
        actual = rewritten(values, pad)
    assert torch.allclose(expected[0], actual[0], atol=1e-5)
    assert torch.allclose(expected[1, :5], actual[1, :5], atol=1e-5)

def test_pre_rotary_checkpoints_still_load():
    """Checkpoints written before `SelfAttention` replaced `nn.MultiheadAttention`."""
    prior = build()
    state = prior.state_dict()
    legacy = {}
    for key, value in state.items():
        key = key.replace(".attention.qkv.weight", ".attention.in_proj_weight")
        key = key.replace(".attention.qkv.bias", ".attention.in_proj_bias")
        key = key.replace(".attention.projection.", ".attention.out_proj.")
        legacy[key] = value
    assert any("in_proj_weight" in k for k in legacy), "test built no legacy names"
    fresh = build()
    load_prior_state_dict(fresh, legacy)
    for key, value in prior.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key
    # A checkpoint already in the new naming must load unchanged.
    load_prior_state_dict(build(), state)
    # A genuinely missing parameter must still be an error.
    broken = dict(legacy)
    broken.pop(next(k for k in broken if "in_proj_weight" in k))
    with pytest.raises(RuntimeError):
        load_prior_state_dict(build(), broken)


def test_distogram_head_is_optional_and_trains():
    """The auxiliary tertiary term: off by default, and only active with coordinates."""
    plain = build()
    assert plain.distogram is None
    with_head = build(distogram_weight=0.5, distogram_bins=16, distogram_max=20.0)
    assert with_head.distogram is not None
    mask = masks()
    tokens = torch.randn(3, 20, 32)
    with_head.set_token_statistics(tokens[mask].mean(0), tokens[mask].std(0))
    coordinates = torch.randn(3, 20, 4, 3) * 8.0


    _, bare = prior_flow_loss(with_head, tokens, mask, "uniform", 0.0)
    assert "prior_distogram_loss" not in bare

    loss, metrics = prior_flow_loss(
        with_head, tokens, mask, "uniform", 0.0, None, coordinates
    )
    assert "prior_distogram_loss" in metrics
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in with_head.distogram.parameters())


def test_distogram_targets_are_symmetric_and_bounded():
    from psjepa.models.token_prior import distogram_targets
    coordinates = torch.randn(2, 12, 4, 3) * 10.0
    target = distogram_targets(coordinates, bins=16, maximum=20.0)
    assert target.shape == (2, 12, 12)
    assert torch.equal(target, target.transpose(1, 2)), "distance map must be symmetric"
    assert int(target.max()) <= 15 and int(target.min()) >= 0

    assert int(target[0].diagonal().max()) == 0


def test_distogram_head_output_is_symmetric():
    prior = build(distogram_weight=1.0, distogram_bins=8)
    tokens = torch.randn(2, 9, 32)
    logits = prior.distogram(tokens)
    assert logits.shape == (2, 9, 9, 8)
    assert torch.allclose(logits, logits.transpose(1, 2), atol=1e-6)


def test_every_config_field_is_serialised():
    """A config field missing from `to_dict` rebuilds the wrong architecture."""
    import dataclasses
    prior = build(self_conditioning=True, rotary_position=True, distogram_weight=0.25)
    stored = prior.to_dict()
    expected = {f.name for f in dataclasses.fields(TokenPriorConfig)}
    assert set(stored) == expected, f"missing from to_dict: {expected - set(stored)}"

    rebuilt = TokenFlowPrior.from_dict(stored)
    assert set(rebuilt.state_dict()) == set(prior.state_dict())


def test_a_distogram_checkpoint_loads_into_a_sampling_model():
    """Sampling builds no auxiliary head, but must still read a trained checkpoint."""
    trained = build(distogram_weight=0.5)
    assert trained.distogram is not None
    sampler = build()
    assert sampler.distogram is None
    load_prior_state_dict(sampler, trained.state_dict())
    for key, value in sampler.state_dict().items():
        assert torch.equal(trained.state_dict()[key], value), key


def test_checkpoint_predating_a_config_field_still_rebuilds():
    """A checkpoint whose `prior_config` lacks a field added later."""
    from psjepa.models.token_prior import prior_from_checkpoint

    trained = build(distogram_weight=0.5, distogram_bins=16, distogram_width=32)
    stored = trained.to_dict()
    for key in [k for k in stored if "distogram" in k]:
        stored.pop(key)
    state = {
        "prior_config": stored,
        "config": {"model": {"distogram_weight": 0.5, "distogram_bins": 16,
                             "distogram_width": 32}},
        "prior": trained.state_dict(),
    }
    rebuilt = prior_from_checkpoint(state)
    assert rebuilt.distogram is not None, "the head was not recovered"
    load_prior_state_dict(rebuilt, state["prior"])
    assert rebuilt.config.distogram_bins == 16

    # A checkpoint with no such field anywhere must still rebuild, at the default.
    plain = build()
    bare = {"prior_config": plain.to_dict(), "prior": plain.state_dict()}
    assert prior_from_checkpoint(bare).distogram is None
