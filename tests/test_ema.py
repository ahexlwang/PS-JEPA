import pytest
import torch
from torch import nn

from psjepa.training.ema import cosine_ema_momentum, resolve_ema_start, update_ema


def test_ema_updates_parameters_and_float_buffers():
    context = nn.BatchNorm1d(3)
    target = nn.BatchNorm1d(3)
    with torch.no_grad():
        context.weight.fill_(2)
        target.weight.fill_(0)
        context.running_mean.fill_(4)
        target.running_mean.fill_(0)
        context.num_batches_tracked.fill_(9)
    update_ema(target, context, 0.75)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))
    assert torch.allclose(target.running_mean, torch.full_like(target.running_mean, 1.0))
    assert target.num_batches_tracked.item() == 9
    assert cosine_ema_momentum(0, 100, 0.996) == 0.996
    assert cosine_ema_momentum(100, 100, 0.996) == 1.0


SCREENING_WINDOW_FRACTION = 250 / 20_000


def test_resolve_ema_start_reproduces_the_screening_default():
    assert resolve_ema_start(20_000, SCREENING_WINDOW_FRACTION) == pytest.approx(0.996)


@pytest.mark.parametrize("max_steps", [625, 1_000, 1_250, 20_000, 100_000])
def test_teacher_window_is_a_constant_share_of_the_run(max_steps):
    start = resolve_ema_start(max_steps, SCREENING_WINDOW_FRACTION)
    window = 1.0 / (1.0 - start)
    assert window / max_steps == pytest.approx(SCREENING_WINDOW_FRACTION)


def test_short_runs_do_not_produce_a_window_longer_than_themselves():
    # A 620-step run under the fixed 0.996 default averaged over 250 steps, 40%
    # of the whole run, leaving the teacher near initialization for most of it.
    start = resolve_ema_start(625, SCREENING_WINDOW_FRACTION)
    assert 1.0 / (1.0 - start) < 625
    assert start < 0.996


def test_degenerate_windows_clamp_instead_of_going_negative():
    assert resolve_ema_start(3, SCREENING_WINDOW_FRACTION) == 0.0
    with pytest.raises(ValueError):
        resolve_ema_start(625, 0.0)

