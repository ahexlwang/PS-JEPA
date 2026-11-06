"""Stopping on the curve rather than on a step counter."""
from __future__ import annotations

from psjepa.training.plateau import PlateauStopper


def feed(stopper, values, start=0, step_size=100, collapsed=False):
    return [
        stopper.update(start + index * step_size, value, collapsed)
        for index, value in enumerate(values)
    ]


def test_a_descending_curve_is_never_stopped():
    stopper = PlateauStopper(patience=3, window=3, min_delta=0.01)
    actions = feed(stopper, [2.0 - 0.05 * index for index in range(30)])
    assert set(actions) == {"continue"}


def test_a_flat_curve_decays_then_stops():
    stopper = PlateauStopper(patience=2, window=2, min_delta=0.01, lr_reductions=1, lr_factor=0.5)
    actions = feed(stopper, [1.0] * 12)
    assert "reduce" in actions
    assert actions.index("reduce") < actions.index("stop")
    assert stopper.level == 0.5


def test_the_decay_happens_before_the_stop_is_allowed():
    """A plateau at one step size is not a plateau, only a reason to try a smaller one."""
    stopper = PlateauStopper(patience=1, window=1, min_delta=0.01, lr_reductions=3)
    actions = feed(stopper, [1.0] * 8)
    assert actions.count("reduce") == 3
    assert actions.index("stop") > max(
        index for index, action in enumerate(actions) if action == "reduce"
    )


def test_noise_of_the_size_the_region_arms_showed_does_not_stop_a_falling_curve():
    """Real numbers: consecutive points came in at 1.109, 0.916, 1.141, 0.982, 1.046."""
    noise = [0.109, -0.084, 0.141, -0.018, 0.046, -0.061, 0.093, -0.102]
    values = [1.30 - 0.02 * index + noise[index % len(noise)] for index in range(40)]
    stopper = PlateauStopper(patience=4, window=3, min_delta=0.005)
    assert "stop" not in feed(stopper, values)


def test_min_steps_holds_the_stop_through_the_collapse_escape():
    """Early training rises out of a collapse, so it looks flat then worse."""
    stopper = PlateauStopper(patience=1, window=1, min_delta=0.01, min_steps=1000,
                             lr_reductions=0)
    actions = feed(stopper, [1.0] * 8, step_size=100)
    assert "stop" not in actions
    assert stopper.update(1000, 1.0) == "stop"


def test_a_collapsed_point_is_ignored_entirely():
    """Its loss is low because the representation went flat, not because it learned."""
    stopper = PlateauStopper(patience=2, window=2, min_delta=0.01)
    assert feed(stopper, [0.01] * 10, collapsed=True) == ["continue"] * 10
    assert stopper.best == float("inf")
    assert stopper.waited == 0


def test_state_is_reportable_and_json_safe():
    stopper = PlateauStopper()
    assert stopper.state()["plateau_best"] is None
    stopper.update(0, 1.0)
    assert set(stopper.state()) == {"plateau_best", "plateau_waited", "plateau_reductions", "lr_level"}


def load_train_module():
    """`scripts/train_encoder.py` is a script, not a package, so it is loaded by path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_encoder.py"
    spec = importlib.util.spec_from_file_location("train_encoder_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_cosine_schedule_is_untouched():
    train = load_train_module()
    assert train.learning_rate(0, 1000, 100, 1.0, "cosine") == 0.01
    assert train.learning_rate(1000, 1000, 100, 1.0, "cosine") < 1e-9
    mid = train.learning_rate(550, 1000, 100, 1.0, "cosine")
    assert 0.49 < mid < 0.51


def test_the_plateau_schedule_holds_the_peak_and_takes_its_level_from_the_stopper():
    train = load_train_module()
    assert train.learning_rate(50, 1000, 100, 1.0, "plateau") == 0.51  # still warming up
    assert train.learning_rate(500, 1000, 100, 1.0, "plateau") == 1.0
    assert train.learning_rate(9999, 1000, 100, 1.0, "plateau") == 1.0
    assert train.learning_rate(500, 1000, 100, 1.0, "plateau", 0.3) == 0.3


def test_a_held_momentum_must_be_capped_below_one():
    """At its horizon the cosine momentum is exactly 1.0, which freezes the teacher."""
    import torch
    from torch import nn

    from psjepa.training.ema import cosine_ema_momentum, update_ema

    assert cosine_ema_momentum(3125, 3125, 0.9744) == 1.0

    student, teacher = nn.Linear(4, 4), nn.Linear(4, 4)
    with torch.no_grad():
        student.weight.fill_(1.0)
        teacher.weight.fill_(0.0)
    update_ema(teacher, student, 1.0)
    assert float(teacher.weight.abs().sum()) == 0.0        # frozen

    update_ema(teacher, student, min(1.0, 0.9999))
    assert float(teacher.weight.abs().sum()) > 0.0         # still moving


def protocol():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_training_protocol.py"
    spec = importlib.util.spec_from_file_location("protocol_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plateau_config(**training):
    base = {
        "batch_size": 512, "accumulation_steps": 1, "max_steps": 20000,
        "warmup_steps": 156, "schedule": "plateau", "ema_horizon": 3125,
        "ema_max": 0.9999, "plateau": {"patience": 6},
    }
    base.update(training)
    return {
        "model": {"levels": [4, 4, 4, 4], "query_mode": "content_conditioned"},
        "training": base,
        "loss": {"code_warmup_steps": 312},
    }


