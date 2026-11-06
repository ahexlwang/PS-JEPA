import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "train_decoder", Path("scripts/train_decoder.py")
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
drop_condition = module.drop_condition


def test_condition_mismatch_rolls_scaffold_but_preserves_motif_tokens():
    condition = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)
    motif = torch.tensor(
        [
            [False, True, False, False, True],
            [True, False, False, True, False],
            [False, False, True, False, True],
        ]
    )
    config = {"motif": {"condition_mismatch": 1.0, "condition_dropout": 0.0}}

    result = drop_condition(condition, config, preserve_mask=motif)
    expected = torch.roll(condition, shifts=1, dims=0)
    expected = torch.where(motif.unsqueeze(-1), condition, expected)

    assert torch.equal(result, expected)


def test_condition_mismatch_keeps_legacy_all_token_roll_without_mask():
    condition = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    config = {"motif": {"condition_mismatch": 1.0, "condition_dropout": 0.0}}

    result = drop_condition(condition, config)

    assert torch.equal(result, torch.roll(condition, shifts=1, dims=0))
