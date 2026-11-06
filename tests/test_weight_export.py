"""Inference exports preserve model tensors without publishing local run metadata."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def test_export_preserves_weights_and_removes_training_metadata(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/export_weights.py"
    spec = spec_from_file_location("weight_export", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "training.pt"
    destination = tmp_path / "encoder.pt"
    tensors = {"encoder.weight": torch.arange(12).reshape(3, 4)}
    torch.save({
        "model": tensors,
        "config": {
            "model_family": "psjepa", "model": {"d_model": 4},
            "masking": {}, "loss": {},
            "experiment": "local-run", "output_dir": str(tmp_path / "run"),
            "data": {"cache": str(tmp_path / "private-data")},
            "training": {"max_steps": 10},
        },
        "optimizer": {"private_state": 1}, "rng": {},
        "args": {"local_file": str(tmp_path / "settings")},
        "step": 10,
    }, source)
    module.export(source, destination)
    result = torch.load(destination, weights_only=True)
    assert torch.equal(result["model"]["encoder.weight"], tensors["encoder.weight"])
    assert result["config"] == {
        "model_family": "psjepa", "model": {"d_model": 4}, "masking": {}, "loss": {},
    }
    assert not {"optimizer", "rng", "args"} & result.keys()
    assert result["psjepa_export"]["license"] == "MIT"
    assert "source" not in result["psjepa_export"]
    assert {"optimizer", "rng"} <= set(result["psjepa_export"]["dropped"])
