"""What has to hold for this checkout to be usable by someone who did not build it."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from psjepa import paths, sota
from psjepa.config import load_config
from psjepa.tokenization.semantic_tokenizer import PSJEPA_MODEL_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(ROOT.glob("configs/*/*.yaml"))


def test_there_are_configs_to_check():
    assert CONFIGS, "configs/ is empty"


def test_the_project_root_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.project_root() == ROOT


def test_the_roots_follow_their_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("PSJEPA_WEIGHTS", str(tmp_path / "w"))
    monkeypatch.setenv("PSJEPA_CACHE", str(tmp_path / "c"))
    monkeypatch.setenv("PSJEPA_OUTPUTS", str(tmp_path / "o"))
    assert paths.weights_root() == tmp_path / "w"
    assert paths.resolve_weights("encoder/x.pt") == tmp_path / "w" / "encoder" / "x.pt"
    assert paths.resolve_cache("cath_backbone_split") == tmp_path / "c" / "cath_backbone_split"
    assert paths.resolve_output("decoder/run") == tmp_path / "o" / "decoder" / "run"


def test_an_absolute_path_in_a_config_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("PSJEPA_WEIGHTS", str(tmp_path / "w"))
    assert paths.resolve_weights("/somewhere/else.pt") == Path("/somewhere/else.pt")


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_config_names_an_absolute_path(config: Path):
    """A tracked config must be portable: absolute paths belong in the environment."""
    raw = config.read_text()
    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip()
        assert not stripped.startswith("/"), line
        for token in stripped.split():
            assert not token.startswith("/home/"), line


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_config_loads_and_resolves(config: Path, tmp_path, monkeypatch):
    monkeypatch.setenv("PSJEPA_WEIGHTS", str(tmp_path / "w"))
    monkeypatch.setenv("PSJEPA_CACHE", str(tmp_path / "c"))
    monkeypatch.setenv("PSJEPA_OUTPUTS", str(tmp_path / "o"))
    resolved = load_config(config)
    assert resolved, f"{config} is empty"
    for key in ("upstream_checkpoint", "decoder_checkpoint", "output_dir"):
        if key in resolved:
            assert Path(resolved[key]).is_absolute(), f"{config}: {key} did not resolve"
            assert str(tmp_path) in resolved[key], f"{config}: {key} ignored its root"


@pytest.mark.parametrize(
    "config", [p for p in CONFIGS if p.parent.name == "encoder"], ids=lambda p: p.name
)
def test_encoder_configs_declare_the_current_model_family(config: Path):
    raw = yaml.safe_load(config.read_text())
    assert raw["model_family"] == PSJEPA_MODEL_FAMILIES[0] == "psjepa"
    assert raw["model"]["quantize"] is False, "this release has no quantiser"


@pytest.mark.parametrize(
    "config", [p for p in CONFIGS if p.parent.name in ("decoder", "prior")],
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_downstream_configs_take_the_reference_encoder(config: Path):
    'Downstream configurations use the distributed encoder.'
    raw = yaml.safe_load(config.read_text())
    assert raw["upstream_checkpoint"] == str(sota.SOTA_ENCODER_WEIGHTS)


def _released(path: Path) -> Path:
    resolved = paths.resolve_weights(path)
    if not resolved.is_file():
        pytest.skip(f"{resolved} is not present; see README.md")
    return resolved


@pytest.mark.parametrize(
    "weights,width,parameters",
    [
        (sota.SOTA_ENCODER_WEIGHTS, sota.SOTA_ENCODER_D_TOKEN, sota.SOTA_ENCODER_PARAMS),
    ],
    ids=["sota_new"],
)
def test_the_released_encoders_load_strictly(weights, width, parameters):
    import torch

    from psjepa.models import PSJEPA

    state = torch.load(_released(weights), map_location="cpu", weights_only=False)
    config = state["config"]
    assert config["model_family"] in PSJEPA_MODEL_FAMILIES
    assert config["model"]["d_model"] == width
    model = PSJEPA.from_dict(config["model"], config["masking"], config["loss"])
    model.load_state_dict(state["model"], strict=True)
    assert sum(p.numel() for p in model.context_encoder.parameters()) == parameters


@pytest.mark.parametrize(
    "weights", [sota.SOTA_ENCODER_WEIGHTS], ids=["sota_new"]
)
def test_a_released_file_records_where_it_came_from(weights):
    import torch

    state = torch.load(_released(weights), map_location="cpu", weights_only=False)
    export = state.get("psjepa_export", state.get("biojepa_export"))
    assert export["source_sha256"]
    assert "optimizer" in export["dropped"], "the optimizer state should not ship"
    assert export["weight_keys"] == ["model"]


def test_the_environment_is_not_carrying_a_stale_root():
    """A leftover PSJEPA_* pointing outside this checkout makes every other test lie."""
    for name in ("PSJEPA_WEIGHTS", "PSJEPA_CACHE", "PSJEPA_OUTPUTS"):
        value = os.environ.get(name)
        if value:
            assert Path(value).is_absolute(), f"{name}={value} is not absolute"


def test_a_moved_upstream_is_found_but_only_when_its_hash_matches(tmp_path, monkeypatch):
    """A distributed decoder records an absolute path to an encoder the recipient lacks."""
    from psjepa.paths import file_sha256, resolve_recorded_checkpoint

    monkeypatch.setenv("PSJEPA_WEIGHTS", str(tmp_path))
    (tmp_path / "encoder").mkdir()
    real = tmp_path / "encoder" / "some_encoder.pt"
    real.write_bytes(b"the weights this model was trained on")
    digest = file_sha256(real)
    recorded = "/a/machine/that/no/longer/exists/some_encoder.pt"

    assert resolve_recorded_checkpoint(recorded, digest) == real

    with pytest.raises(RuntimeError):
        resolve_recorded_checkpoint(recorded, None)          # no hash to check against
    with pytest.raises(RuntimeError):
        resolve_recorded_checkpoint(recorded, "0" * 64)      # a different encoder
    with pytest.raises(FileNotFoundError):
        resolve_recorded_checkpoint("/gone/absent.pt", digest)

    present = tmp_path / "present.pt"
    present.write_bytes(b"x")
    assert resolve_recorded_checkpoint(present, None) == present   # still there: used as is


def test_env_local_supplies_defaults_and_does_not_override_the_caller(tmp_path):
    """`PSJEPA_OUTPUTS=... command` must reach the run."""
    import subprocess

    env_local = ROOT / ".env.local"
    if not env_local.is_file():
        pytest.skip(".env.local is machine-local and absent here")

    def outputs(**environment):
        result = subprocess.run(
            ["bash", "-c", f'source "{ROOT}/scripts/env.sh"; echo "$PSJEPA_OUTPUTS"'],
            capture_output=True, text=True, check=True,
            env={**os.environ, **environment},
        )
        return result.stdout.strip()

    override = str(tmp_path / "elsewhere")
    assert outputs(PSJEPA_OUTPUTS=override) == override
