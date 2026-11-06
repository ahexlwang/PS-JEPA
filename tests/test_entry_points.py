"""The shell entry points have to name files that exist and arguments programs accept."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = sorted(ROOT.glob("scripts/**/*.sh"))


@pytest.mark.parametrize("script", SHELL, ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_scripts_parse(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SHELL, ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_scripts_only_name_siblings_that_exist(script: Path):
    """A rename that misses a caller leaves `exit 127` for whoever runs it next."""
    text = script.read_text()
    referenced = set(re.findall(r'\$(?:PROJECT|HERE|SCRIPTS)[\w/${}."-]*/([\w.]+\.(?:sh|py))', text))
    # `./src/script/run_supervised_task.py` is the external benchmark's entry point, reached
    # after `cd "$STB_ROOT"`, so it is deliberately not a file in this repository.
    referenced -= {"run_supervised_task.py"}
    for name in sorted(referenced):
        candidates = list(ROOT.glob(f"scripts/**/{name}"))
        assert candidates, f"{script.relative_to(ROOT)} names {name}, which is not in this repository"


def _parser_flags(script: Path) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', script.read_text()))


@pytest.mark.parametrize("script", ["train_encoder.py", "train_decoder.py", "train_prior.py"])
def test_training_entry_points_accept_config_and_resume(script):
    flags = _parser_flags(ROOT / "scripts" / script)
    assert {"--config", "--resume"} <= flags


READOUT_VARIABLE = "PSJEPA_READOUT"


def test_the_runners_and_the_loader_read_the_same_readout_variable():
    """When they disagreed, switching the readout changed the model but not the cache key."""
    loader = (ROOT / "src/psjepa/tokenization/semantic_tokenizer.py").read_text()
    assert READOUT_VARIABLE in loader
    for runner in ("run_test.sh", "run_validation.sh"):
        text = (ROOT / "scripts/structtokenbench" / runner).read_text()
        assert READOUT_VARIABLE in text, f"{runner} does not read {READOUT_VARIABLE}"
        assert f'export {READOUT_VARIABLE}=' in text, (
            f"{runner} must export the readout it tagged its paths with, or the loader can "
            "build a different model than the paths describe"
        )


def test_the_runners_default_to_the_same_readout_as_the_loader():
    """An untagged results directory holds teacher numbers, so teacher is the default."""
    loader = (ROOT / "src/psjepa/tokenization/semantic_tokenizer.py").read_text()
    assert '"PSJEPA_FLEX_READOUT", "teacher"' in loader
    for runner in ("run_test.sh", "run_validation.sh"):
        text = (ROOT / "scripts/structtokenbench" / runner).read_text()
        assert ":-teacher}" in text, f"{runner} does not default to the loader's readout"


@pytest.mark.parametrize("runner", ["run_test.sh", "run_validation.sh"], ids=lambda s: s)
def test_both_runners_take_a_representation(runner: str):
    """The validation runner ignored its third argument and asked for codebook ids."""
    text = (ROOT / "scripts/structtokenbench" / runner).read_text()
    assert re.search(r"^REPRESENTATION=\$\{3", text, flags=re.M), f"{runner} ignores argument 3"
    assert 'data.use_continuous="$USE_CONTINUOUS"' in text
    assert '+psjepa_representation="$REPRESENTATION"' in text
    assert "$REPRESENTATION/$TASK.json" in text, "results must not collide across representations"


def test_the_benchmark_gate_runs_before_the_benchmark():
    """An unpatched StructTokenBench should fail in a second, not minutes into a job."""
    for runner in ("run_test.sh", "run_validation.sh"):
        text = (ROOT / "scripts/structtokenbench" / runner).read_text()
        gate = text.index("check_integration.py")
        work = text.index("run_supervised_task.py")
        assert gate < work, f"{runner} starts the benchmark before checking the integration"


def test_the_integration_checker_rejects_a_directory_that_is_not_a_checkout(tmp_path):
    result = subprocess.run(
        ["python", str(ROOT / "scripts/structtokenbench/check_integration.py"),
         "--root", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "does not look like a StructTokenBench checkout" in result.stderr


SHELL_FILES = sorted(ROOT.glob("scripts/**/*.sh"))


@pytest.mark.parametrize("script", SHELL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_shell_file_declares_its_shell(script: Path):
    """shellcheck refuses to guess, and CI fails on the guess it cannot make (SC2148)."""
    first_lines = script.read_text().splitlines()[:3]
    has_shebang = first_lines and first_lines[0].startswith("#!")
    has_directive = any(line.startswith("# shellcheck shell=") for line in first_lines)
    assert has_shebang or has_directive, (
        f"{script.name} declares neither a shebang nor `# shellcheck shell=bash`"
    )


@pytest.mark.parametrize("script", SHELL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_executable_bit_matches_how_a_script_is_used(script: Path):
    """A sourced library that is executable invites someone to run it and get nothing."""
    import os

    executable = os.access(script, os.X_OK)
    has_shebang = script.read_text().startswith("#!")
    assert executable == has_shebang, (
        f"{script.name}: executable={executable} but shebang={has_shebang}"
    )
