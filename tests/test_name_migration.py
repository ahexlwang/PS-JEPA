"""The rename must preserve checkpoint classes and explicitly selected readouts/roots."""
import os
import subprocess
import sys
from pathlib import Path


def test_old_namespace_resolves_to_the_same_classes():
    from psjepa.models import PSJEPA, PSJEPAConfig
    from biojepa.models import BioJEPA, BioJEPAConfig
    from biojepa.models.biojepa import BioJEPA as direct
    from psjepa.models.mamba_encoder import BiMambaStack
    from biojepa.models.mamba_encoder import BiMambaStack as old_stack
    assert PSJEPA is BioJEPA is direct
    assert PSJEPAConfig is BioJEPAConfig
    assert BiMambaStack is old_stack


def test_environment_alias_preserves_explicit_settings(monkeypatch):
    from psjepa.environment import getenv
    monkeypatch.delenv('PSJEPA_READOUT', raising=False)
    monkeypatch.setenv('BIOJEPA_READOUT', 'student')
    assert getenv('PSJEPA_READOUT', 'teacher') == 'student'
    monkeypatch.setenv('PSJEPA_READOUT', 'teacher')
    assert getenv('PSJEPA_READOUT', 'student') == 'teacher'


def test_legacy_shell_root_is_not_overwritten_by_local_defaults(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {k:v for k,v in os.environ.items() if not k.startswith(('PSJEPA_', 'BIOJEPA_'))}
    env['BIOJEPA_OUTPUTS'] = str(tmp_path/'selected')
    result = subprocess.run(['bash','-c','source scripts/env.sh; echo "$PSJEPA_OUTPUTS"'],
                            cwd=root, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == env['BIOJEPA_OUTPUTS']
