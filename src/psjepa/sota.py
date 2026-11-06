'Default PS-JEPA encoder checkpoint and representation settings.'
from pathlib import Path
from psjepa.paths import resolve_weights

SOTA_ENCODER_NAME = "PS-JEPA (512 channels, 8 layers)"
SOTA_ENCODER_WEIGHTS = Path("encoder/psjepa_d512.pt")
SOTA_ENCODER_D_TOKEN = 512
SOTA_ENCODER_PARAMS = 13_981_939
SOTA_ENCODER_MASKING = "context 0.85 / target 0.125, four blocks, exclude overlap"
SOTA_ENCODER_STEPS = 35_000
SOTA_ENCODER_BATCH_SIZE = 64
SOTA_ENCODER_ACCUMULATION = 8
SOTA_ENCODER_EFFECTIVE_BATCH = 512
SOTA_ENCODER_READOUT = "teacher"
SOTA_SOURCE_SHA256 = "cc47c892b4069bed990a24cbf4bc7489297096427669b339cc1c53968c5df83b"

def encoder_checkpoint() -> Path:
    """Resolve the current encoder under PSJEPA_WEIGHTS."""
    return resolve_weights(SOTA_ENCODER_WEIGHTS)
