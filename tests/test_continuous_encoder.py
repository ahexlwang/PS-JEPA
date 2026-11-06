"""Reject unsupported quantized configurations before constructing a model."""
import pytest

from psjepa.models import PSJEPAConfig
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer


@pytest.mark.parametrize(
    "model,masking",
    [
        ({"quantize": True}, {}),
        ({"quantize_context": True}, {}),
    ],
)
def test_quantized_training_is_rejected(model, masking):
    with pytest.raises(ValueError, match="continuous PS-JEPA encoders only"):
        PSJEPAConfig.from_dict(model, masking, {})


def test_unsupported_masking_is_rejected():
    with pytest.raises(ValueError, match="Unsupported masking scheme"):
        PSJEPAConfig.from_dict({}, {"scheme": "quantized"}, {})


def test_continuous_tokenizer_does_not_read_an_external_codebook(model, monkeypatch):
    monkeypatch.setenv("PSJEPA_CODEBOOK_TABLE", "/nonexistent/table.pt")
    tokenizer = SemanticTokenizer(model)
    with pytest.raises(RuntimeError, match="has no codebook"):
        tokenizer.get_codebook_embedding()
    with pytest.raises(RuntimeError, match="has no codebook"):
        tokenizer.get_num_tokens()
