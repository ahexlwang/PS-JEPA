from __future__ import annotations

from psjepa.environment import getenv

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from psjepa.models import PSJEPA


PSJEPA_MODEL_FAMILIES = ("psjepa", "biojepa", "flexjepa", "bjepa")


def _mark_legacy_backbone_input(config: dict, weights: dict) -> None:
    "Detect coordinate-plus-mask backbone inputs from the saved projection width."
    model = config.get("model")
    if not isinstance(model, dict) or model.get("input_mode", "backbone") != "backbone":
        return
    for name, tensor in weights.items():
        if name.endswith("projection.net.0.weight") and tensor.ndim == 2:
            model["backbone_presence_bits"] = int(tensor.shape[1]) == 16
            return
    for name, tensor in weights.items():
        if name.endswith("upstream.context_encoder.projection.net.0.weight") and tensor.ndim == 2:
            up = config.get("upstream", {}).get("model")
            if isinstance(up, dict) and up.get("input_mode") == "backbone":
                up["backbone_presence_bits"] = int(tensor.shape[1]) == 16
            return


class SemanticTokenizer:
    def __init__(self, model: PSJEPA, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.eval().requires_grad_(False).to(self.device)
        # PSJEPA has no quantiser, so there is no id reserved for padding. The
        # continuous wrappers never read this, and the discrete ones are refused
        # below, so 0 is a value that cannot be mistaken for a real codebook size.
        self.pad_token_id = 0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        *,
        flex_readout: str | None = None,
    ) -> "SemanticTokenizer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = deepcopy(state["config"])
        _mark_legacy_backbone_input(config, state["model"])
        family = config.get("model_family")
        if family not in PSJEPA_MODEL_FAMILIES:
            raise RuntimeError(
                f"{path} declares model_family={family!r}. This release loads the continuous "
                f"PSJEPA encoder only; accepted values are {PSJEPA_MODEL_FAMILIES}. The "
                "checkpoint must contain a continuous PS-JEPA encoder."
            )
        model = PSJEPA.from_dict(config["model"], config["masking"], config["loss"])
        model.load_state_dict(state["model"], strict=True)
        readout = (
            flex_readout
            if flex_readout is not None
            else getenv("PSJEPA_READOUT", getenv("PSJEPA_FLEX_READOUT", "teacher"))
        )
        if readout not in ("teacher", "student"):
            raise ValueError(
                f"PSJEPA readout must be 'teacher' or 'student', got {readout!r}"
            )
        model.readout = readout
        print(f"PSJEPA readout encoder: {readout}", flush=True)
        return cls(model, device)

    @torch.inference_mode()
    def encode_batch(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_type: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.model.tokenize(
            coords.to(self.device),
            atom_mask.to(self.device),
            residue_mask.to(self.device),
            residue_type.to(self.device) if residue_type is not None else None,
        )

    @torch.inference_mode()
    def encode_chunks(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_type: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        maximum = self.model.config.max_residues
        if getattr(self.model.config, "semantic_slots_only", False):
            # A fixed-slot stream summarises whatever it is shown, so chunking would
            # produce L slots per chunk and a token count that grows with length —
            # the property this mode exists to remove. Encode one window instead and
            # truncate, which is visible in the returned length rather than silent.
            window = coords[:maximum].unsqueeze(0)
            window_mask = atom_mask[:maximum].unsqueeze(0)
            window_type = (
                residue_type[:maximum].unsqueeze(0) if residue_type is not None else None
            )
            present = torch.ones(1, window.shape[1], dtype=torch.bool)
            output = self.encode_batch(window, window_mask, present, window_type)
            return {key: value[0].detach() for key, value in output.items()}
        pieces: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(coords), maximum):
            chunk_coords = coords[start : start + maximum].unsqueeze(0)
            chunk_mask = atom_mask[start : start + maximum].unsqueeze(0)
            chunk_type = (
                residue_type[start : start + maximum].unsqueeze(0)
                if residue_type is not None
                else None
            )
            residue_mask = torch.ones(1, len(chunk_coords[0]), dtype=torch.bool)
            output = self.encode_batch(chunk_coords, chunk_mask, residue_mask, chunk_type)
            for key, value in output.items():
                pieces.setdefault(key, []).append(value[0].detach())
        return {key: torch.cat(value, dim=0) for key, value in pieces.items()}

    def _require_quantizer(self) -> None:
        raise RuntimeError(
            f"{type(self.model).__name__} has no codebook: it can only be read as "
            "continuous_semantic. Use WrappedPSJEPAContinuousTokenizer."
        )

    def get_num_tokens(self) -> int:
        self._require_quantizer()

    def get_codebook_embedding(self) -> torch.Tensor:
        """Discrete codebook readout is unavailable for this continuous encoder."""
        self._require_quantizer()


BIOJEPA_MODEL_FAMILIES = PSJEPA_MODEL_FAMILIES
