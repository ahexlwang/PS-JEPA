"""StructTokenBench-compatible residue tokenizer adapter."""
from __future__ import annotations

from psjepa.environment import getenv

import os
from typing import Literal

import numpy as np
import torch

from .semantic_tokenizer import SemanticTokenizer


class WrappedPSJEPATokenizer:
    # `slot_ids` and `slot_embeddings` exist only for a checkpoint trained with
    # `semantic_slots_only`. They return L tokens for the whole chain rather than one
    # per residue, so only protein-level tasks can read them; a residue-level task
    # will fail on label alignment, loudly, which is the intended behaviour.
    REPRESENTATIONS = {
        "continuous_semantic", "quantized_semantic", "discrete_ids",
        "slot_ids", "slot_embeddings",
    }
    SLOT_REPRESENTATIONS = {"slot_ids", "slot_embeddings"}
    DISCRETE_REPRESENTATIONS = {"discrete_ids", "slot_ids"}
    # StructTokenBench builds its head at the tokenizer's own width. Its task configs
    # ship `d_model: null # hidden dimention from the tokenizer embedding dim`, and each
    # baseline supplies its own: FoldSeek 2, ProTokens 32, ESM3 128, AIDO 384,
    # AminoAseed 1024. So the default here is to emit the encoder's native width and let
    # the head be built to match -- `scripts/checkpoint_width.py` reads that width off the
    # same checkpoint, so the shell cannot pass one that disagrees.
    AMINOASEED_PROBE_DIM = 1024

    def __init__(
        self,
        device: str | torch.device = "cpu",
        checkpoint_path: str | None = None,
        ckpt_name: str | None = None,
        representation: str = "discrete_ids",
        **_: object,
    ) -> None:
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if representation not in self.REPRESENTATIONS:
            raise ValueError(f"Unsupported PS-JEPA representation: {representation}")
        self.tokenizer = SemanticTokenizer.from_checkpoint(checkpoint_path, device)


        if (getattr(self.tokenizer.model, "quantizer", None) is None
                and representation != "continuous_semantic"):
            raise ValueError(
                f"{type(self.tokenizer.model).__name__} has no quantiser: "
                f"representation must be continuous_semantic, got {representation!r}"
            )
        self.representation = representation
        self.ckpt_name = ckpt_name or os.path.basename(checkpoint_path)
        self.pad_token_id = self.tokenizer.pad_token_id

    def get_num_tokens(self) -> int:
        return self.tokenizer.get_num_tokens()

    def get_codebook_embedding(self) -> torch.Tensor:
        return self._match_aminoaseed_width(self.tokenizer.get_codebook_embedding())

    @classmethod
    def _match_aminoaseed_width(cls, values: torch.Tensor) -> torch.Tensor:
        "Emit the encoder's own width; pad only when `PSJEPA_PAD_WIDTH` asks."
        if getenv("PSJEPA_NATIVE_WIDTH") is not None:
            raise ValueError(
                "PSJEPA_NATIVE_WIDTH is gone: the native width is now the default. "
                "Set PSJEPA_PAD_WIDTH=1024 to pad instead."
            )
        target = getenv("PSJEPA_PAD_WIDTH")
        if target:
            target = int(target)
            width = values.shape[-1]
            if width > target:
                raise ValueError(
                    f"PS-JEPA representation width {width} exceeds PSJEPA_PAD_WIDTH {target}"
                )
            if width < target:
                values = torch.nn.functional.pad(values, (0, target - width))
        return cls._assert_expected_width(values)

    @staticmethod
    def _assert_expected_width(values: torch.Tensor) -> torch.Tensor:
        'Refuse to emit features the head was not built for.'
        expected = getenv("PSJEPA_EXPECTED_WIDTH")
        if not expected:
            return values
        expected = int(expected)
        if values.shape[-1] != expected:
            raise ValueError(
                f"the head was built at d_model={expected} but this tokenizer emits "
                f"{values.shape[-1]}-wide features. The runner script and this adapter "
                f"disagree -- most likely src/ was edited while the job was running."
            )
        return values

    @torch.inference_mode()
    def encode_structure(
        self,
        pdb_chain,
        use_continuous: bool = False,
        use_sequence: bool = False,
        selected_indices=None,
    ):
        if self.representation in self.DISCRETE_REPRESENTATIONS and use_continuous:
            raise ValueError(f"{self.representation} requires data.use_continuous=false")
        if self.representation not in self.DISCRETE_REPRESENTATIONS and not use_continuous:
            raise ValueError(f"{self.representation} requires data.use_continuous=true")
        atom37 = torch.as_tensor(pdb_chain.atom37_positions, dtype=torch.float32)
        if selected_indices is not None:
            atom37 = atom37[selected_indices]
        sequence = list(pdb_chain.sequence)
        if selected_indices is not None:
            sequence = np.asarray(sequence)[np.asarray(selected_indices)].tolist()
        # AlphaFold atom37 ordering: N=0, CA=1, C=2, O=4.
        coords = (
            atom37
            if self.tokenizer.model.config.input_mode == "all_atom"
            else atom37[:, [0, 1, 2, 4]]
        )
        atom_mask = torch.isfinite(coords).all(dim=-1)
        coords = torch.nan_to_num(coords)
        valid_ca = atom_mask[:, 1]
        center = coords[valid_ca, 1].mean(0) if valid_ca.any() else coords[atom_mask].reshape(-1, 3).mean(0)
        coords = (coords - center) * atom_mask.unsqueeze(-1)
        try:
            import Bio.PDB.Polypeptide
            residue_type = torch.tensor(
                [Bio.PDB.Polypeptide.one_to_index(item) if item != "X" else 20 for item in sequence],
                dtype=torch.long,
            )
        except Exception:
            residue_type = torch.full((len(sequence),), 20, dtype=torch.long)
        output = self.tokenizer.encode_chunks(coords, atom_mask, residue_type)
        key = {
            "continuous_semantic": "continuous_semantic",
            "quantized_semantic": "semantic_embeddings",
            "discrete_ids": "semantic_ids",
            "slot_ids": "slot_ids",
            "slot_embeddings": "slot_embeddings",
        }[self.representation]
        values = output[key]
        if self.representation not in self.DISCRETE_REPRESENTATIONS:
            values = self._match_aminoaseed_width(values)
        if self.representation in self.SLOT_REPRESENTATIONS:
            # There is no residue for a slot to sit on. Report one index and one
            # sequence entry per slot so the harness's length checks hold, and let
            # the unknown-residue code stand in for a residue type that does not
            # exist rather than inventing one.
            count = int(values.shape[0])
            return values, np.arange(count), [20] * count
        if selected_indices is None:
            positions = np.asarray(pdb_chain.residue_index)
        else:
            positions = np.asarray(pdb_chain.residue_index)[np.asarray(selected_indices)]
        try:
            import Bio.PDB.Polypeptide
            seqs = [Bio.PDB.Polypeptide.one_to_index(item) if item != "X" else 20 for item in sequence]
        except Exception:
            seqs = [20] * len(sequence)
        return values, positions, seqs
