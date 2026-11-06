'Continuous masked latent prediction with online and EMA target encoders.'
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from psjepa.data.masking import sample_context_and_targets, sample_window_and_flanks
from psjepa.layers.predictors import ContextPredictor, TransformerContextPredictor
from psjepa.layers.residue_projection import ResidueProjection
from psjepa.layers.sigreg import sigreg_tokens
from psjepa.models.encoder import (EncoderConfig, StructureEncoder, _rotate_batch,
                                    masked_mse)
from psjepa.models.mamba_encoder import (BiMambaStack, MambaEncoderConfig,
                                          set_padding_invariant)
from psjepa.training.ema import copy_encoder, update_ema
from psjepa.training.representation_diagnostics import masked_mean, token_diagnostics


@dataclass
class PSJEPAConfig:
    d_model: int = 256
    n_layer: int = 6
    predictor_layers: int = 3
    max_residues: int = 512
    mixer_backend: str = "mamba"
    input_mode: str = "gvp"
    # Rotate the target encoder's coordinates independently of the context encoder's.
    # Irrelevant for `input_mode: gvp`, whose features are invariant by construction, but
    # decisive for `backbone`: with one shared rotation per sample, context and target sit
    # in the SAME frame, so absolute position is a shortcut for "is this target mine?" that
    # needs no structural understanding.
    independent_rotation: bool = False
    gvp_neighbours: int = 12
    gvp_global_radial: int = 16
    gvp_layers: int = 3
    gvp_message_layers: int = 2
    gvp_node_vectors: int = 16
    gvp_edge_positional: bool = False
    gvp_node_features: str = "atom"
    gvp_anchor_mode: str = "mean"
    gvp_rbf_width_mode: str = "spacing"
    gvp_positional_order: str = "sin_cos"
    # "legacy" or "bonded": paper-node chain ends and breaks (gvp_message.paper_node_features).
    gvp_chain_ends: str = "legacy"


    gvp_message_scalar_hidden: int = 128
    gvp_message_vector_hidden: int = 16
    gvp_featurizer_layers: int = 2
    gvp_scalar_hidden: int = 128
    backbone_presence_bits: bool = False  # legacy 16-wide xyz input; see ResidueProjection
    context_ratio: float = 0.5
    target_ratio: float = 0.125
    target_blocks: int = 4
    spatial_probability: float = 0.5


    exclude_context: bool = True


    scheme: str = "regions"
    encoder_layout: str = "legacy_slot"


    padding_invariant: bool = False
    # A Transformer encoder is what makes an arbitrary residue set encodable. Mamba's
    # recurrence *is* its position, so dropping interior residues renumbers everyone after
    # them and the crop had to be a contiguous run. Attention carries position in an added
    # table instead, so the crop can be any set of indices -- which is what lets it be a
    # 3D neighbourhood rather than a stretch of chain.
    encoder_backend: str = "mamba"
    encoder_heads: int = 8
    encoder_mlp_ratio: float = 4.0
    # 0.0 reproduces a pure sequence window, 1.0 a pure 3D ball. Mixed on ranks rather
    # than on raw distances so angstroms and residue counts do not need a common scale.
    crop_spatial_fraction: float = 0.5
    predictor_backend: str = "mamba"
    predictor_heads: int = 8
    predictor_mlp_ratio: float = 4.0


    predictor_width: int | None = None
    crop_views: int = 3
    crop_ratio_min: float = 0.45
    crop_ratio_max: float = 0.55
    # A ladder of crop scales instead of one narrow band. When its length equals
    # `crop_views` each view takes one rung, so every step sees every scale rather than
    # sampling one and hoping the average works out. Empty falls back to the uniform draw
    # between min and max. Motivated by measurement: pooled effective rank came out 50.6 at
    # crop 30% against 22.9 at crop 50%, so scale matters and the best rung is not known.
    crop_ratio_choices: tuple = ()


    radial_reference: str = "crop"
    trim_ratio: float = 0.20
    prediction_ratio: float = 0.30
    normalize_prediction: bool = False
    loss_weights: dict = field(default_factory=lambda: {"s2s": 1.0})
    # LePSJEPA: one shared encoder, direct target gradients, and SIGReg on
    # unnormalised tokens (the exact values the decoder and sampler consume).
    le_shared_encoder: bool = False
    sigreg_weight: float = 0.0
    sigreg_ramp_steps: int = 0
    sigreg_slices: int = 64
    sigreg_per_chain: int = 16
    # False drops the per-token LayerNorm on the regression target, as in the LeJEPA
    # paper, so the MSE sees the same raw tokens SIGReg constrains.
    target_layernorm: bool = True
    # Whole-protein term: the mean of the student's context tokens, through `global_head`,
    # regresses onto the LayerNormed mean of the full-view target tokens. The per-residue
    # loss only asks each token to name its neighbourhood; this asks the visible residues,
    # together, to carry what the whole chain is. The pooled target is always detached, so
    # a shared encoder cannot make it easy the way it made the per-residue target easy.
    global_weight: float = 0.0
    # Amino-acid identity as an input (an embedding added to each residue's geometric
    # feature). Off for every headline encoder, which are backbone-only by protocol; on for
    # the sequence-aware variants compared against sequence+structure encoders (GearNet).
    # A hidden residue's identity is hidden with it: `mask_x` replaces the whole projected
    # feature, embedding included.
    use_residue_identity: bool = False

    @classmethod
    def from_dict(cls, model: dict, masking: dict, loss: dict) -> "PSJEPAConfig":
        if model.get("quantize", False) or model.get("quantize_context", False):
            raise ValueError(
                "This release supports continuous PS-JEPA encoders only; "
                "quantized training is not supported."
            )
        scheme = masking.get("scheme", "regions")
        if scheme not in ("regions", "flanks", "crop_flanks_v1_1", "spatial_crop_v1_2"):
            raise ValueError(f"Unsupported masking scheme: {scheme!r}")
        return cls(
            d_model=int(model["d_model"]), n_layer=int(model["n_layer"]),
            predictor_layers=int(model["predictor_layers"]),
            max_residues=int(model["max_residues"]),
            mixer_backend=model.get("mixer_backend", "mamba"),
            input_mode=model.get("input_mode", "gvp"),
            independent_rotation=bool(model.get("independent_rotation", False)),
            gvp_neighbours=int(model.get("gvp_neighbours", 12)),
            gvp_global_radial=int(model.get("gvp_global_radial", 16)),
            gvp_layers=int(model.get("gvp_layers", 3)),
            gvp_message_layers=int(model.get("gvp_message_layers", 2)),
            gvp_node_vectors=int(model.get("gvp_node_vectors", 16)),
            gvp_edge_positional=bool(model.get("gvp_edge_positional", False)),
            gvp_node_features=str(model.get("gvp_node_features", "atom")),
            gvp_anchor_mode=str(model.get("gvp_anchor_mode", "mean")),
            gvp_rbf_width_mode=str(model.get("gvp_rbf_width_mode", "spacing")),
            gvp_positional_order=str(model.get("gvp_positional_order", "sin_cos")),
            gvp_chain_ends=str(model.get("gvp_chain_ends", "legacy")),
            gvp_message_scalar_hidden=int(model.get("gvp_message_scalar_hidden", 128)),
            gvp_message_vector_hidden=int(model.get("gvp_message_vector_hidden", 16)),
            gvp_featurizer_layers=int(model.get("gvp_featurizer_layers", 2)),
            gvp_scalar_hidden=int(model.get("gvp_scalar_hidden", 128)),
            backbone_presence_bits=bool(model.get("backbone_presence_bits", False)),
            context_ratio=float(masking.get("context_ratio", 0.5)),
            target_ratio=float(masking.get("target_ratio", 0.125)),
            target_blocks=int(masking.get("target_blocks", 4)),
            spatial_probability=float(masking.get("spatial_probability", 0.5)),
            exclude_context=bool(masking.get("exclude_context", True)),
            scheme=str(masking.get("scheme", "regions")),
            encoder_layout=str(model.get("encoder_layout", "legacy_slot")),
            padding_invariant=bool(model.get("padding_invariant", False)),
            predictor_backend=str(model.get("predictor_backend", "mamba")),
            encoder_backend=str(model.get("encoder_backend", "mamba")),
            encoder_heads=int(model.get("encoder_heads", 8)),
            encoder_mlp_ratio=float(model.get("encoder_mlp_ratio", 4.0)),
            predictor_heads=int(model.get("predictor_heads", 8)),
            predictor_mlp_ratio=float(model.get("predictor_mlp_ratio", 4.0)),
            predictor_width=(int(model["predictor_width"])
                             if model.get("predictor_width") is not None else None),
            crop_views=int(masking.get("crop_views", 3)),
            crop_spatial_fraction=float(masking.get("crop_spatial_fraction", 0.5)),
            crop_ratio_choices=tuple(masking.get("crop_ratio_choices", ()) or ()),
            radial_reference=str(model.get("radial_reference", "crop")),
            crop_ratio_min=float(masking.get("crop_ratio_min", 0.45)),
            crop_ratio_max=float(masking.get("crop_ratio_max", 0.55)),
            trim_ratio=float(masking.get("trim_ratio", 0.20)),
            prediction_ratio=float(masking.get("prediction_ratio", 0.30)),
            normalize_prediction=bool(loss.get("normalize_prediction", False)),
            loss_weights={"s2s": float(loss.get("s2s", 1.0))},
            le_shared_encoder=bool(model.get("le_shared_encoder", False)),
            sigreg_weight=float(loss.get("sigreg_weight", 0.0)),
            sigreg_ramp_steps=int(loss.get("sigreg_ramp_steps", 0)),
            sigreg_slices=int(loss.get("sigreg_slices", 64)),
            sigreg_per_chain=int(loss.get("sigreg_per_chain", 16)),
            target_layernorm=bool(loss.get("target_layernorm", True)),
            global_weight=float(loss.get("global", 0.0)),
            use_residue_identity=bool(model.get("use_residue_identity", False)),
        )


class ResidueOnlyStructureEncoder(nn.Module):
    """A structure encoder whose Mamba sequence is exactly the L residue sites."""

    def __init__(self, config: PSJEPAConfig) -> None:
        super().__init__()
        self.projection = ResidueProjection(
            config.d_model,
            use_residue_identity=config.use_residue_identity,
            input_mode=config.input_mode,
            gvp_neighbours=config.gvp_neighbours,
            gvp_global_radial=config.gvp_global_radial,
            # Without these the residue_only encoders silently build the featuriser's own
            # defaults, so `input_mode: gvp_message` would ignore every knob in the config.
            gvp_layers=config.gvp_layers,
            gvp_message_layers=config.gvp_message_layers,
            gvp_node_vectors=config.gvp_node_vectors,
            gvp_edge_positional=config.gvp_edge_positional,
            gvp_node_features=config.gvp_node_features,
            gvp_anchor_mode=config.gvp_anchor_mode,
            gvp_rbf_width_mode=config.gvp_rbf_width_mode,
            gvp_positional_order=config.gvp_positional_order,
            gvp_chain_ends=config.gvp_chain_ends,
            gvp_message_scalar_hidden=config.gvp_message_scalar_hidden,
            gvp_message_vector_hidden=config.gvp_message_vector_hidden,
            gvp_featurizer_layers=config.gvp_featurizer_layers,
            gvp_scalar_hidden=config.gvp_scalar_hidden,
            backbone_presence_bits=config.backbone_presence_bits,
        )
        self.mask_x = nn.Parameter(torch.zeros(config.d_model))
        nn.init.normal_(self.mask_x, std=0.02)
        self.encoder = BiMambaStack(MambaEncoderConfig(
            d_model=config.d_model,
            n_layer=config.n_layer,
            bidirectional=True,
            backend=config.mixer_backend,
        ))

    def forward(self, coords, atom_mask, residue_mask, prediction_mask=None,
                residue_type=None, positions=None, radial_reference=None):
        visible = None if prediction_mask is None else ~prediction_mask
        # The scan itself reads no positions; the featuriser uses them for sequence order.
        residues = self.projection(coords, atom_mask, residue_type, visible, radial_reference,
                                   positions)
        if prediction_mask is not None:
            residues = torch.where(prediction_mask.unsqueeze(-1), self.mask_x, residues)
        residues = residues * residue_mask.unsqueeze(-1)
        encoded = self.encoder(residues, ~residue_mask)
        return None, encoded, None


class TransformerResidueEncoder(nn.Module):
    'A structure encoder over an arbitrary set of residues, ordered by attention.'

    def __init__(self, config: PSJEPAConfig) -> None:
        super().__init__()
        self.projection = ResidueProjection(
            config.d_model,
            use_residue_identity=config.use_residue_identity,
            input_mode=config.input_mode,
            gvp_neighbours=config.gvp_neighbours,
            gvp_global_radial=config.gvp_global_radial,
            # Without these the residue_only encoders silently build the featuriser's own
            # defaults, so `input_mode: gvp_message` would ignore every knob in the config.
            gvp_layers=config.gvp_layers,
            gvp_message_layers=config.gvp_message_layers,
            gvp_node_vectors=config.gvp_node_vectors,
            gvp_edge_positional=config.gvp_edge_positional,
            gvp_node_features=config.gvp_node_features,
            gvp_anchor_mode=config.gvp_anchor_mode,
            gvp_rbf_width_mode=config.gvp_rbf_width_mode,
            gvp_positional_order=config.gvp_positional_order,
            gvp_chain_ends=config.gvp_chain_ends,
            gvp_message_scalar_hidden=config.gvp_message_scalar_hidden,
            gvp_message_vector_hidden=config.gvp_message_vector_hidden,
            gvp_featurizer_layers=config.gvp_featurizer_layers,
            gvp_scalar_hidden=config.gvp_scalar_hidden,
            backbone_presence_bits=config.backbone_presence_bits,
        )
        self.mask_x = nn.Parameter(torch.zeros(config.d_model))
        nn.init.normal_(self.mask_x, std=0.02)
        self.max_residues = int(config.max_residues)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.position = nn.Parameter(torch.empty(self.max_residues, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        if config.d_model % config.encoder_heads:
            raise ValueError(f"d_model={config.d_model} not divisible by "
                             f"encoder_heads={config.encoder_heads}")
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_heads,
            dim_feedforward=int(round(config.encoder_mlp_ratio * config.d_model)),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layer, norm=nn.LayerNorm(config.d_model)
        )
        # Off unless `enable_cls_token` is called, so existing checkpoints load unchanged.
        self.cls_token = None

    def enable_cls_token(self) -> None:
        """Prepend one learned summary token; forward then returns its output first."""
        d_model = self.input_norm.normalized_shape[0]
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, coords, atom_mask, residue_mask, prediction_mask=None,
                residue_type=None, positions=None, radial_reference=None):
        visible = None if prediction_mask is None else ~prediction_mask
        residues = self.projection(coords, atom_mask, residue_type, visible, radial_reference,
                                   positions)
        if prediction_mask is not None:
            residues = torch.where(prediction_mask.unsqueeze(-1), self.mask_x, residues)
        residues = self.input_norm(residues)
        width = residues.shape[1]
        if positions is None:
            if width > self.max_residues:
                raise ValueError(f"length {width} exceeds position table {self.max_residues}")
            index = torch.arange(width, device=residues.device).unsqueeze(0).expand(
                residues.shape[0], width)
        else:
            index = positions
            if int(index.max()) >= self.max_residues:
                raise ValueError(f"position {int(index.max())} exceeds table {self.max_residues}")
        hidden = residues + self.position[index.clamp(0, self.max_residues - 1)].to(residues.dtype)
        hidden = hidden * residue_mask.unsqueeze(-1).to(hidden.dtype)
        # A row with every key padded makes attention return NaN, so one placeholder key is
        # admitted internally and masked back out of the result.
        safe = residue_mask.clone()
        summary = None
        if self.cls_token is not None:
            # The summary token is always a valid key, so no row is ever empty.
            cls = self.cls_token.to(hidden.dtype).expand(hidden.shape[0], 1, -1)
            keys = torch.cat([safe.new_ones(safe.shape[0], 1), safe], dim=1)
            encoded = self.encoder(torch.cat([cls, hidden], dim=1), src_key_padding_mask=~keys)
            summary, encoded = encoded[:, 0], encoded[:, 1:]
        else:
            empty = ~safe.any(dim=1)
            if empty.any() and width:
                safe[empty, 0] = True
            encoded = self.encoder(hidden, src_key_padding_mask=~safe)
        return summary, encoded * residue_mask.unsqueeze(-1).to(encoded.dtype), None


class PSJEPA(nn.Module):
    """A single-stream JEPA over protein backbones."""

    def __init__(self, config: PSJEPAConfig) -> None:
        super().__init__()
        self.config = config


        if config.encoder_layout == "residue_only":
            build = (TransformerResidueEncoder if config.encoder_backend == "transformer"
                     else ResidueOnlyStructureEncoder)
            if config.encoder_backend not in ("mamba", "transformer"):
                raise ValueError(f"unknown encoder_backend: {config.encoder_backend!r}")
            self.context_encoder = build(config)
            self.target_encoder = build(config)
        elif config.encoder_layout == "legacy_slot":
            if (config.gvp_message_scalar_hidden, config.gvp_message_vector_hidden) != (128, 16):
                raise ValueError("gvp_message widths are wired for encoder_layout=residue_only only")
            inner = EncoderConfig(
                d_model=config.d_model, n_layer=config.n_layer,
                predictor_layers=config.predictor_layers, max_residues=config.max_residues,
                quantize=False, mixer_backend=config.mixer_backend,
                query_mode="content_conditioned", input_mode=config.input_mode,
                gvp_neighbours=config.gvp_neighbours,
                gvp_global_radial=config.gvp_global_radial,
                gvp_layers=config.gvp_layers,
                gvp_message_layers=config.gvp_message_layers,
                gvp_node_vectors=config.gvp_node_vectors,
                gvp_edge_positional=config.gvp_edge_positional,
                gvp_node_features=config.gvp_node_features,
                gvp_featurizer_layers=config.gvp_featurizer_layers,
                gvp_scalar_hidden=config.gvp_scalar_hidden,
                backbone_presence_bits=config.backbone_presence_bits,
                global_slots=1, semantic_slots_only=True,
                use_residue_identity=config.use_residue_identity,
                independent_rotation=False,
            )
            self.context_encoder = StructureEncoder(inner)
            self.target_encoder = StructureEncoder(inner)
        else:
            raise ValueError(f"unknown encoder_layout: {config.encoder_layout!r}")
        if not config.target_layernorm and config.scheme not in ("flanks", "regions"):
            raise ValueError("target_layernorm=False is only implemented for flanks/regions")
        if config.le_shared_encoder:
            if config.sigreg_weight <= 0:
                raise ValueError("LePSJEPA requires positive SIGReg weight and continuous tokens")
            self.target_encoder = self.context_encoder


            print("warning: le_shared_encoder gives the per-residue target a gradient; "
                  "watch val_positional_r2 and val_loss_prediction_rms", flush=True)
        else:
            copy_encoder(self.target_encoder, self.context_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        if config.predictor_backend == "transformer":
            if config.predictor_width is not None:
                raise ValueError("predictor_width is only implemented for predictor_backend=mamba")
            self.predictor = TransformerContextPredictor(
                config.d_model,
                config.predictor_layers,
                config.max_residues,
                config.predictor_heads,
                config.predictor_mlp_ratio,
            )
        elif config.predictor_backend == "mamba":
            self.predictor = ContextPredictor(config.d_model, config.predictor_layers,
                                              config.mixer_backend, width=config.predictor_width)
        else:
            raise ValueError(f"unknown predictor_backend: {config.predictor_backend!r}")
        if config.padding_invariant:
            # This is a computation flag rather than a parameter. Applying it after all
            # modules are built covers the online encoder, EMA encoder and Mamba predictor.
            set_padding_invariant(self.context_encoder, True)
            set_padding_invariant(self.target_encoder, True)
            set_padding_invariant(self.predictor, True)
        self.target_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)

        self.global_head = None
        if config.global_weight > 0:
            if config.scheme not in ("flanks", "regions"):
                raise ValueError("global_weight is only implemented for flanks/regions")
            self.global_head = nn.Sequential(
                nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model),
                nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.prediction_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)
        # Not a submodule and not in the state dict: a read-time switch, set by
        # `SemanticTokenizer.from_checkpoint` so the same checkpoint can be probed
        # either way. See `readout_encoder`. The EMA target encoder is the representation a
        # JEPA is defined to produce; the student is only the branch that chases it.
        self.readout = "teacher"

    def context_parameters(self) -> list[torch.Tensor]:
        "The encoder's parameters only — for per-loss gradient-norm diagnostics."
        return [p for p in self.context_encoder.parameters() if p.requires_grad]

    def trainable_parameters(self) -> list[torch.Tensor]:
        'Everything the optimiser must own: the context encoder and the predictor.'
        return [p for p in self.parameters() if p.requires_grad]

    def readout_encoder(self) -> StructureEncoder:
        """Which of the two encoders a downstream task reads.

        The EMA teacher is the default: every training step gave it a complete structure,
        and I-JEPA evaluates the target encoder. The online student learned from a masked
        input and therefore sees a complete structure only if explicitly requested as a
        diagnostic. Both are in the checkpoint, so this remains a read-time choice.
        """
        if self.readout not in ("student", "teacher"):
            raise ValueError(f"readout must be student or teacher, got {self.readout!r}")
        return self.target_encoder if self.readout == "teacher" else self.context_encoder

    def encode(self, coords, atom_mask, residue_mask, residue_type=None) -> torch.Tensor:
        """The stream a downstream task reads: one continuous vector per residue."""
        _slot, patch, _world = self.readout_encoder()(
            coords, atom_mask, residue_mask, None, residue_type)
        return patch

    @torch.no_grad()
    def tokenize(self, coords, atom_mask, residue_mask, residue_type=None) -> dict[str, Any]:
        """The StructTokenBench contract, named to match `PSJEPA.tokenize`.

        There is no quantiser here, so only the continuous keys exist. Asking this
        checkpoint for `discrete_ids` or `quantized_semantic` fails in the adapter
        rather than silently returning something that is not a token.
        """
        patch = self.encode(coords, atom_mask, residue_mask, residue_type)
        invalid = ~residue_mask
        zeroed = patch.masked_fill(invalid.unsqueeze(-1), 0.0)
        output = {"continuous_semantic": zeroed, "patch_embeddings": zeroed}
        return output

    def forward(self, batch: dict, step: int = 0) -> dict[str, Any]:
        coords, atom_mask = batch["coords"], batch["atom_mask"]
        residue_mask = batch["residue_mask"]
        residue_type = batch.get("residue_type")
        if self.config.scheme in ("crop_flanks_v1_1", "spatial_crop_v1_2"):
            return self._forward_crop_flanks(
                coords, atom_mask, residue_mask, residue_type
            )
        if self.config.scheme == "flanks":
            regions = sample_window_and_flanks(
                coords, atom_mask, residue_mask,
                self.config.context_ratio, self.config.target_ratio,
            )
        elif self.config.scheme == "regions":
            regions = sample_context_and_targets(
                coords, atom_mask, residue_mask, self.config.context_ratio,
                self.config.target_ratio, self.config.target_blocks,
                self.config.spatial_probability,
                exclude_context=self.config.exclude_context,
            )
        else:
            raise ValueError(f"unknown masking scheme: {self.config.scheme!r}")
        hidden = residue_mask & ~regions["context_mask"]
        blocks = regions["target_blocks"]

        _slot, context, _w = self.context_encoder(
            coords, atom_mask, residue_mask, hidden, residue_type)
        target_coords = (_rotate_batch(coords, atom_mask)
                         if self.config.independent_rotation else coords)
        with (nullcontext() if self.config.le_shared_encoder else torch.no_grad()):
            _slot_t, target, _wt = self.target_encoder(
                target_coords, atom_mask, residue_mask, None, residue_type)
        raw_target = target
        target = target if self.config.le_shared_encoder else target.detach()
        if self.config.target_layernorm:
            target = self.target_norm(target)
        shape = self.prediction_norm if self.config.normalize_prediction else (lambda x: x)

        count = blocks.shape[1]
        if self.config.predictor_backend == "transformer":
            parts = [shape(self.predictor(
                context, blocks[:, i], residue_mask, regions["context_mask"]
            )) for i in range(count)]
        else:
            parts = [shape(self.predictor(context, blocks[:, i], residue_mask))
                     for i in range(count)]
        loss_s2s = sum(masked_mse(part, target, blocks[:, i])
                       for i, part in enumerate(parts)) / count
        total = self.config.loss_weights["s2s"] * loss_s2s
        sigreg_target = target.new_zeros((), dtype=torch.float32)
        sigreg_context = target.new_zeros((), dtype=torch.float32)
        sigreg_weight = self.config.sigreg_weight
        if self.config.sigreg_ramp_steps > 0:
            sigreg_weight *= min(1.0, (step + 1) / self.config.sigreg_ramp_steps)
        if self.config.sigreg_weight > 0:
            options = {"slices": self.config.sigreg_slices,
                       "per_chain": self.config.sigreg_per_chain}
            sigreg_target = sigreg_tokens(raw_target, residue_mask, **options)
            sigreg_context = sigreg_tokens(context, regions["context_mask"], **options)
            # An EMA target has no gradient. Regularise the online context tokens;
            # still measure the target statistic to see whether EMA tracks it.
            penalty = ((sigreg_target + sigreg_context) * 0.5
                       if self.config.le_shared_encoder else sigreg_context)
            total = total + sigreg_weight * penalty
        loss_global = target.new_zeros((), dtype=torch.float32)
        if self.global_head is not None:
            visible = regions["context_mask"] & residue_mask
            pooled_target = masked_mean(raw_target.detach().float(), residue_mask)
            pooled_target = nn.functional.layer_norm(pooled_target, pooled_target.shape[-1:])
            guess = self.global_head(masked_mean(context, visible)).float()
            loss_global = (guess - pooled_target).square().mean()
            total = total + self.config.global_weight * loss_global

        with torch.no_grad():
            seen, unseen = regions["target_seen"], regions["target_unseen"]
            rolled = torch.roll(target, 1, dims=0)

            def block_error(reference, selector):
                numerator = target.new_zeros((), dtype=torch.float32)
                denominator = target.new_zeros((), dtype=torch.float32)
                for part, block in zip(parts, blocks.unbind(dim=1)):
                    mask = block & selector
                    per_residue = (part - reference).float().square().mean(dim=-1)
                    numerator += per_residue.masked_fill(~mask, 0.0).sum()
                    denominator += mask.sum()
                return numerator / denominator.clamp_min(1.0)

            prediction_energy = target.new_zeros((), dtype=torch.float32)
            prediction_values = target.new_zeros((), dtype=torch.float32)
            for part, block in zip(parts, blocks.unbind(dim=1)):
                mask = block & residue_mask
                prediction_energy += part.float().square().masked_fill(
                    ~mask.unsqueeze(-1), 0.0
                ).sum()
                prediction_values += mask.sum() * part.shape[-1]

            # Collapse belongs to the full-view EMA target that downstream reads,
            # not to the student's mostly-masked training input.
            centred = target - target[residue_mask].mean(dim=0, keepdim=True)
            flat = centred[residue_mask]
            singular = torch.linalg.svdvals(flat.float()) if flat.numel() else torch.zeros(1)
            share = (singular / singular.sum().clamp_min(1e-12)).clamp_min(1e-12)
            diagnostics = {
                "loss_target_seen": block_error(target, seen),
                "loss_target_unseen": block_error(target, unseen),
                "loss_target_shuffled": block_error(
                    rolled, residue_mask & residue_mask.roll(1, dims=0)
                ),
                "loss_prediction_rms": (
                    prediction_energy / prediction_values.clamp_min(1.0)
                ).sqrt(),
                "effective_rank": torch.exp(-(share * share.log()).sum()),
                "feature_std_mean": (
                    flat.float().std(dim=0).mean()
                    if flat.shape[0] > 1 else prediction_energy * 0.0
                ),
                "target_coverage": regions["target_mask"].float().mean(),
                "unseen_coverage": unseen.float().mean(),
            }
            if not self.training:
                # A least-squares fit per batch; validation only, to keep it off the step time.
                diagnostics.update(token_diagnostics(raw_target, residue_mask))
        return {
            "loss": total, "loss_s2s": loss_s2s,
            "loss_sigreg_target": sigreg_target, "loss_sigreg_context": sigreg_context,
            "loss_global": loss_global,
            "sigreg_weight_effective": target.new_tensor(sigreg_weight),
            "prediction_mask": regions["target_mask"],
            "mask_modes": regions["modes"],
            **diagnostics,
        }

    def _sample_crop_view(self, residue_mask: torch.Tensor, ratio_override=None) -> list[dict[str, torch.Tensor]]:
        """Sample one crop/core/flank problem independently for every protein."""
        sampled = []
        for row in residue_mask:
            valid = torch.nonzero(row, as_tuple=False).flatten()
            length = int(valid.numel())
            if length < 6:
                sampled.append({
                    "crop": valid,
                    "core": valid[:0],
                    "target": valid[:0],
                    "target_inside": valid[:0],
                })
                continue
            ratio = ratio_override if ratio_override is not None else float(
                torch.empty((), device=row.device).uniform_(
                    self.config.crop_ratio_min, self.config.crop_ratio_max))
            crop_length = max(3, min(length - 2, int(round(ratio * length))))
            crop_start = int(torch.randint(length - crop_length + 1, (), device=row.device))
            crop = valid[crop_start:crop_start + crop_length]

            trim = max(2, min(crop_length - 1, int(round(self.config.trim_ratio * length))))
            # Both sides lose at least one token; the asymmetric split is redrawn per view.
            trim_left = int(torch.randint(1, trim, (), device=row.device))
            trim_right = trim - trim_left
            core_start = crop_start + trim_left
            core_end = crop_start + crop_length - trim_right
            core = valid[core_start:core_end]

            budget = max(2, min(length - len(core), int(round(
                self.config.prediction_ratio * length
            ))))
            # The removed margins are precisely the context/target overlap.  Every
            # removed token is predicted; the remaining target budget extends outward.
            budget = max(trim, budget)
            extra = budget - trim
            extra_left = int(torch.randint(extra + 1, (), device=row.device))
            desired_left = trim_left + extra_left
            room_left, room_right = core_start, length - core_end
            left = min(room_left, desired_left)
            right = min(room_right, budget - left)
            left = min(room_left, budget - right)
            target = torch.cat([
                valid[core_start - left:core_start],
                valid[core_end:core_end + right],
            ])
            # The trimmed margins are inside the crop, so the encoder read their
            # coordinates even though the predictor never sees their tokens; the
            # outward extension it never read at all. Kept apart because a
            # bidirectional mixer folds the margins into the retained core, and that
            # is a feature-level version of exactly the overlap this scheme removes.
            inside = torch.isin(target, crop).long()
            sampled.append({"crop": crop, "core": core, "target": target,
                            "target_inside": inside})
        return sampled

    def _full_structure_reference(self, coords, atom_mask, residue_mask):
        """The whole chain's centroid and radius of gyration, per protein.

        Computed here rather than inside the featuriser because `_encode_crop` gathers only
        the crop's residues into their own tensor: by the time the featuriser runs, the rest
        of the chain is not in the batch at all. Only used when
        `radial_reference: full`, and it is deliberately an aggregate over residues the
        encoder does not see.
        """
        present = atom_mask.unsqueeze(-1).to(coords.dtype)
        counts = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(coords.dtype)
        anchors = (coords * present).sum(dim=-2) / counts
        weight = residue_mask.unsqueeze(-1).to(coords.dtype)
        total = weight.sum(dim=-2).clamp_min(1.0)
        centroid = (anchors * weight).sum(dim=-2) / total
        distance = torch.linalg.vector_norm(anchors - centroid.unsqueeze(-2), dim=-1)
        gyration = ((distance ** 2 * weight.squeeze(-1)).sum(dim=-1)
                    / total.squeeze(-1)).clamp_min(1e-6).sqrt()
        return centroid, gyration

    def _view_ratio(self, index: int):
        """The crop scale for view `index`, or None to let the sampler draw one.

        With as many rungs as views each view gets its own, in order, so a step always
        covers the whole ladder. With a different count the rung is chosen by index modulo
        length, which still cycles through all of them across views and steps.
        """
        choices = self.config.crop_ratio_choices
        if not choices:
            return None
        return float(choices[index % len(choices)])

    def _sample_spatial_crop_view(self, coords, residue_mask, ratio_override=None):
        """Crop, core and target as concentric shells around a seed residue.

        `_sample_crop_view` orders residues by their distance along the chain and takes a
        run. This orders them by a mix of that distance and the 3D distance between CA atoms,
        then takes the same three shells from the ordering: the nearest `crop` are encoded,
        the innermost of those are the predictor's context, and the shell that was removed
        plus the shell just outside the crop are the targets. At
        `crop_spatial_fraction: 0.0` it reproduces a sequence window centred on the seed; at
        1.0 the crop is a ball in space and can be a set of residues far apart in the chain.

        Mixing is on ranks, not on raw distances, because one term is in angstroms and the
        other in residue counts and no fixed conversion between them is right for every
        protein. Ranks make the parameter mean "what share of the ordering is decided by
        geometry", which is the quantity being swept.
        """
        alpha = float(self.config.crop_spatial_fraction)
        sampled = []
        for row in range(residue_mask.shape[0]):
            valid = torch.nonzero(residue_mask[row], as_tuple=False).flatten()
            length = int(valid.numel())
            if length < 6:
                empty = valid[:0]
                sampled.append({"crop": empty, "core": empty, "target": empty,
                                "target_inside": empty.long()})
                continue
            ca = coords[row, :, 1]
            seed = int(torch.randint(length, (), device=residue_mask.device))
            sequence_rank = (valid - valid[seed]).abs().argsort().argsort().float()
            distance = (ca[valid] - ca[valid[seed]]).square().sum(dim=-1)
            spatial_rank = distance.argsort().argsort().float()
            order = (alpha * spatial_rank + (1.0 - alpha) * sequence_rank).argsort()

            ratio = ratio_override if ratio_override is not None else float(
                torch.empty((), device=residue_mask.device).uniform_(
                    self.config.crop_ratio_min, self.config.crop_ratio_max))
            crop_size = max(3, min(length - 2, int(round(ratio * length))))
            trim = max(1, min(crop_size - 1, int(round(self.config.trim_ratio * length))))
            budget = max(trim, int(round(self.config.prediction_ratio * length)))
            budget = min(budget, length - (crop_size - trim))
            core_size = crop_size - trim

            crop = valid[order[:crop_size]].sort().values
            core = valid[order[:core_size]].sort().values
            target = valid[order[core_size:core_size + budget]].sort().values


            inside = torch.isin(target, crop).long()
            sampled.append({"crop": crop, "core": core, "target": target,
                            "target_inside": inside})
        return sampled

    @staticmethod
    def _pack_indices(rows, key, device):
        width = max((int(item[key].numel()) for item in rows), default=0)
        indices = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        valid = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
        for batch_index, item in enumerate(rows):
            count = int(item[key].numel())
            if count:
                indices[batch_index, :count] = item[key]
                valid[batch_index, :count] = True
        return indices, valid

    def _encode_crop(self, coords, atom_mask, residue_type, rows, radial_reference=None):
        crop_indices, crop_valid = self._pack_indices(rows, "crop", coords.device)
        batch, width = crop_indices.shape
        atoms = coords.shape[-2]
        gather_coords = crop_indices[:, :, None, None].expand(batch, width, atoms, 3)
        gather_atoms = crop_indices[:, :, None].expand(batch, width, atoms)
        crop_coords = coords.gather(1, gather_coords)
        crop_atoms = atom_mask.gather(1, gather_atoms)
        crop_coords = crop_coords * crop_valid[:, :, None, None]
        crop_atoms = crop_atoms & crop_valid[:, :, None]
        crop_types = None
        if residue_type is not None:
            crop_types = residue_type.gather(1, crop_indices)
        # The crop's absolute residue numbers, not 0..len(crop). A Mamba encoder ignores the
        # argument and reads order from the recurrence; a Transformer needs it, and it is the
        # only thing telling the encoder that residue 91 sits far from residue 14 in chain
        # even when they are neighbours in space.
        _slot, encoded, _world = self.context_encoder(
            crop_coords, crop_atoms, crop_valid, None, crop_types, positions=crop_indices,
            radial_reference=radial_reference,
        )

        core_indices, core_valid = self._pack_indices(rows, "core", coords.device)
        core = encoded.new_zeros(batch, core_indices.shape[1], encoded.shape[-1])
        for batch_index, item in enumerate(rows):
            count = int(item["core"].numel())
            if not count:
                continue
            local = torch.searchsorted(item["crop"], item["core"])
            core[batch_index, :count] = encoded[batch_index, local]
        return core, core_indices, core_valid

    def _forward_crop_flanks(self, coords, atom_mask, residue_mask, residue_type):
        if self.config.encoder_layout != "residue_only":
            raise ValueError(f"{self.config.scheme} requires encoder_layout=residue_only")
        if self.config.predictor_backend != "transformer":
            raise ValueError(f"{self.config.scheme} requires predictor_backend=transformer")


        radial_reference = (self._full_structure_reference(coords, atom_mask, residue_mask)
                            if self.config.radial_reference == "full" else None)
        with torch.no_grad():
            _slot, teacher, _world = self.target_encoder(
                coords, atom_mask, residue_mask, None, residue_type,
                radial_reference=radial_reference,
            )
        target = self.target_norm(teacher.detach())
        shape = self.prediction_norm if self.config.normalize_prediction else (lambda x: x)

        losses, shuffled_losses, energies, value_counts = [], [], [], []
        seen_losses, unseen_losses, seen_counts, unseen_counts = [], [], [], []
        prediction_mask = torch.zeros_like(residue_mask)
        all_rows = []
        for view in range(self.config.crop_views):
            ratio = self._view_ratio(view)
            rows = (self._sample_spatial_crop_view(coords, residue_mask, ratio)
                    if self.config.scheme == "spatial_crop_v1_2"
                    else self._sample_crop_view(residue_mask, ratio))
            all_rows.append(rows)
            context, context_positions, context_valid = self._encode_crop(
                coords, atom_mask, residue_type, rows, radial_reference
            )
            target_positions, target_valid = self._pack_indices(rows, "target", coords.device)
            prediction = shape(self.predictor.forward_sparse(
                context, context_positions, context_valid, target_positions, target_valid
            ))
            gather = target_positions.unsqueeze(-1).expand(-1, -1, target.shape[-1])
            matched_target = target.gather(1, gather)
            rolled = torch.roll(target, 1, dims=0).gather(1, gather)
            rolled_valid = residue_mask.roll(1, dims=0).gather(1, target_positions)
            per_target = (prediction - matched_target).float().square().mean(dim=-1)
            per_shuffled = (prediction - rolled).float().square().mean(dim=-1)
            inside, _ = self._pack_indices(rows, "target_inside", coords.device)
            seen = target_valid & inside.bool()
            unseen = target_valid & ~inside.bool()
            seen_losses.append(per_target[seen].mean() if seen.any()
                               else per_target.sum() * 0.0)
            unseen_losses.append(per_target[unseen].mean() if unseen.any()
                                 else per_target.sum() * 0.0)
            seen_counts.append(seen.sum())
            unseen_counts.append(unseen.sum())
            losses.append(per_target[target_valid].mean() if target_valid.any()
                          else per_target.sum() * 0.0)
            shuffled_mask = target_valid & rolled_valid
            shuffled_losses.append(per_shuffled[shuffled_mask].mean() if shuffled_mask.any()
                                   else per_shuffled.sum() * 0.0)
            energies.append(prediction.float().square().masked_fill(
                ~target_valid.unsqueeze(-1), 0.0
            ).sum())
            value_counts.append(target_valid.sum() * prediction.shape[-1])
            for batch_index, item in enumerate(rows):
                prediction_mask[batch_index, item["target"]] = True

        loss_s2s = sum(losses) / max(len(losses), 1)
        total = self.config.loss_weights["s2s"] * loss_s2s
        with torch.no_grad():
            valid_target = target[residue_mask]
            centred = target - valid_target.mean(dim=0, keepdim=True)
            flat = centred[residue_mask]
            singular = torch.linalg.svdvals(flat.float()) if flat.numel() else torch.zeros(1)
            share = (singular / singular.sum().clamp_min(1e-12)).clamp_min(1e-12)
            energy = sum(energies)
            values = sum(value_counts)
            diagnostics = {
                # Not a hardcoded zero any more. Roughly two thirds of each view's
                # targets are the trimmed margins, whose coordinates the encoder did
                # read; if `seen` runs far below `unseen` the crop is leaking through
                # the recurrence and the scheme is buying less than it looks.
                "loss_target_seen": sum(seen_losses) / max(len(seen_losses), 1),
                "loss_target_unseen": sum(unseen_losses) / max(len(unseen_losses), 1),
                "target_seen_fraction": (sum(seen_counts).float()
                                         / (sum(seen_counts) + sum(unseen_counts)).clamp_min(1)),
                "loss_target_shuffled": sum(shuffled_losses) / max(len(shuffled_losses), 1),
                "loss_prediction_rms": (energy / values.clamp_min(1)).sqrt(),
                "effective_rank": torch.exp(-(share * share.log()).sum()),
                "feature_std_mean": flat.float().std(dim=0).mean(),
                "target_coverage": prediction_mask.float().sum() / residue_mask.sum().clamp_min(1),
                "unseen_coverage": prediction_mask.float().sum() / residue_mask.sum().clamp_min(1),
            }
        return {
            "loss": total,
            "loss_s2s": loss_s2s,
            "prediction_mask": prediction_mask,
            "mask_modes": [self.config.scheme] * residue_mask.shape[0],
            **diagnostics,
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not self.config.le_shared_encoder:
            update_ema(self.target_encoder, self.context_encoder, momentum)

    @classmethod
    def from_dict(cls, model: dict, masking: dict, loss: dict) -> "PSJEPA":
        return cls(PSJEPAConfig.from_dict(model, masking, loss))


BioJEPA = PSJEPA
BioJEPAConfig = PSJEPAConfig
