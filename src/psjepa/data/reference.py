'Dataset and split identifiers for backbone training.'
from __future__ import annotations

REFERENCE_DATASET = "afdb"
REFERENCE_CACHE = "cache/afdb"
REFERENCE_TRAIN_SPLIT = "afdb/train"
REFERENCE_VALIDATION_SPLIT = "afdb/validation"

LEGACY_CATH_CACHE = "cache/cath_backbone_split"
LEGACY_CATH_TRAIN_SPLIT = "train"
LEGACY_CATH_VALIDATION_SPLIT = "validation"

__all__ = [
    "REFERENCE_DATASET",
    "REFERENCE_CACHE",
    "REFERENCE_TRAIN_SPLIT",
    "REFERENCE_VALIDATION_SPLIT",
    "LEGACY_CATH_CACHE",
    "LEGACY_CATH_TRAIN_SPLIT",
    "LEGACY_CATH_VALIDATION_SPLIT",
]
