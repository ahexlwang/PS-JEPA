from .psjepa import PSJEPA, PSJEPAConfig, ResidueOnlyStructureEncoder
from .encoder import EncoderConfig, StructureEncoder

__all__ = ["PSJEPA", "PSJEPAConfig", "EncoderConfig", "StructureEncoder", "ResidueOnlyStructureEncoder"]

BioJEPA = PSJEPA
BioJEPAConfig = PSJEPAConfig
