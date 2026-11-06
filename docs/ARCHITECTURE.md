# Architecture

## Encoder

The encoder maps N/CA/C/O coordinates to a continuous 512-dimensional vector per
residue. Atom and residue masks exclude missing and padded data.

| Component | Configuration |
|---|---|
| Graph anchor | Mean of observed backbone atom positions |
| Neighborhood | 12 nearest visible residues |
| Node scalars | Torsion sine/cosine channels and 17 global radial channels |
| Node vectors | Forward/backward Cα and pseudo-Cβ directions |
| Edge scalars | 16 distance RBF and 16 sequence-separation channels |
| GVP hidden channels | 128 scalars and 16 vectors |
| GVP message blocks | 3, with 2 GVP stages per message |
| Sequence mixer | 8 bidirectional Mamba layers, width 512 |
| Output | `[batch, residues, 512]` |

Each bidirectional Mamba block shares parameters between its forward and reverse
passes. Reverse scans operate over each chain's valid residues. The Mamba backend
is provided by the separately installed `mamba-ssm` package.

## Masked latent prediction

Each example samples a sequence-contiguous or spatial masking mode with equal
probability. Four target blocks are excluded from the visible context. The online
encoder observes the visible backbone; a predictor estimates the full-view EMA
encoder's features at target positions.

Targets receive non-affine per-residue LayerNorm. Mean squared error averages
over feature channels and valid positions in each block, then equally across
blocks. The teacher parameters follow an exponential moving average of the
online encoder. Downstream inference uses the teacher's raw continuous features.

## Downstream modules

The source package provides frozen-feature prediction adapters, coordinate flow
decoders and a latent flow prior. Decoder and prior configurations describe new
training runs; pair their checkpoints with the exact encoder used to train them.
Checkpoint shapes alone do not identify a compatible representation space.
