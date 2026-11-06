# Training

## Encoder

Prepare the `afdb` cache as described in [data preparation](DATA.md), install the
CUDA dependencies and run:

```bash
python scripts/train_encoder.py --config configs/encoder/sota_d512.yaml --resume auto
```

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas 0.9/0.95 |
| Peak learning rate | 0.00085 |
| Weight decay | 0.05 |
| Microbatch / accumulation | 64 / 8 |
| Training budget | 35,000 optimizer updates |
| Precision | BF16 |
| Seed | 42 |
| Maximum crop | 512 residues |

`last.pt` saves model, optimizer, RNG, data cursor and scheduler state. Automatic
resume requires compatible stored and requested configurations. Use `--resume never`
for a new output directory. An interrupted run can resume from its full training
checkpoint; inference exports cannot resume optimizer state.

The distributed checkpoint was selected after plateau scheduling and continuation
to 35,000 total updates. The supplied configuration trains a fresh model with the
same architecture and budget; independent runs need not reproduce its exact tensors
or scores. For inference and published-checkpoint evaluation, download the weights
listed in [the download instructions](../README.md).

## Coordinate decoder and latent prior

Training utilities accept frozen encoder features:

```bash
python scripts/train_decoder.py --config configs/decoder/sota_d512.yaml --resume auto
python scripts/train_prior.py --config configs/prior/sota_d512.yaml --resume auto
```

These configurations are starting points for training new downstream models.
Set checkpoint paths, corpus, length range and training budget for your run.
Decoder and prior weights are not part of the active downloadable model set.
Checkpoints record the encoder used to produce their conditioning features;
keep that encoder paired with the downstream model.

## Paths

Configurations resolve datasets, weights and output through `PSJEPA_CACHE`,
`PSJEPA_WEIGHTS` and `PSJEPA_OUTPUTS`. Run the Python entry points directly from
this checkout. Full training checkpoints retain optimizer and RNG state for resume.
