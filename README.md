# PS-JEPA

Protein Structure Joint-Embedding Predictive Architecture.

PS-JEPA maps N/CA/C/O backbone coordinates to continuous residue representations
using a GVP encoder and eight bidirectional Mamba layers. The encoder predicts
masked EMA-target features without amino acid identity or coordinate reconstruction.

## Install

Linux, Python 3.11 and a CUDA toolchain compatible with PyTorch are required for
Mamba training. The tested stack uses PyTorch 2.9.1 and CUDA 12.8.

```bash
git clone https://github.com/ahexlwang/PS-JEPA.git
cd PS-JEPA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cuda,benchmark]"
python tools/fetch_weights.py
```

Weights are downloaded and checksum-verified using `weights/MANIFEST.tsv`.
The encoder checkpoint is `weights/encoder/psjepa_d512.pt`.
A base install without `mamba-ssm` cannot run the released encoders.

## Encode

```bash
python examples/encode_backbone.py --pdb protein.pdb --device cuda
```

```python
from psjepa.sota import encoder_checkpoint
from psjepa.tokenization.semantic_tokenizer import SemanticTokenizer

encoder = SemanticTokenizer.from_checkpoint(encoder_checkpoint(), device="cuda")
features = encoder.encode_batch(coords, atom_mask, residue_mask)["continuous_semantic"]
```

Inputs are `coords [B,L,4,3]` in N/CA/C/O order, boolean `atom_mask [B,L,4]`
and `residue_mask [B,L]`. Output is `[B,L,512]`; padding is zeroed.
The default readout is the frozen EMA encoder with 13,981,939 parameters.

## Train and evaluate

Prepare the data following [DATA.md](docs/DATA.md), then run:

```bash
python scripts/train_encoder.py --config configs/encoder/sota_d512.yaml --resume auto
```

[Training](docs/TRAINING.md) covers encoder, coordinate decoder and latent prior
entry points. [Evaluation](docs/EVALUATION.md) covers frozen-feature benchmarking
and generated-backbone scoring. Downloadable weights currently include the encoder.

| Encoder | Functional | Flexibility | Homology | Overall |
|---|---:|---:|---:|---:|
| PS-JEPA | 80.99 | 56.29 | 73.03 | 70.10 |

StructTokenBench scores are metric × 100. Overall equally weights the three
category means. [All 24 conditions](docs/benchmark_results.csv).

## Paths

`PSJEPA_WEIGHTS`, `PSJEPA_CACHE` and `PSJEPA_OUTPUTS` default to `weights/`,
`cache/` and `outputs/`. Set environment variables to relocate them. Benchmark
shell scripts also read optional settings from `.env.local.example` copied to
`.env.local`.

## Development

```bash
python -m pip install -e . --group dev
pytest -q
```

CPU tests use small GRU-backed models; released Mamba weights require the CUDA
extra. Data and checkpoints are distributed separately from source code.

## Architecture and license

See [architecture](docs/ARCHITECTURE.md) for the encoder and objective.
Source code and released encoder weights are MIT licensed:
[code](LICENSE), [weights](weights/LICENSE). External package and benchmark-patch
notices are in [NOTICE](NOTICE) and `LICENSES/`. Datasets retain provider terms.
