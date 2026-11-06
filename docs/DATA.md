# Data preparation

Datasets are downloaded from their providers and prepared locally. They are not
included in the source distribution. `PSJEPA_CACHE` defaults to `cache/`.

## Cache format

Each split contains:

| File | Shape / contents |
|---|---|
| `coords.npy` | float32 `[total_residues, 4, 3]`, atom order N/CA/C/O |
| `atom_mask.npy` | `[total_residues, 4]`, observed-atom indicators |
| `offsets.npy` | int64 chain boundaries, including the terminal offset |
| `metadata.npz` | Chain identifiers and labels |
| `residue_type.npy` | Optional residue identities; unused by the released encoder |

Arrays concatenate chains and are memory-mapped by `BackboneDataset`. Atom masks
mark missing coordinates; residue masks are created when chains are batched.

## AFDB pretraining cache

Use `scripts/build_afdb_cache.py` with the Hugging Face dataset that supplies the
AFDB domain parquet shards. The provider's split name and the local destination
can differ; for example, a provider split named `val` is stored locally as `validation`.

```bash
export PSJEPA_CACHE="${PSJEPA_CACHE:-$PWD/cache}"
python scripts/build_afdb_cache.py --repo "$AFDB_DATASET" --split train \
    --out "$PSJEPA_CACHE/afdb/train"
python scripts/build_afdb_cache.py --repo "$AFDB_DATASET" --split val \
    --out "$PSJEPA_CACHE/afdb/validation"
```

Set `AFDB_DATASET` to the provider's dataset repository ID. The `afdb` training
configuration reads the two local splits above. Input parquet requires `name`
and N/CA/C coordinates, with optional labels and confidence scores. The builder
infers oxygen coordinates from neighboring backbone geometry and writes
`summary.json` with chain counts and reconstruction statistics. Confidence
values are preserved from the supplied data.

## CATH cache

Prepare CATH backbone parquet with `scripts/build_cache.py`:

```bash
python scripts/build_cache.py --data-root /path/to/cath/processed \
    --cache-root "$PSJEPA_CACHE/cath_backbone_split" --splits train validation test
```

The `cath_backbone_split` cache is used by the downstream example configurations.
Run either builder with `--help` for input and filtering options. Use identical
splits and preprocessing when comparing checkpoints.

## Benchmark data and terms

StructTokenBench provides its own task data and split definitions; see
[evaluation](EVALUATION.md). AFDB, CATH and benchmark datasets remain subject to
their providers' terms. The source repository does not redistribute those datasets.
