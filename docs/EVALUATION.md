# Evaluation

## StructTokenBench

Install StructTokenBench and its released task data, then apply the integration
patch shipped with this repository:

```bash
git clone https://github.com/KatarinaYuan/StructTokenBench.git external/StructTokenBench
export STB_ROOT="$PWD/external/StructTokenBench"
git -C "$STB_ROOT" checkout "$(sed -n 's/^Base commit: //p' external/structtokenbench.patch)"
scripts/structtokenbench/apply_patch.sh "$STB_ROOT"
```

Follow StructTokenBench's installation and data preparation instructions. Its
dependencies are separate from this package's `benchmark` extra, which supplies
the adapter's mmCIF dependency.

```bash
scripts/structtokenbench/run_validation.sh encoder/psjepa_d512.pt homo continuous_semantic
scripts/structtokenbench/run_test.sh encoder/psjepa_d512.pt homo continuous_semantic
```

`scripts/structtokenbench/stb_tasks.sh` lists the twelve supported tasks. Run each
task to obtain the full 24-condition table. Readout defaults to the frozen EMA
teacher. Cache keys include the checkpoint, width, readout and representation.

## Metrics

The released evaluation uses supervised heads over frozen continuous features:
residue heads for functional sites and flexibility; masked mean pooling followed
by a classification head for homology. Heads are trained on the training split
and selected by validation performance before test evaluation.

The reported category means use 15 functional AUROC conditions, six flexibility
Spearman conditions and three homology macro-F1 conditions. All reported values
are multiplied by 100. The overall score equally weights the three category means.
The [24 recorded conditions](benchmark_results.csv) use StructTokenBench's released
labels and splits. Classification metrics follow its batchwise aggregation;
regression correlations use concatenated split outputs. Changing labels or
aggregation defines a different evaluation protocol.

## Geometry and designability

`scripts/evaluate_decoder.py` evaluates a trained coordinate decoder;
`scripts/sample.py` samples a trained latent prior. Specify the matching frozen
encoder and decoder checkpoints.

`scripts/designability_eval.py` designs sequences with ProteinMPNN and refolds
them with ESMFold. Install the `designability` extra and a separate ProteinMPNN
checkout, then set `PROTEINMPNN_DIR`. Run the script with `--help` for its inputs.
Keep failed designs in the evaluation denominator and report the number of
sequence attempts per generated backbone.
