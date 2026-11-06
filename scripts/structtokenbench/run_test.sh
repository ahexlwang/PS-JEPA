#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
psjepa_require STB_ROOT "a StructTokenBench checkout" \
  "git clone https://github.com/KatarinaYuan/StructTokenBench; see docs/EVALUATION.md"
# A plain clone cannot run this encoder: the tokenizer classes and four config keys are
# additions this project made to the benchmark. Fail here, naming what is absent, rather than
# minutes into the job on a Hydra error about an unknown key.
"$PYTHON" "$PROJECT/scripts/structtokenbench/check_integration.py" --root "$STB_ROOT"
DATA_ROOT="${STB_DATA_ROOT:-$STB_ROOT/struct_token_bench_release_data}"
PDB_ROOT="${STB_PDB_ROOT:-$STB_ROOT/pdb_data/mmcif_files}"

RUN=${1:?Usage: run_test.sh RUN_DIR TASK REPRESENTATION}
TASK=${2:?Usage: run_test.sh RUN_DIR TASK REPRESENTATION}
REPRESENTATION=${3:?Usage: run_test.sh RUN_DIR TASK REPRESENTATION}
# `RUN` is either a training run directory -- relative to PSJEPA_OUTPUTS, or absolute -- or
# a released weights file, which is what someone evaluating the distributed encoder has. A
# released file has no run directory to write beside, so results go under PSJEPA_OUTPUTS in
# a directory named after it.
if [[ -f "$RUN" ]]; then
  CHECKPOINT_OVERRIDE="$RUN"
  RUN_DIR="$PSJEPA_OUTPUTS/benchmark/$(basename "${RUN%.pt}")"
elif [[ -f "$PSJEPA_WEIGHTS/$RUN" ]]; then
  CHECKPOINT_OVERRIDE="$PSJEPA_WEIGHTS/$RUN"
  RUN_DIR="$PSJEPA_OUTPUTS/benchmark/$(basename "${RUN%.pt}")"
else
  CHECKPOINT_OVERRIDE=""
  case "$RUN" in /*) RUN_DIR="$RUN" ;; *) RUN_DIR="$PSJEPA_OUTPUTS/$RUN" ;; esac
fi
CHECKPOINT="${CHECKPOINT_OVERRIDE:-$RUN_DIR/${PSJEPA_CHECKPOINT_NAME:-best.pt}}"
RUN_SLUG=$(echo "${RUN#outputs/}" | tr '/' '-')
# Results from a non-default checkpoint go to their own directory, so a
# best.pt run and a last.pt run of the same arm can be compared rather
# than one silently overwriting the other.
SUFFIX=""
[[ "${PSJEPA_CHECKPOINT_NAME:-best.pt}" == "best.pt" ]] || SUFFIX="_${PSJEPA_CHECKPOINT_NAME%.pt}"
# StructTokenBench builds its head at the tokenizer's own width -- its task configs ship
# `d_model: null # hidden dimention from the tokenizer embedding dim`, and each baseline
# supplies its own (FoldSeek 2, ProTokens 32, ESM3 128, AIDO 384, AminoAseed 1024). This was
# hardcoded to 1024, which zero-padded every 128- and 256-wide arm into AminoAseed's width;
# model_module.py:167 then added a 1024-d position embedding and LayerNormed across all of
# it, so the padding became a position-only channel no baseline receives. The width now
# comes from the checkpoint itself so it cannot disagree with the features, and it is tagged
# into both the results directory and the cache key: a d128 run must never land on a d1024
# run's numbers or read its cached features.
WIDTH=$("$PYTHON" "$PROJECT/scripts/checkpoint_width.py" "$CHECKPOINT")
[[ "$WIDTH" =~ ^[0-9]+$ ]] || { echo "bad width from checkpoint_width.py: $WIDTH" >&2; exit 2; }
SUFFIX="${SUFFIX}_d${WIDTH}"
READOUT="${PSJEPA_READOUT:-${PSJEPA_FLEX_READOUT:-teacher}}"
# Export both spellings so the loader sees the same choice this script tagged the paths with,
# whichever one the caller used.
export PSJEPA_READOUT="$READOUT" PSJEPA_FLEX_READOUT="$READOUT"
[[ "$READOUT" == "teacher" ]] || SUFFIX="${SUFFIX}_${READOUT}"
# The adapter checks this against what it actually emits; see _assert_expected_width.
export PSJEPA_EXPECTED_WIDTH="$WIDTH"
# A run that adds test splits (STB_EXTRA_TEST_SPLITS, e.g. the post-2025 BioLIP2 temporal
# sets) retrains the same head but writes different metrics, so it gets its own directory
# rather than overwriting the standard fold / superfamily result.
[[ -z "${STB_RESULT_TAG:-}" ]] || SUFFIX="${SUFFIX}_${STB_RESULT_TAG}"
OUTPUT="$RUN_DIR/structtokenbench_final${SUFFIX}/$REPRESENTATION/$TASK.json"

# A rerun (after a TIMEOUT or a port collision) recomputes only what is missing:
# a metrics file that already parses as JSON with the expected block is final.
# PSJEPA_FORCE_RERUN=1 disables the guard.
if [[ -z "${PSJEPA_FORCE_RERUN:-}" && -s "$OUTPUT" ]] && "$PYTHON" - "$OUTPUT" "test" <<'EOF'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
key = sys.argv[2]
ok = bool(data.get(key)) if key != "validation_" else any(k.startswith("validation_") for k in data)
sys.exit(0 if ok else 1)
EOF
then
  echo "[skip] $OUTPUT already complete"; exit 0
fi

LOG_ROOT="$RUN_DIR/structtokenbench_logs${SUFFIX}/$REPRESENTATION"
HEAD="$RUN_DIR/structtokenbench_heads${SUFFIX}/$REPRESENTATION/$TASK.ckpt"

# Fit and select heads on validation before evaluating the test split.
cd "$PROJECT"

# The twelve-task table lives in one place so every runner and the aggregator
# agree on which config and label field each paper column means.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stb_tasks.sh"
read -r CONFIG TARGET METRIC BATCH <<<"$(stb_task_spec "$TASK")"
[[ -n "$CONFIG" ]] || { echo "Unknown StructTokenBench task: $TASK" >&2; exit 2; }
MONITOR_ARGS=()
if [[ "$METRIC" == spearmanr ]]; then
  MONITOR_ARGS+=(
    "lightning.callbacks.checkpoint.monitor=validation_spearmanr"
    "lightning.callbacks.checkpoint.mode=max"
    "lightning.callbacks.checkpoint.filename='{epoch}-{step}-{validation_spearmanr:.4f}'"
  )
fi


case "$REPRESENTATION" in
  discrete_ids)
    TOKENIZER=WrappedPSJEPATokenizer
    USE_CONTINUOUS=false
    ;;
  continuous_semantic|quantized_semantic)
    TOKENIZER=WrappedPSJEPAContinuousTokenizer
    USE_CONTINUOUS=true
    ;;
  *) echo "Unknown representation: $REPRESENTATION" >&2; exit 2 ;;
esac

[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$HEAD")" "$LOG_ROOT"
cd "$STB_ROOT"
export PYTHONPATH="$PROJECT/src:$STB_ROOT/src"

# The head is trained and selected using validation only. post_selection_test
# then evaluates each named test split once with that locked head.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" ./src/script/run_supervised_task.py \
  --config-name="$CONFIG" \
  tokenizer="$TOKENIZER" \
  tokenizer_device=cuda \
  deepspeed_path=null \
  model.d_model="$WIDTH" \
  trainer.devices='[0]' \
  optimization.micro_batch_size="$BATCH" \
  optimization.optimizer.lr=0.001 \
  data.target_field="$TARGET" \
  data.use_continuous="$USE_CONTINUOUS" \
  experiment_name="${TASK}_psjepa_$(basename "$RUN")${SUFFIX}_${REPRESENTATION}" \
  run_name="${TASK}_psjepa_${REPRESENTATION}.ckpt" \
  save_dir_path="$HEAD" \
  default_data_dir="$DATA_ROOT" \
  data.pdb_data_dir="$PDB_ROOT" \
  trainer.default_root_dir="$LOG_ROOT" \
  tokenizer_ckpt_name="psjepa_${RUN_SLUG}${SUFFIX}_${REPRESENTATION}" \
  +psjepa_root="$PROJECT" \
  +psjepa_checkpoint_path="$CHECKPOINT" \
  +psjepa_representation="$REPRESENTATION" \
  +validation_only=true \
  +post_selection_test=true \
  +metrics_output_path="$OUTPUT" \
  "${MONITOR_ARGS[@]}"
