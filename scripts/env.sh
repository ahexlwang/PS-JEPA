#!/bin/bash
# Shared environment resolution for every shell entry point in this repository.
#
# Source it, do not execute it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Nothing here is site-specific. Each value comes from an environment variable when
# one is set and from a repository-relative default otherwise, so a fresh checkout on
# another machine needs no edits to tracked files. Put your own values in
# `.env.local` at the repository root (it is git-ignored); this file reads it.
#
#   PROJECT            the checkout root, derived from this script's own location
#   PYTHON             interpreter to run (default: whatever `python` resolves to)
#   PSJEPA_WEIGHTS    released checkpoints          (default: $PROJECT/weights)
#   PSJEPA_CACHE      preprocessed datasets         (default: $PROJECT/cache)
#   PSJEPA_OUTPUTS    run directories               (default: $PROJECT/outputs)
#
# Optional, only needed by the evaluations that call an external program. They have no
# default, and the helper `psjepa_require` below turns a missing one into a clear message
# instead of a path that does not exist:
#
#   STB_ROOT           a StructTokenBench checkout   (downstream benchmark)
#   PROTEINMPNN_DIR    a ProteinMPNN checkout        (designability)
#   FOLDSEEK           the foldseek executable       (novelty / diversity scoring)
set -euo pipefail

if [[ -z "${PROJECT:-}" ]]; then
  PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PROJECT

if [[ -f "$PROJECT/.env.local" ]]; then
  _psjepa_preset=()
  for _name in PYTHON PSJEPA_WEIGHTS PSJEPA_CACHE PSJEPA_OUTPUTS \
               BIOJEPA_WEIGHTS BIOJEPA_CACHE BIOJEPA_OUTPUTS \
               STB_ROOT PROTEINMPNN_DIR FOLDSEEK; do
    [[ -n "${!_name:-}" ]] && _psjepa_preset+=("$_name=${!_name}")
  done
  # shellcheck disable=SC1091
  source "$PROJECT/.env.local"
  for _entry in ${_psjepa_preset+"${_psjepa_preset[@]}"}; do
    export "${_entry?}"
  done
  unset _name _entry _psjepa_preset
fi

# Existing machine-local roots remain usable during the namespace migration.
export PSJEPA_WEIGHTS="${PSJEPA_WEIGHTS:-${BIOJEPA_WEIGHTS:-$PROJECT/weights}}"
export PSJEPA_CACHE="${PSJEPA_CACHE:-${BIOJEPA_CACHE:-$PROJECT/cache}}"
export PSJEPA_OUTPUTS="${PSJEPA_OUTPUTS:-${BIOJEPA_OUTPUTS:-$PROJECT/outputs}}"

while IFS= read -r _psjepa_legacy; do
  [[ "$_psjepa_legacy" == BIOJEPA_* ]] || continue
  _psjepa_current="PSJEPA_${_psjepa_legacy#BIOJEPA_}"
  if [[ ! -v "$_psjepa_current" ]]; then
    export "$_psjepa_current=${!_psjepa_legacy}"
  fi
done < <(compgen -e)
unset _psjepa_legacy _psjepa_current

export PYTHON="${PYTHON:-python}"
export PSJEPA_WEIGHTS="${PSJEPA_WEIGHTS:-$PROJECT/weights}"
export PSJEPA_CACHE="${PSJEPA_CACHE:-$PROJECT/cache}"
export PSJEPA_OUTPUTS="${PSJEPA_OUTPUTS:-$PROJECT/outputs}"
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts${PYTHONPATH:+:$PYTHONPATH}"

# psjepa_require VARNAME "what it is" "how to get it"
# Fails with an actionable message when an optional external dependency is not configured.
psjepa_require() {
  local name="$1" what="$2" how="$3"
  if [[ -z "${!name:-}" ]]; then
    echo "error: \$$name is not set. It must point at $what." >&2
    echo "       $how" >&2
    echo "       Set it in $PROJECT/.env.local or in your environment." >&2
    exit 78
  fi
}
