#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

TARGET="${1:-${STB_ROOT:-}}"
PATCH="$PROJECT/external/structtokenbench.patch"

[[ -n "$TARGET" ]] || { echo "usage: $0 /path/to/StructTokenBench (or set STB_ROOT)" >&2; exit 2; }
[[ -f "$PATCH" ]] || { echo "missing patch: $PATCH" >&2; exit 2; }
[[ -f "$TARGET/src/script/run_supervised_task.py" ]] || {
  echo "$TARGET does not look like a StructTokenBench checkout" >&2; exit 2; }

BASE=$(sed -n 's/^Base commit: *//p' "$PATCH" | head -1)
if [[ -d "$TARGET/.git" && -n "$BASE" ]]; then
  CURRENT=$(git -C "$TARGET" rev-parse HEAD)
  if [[ "$CURRENT" != "$BASE" ]]; then
    echo "warning: $TARGET is at $CURRENT, the patch was made against $BASE" >&2
    if [[ "${PSJEPA_PATCH_ANY_COMMIT:-}" != "1" ]]; then
      echo "         checkout that commit, or set PSJEPA_PATCH_ANY_COMMIT=1 to try anyway" >&2
      exit 1
    fi
  fi
fi

# Strip this file's own header; everything after the marker is the diff.
BODY=$(mktemp); trap 'rm -f "$BODY"' EXIT
sed -n '/^---- 8< ---- patch begins/,$p' "$PATCH" | tail -n +2 > "$BODY"

if git -C "$TARGET" apply --check "$BODY" 2>/dev/null; then
  git -C "$TARGET" apply "$BODY"
  echo "applied the PS-JEPA integration to $TARGET"
elif git -C "$TARGET" apply --reverse --check "$BODY" 2>/dev/null; then
  echo "$TARGET already carries this patch; nothing to do"
else
  echo "error: the patch does not apply cleanly to $TARGET" >&2
  echo "       run: git -C '$TARGET' apply --check --verbose <the diff> to see why" >&2
  exit 1
fi

"$PYTHON" "$HERE/check_integration.py" --root "$TARGET"
