#!/usr/bin/env python3
"Check that a StructTokenBench checkout can actually run this repository's encoder."
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# file -> (what it must contain, why the runners need it)
REQUIRED = {
    "src/tokenizer.py": [
        ("WrappedPSJEPATokenizer",
         "the discrete-id tokenizer the runners name as `tokenizer=`"),
        ("WrappedPSJEPAContinuousTokenizer",
         "the continuous tokenizer, which is what the released encoders need: they have no "
         "quantiser, so the discrete one refuses them"),
    ],
    "src/script/run_supervised_task.py": [
        ("psjepa_checkpoint_path",
         "the `+psjepa_checkpoint_path=` key every runner passes"),
        ("psjepa_representation",
         "the `+psjepa_representation=` key that selects continuous or discrete reading"),
        ("metrics_output_path",
         "where the runners expect the metrics JSON to be written; without it a finished "
         "job leaves nothing for the aggregation to read"),
        ("post_selection_test",
         "train and select the head on validation, then evaluate test once -- the protocol "
         "the published numbers were measured under"),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("STB_ROOT", "")),
        help="a StructTokenBench checkout (default: $STB_ROOT)",
    )
    args = parser.parse_args()

    if not args.root or str(args.root) == ".":
        print("error: pass --root or set STB_ROOT to a StructTokenBench checkout", file=sys.stderr)
        return 2
    if not (args.root / "src" / "script" / "run_supervised_task.py").is_file():
        print(f"error: {args.root} does not look like a StructTokenBench checkout", file=sys.stderr)
        return 2

    missing: list[str] = []
    for relative, requirements in REQUIRED.items():
        path = args.root / relative
        text = path.read_text(errors="ignore") if path.is_file() else ""
        if not text:
            missing.append(f"{relative}: the file itself is absent")
            continue
        for needle, why in requirements:
            if needle not in text:
                missing.append(f"{relative}: no `{needle}` — {why}")

    if not missing:
        print(f"ok: {args.root} carries the PS-JEPA integration")
        return 0

    print(f"{args.root} is missing the PS-JEPA integration:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print(
        "\nA plain clone of StructTokenBench does not have these; they are additions this\n"
        "project made to it. See docs/EVALUATION.md, 'Patching StructTokenBench'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
