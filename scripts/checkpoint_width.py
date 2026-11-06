#!/usr/bin/env python3
'Read the native encoder width used by a downstream prediction head.'
from __future__ import annotations

from psjepa.environment import getenv

import argparse
import os
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        width = int(state["config"]["model"]["d_model"])
    except (KeyError, TypeError) as error:
        raise SystemExit(f"cannot read d_model from {args.checkpoint}: {error}")

    pad = getenv("PSJEPA_PAD_WIDTH")
    if pad:
        pad = int(pad)
        if pad < width:
            raise SystemExit(f"PSJEPA_PAD_WIDTH={pad} is narrower than the encoder's {width}")
        width = pad
    print(width)


if __name__ == "__main__":
    main()
