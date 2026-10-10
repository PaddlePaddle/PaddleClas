#!/usr/bin/env python3
"""Adapt PaddleClas 2.6 TIPC's model-subdirectory output convention."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

MODEL_NAME = "MambaVision_T"


def main() -> None:
    for index, argument in enumerate(sys.argv):
        prefix = "Global.output_dir="
        if not argument.startswith(prefix):
            continue
        output_dir = Path(argument[len(prefix):])
        if output_dir.name != MODEL_NAME:
            sys.argv[index] = prefix + str(output_dir / MODEL_NAME)
        break

    train_entry = Path(__file__).resolve().parents[1] / "tools" / "train.py"
    if not train_entry.is_file():
        raise SystemExit(
            f"PaddleClas tools/train.py was not found: {train_entry}")
    sys.argv[0] = str(train_entry)
    runpy.run_path(str(train_entry), run_name="__main__")


if __name__ == "__main__":
    main()
