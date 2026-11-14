#!/usr/bin/env python3
"""Create a one-row classification manifest for CT_CLS8."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    image = Path(args.image).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    pd.DataFrame(
        [{"series_id": args.case_id, "image_path": str(image)}]
    ).to_csv(args.out, index=False)
    print(f"{args.out}: {args.case_id} -> {image}")


if __name__ == "__main__":
    main()
