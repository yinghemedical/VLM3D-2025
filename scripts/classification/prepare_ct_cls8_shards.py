#!/usr/bin/env python3
"""Create flat shard directories of NIfTI symlinks for CT_CLS8 inference.

CT_CLS8's dataset scans a directory recursively and uses the basename as the
case id, so a flat directory of symlinks is sufficient.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.gt_csv, dtype=str)
    if "image_path" not in df.columns:
        raise ValueError("gt.csv must contain image_path")
    if args.limit is not None:
        df = df.head(args.limit)
    rows = list(df[["series_id", "image_path"]].itertuples(index=False))

    out = Path(args.out_dir)
    shard_root = out / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_lists = [[] for _ in range(args.num_shards)]
    for i, row in enumerate(rows):
        shard_lists[i % args.num_shards].append(row)

    for rank, items in enumerate(shard_lists):
        d = shard_root / f"shard{rank:02d}"
        d.mkdir(parents=True, exist_ok=True)
        # Remove stale symlinks only.
        for old in d.iterdir():
            if old.is_symlink():
                old.unlink()
        lines = []
        for series_id, image_path in items:
            src = Path(image_path).resolve()
            if not src.is_file():
                raise FileNotFoundError(src)
            link = d / f"{series_id}.nii.gz"
            if not link.exists():
                os.symlink(src, link)
            lines.append(f"{series_id},{link}")
        (out / f"cases_shard{rank:02d}.csv").write_text(
            "series_id,image_path\n" + "\n".join(lines) + "\n"
        )
        print(f"shard {rank:02d}: {len(items)} cases -> {d}")

    (out / "shard_manifest.json").write_text(
        __import__("json").dumps(
            {
                "gt_csv": args.gt_csv,
                "num_cases": len(rows),
                "num_shards": args.num_shards,
                "shards": [
                    {"rank": r, "cases": len(x), "dir": str(shard_root / f"shard{r:02d}")}
                    for r, x in enumerate(shard_lists)
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
