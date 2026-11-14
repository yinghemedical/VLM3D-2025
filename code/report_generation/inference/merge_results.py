#!/usr/bin/env python3
"""Merge per-shard MedM-VL report predictions in GT order."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gt-csv", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    gt_csv = Path(args.gt_csv) if args.gt_csv else run_dir / "gt.csv"
    out_json = Path(args.output_json) if args.output_json else run_dir / "predictions.json"
    out_csv = Path(args.output_csv) if args.output_csv else run_dir / "predictions.csv"

    shard_files = sorted(glob.glob(str(run_dir / "predictions.shard*.json")))
    if not shard_files:
        raise RuntimeError(f"no predictions.shard*.json under {run_dir}")

    by_id: dict[str, dict] = {}
    for path in shard_files:
        for rec in json.loads(Path(path).read_text()):
            sid = rec["series_id"]
            if sid not in by_id or (not by_id[sid].get("success") and rec.get("success")):
                by_id[sid] = rec

    gt = pd.read_csv(gt_csv)
    order = gt["series_id"].astype(str).tolist()
    merged = []
    for sid in order:
        rec = by_id.get(sid)
        if rec is None:
            rec = {
                "series_id": sid,
                "generated_report": "",
                "success": False,
                "error": "missing prediction",
            }
        merged.append(rec)

    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(merged).to_csv(out_csv, index=False)
    print(
        json.dumps(
            {
                "shards": len(shard_files),
                "gt_cases": len(order),
                "predictions": len(by_id),
                "merged": len(merged),
                "success": sum(bool(r.get("success")) for r in merged),
                "output_json": str(out_json),
                "output_csv": str(out_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
