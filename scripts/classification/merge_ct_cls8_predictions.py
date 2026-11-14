#!/usr/bin/env python3
"""Merge CT_CLS8 per-shard results.json files into a prediction CSV."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def clean_id(name: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", Path(str(name)).name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    gt = pd.read_csv(args.gt_csv, dtype=str)
    order = gt["series_id"].astype(str).tolist()

    rows: dict[str, dict] = {}
    files = sorted(glob.glob(str(run_dir / "predictions.shard*.json")))
    if not files:
        raise RuntimeError(f"no predictions.shard*.json under {run_dir}")
    for path in files:
        obj = json.loads(Path(path).read_text())
        for rec in obj.get("predictions", []):
            sid = clean_id(rec["input_image_name"])
            probs = rec.get("probabilities", {})
            rows[sid] = {c: float(probs.get(c, 0.0)) for c in LABELS}

    out = []
    for sid in order:
        rec = {"series_id": sid}
        if sid in rows:
            rec.update(rows[sid])
            rec["success"] = True
        else:
            rec.update({c: 0.0 for c in LABELS})
            rec["success"] = False
        out.append(rec)

    out_csv = Path(args.output_csv) if args.output_csv else run_dir / "ct_cls8_predictions.csv"
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(
        json.dumps(
            {
                "shards": len(files),
                "gt_cases": len(order),
                "predicted": len(rows),
                "output_csv": str(out_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
