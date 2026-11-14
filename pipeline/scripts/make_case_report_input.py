#!/usr/bin/env python3
"""Build one MedM-VL report input from a CT_CLS8 prediction row."""
from __future__ import annotations

import argparse
import json
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

PROMPT = (
    "<image3d>\n"
    "Generate radiology report for the CT."
    "This patient may have the following diseases: {pos}\n"
    "This patient does not have the following diseases: {neg}\n"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--classification-csv", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    image = str(Path(args.image).resolve())
    df = pd.read_csv(args.classification_csv, dtype={"series_id": str})
    row = df[df["series_id"].astype(str) == str(args.case_id)]
    if len(row) != 1:
        raise RuntimeError(f"expected one row for {args.case_id}, got {len(row)}")
    row = row.iloc[0]
    positive = [c for c in LABELS if float(row[c]) >= args.threshold]
    negative = [c for c in LABELS if c not in positive]

    item = {
        "id": args.case_id,
        "series_id": args.case_id,
        "image3d": [image],
        "conversations": [
            {
                "from": "human",
                "value": PROMPT.format(
                    pos=", ".join(positive) if positive else "None",
                    neg=", ".join(negative) if negative else "None",
                ),
            }
        ],
    }
    Path(args.out).write_text(json.dumps([item], indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"positive": positive, "negative": negative}, indent=2))


if __name__ == "__main__":
    main()
